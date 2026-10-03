from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_sample(path: str | Path, seed: int = 42) -> pd.DataFrame:
    """Create deterministic, portfolio-safe monthly publishing data."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", "2024-12-01", freq="MS")
    genres = {"Fiction": 1.25, "Nonfiction": 1.0, "Children": 0.8}
    rows = []
    for genre, scale in genres.items():
        t = np.arange(len(dates))
        seasonal = 18 * np.sin(2 * np.pi * (t - 1) / 12) + 10 * np.cos(4 * np.pi * t / 12)
        counts = np.maximum(1, 110 * scale + 0.45 * t * scale + seasonal + rng.normal(0, 8, len(t)))
        rows.extend(zip(dates, [genre] * len(dates), np.rint(counts).astype(int)))
    df = pd.DataFrame(rows, columns=["publication_date", "genre", "titles_published"])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def _read_dated(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if "publication_date" not in raw:
        raise ValueError("Input must contain publication_date")
    raw["publication_date"] = pd.to_datetime(raw["publication_date"], errors="coerce")
    if raw["publication_date"].isna().any():
        raise ValueError("publication_date contains invalid values")
    if "titles_published" in raw and raw["titles_published"].lt(0).any():
        raise ValueError("titles_published contains negative values")
    raw["month"] = raw["publication_date"].dt.to_period("M").dt.to_timestamp()
    return raw


def load_and_aggregate(path: str | Path) -> pd.DataFrame:
    raw = _read_dated(path)
    if "titles_published" in raw:
        monthly = raw.groupby("month", as_index=False)["titles_published"].sum()
    elif "title_id" in raw:
        monthly = raw.groupby("month", as_index=False)["title_id"].nunique()
        monthly = monthly.rename(columns={"title_id": "titles_published"})
    else:
        raise ValueError("Input needs titles_published or title_id")
    full = pd.DataFrame({"month": pd.date_range(monthly.month.min(), monthly.month.max(), freq="MS")})
    monthly = full.merge(monthly, on="month", how="left").fillna({"titles_published": 0})
    monthly["titles_published"] = monthly["titles_published"].astype(float)
    return monthly.sort_values("month").reset_index(drop=True)



def genre_monthly(path: str | Path) -> pd.DataFrame | None:
    """Monthly titles per genre for EDA, or None when the input has no genre column."""
    raw = _read_dated(path)
    if "genre" not in raw:
        return None
    raw["genre"] = raw["genre"].fillna("Unknown")
    if "titles_published" in raw:
        out = raw.groupby(["month", "genre"], as_index=False)["titles_published"].sum()
    else:
        out = raw.groupby(["month", "genre"], as_index=False)["title_id"].nunique()
        out = out.rename(columns={"title_id": "titles_published"})
    return out.sort_values(["month", "genre"]).reset_index(drop=True)
