from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = (1, 2, 3, 6, 12)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("month").reset_index(drop=True)
    y = out["titles_published"]
    for lag in LAGS:
        out[f"lag_{lag}"] = y.shift(lag)
    for window in (3, 6, 12):
        out[f"rolling_mean_{window}"] = y.shift(1).rolling(window).mean()
    out["trend"] = np.arange(len(out))
    out["month_sin"] = np.sin(2 * np.pi * out["month"].dt.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"].dt.month / 12)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in {"month", "titles_published"}]

