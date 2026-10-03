from __future__ import annotations

import pandas as pd


def eda_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    """Return compact EDA statistics that are easy to version and discuss."""
    y = monthly.set_index("month")["titles_published"]
    return pd.DataFrame({"value": {
        "start_month": y.index.min().date().isoformat(),
        "end_month": y.index.max().date().isoformat(),
        "months": len(y),
        "missing_target": int(y.isna().sum()),
        "zero_months": int(y.eq(0).sum()),
        "mean": round(float(y.mean()), 2),
        "std": round(float(y.std()), 2),
        "min": round(float(y.min()), 2),
        "max": round(float(y.max()), 2),
        "lag_12_autocorrelation": round(float(y.autocorr(12)), 3),
    }}).rename_axis("statistic").reset_index()


def monthly_seasonality(monthly: pd.DataFrame) -> pd.DataFrame:
    """Calendar-month profile in January-to-December order."""
    stats = (
        monthly.groupby(monthly["month"].dt.month)["titles_published"]
        .agg(["mean", "median", "std"])
        .rename_axis("month_number")
        .reset_index()
    )
    stats.insert(1, "calendar_month", pd.to_datetime(stats["month_number"], format="%m").dt.month_name())
    return stats


def residual_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Backtest error diagnostics per model: bias, spread, and the worst calendar month."""
    data = predictions.assign(
        error=predictions["actual"] - predictions["prediction"],
        calendar_month=pd.to_datetime(predictions["month"]).dt.month_name(),
    )
    rows = []
    for model, group in data.groupby("model"):
        by_month = group.groupby("calendar_month")["error"].apply(lambda e: e.abs().mean())
        rows.append({
            "model": model,
            "mean_error": group["error"].mean(),
            "error_std": group["error"].std(),
            "abs_error_p80": group["error"].abs().quantile(0.8),
            "share_under_forecast": group["error"].gt(0).mean(),
            "worst_calendar_month": by_month.idxmax(),
            "worst_month_mae": by_month.max(),
        })
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)
