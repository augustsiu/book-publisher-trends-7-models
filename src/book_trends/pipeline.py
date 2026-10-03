from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .data import generate_sample, genre_monthly, load_and_aggregate
from .eda import eda_summary, monthly_seasonality, residual_summary
from .models import seasonal_naive
from .validation import MODEL_FUNCS, backtest, holdout_start, tune_xgboost

FOLDS = 3


def run(input_path: Path, output: Path, models: list[str], horizon: int, tune: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    data = load_and_aggregate(input_path)
    processed = Path("data/processed/monthly_titles.csv")
    processed.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(processed, index=False)
    # The app reads only from the artifact directory, so keep a copy of the history there.
    data.to_csv(output / "history.csv", index=False)
    genres = genre_monthly(input_path)
    if genres is not None:
        genres.to_csv(output / "genre_monthly.csv", index=False)
    eda_summary(data).to_csv(output / "eda_summary.csv", index=False)
    monthly_seasonality(data).to_csv(output / "monthly_seasonality.csv", index=False)
    backtest_horizon = min(6, horizon)
    params = {}
    if tune and "xgboost" in models:
        # Tune only on months before the first backtest fold so scores stay out-of-sample.
        cutoff = holdout_start(len(data), horizon=backtest_horizon, folds=FOLDS)
        params["xgboost"], tuning = tune_xgboost(data.iloc[:cutoff])
        tuning.to_csv(output / "tuning_results.csv", index=False)
    scores, predictions = backtest(
        data, models, horizon=backtest_horizon, folds=FOLDS, params=params
    )
    scores.to_csv(output / "metrics.csv", index=False)
    predictions.to_csv(output / "backtest_predictions.csv", index=False)
    residuals = residual_summary(predictions)
    residuals.to_csv(output / "residual_summary.csv", index=False)
    champion = scores.groupby("model").smape.mean().idxmin()
    dates = pd.Series(pd.date_range(data.month.max() + pd.offsets.MonthBegin(), periods=horizon, freq="MS"))
    forecasts = []
    for name in models:
        pred = (seasonal_naive(data, dates) if name == "seasonal_naive"
                else MODEL_FUNCS[name](data, dates, params.get(name)))
        # Empirical 80% band from backtest errors; comparable across model families.
        width = float(residuals.loc[residuals["model"] == name, "abs_error_p80"].iloc[0])
        forecasts.append(pd.DataFrame({
            "month": dates,
            "forecast": pred,
            "lower_80": (pred - width).clip(min=0),
            "upper_80": pred + width,
            "model": name,
        }))
    all_forecasts = pd.concat(forecasts, ignore_index=True)
    all_forecasts.to_csv(output / "all_model_forecasts.csv", index=False)
    all_forecasts.query("model == @champion").to_csv(output / "forecast.csv", index=False)
    metadata = {"run_at_utc": datetime.now(timezone.utc).isoformat(), "rows": len(data),
                "horizon": horizon, "backtest_horizon": backtest_horizon, "folds": FOLDS,
                "models": models, "champion": champion, "parameters": params}
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(f"Champion: {champion}; artifacts written to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast monthly publishing trends")
    parser.add_argument("--input", type=Path, default=Path("data/raw/sample_books.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--generate-sample", action="store_true")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["seasonal_naive", "arima", "sarima", "ets", "prophet", "xgboost", "lstm"],
        choices=MODEL_FUNCS,
    )
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    if args.generate_sample:
        generate_sample(args.input)
    run(args.input, args.output, args.models, args.horizon, args.tune)


if __name__ == "__main__":
    main()
