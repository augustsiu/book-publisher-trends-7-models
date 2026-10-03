from __future__ import annotations

import itertools

import pandas as pd

from .metrics import forecast_metrics
from .models import (
    arima_forecast,
    ets_forecast,
    lstm_recursive,
    prophet_forecast,
    sarima_forecast,
    seasonal_naive,
    xgb_recursive,
)

MODEL_FUNCS = {
    "seasonal_naive": seasonal_naive,
    "arima": arima_forecast,
    "sarima": sarima_forecast,
    "ets": ets_forecast,
    "prophet": prophet_forecast,
    "xgboost": xgb_recursive,
    "lstm": lstm_recursive,
}


def expanding_splits(n: int, min_train: int = 48, horizon: int = 6, folds: int = 3):
    starts = list(range(n - horizon * folds, n, horizon))
    return [(range(s), range(s, min(s + horizon, n))) for s in starts if s >= min_train]


def holdout_start(n: int, horizon: int = 6, folds: int = 3) -> int:
    """Index of the first month used by backtest folds; tuning must stop before it."""
    return n - horizon * folds


def backtest(df: pd.DataFrame, models: list[str], horizon: int = 6, folds: int = 3,
             params: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    splits = expanding_splits(len(df), horizon=horizon, folds=folds)
    if not splits:
        raise ValueError(
            f"Series has {len(df)} months; backtesting needs at least "
            f"{48 + horizon * folds} (48 training + {folds} folds of {horizon})"
        )
    scores, predictions = [], []
    for fold, (train_idx, test_idx) in enumerate(splits, 1):
        train, test = df.iloc[list(train_idx)], df.iloc[list(test_idx)]
        for name in models:
            pred = MODEL_FUNCS[name](train, test["month"], (params or {}).get(name)) if name != "seasonal_naive" else seasonal_naive(train, test["month"])
            scores.append({"fold": fold, "model": name, **forecast_metrics(test["titles_published"], pred)})
            predictions.extend({"fold": fold, "model": name, "month": m, "actual": a, "prediction": p}
                               for m, a, p in zip(test["month"], test["titles_published"], pred))
    return pd.DataFrame(scores), pd.DataFrame(predictions)


def tune_xgboost(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    grid = {"max_depth": [2, 3, 5], "learning_rate": [0.03, 0.08], "n_estimators": [200, 400]}
    trials = []
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid, values))
        scores, _ = backtest(df, ["xgboost"], horizon=6, folds=2, params={"xgboost": params})
        trials.append({**params, "mean_smape": scores.smape.mean()})
    results = pd.DataFrame(trials).sort_values("mean_smape")
    best = results.iloc[0].drop("mean_smape").to_dict()
    best["max_depth"] = int(best["max_depth"])
    best["n_estimators"] = int(best["n_estimators"])
    best["learning_rate"] = float(best["learning_rate"])
    return best, results
