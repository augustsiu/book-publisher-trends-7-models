import numpy as np
import pandas as pd
import pytest

from book_trends.data import generate_sample, load_and_aggregate
from book_trends.eda import monthly_seasonality, residual_summary
from book_trends.validation import backtest, expanding_splits, holdout_start


def test_splits_train_strictly_before_test():
    for train, test in expanding_splits(120):
        assert max(train) < min(test)
        assert min(train) == 0


def test_tuning_window_ends_before_first_backtest_fold():
    first_test = min(expanding_splits(120, horizon=6, folds=3)[0][1])
    assert holdout_start(120, horizon=6, folds=3) == first_test


def test_short_series_raises_clear_error():
    short = pd.DataFrame({"month": pd.date_range("2020-01-01", periods=30, freq="MS"),
                          "titles_published": np.arange(30, dtype=float)})
    with pytest.raises(ValueError, match="backtesting needs"):
        backtest(short, ["seasonal_naive"])


def test_seasonality_is_in_calendar_order(tmp_path):
    path = tmp_path / "sample.csv"
    generate_sample(path)
    stats = monthly_seasonality(load_and_aggregate(path))
    assert stats["calendar_month"].tolist()[:3] == ["January", "February", "March"]
    assert len(stats) == 12


def test_residual_summary_reports_bias_direction():
    preds = pd.DataFrame({
        "model": ["m"] * 4,
        "month": pd.date_range("2024-01-01", periods=4, freq="MS"),
        "actual": [10.0, 10.0, 10.0, 30.0],
        "prediction": [8.0, 8.0, 8.0, 8.0],
    })
    row = residual_summary(preds).iloc[0]
    assert row["mean_error"] > 0
    assert row["share_under_forecast"] == 1.0
    assert row["worst_calendar_month"] == "April"
