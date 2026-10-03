"""Smoke tests for the optional heavy models; skipped unless the models extra is installed."""
import numpy as np
import pandas as pd
import pytest

from book_trends.validation import MODEL_FUNCS


@pytest.fixture
def series():
    t = np.arange(72)
    return pd.DataFrame({
        "month": pd.date_range("2018-01-01", periods=72, freq="MS"),
        "titles_published": 100 + t + 10 * np.sin(2 * np.pi * t / 12),
    })


@pytest.mark.parametrize("name,module", [
    ("arima", "statsmodels"),
    ("sarima", "statsmodels"),
    ("ets", "statsmodels"),
    ("xgboost", "xgboost"),
])
def test_model_returns_nonnegative_forecast_of_requested_length(series, name, module):
    pytest.importorskip(module)
    future = pd.Series(pd.date_range("2024-01-01", periods=6, freq="MS"))
    pred = MODEL_FUNCS[name](series, future, None)
    assert pred.shape == (6,)
    assert np.isfinite(pred).all() and (pred >= 0).all()
