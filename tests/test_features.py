import numpy as np
import pandas as pd

from book_trends.features import make_features
from book_trends.metrics import forecast_metrics


def test_features_do_not_use_current_target():
    df = pd.DataFrame({"month": pd.date_range("2020-01-01", periods=24, freq="MS"),
                       "titles_published": np.arange(24)})
    features = make_features(df)
    assert features.loc[12, "lag_12"] == 0
    assert features.loc[12, "rolling_mean_3"] == 10


def test_metrics_are_zero_for_perfect_forecast():
    assert all(value == 0 for value in forecast_metrics([1, 2], [1, 2]).values())

