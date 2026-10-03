import pandas as pd

from book_trends.data import generate_sample, load_and_aggregate
from book_trends.pipeline import run
from book_trends.validation import backtest


def test_baseline_pipeline(tmp_path):
    path = tmp_path / "sample.csv"
    generate_sample(path)
    scores, predictions = backtest(load_and_aggregate(path), ["seasonal_naive"])
    assert len(scores) == 3
    assert len(predictions) == 18
    assert scores.smape.notna().all()


def test_pipeline_writes_forecast_for_each_requested_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "sample.csv"
    output = tmp_path / "artifacts"
    generate_sample(path)
    run(path, output, ["seasonal_naive"], horizon=3, tune=False)
    saved = pd.read_csv(output / "all_model_forecasts.csv")
    assert len(saved) == 3
    assert saved["model"].unique().tolist() == ["seasonal_naive"]


def test_pipeline_writes_app_artifacts_with_intervals(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "sample.csv"
    output = tmp_path / "artifacts"
    generate_sample(path)
    run(path, output, ["seasonal_naive"], horizon=6, tune=False)
    for name in ["history.csv", "genre_monthly.csv", "residual_summary.csv", "forecast.csv"]:
        assert (output / name).exists(), name
    forecast = pd.read_csv(output / "forecast.csv")
    assert (forecast.lower_80 <= forecast.forecast).all()
    assert (forecast.forecast <= forecast.upper_80).all()
