import numpy as np


def forecast_metrics(y_true, y_pred) -> dict[str, float]:
    actual = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    error = actual - pred
    nonzero = np.abs(actual) > 1e-8
    mape = np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100 if nonzero.any() else np.nan
    denom = np.abs(actual) + np.abs(pred)
    smape = np.mean(np.divide(2 * np.abs(error), denom, out=np.zeros_like(error), where=denom > 0)) * 100
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape": float(mape),
        "smape": float(smape),
    }

