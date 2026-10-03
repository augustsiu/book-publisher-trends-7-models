from __future__ import annotations

import numpy as np
import pandas as pd

from .features import feature_columns, make_features


def seasonal_naive(train: pd.DataFrame, future_dates: pd.Series) -> np.ndarray:
    history = train["titles_published"].to_numpy()
    return np.array([history[-12 + (i % 12)] for i in range(len(future_dates))], dtype=float)


def arima_forecast(train: pd.DataFrame, future_dates: pd.Series, params=None) -> np.ndarray:
    """Non-seasonal ARIMA; useful as a classical statistical benchmark."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError as exc:
        raise ImportError("Install the models extra: pip install -e '.[models]'") from exc
    order = (params or {}).get("order", (2, 1, 2))
    fitted = ARIMA(train["titles_published"], order=order).fit()
    return np.maximum(0, np.asarray(fitted.forecast(len(future_dates)), dtype=float))


def sarima_forecast(train: pd.DataFrame, future_dates: pd.Series, params=None) -> np.ndarray:
    """Seasonal ARIMA with annual seasonality for monthly publishing counts."""
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError as exc:
        raise ImportError("Install the models extra: pip install -e '.[models]'") from exc
    p = params or {}
    fitted = SARIMAX(
        train["titles_published"],
        order=p.get("order", (1, 1, 1)),
        seasonal_order=p.get("seasonal_order", (1, 1, 1, 12)),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    return np.maximum(0, np.asarray(fitted.forecast(len(future_dates)), dtype=float))


def ets_forecast(train: pd.DataFrame, future_dates: pd.Series, params=None) -> np.ndarray:
    """Holt-Winters exponential smoothing with trend and annual seasonality."""
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ImportError as exc:
        raise ImportError("Install the models extra: pip install -e '.[models]'") from exc
    p = {"trend": "add", "seasonal": "add", "seasonal_periods": 12, "damped_trend": True}
    p.update(params or {})
    fitted = ExponentialSmoothing(train["titles_published"], **p).fit(optimized=True)
    return np.maximum(0, np.asarray(fitted.forecast(len(future_dates)), dtype=float))


def prophet_forecast(train: pd.DataFrame, future_dates: pd.Series, params=None) -> np.ndarray:
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise ImportError("Install the models extra: pip install -e '.[models]'") from exc
    p = {"yearly_seasonality": True, "weekly_seasonality": False, "daily_seasonality": False}
    p.update(params or {})
    model = Prophet(**p).fit(train.rename(columns={"month": "ds", "titles_published": "y"}))
    return model.predict(pd.DataFrame({"ds": future_dates}))["yhat"].clip(lower=0).to_numpy()


def xgb_recursive(train: pd.DataFrame, future_dates: pd.Series, params=None) -> np.ndarray:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("Install the models extra: pip install -e '.[models]'") from exc
    defaults = {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.04,
                "subsample": 0.9, "colsample_bytree": 0.9, "random_state": 42}
    defaults.update(params or {})
    history = train.copy()
    feat = make_features(history).dropna()
    cols = feature_columns(feat)
    model = XGBRegressor(**defaults).fit(feat[cols], feat["titles_published"])
    predictions = []
    for date in pd.to_datetime(future_dates):
        candidate = pd.concat([history, pd.DataFrame({"month": [date], "titles_published": [np.nan]})], ignore_index=True)
        row = make_features(candidate).iloc[[-1]]
        pred = max(0.0, float(model.predict(row[cols])[0]))
        predictions.append(pred)
        history = pd.concat([history, pd.DataFrame({"month": [date], "titles_published": [pred]})], ignore_index=True)
    return np.asarray(predictions)


def lstm_recursive(train: pd.DataFrame, future_dates: pd.Series, params=None) -> np.ndarray:
    """Forecast recursively with an LSTM trained on normalized monthly sequences."""
    try:
        import os

        os.environ["KERAS_BACKEND"] = "jax"
        import keras
        from sklearn.preprocessing import MinMaxScaler
    except ImportError as exc:
        raise ImportError("Install the models extra: pip install -e '.[models]'") from exc

    p = {
        "lookback": 12,
        "units": 32,
        "dropout": 0.1,
        "epochs": 200,
        "batch_size": 16,
        "learning_rate": 0.005,
        "patience": 12,
        "random_state": 42,
    }
    p.update(params or {})
    keras.utils.set_random_seed(int(p["random_state"]))

    values = train[["titles_published"]].to_numpy(dtype=np.float32)
    lookback = int(p["lookback"])
    if len(values) <= lookback:
        raise ValueError(f"LSTM requires more than {lookback} training observations")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values).astype(np.float32)
    x_train = np.asarray(
        [scaled[i - lookback:i] for i in range(lookback, len(scaled))], dtype=np.float32
    )
    y_train = np.asarray(
        [scaled[i, 0] for i in range(lookback, len(scaled))], dtype=np.float32
    ).reshape(-1, 1)

    model = keras.Sequential([
        keras.layers.Input(shape=(lookback, 1)),
        keras.layers.LSTM(int(p["units"])),
        keras.layers.Dropout(float(p["dropout"])),
        keras.layers.Dense(1),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(p["learning_rate"])),
        loss="mse",
    )
    model.fit(
        x_train,
        y_train,
        epochs=int(p["epochs"]),
        batch_size=int(p["batch_size"]),
        validation_split=0.15,
        shuffle=False,
        verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=int(p["patience"]), restore_best_weights=True
        )],
    )

    sequence = scaled[-lookback:, 0].tolist()
    predictions = []
    for _ in future_dates:
        x_next = np.asarray(sequence[-lookback:], dtype=np.float32).reshape(1, lookback, 1)
        next_scaled = float(model.predict(x_next, verbose=0)[0, 0])
        next_value = max(0.0, float(scaler.inverse_transform([[next_scaled]])[0, 0]))
        predictions.append(next_value)
        sequence.append(next_scaled)
    return np.asarray(predictions)
