from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from book_trends.features import make_features
from book_trends.validation import expanding_splits

st.set_page_config(page_title="Book Publisher Trends (7 Models)", page_icon="📚", layout="wide")
st.title("📚 Book Publisher Trends (7 Models)")
st.caption("Seven forecasting approaches evaluated with chronological rolling-origin validation")

artifact_dir = Path("artifacts")
required = ["metrics.csv", "backtest_predictions.csv", "all_model_forecasts.csv", "history.csv"]
if any(not (artifact_dir / name).exists() for name in required):
    st.warning("Run the full pipeline first: `python -m book_trends.pipeline --generate-sample --tune`")
    st.stop()

metrics = pd.read_csv(artifact_dir / "metrics.csv")
backtest = pd.read_csv(artifact_dir / "backtest_predictions.csv", parse_dates=["month"])
forecasts = pd.read_csv(artifact_dir / "all_model_forecasts.csv", parse_dates=["month"])
history = pd.read_csv(artifact_dir / "history.csv", parse_dates=["month"])


def optional_csv(name: str, **kwargs) -> pd.DataFrame | None:
    path = artifact_dir / name
    return pd.read_csv(path, **kwargs) if path.exists() else None


eda = optional_csv("eda_summary.csv")
seasonality = optional_csv("monthly_seasonality.csv")
genres = optional_csv("genre_monthly.csv", parse_dates=["month"])
residuals = optional_csv("residual_summary.csv")

summary = (
    metrics.groupby("model", as_index=False)[["mae", "rmse", "mape", "smape"]]
    .mean()
    .sort_values("smape")
)
summary.insert(0, "rank", range(1, len(summary) + 1))
champion = summary.iloc[0]["model"]

left, right = st.columns([1, 3])
left.metric("Champion model", champion.replace("_", " ").title())
left.metric("Champion mean sMAPE", f"{summary.iloc[0]['smape']:.2f}%")
right.markdown(
    "The champion has the lowest mean **sMAPE** across three expanding-window folds. "
    "The LSTM is a deep-learning challenger; 120 months is a small neural-network dataset, "
    "so it should earn its place against the simpler benchmarks."
)

st.subheader("Exploratory analysis")
eda_tabs = st.tabs(["Trend", "Seasonality", "Genre mix", "Summary statistics"])
with eda_tabs[0]:
    trend = history.assign(
        rolling_12=history["titles_published"].rolling(12).mean()
    ).melt(id_vars="month", value_vars=["titles_published", "rolling_12"],
           var_name="series", value_name="titles")
    st.plotly_chart(
        px.line(trend, x="month", y="titles", color="series",
                labels={"titles": "Titles published", "month": "Month"}),
        width="stretch",
        key="eda-trend",
    )
    st.caption("The 12-month rolling mean removes annual seasonality to expose the trend.")
with eda_tabs[1]:
    if seasonality is not None:
        st.plotly_chart(
            px.bar(seasonality, x="calendar_month", y="mean", error_y="std",
                   labels={"calendar_month": "Calendar month", "mean": "Mean titles"}),
            width="stretch",
            key="eda-seasonality",
        )
        st.caption("Bars show the mean per calendar month; whiskers show one standard deviation.")
    else:
        st.info("monthly_seasonality.csv not found; rerun the pipeline.")
with eda_tabs[2]:
    if genres is not None:
        st.plotly_chart(
            px.area(genres, x="month", y="titles_published", color="genre",
                    labels={"titles_published": "Titles published", "month": "Month"}),
            width="stretch",
            key="eda-genre",
        )
        share = genres.groupby("genre")["titles_published"].sum()
        st.dataframe(
            (share / share.sum() * 100).round(1).rename("share_%").reset_index(),
            hide_index=True,
        )
    else:
        st.info("The input data has no genre column.")
with eda_tabs[3]:
    if eda is not None:
        st.dataframe(eda, hide_index=True)

st.subheader("Model leaderboard")
st.dataframe(
    summary.style.format(
        {"mae": "{:.2f}", "rmse": "{:.2f}", "mape": "{:.2f}%", "smape": "{:.2f}%"}
    ).highlight_min(subset=["mae", "rmse", "mape", "smape"], color="#b7e4c7"),
    width="stretch",
    hide_index=True,
)

with st.expander("Understand lags, rolling means, and expanding windows", expanded=True):
    concept_tabs = st.tabs(["Lag features", "Rolling means", "Expanding validation"])
    features = make_features(history)
    with concept_tabs[0]:
        st.markdown(
            "A **lag** is an earlier target value aligned with the current month. `lag_1` is "
            "last month and `lag_12` is the same month one year earlier. XGBoost uses lags "
            "1, 2, 3, 6, and 12; the LSTM receives the latest 12 values as an ordered sequence."
        )
        lag_plot = features.tail(30).melt(
            id_vars="month",
            value_vars=["titles_published", "lag_1", "lag_12"],
            var_name="series",
            value_name="titles",
        )
        st.plotly_chart(
            px.line(lag_plot, x="month", y="titles", color="series", markers=True),
            width="stretch",
            key="lag-explainer",
        )
    with concept_tabs[1]:
        st.markdown(
            "A **rolling mean** smooths volatility over a moving 3-, 6-, or 12-month window. "
            "Each feature is shifted one month, so it uses only information available before "
            "the forecast date and cannot leak the current target."
        )
        rolling_plot = features.tail(36).melt(
            id_vars="month",
            value_vars=[
                "titles_published", "rolling_mean_3", "rolling_mean_6", "rolling_mean_12"
            ],
            var_name="series",
            value_name="titles",
        )
        st.plotly_chart(
            px.line(rolling_plot, x="month", y="titles", color="series"),
            width="stretch",
            key="rolling-explainer",
        )
    with concept_tabs[2]:
        st.markdown(
            "In **expanding-window rolling-origin validation**, the forecast origin advances six "
            "months while all earlier history is retained. This simulates three historical planning "
            "cycles without training on future information."
        )
        fold_rows = []
        for fold, (train_idx, test_idx) in enumerate(expanding_splits(len(history)), 1):
            train_dates = history.iloc[list(train_idx)]["month"]
            test_dates = history.iloc[list(test_idx)]["month"]
            fold_rows.append({
                "fold": fold,
                "training period": f"{train_dates.min():%b %Y} – {train_dates.max():%b %Y}",
                "training months": len(train_dates),
                "validation period": f"{test_dates.min():%b %Y} – {test_dates.max():%b %Y}",
                "validation months": len(test_dates),
            })
        st.dataframe(pd.DataFrame(fold_rows), width="stretch", hide_index=True)

model_order = ["seasonal_naive", "arima", "sarima", "ets", "prophet", "xgboost", "lstm"]
labels = {
    "seasonal_naive": "Seasonal Naïve",
    "arima": "ARIMA",
    "sarima": "SARIMA",
    "ets": "Holt-Winters ETS",
    "prophet": "Prophet",
    "xgboost": "XGBoost",
    "lstm": "LSTM",
}
descriptions = {
    "seasonal_naive": "Repeats the value observed in the same month last year.",
    "arima": "Models non-seasonal autoregression, differencing, and forecast errors.",
    "sarima": "Extends ARIMA with explicit 12-month seasonal dynamics.",
    "ets": "Smooths level, damped trend, and annual seasonality.",
    "prophet": "Combines trend changepoints with annual seasonal components.",
    "xgboost": "Learns nonlinear relationships from lag, rolling, trend, and calendar features.",
    "lstm": "Learns sequential dependencies from the previous 12 normalized monthly values.",
}
available = [model for model in model_order if model in set(summary["model"])]
tabs = st.tabs([labels[model] for model in available])

for tab, model in zip(tabs, available):
    with tab:
        row = summary.loc[summary["model"] == model].iloc[0]
        heading = f"{labels[model]} — overall champion" if model == champion else labels[model]
        st.subheader(heading)
        st.caption(descriptions[model])
        if model == "lstm":
            st.info(
                "Configuration: 12-month lookback → 32 LSTM units → 10% dropout → "
                "1 output; Adam learning rate 0.005, up to 200 epochs, batch size 16, "
                "15% chronological training-tail validation, and early stopping patience 12."
            )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean MAE", f"{row['mae']:.2f} titles")
        c2.metric("Mean RMSE", f"{row['rmse']:.2f} titles")
        c3.metric("Mean MAPE", f"{row['mape']:.2f}%")
        c4.metric("Mean sMAPE", f"{row['smape']:.2f}%")

        st.markdown("#### Rolling-origin backtest")
        model_backtest = backtest.loc[backtest["model"] == model].copy()
        long_backtest = model_backtest.melt(
            id_vars=["month", "fold"],
            value_vars=["actual", "prediction"],
            var_name="series",
            value_name="titles",
        )
        chart = px.line(
            long_backtest,
            x="month",
            y="titles",
            color="series",
            line_dash="series",
            markers=True,
            color_discrete_map={"actual": "#202124", "prediction": "#2563eb"},
            labels={"titles": "Titles published", "month": "Validation month"},
        )
        chart.update_layout(legend_title_text="Series", hovermode="x unified")
        st.plotly_chart(chart, width="stretch", key=f"backtest-{model}")

        fold_metrics = metrics.loc[
            metrics["model"] == model, ["fold", "mae", "rmse", "mape", "smape"]
        ]
        st.dataframe(
            fold_metrics.style.format(
                {"mae": "{:.2f}", "rmse": "{:.2f}", "mape": "{:.2f}%", "smape": "{:.2f}%"}
            ),
            width="stretch",
            hide_index=True,
        )

        if residuals is not None and model in set(residuals["model"]):
            diag = residuals.loc[residuals["model"] == model].iloc[0]
            bias = "under-forecasts" if diag["mean_error"] > 0 else "over-forecasts"
            st.markdown(
                f"**Residuals:** on average the model {bias} by "
                f"{abs(diag['mean_error']):.1f} titles "
                f"({diag['share_under_forecast']:.0%} of validation months were under-forecast). "
                f"Errors are largest in **{diag['worst_calendar_month']}** "
                f"(MAE {diag['worst_month_mae']:.1f})."
            )

        st.markdown("#### Twelve-month planning forecast")
        future = forecasts.loc[forecasts["model"] == model]
        context = history.tail(24)
        forecast_chart = go.Figure()
        if {"lower_80", "upper_80"} <= set(future.columns):
            forecast_chart.add_trace(go.Scatter(
                x=pd.concat([future["month"], future["month"][::-1]]),
                y=pd.concat([future["upper_80"], future["lower_80"][::-1]]),
                fill="toself",
                fillcolor="rgba(220, 38, 38, 0.15)",
                line={"width": 0},
                hoverinfo="skip",
                name="80% backtest-error band",
            ))
        forecast_chart.add_trace(go.Scatter(
            x=context["month"],
            y=context["titles_published"],
            mode="lines+markers",
            name="Historical actual",
            line={"color": "#202124"},
        ))
        forecast_chart.add_trace(go.Scatter(
            x=future["month"],
            y=future["forecast"],
            mode="lines+markers",
            name=f"{labels[model]} forecast",
            line={"color": "#dc2626", "dash": "dash"},
        ))
        forecast_chart.update_layout(
            xaxis_title="Month", yaxis_title="Titles published", hovermode="x unified"
        )
        st.plotly_chart(forecast_chart, width="stretch", key=f"forecast-{model}")
        st.download_button(
            f"Download {labels[model]} forecast",
            future.to_csv(index=False),
            f"{model}_publishing_forecast.csv",
            "text/csv",
            key=f"download-{model}",
        )
