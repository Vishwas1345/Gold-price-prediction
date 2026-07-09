"""Streamlit app for live gold price prediction.

Run with:  streamlit run app/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_data
from src.features import MARKET_COLUMNS, WARMUP_ROWS, build_live_features
from src.model import MODEL_PATH, GoldPriceModel

st.set_page_config(page_title="Gold Price Predictor", page_icon="🪙", layout="wide")

INPUT_LABELS = {
    "SPX": "S&P 500 (SPX)",
    "USO": "Oil ETF (USO)",
    "SLV": "Silver ETF (SLV)",
    "EUR/USD": "EUR/USD rate",
}


@st.cache_resource
def get_model() -> GoldPriceModel:
    return GoldPriceModel.load(MODEL_PATH)


@st.cache_data(ttl=15 * 60, show_spinner="Fetching latest market data...")
def get_market_data():
    """Latest values + recent history from Yahoo Finance.

    Falls back to the tail of the training CSV when the fetch fails
    (offline, rate limited), so the app stays usable in demo settings.
    """
    try:
        from src.live_data import fetch_latest

        today, history = fetch_latest()
        return today, history, True
    except Exception:
        df = load_data()
        latest = df.iloc[-1]
        today = {c: float(latest[c]) for c in MARKET_COLUMNS}
        history = df.iloc[:-1].tail(WARMUP_ROWS + 10).reset_index(drop=True)
        return today, history, False


def predict(model: GoldPriceModel, history: pd.DataFrame, today: dict):
    """Returns (point_prediction, band_low, band_high, last_known_gld)."""
    X, gld_lag1 = build_live_features(history, today)
    price = float(model.predict_price(X, gld_lag1)[0])
    lo, hi = model.predict_price_interval(X, gld_lag1)
    return price, float(lo[0]), float(hi[0]), gld_lag1


def show_prediction(price: float, lo: float, hi: float, last_gld: float):
    a, b, c = st.columns(3)
    a.metric(
        "Predicted GLD price",
        f"${price:.2f}",
        delta=f"{price - last_gld:+.2f} vs last close",
    )
    b.metric("Model uncertainty band", f"${lo:.2f} – ${hi:.2f}")
    c.metric("Last known GLD close", f"${last_gld:.2f}")
    st.caption(
        "The band is the 5th–95th percentile spread of the forest's individual "
        "trees — a measure of model disagreement, not a calibrated confidence interval."
    )


def main():
    st.title("🪙 Gold Price Predictor")
    st.markdown(
        "Predicts the **GLD gold ETF price** from same-day moves in the S&P 500, "
        "oil (USO), silver (SLV) and the EUR/USD rate, using a Random Forest "
        "trained on returns with a time-based evaluation split."
    )

    if not MODEL_PATH.exists():
        st.error("No trained model found. Run `python -m src.train` first.")
        st.stop()
    model = get_model()

    today, history, is_live = get_market_data()
    if not is_live:
        st.warning(
            "Could not reach Yahoo Finance — using the last rows of the bundled "
            "historical dataset instead of live prices."
        )

    mode = st.sidebar.radio("Mode", ["Live", "Manual"])
    st.sidebar.markdown("---")
    st.sidebar.warning(
        "**Disclaimer** — This is an educational / portfolio project. "
        "It is **not financial advice** and should not be used to make "
        "investment decisions."
    )

    if mode == "Live":
        st.subheader("Live prediction from today's market data")
        cols = st.columns(4)
        for col, name in zip(cols, MARKET_COLUMNS):
            prev = float(history[name].iloc[-1])
            col.metric(
                INPUT_LABELS[name],
                f"{today[name]:,.2f}",
                delta=f"{(today[name] / prev - 1) * 100:+.2f}% today",
            )

        price, lo, hi, last_gld = predict(model, history, today)
        show_prediction(price, lo, hi, last_gld)

        st.subheader("Recent GLD trend vs prediction")
        trend = history[["Date", "GLD"]].tail(60).copy() if "Date" in history else None
        if trend is not None:
            trend = trend.rename(columns={"GLD": "Actual GLD close"})
            pred_row = pd.DataFrame(
                {
                    "Date": [trend["Date"].iloc[-1] + pd.Timedelta(days=1)],
                    "Predicted": [price],
                }
            )
            chart_df = pd.concat([trend, pred_row]).set_index("Date")
            st.line_chart(chart_df)

    else:
        st.subheader("Manual what-if prediction")
        st.markdown(
            "Enter your own market values — the prediction updates immediately. "
            "Inputs are pre-filled with the latest available data."
        )
        left, right = st.columns([1, 2])
        user_today = {}
        with left:
            for name in MARKET_COLUMNS:
                step = 0.001 if name == "EUR/USD" else 0.5
                user_today[name] = st.number_input(
                    INPUT_LABELS[name],
                    min_value=0.0,
                    value=float(today[name]),
                    step=step,
                    format="%.3f" if name == "EUR/USD" else "%.2f",
                )

        price, lo, hi, last_gld = predict(model, history, user_today)
        with right:
            show_prediction(price, lo, hi, last_gld)

            # Sensitivity: what happens to the prediction if each input had
            # not moved today (set back to yesterday's close)? The delta is
            # how much that input's move is pulling the prediction.
            st.subheader("What's pulling the prediction?")
            rows = []
            for name in MARKET_COLUMNS:
                frozen = dict(user_today)
                frozen[name] = float(history[name].iloc[-1])
                frozen_price, *_ = predict(model, history, frozen)
                rows.append(
                    {"Input": INPUT_LABELS[name], "Effect on prediction ($)": price - frozen_price}
                )
            effect = pd.DataFrame(rows).set_index("Input")
            st.bar_chart(effect)
            st.caption(
                "Each bar shows how much the prediction changes because of that "
                "input's move versus its previous close. Positive bars pull the "
                "predicted gold price up."
            )

    st.markdown("---")
    st.caption(
        "Educational project — model trained on 2008–2018 daily data. "
        "See the README for honest accuracy numbers and methodology."
    )


main()
