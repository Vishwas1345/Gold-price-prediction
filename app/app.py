"""Gradio app for live gold price prediction.

Run locally with:  python app/app.py
Deployed on Hugging Face Spaces (Gradio SDK) -- see the README.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.features import MARKET_COLUMNS
from src.inference import get_market_data, input_effects, predict_with_band
from src.model import MODEL_PATH, GoldPriceModel

INPUT_LABELS = {
    "SPX": "S&P 500 (SPX)",
    "USO": "Oil ETF (USO)",
    "SLV": "Silver ETF (SLV)",
    "EUR/USD": "EUR/USD rate",
}

INTRO = (
    "Predicts the **GLD gold ETF price** from same-day moves in the S&P 500, "
    "oil (USO), silver (SLV) and the EUR/USD rate, using a Random Forest "
    "trained on returns with a time-based evaluation split."
)

DISCLAIMER = (
    "⚠️ **Disclaimer** — This is an educational / portfolio project. "
    "It is **not financial advice** and should not be used to make "
    "investment decisions."
)

FOOTER = (
    "Educational project — model trained on 2008–2018 daily data. "
    "See the README for honest accuracy numbers and methodology."
)

BAND_CAPTION = (
    "The band is the 5th–95th percentile spread of the forest's individual "
    "trees — a measure of model disagreement, not a calibrated confidence interval."
)

EFFECTS_CAPTION = (
    "Each bar shows how much the prediction changes because of that "
    "input's move versus its previous close. Positive bars pull the "
    "predicted gold price up."
)

OFFLINE_WARNING = (
    "⚠️ Could not reach Yahoo Finance — using the last rows of the bundled "
    "historical dataset instead of live prices."
)


def _prediction_md(price: float, lo: float, hi: float, last_gld: float) -> str:
    delta = price - last_gld
    return (
        f"### Predicted GLD price: ${price:.2f}  ({delta:+.2f} vs last close)\n\n"
        f"**Model uncertainty band:** ${lo:.2f} – ${hi:.2f} &nbsp;·&nbsp; "
        f"**Last known GLD close:** ${last_gld:.2f}\n\n"
        f"<sub>{BAND_CAPTION}</sub>"
    )


def _market_md(today: dict, history: pd.DataFrame) -> str:
    rows = ["| Input | Today | Move |", "|---|---|---|"]
    for name in MARKET_COLUMNS:
        prev = float(history[name].iloc[-1])
        move = (today[name] / prev - 1) * 100
        rows.append(f"| {INPUT_LABELS[name]} | {today[name]:,.2f} | {move:+.2f}% today |")
    return "\n".join(rows)


def _trend_figure(history: pd.DataFrame, price: float):
    """Recent actual GLD closes with the prediction as a next-day point."""
    if "Date" not in history:
        return None
    trend = history[["Date", "GLD"]].tail(60)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(trend["Date"], trend["GLD"], color="goldenrod", label="Actual GLD close")
    next_day = trend["Date"].iloc[-1] + pd.Timedelta(days=1)
    ax.scatter([next_day], [price], color="steelblue", zorder=5, label="Predicted")
    ax.set_ylabel("GLD ($)")
    ax.set_title("Recent GLD trend vs prediction")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.close(fig)
    return fig


def _effects_figure(effects: pd.DataFrame):
    labels = [INPUT_LABELS[n] for n in effects["Input"]]
    values = effects["effect_usd"]
    colors = ["goldenrod" if v >= 0 else "steelblue" for v in values]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Effect on prediction ($)")
    ax.set_title("What's pulling the prediction?")
    fig.tight_layout()
    plt.close(fig)
    return fig


def build_app() -> gr.Blocks:
    if not MODEL_PATH.exists():
        with gr.Blocks(title="Gold Price Predictor") as demo:
            gr.Markdown(
                "## 🚫 No trained model found\nRun `python -m src.train` first."
            )
        return demo

    model = GoldPriceModel.load(MODEL_PATH)
    # Initial fetch pre-fills the manual inputs; falls back to the bundled
    # CSV when offline, so building the UI never fails.
    today0, _, _ = get_market_data()

    def run_live(force_refresh: bool = False):
        today, history, is_live = get_market_data(force_refresh=force_refresh)
        warning = "" if is_live else OFFLINE_WARNING
        price, lo, hi, last_gld = predict_with_band(model, history, today)
        return (
            warning,
            _market_md(today, history),
            _prediction_md(price, lo, hi, last_gld),
            _trend_figure(history, price),
        )

    def refresh_live():
        return run_live(force_refresh=True)

    def run_manual(spx, uso, slv, eurusd):
        values = dict(zip(MARKET_COLUMNS, [spx, uso, slv, eurusd]))
        bad = [INPUT_LABELS[k] for k, v in values.items() if v is None or v <= 0]
        if bad:
            return f"Enter a positive value for: {', '.join(bad)}", None
        user_today = {k: float(v) for k, v in values.items()}
        _, history, _ = get_market_data()
        price, lo, hi, last_gld = predict_with_band(model, history, user_today)
        effects = input_effects(model, history, user_today)
        return _prediction_md(price, lo, hi, last_gld), _effects_figure(effects)

    with gr.Blocks(title="Gold Price Predictor") as demo:
        gr.Markdown("# 🪙 Gold Price Predictor")
        gr.Markdown(INTRO)
        gr.Markdown(DISCLAIMER)

        with gr.Tabs():
            with gr.Tab("Live"):
                gr.Markdown("### Live prediction from today's market data")
                refresh_btn = gr.Button(
                    "🔄 Refresh live data & predict", variant="primary"
                )
                live_warning = gr.Markdown()
                live_market = gr.Markdown()
                live_pred = gr.Markdown()
                live_plot = gr.Plot(show_label=False)
                refresh_btn.click(
                    refresh_live,
                    outputs=[live_warning, live_market, live_pred, live_plot],
                )

            with gr.Tab("Manual"):
                gr.Markdown("### Manual what-if prediction")
                gr.Markdown(
                    "Enter your own market values — the prediction updates when "
                    "you change an input. Inputs are pre-filled with the latest "
                    "available data."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        manual_inputs = []
                        for name in MARKET_COLUMNS:
                            precision = 3 if name == "EUR/USD" else 2
                            step = 0.001 if name == "EUR/USD" else 0.5
                            manual_inputs.append(
                                gr.Number(
                                    label=INPUT_LABELS[name],
                                    value=round(float(today0[name]), precision),
                                    minimum=0.0,
                                    step=step,
                                    precision=precision,
                                )
                            )
                    with gr.Column(scale=2):
                        manual_pred = gr.Markdown()
                        manual_plot = gr.Plot(show_label=False)
                        gr.Markdown(f"<sub>{EFFECTS_CAPTION}</sub>")
                for inp in manual_inputs:
                    inp.change(
                        run_manual,
                        inputs=manual_inputs,
                        outputs=[manual_pred, manual_plot],
                    )

        gr.Markdown("---")
        gr.Markdown(f"<sub>{FOOTER}</sub>")

        # Populate both tabs on page load (mirrors the Streamlit behavior).
        demo.load(run_live, outputs=[live_warning, live_market, live_pred, live_plot])
        demo.load(run_manual, inputs=manual_inputs, outputs=[manual_pred, manual_plot])

    return demo


demo = build_app()

if __name__ == "__main__":
    demo.launch()
