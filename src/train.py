"""Train and evaluate the gold price model.

Usage:  python -m src.train

Reports three honest, time-split numbers side by side:
  1. the original approach (raw price levels, random split) -- leaky
  2. the original approach on a proper time-based split -- the real baseline
  3. the return-space model with lag/rolling features -- the shipped model

Saves the final model to models/model.pkl, evaluation metrics to
results/metrics.json, and diagnostic charts to results/.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from .data_loader import load_data, time_based_split
from .features import FEATURE_COLUMNS, MARKET_COLUMNS, build_features
from .model import MODEL_PATH, GoldPriceModel

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def evaluate_original_approach(df: pd.DataFrame) -> dict:
    """Reproduce the original notebook (raw levels) under both split types."""
    X = df[MARKET_COLUMNS]
    y = df["GLD"]

    # Random split -- how the original 98.94% was produced. Leaky for
    # time series: test days sit between their own training neighbors.
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    leaky = RandomForestRegressor(random_state=42).fit(Xtr, ytr)
    leaky_pred = leaky.predict(Xte)

    # Same model, chronological split -- the honest version.
    train_df, test_df = time_based_split(df)
    honest = RandomForestRegressor(random_state=42).fit(
        train_df[MARKET_COLUMNS], train_df["GLD"]
    )
    honest_pred = honest.predict(test_df[MARKET_COLUMNS])

    return {
        "original_random_split": {
            "r2": r2_score(yte, leaky_pred),
            "mae_usd": mean_absolute_error(yte, leaky_pred),
        },
        "original_time_split": {
            "r2": r2_score(test_df["GLD"], honest_pred),
            "mae_usd": mean_absolute_error(test_df["GLD"], honest_pred),
        },
    }


def evaluate_final_model(features: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Time-split evaluation of the return-space model.

    Returns the metrics plus the test frame with predictions attached
    (used for the diagnostic charts).
    """
    train_f, test_f = time_based_split(features)
    model = GoldPriceModel().train(train_f, train_f["target_ret"])

    test_f = test_f.copy()
    test_f["pred_price"] = model.predict_price(test_f, test_f["GLD_lag1"])
    naive = test_f["GLD_lag1"]  # "tomorrow = today" persistence baseline

    metrics = {
        "final_time_split": {
            "price_r2": r2_score(test_f["GLD"], test_f["pred_price"]),
            "price_mae_usd": mean_absolute_error(test_f["GLD"], test_f["pred_price"]),
            "return_r2": r2_score(test_f["target_ret"], model.predict_return(test_f)),
        },
        "naive_persistence_baseline": {
            "price_r2": r2_score(test_f["GLD"], naive),
            "price_mae_usd": mean_absolute_error(test_f["GLD"], naive),
        },
        "test_period": {
            "start": str(test_f["Date"].iloc[0].date()),
            "end": str(test_f["Date"].iloc[-1].date()),
            "n_days": len(test_f),
        },
    }
    return metrics, test_f


def save_charts(model: GoldPriceModel, test_f: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    importance = model.feature_importance()
    fig, ax = plt.subplots(figsize=(8, 5))
    importance.sort_values().plot.barh(ax=ax, color="goldenrod")
    ax.set_title("Feature importance (return-space model)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(test_f["GLD"], test_f["pred_price"], s=8, alpha=0.5, color="goldenrod")
    lims = [test_f["GLD"].min(), test_f["GLD"].max()]
    ax.plot(lims, lims, "k--", linewidth=1, label="perfect prediction")
    ax.set_xlabel("Actual GLD price ($)")
    ax.set_ylabel("Predicted GLD price ($)")
    ax.set_title("Actual vs predicted (chronological test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)

    residuals = test_f["GLD"] - test_f["pred_price"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(test_f["Date"], residuals, s=8, alpha=0.5, color="steelblue")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Residual ($)")
    ax.set_title("Residuals over the test period")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "residuals.png", dpi=150)
    plt.close(fig)


def main() -> None:
    df = load_data()
    print(f"Loaded {len(df)} rows: {df['Date'].iloc[0].date()} to {df['Date'].iloc[-1].date()}\n")

    metrics = evaluate_original_approach(df)
    o = metrics["original_random_split"]
    print(f"Original (raw levels, RANDOM split)   R2={o['r2']:.4f}  MAE=${o['mae_usd']:.2f}   <- leaky")
    o = metrics["original_time_split"]
    print(f"Original (raw levels, TIME split)     R2={o['r2']:.4f}  MAE=${o['mae_usd']:.2f}   <- honest baseline")

    features = build_features(df)
    final_metrics, test_f = evaluate_final_model(features)
    metrics.update(final_metrics)
    f = metrics["final_time_split"]
    n = metrics["naive_persistence_baseline"]
    print(f"Final (return space, TIME split)      R2={f['price_r2']:.4f}  MAE=${f['price_mae_usd']:.2f}   return R2={f['return_r2']:.4f}")
    print(f"Naive persistence (predict yesterday) R2={n['price_r2']:.4f}  MAE=${n['price_mae_usd']:.2f}")

    # Retrain on the full history for deployment, then persist everything.
    final_model = GoldPriceModel().train(features, features["target_ret"])
    final_model.save(MODEL_PATH)
    save_charts(final_model, test_f)
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved metrics and charts to {RESULTS_DIR}/")
    print("\nTop feature importances:")
    print(final_model.feature_importance().head(5).to_string())


if __name__ == "__main__":
    main()
