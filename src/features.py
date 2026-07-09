"""Feature engineering for the gold price nowcast model.

The model predicts GLD's same-day return from same-day moves in SPX, USO,
SLV and EUR/USD, plus short-term context (5-day returns and each series'
position relative to its 20-day mean). Working in return space rather than
raw price levels matters for a Random Forest: trees cannot extrapolate, so
a model trained on raw prices fails as soon as prices leave the range seen
in training. Returns are stationary, so the model generalizes across
price regimes.

A predicted return is converted back to a price with yesterday's GLD close:
    price = GLD_lag1 * (1 + predicted_return)
"""

import numpy as np
import pandas as pd

MARKET_COLUMNS = ["SPX", "USO", "SLV", "EUR/USD"]
ROLLING_WINDOW = 20
# Rows consumed by lag/rolling warm-up before all features are defined.
# The binding constraint is the 20-day rolling mean of lagged GLD, which
# first resolves at row 20 (it needs GLD closes from rows 0-19).
WARMUP_ROWS = ROLLING_WINDOW

FEATURE_COLUMNS = (
    [f"{c}_ret1" for c in MARKET_COLUMNS]
    + [f"{c}_ret5" for c in MARKET_COLUMNS]
    + [f"{c}_rel20" for c in MARKET_COLUMNS]
    + ["GLD_lag_rel20"]
)


def _compute_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute FEATURE_COLUMNS for every row of a chronological price frame.

    Only uses GLD through its lag, so the current row's GLD may be NaN
    (as it is for a live prediction, where GLD is the unknown).
    """
    out = pd.DataFrame(index=frame.index)
    for c in MARKET_COLUMNS:
        out[f"{c}_ret1"] = frame[c].pct_change(1)
        out[f"{c}_ret5"] = frame[c].pct_change(5)
        out[f"{c}_rel20"] = frame[c] / frame[c].rolling(ROLLING_WINDOW).mean() - 1
    gld_lag = frame["GLD"].shift(1)
    out["GLD_lag_rel20"] = gld_lag / gld_lag.rolling(ROLLING_WINDOW).mean() - 1
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the training table from the historical price frame.

    Returns a DataFrame with FEATURE_COLUMNS plus:
      target_ret -- GLD same-day return (the regression target)
      GLD_lag1   -- previous day's GLD close, to reconstruct prices
      GLD        -- actual GLD close, for evaluation
      Date       -- carried through for plotting and time-based splits
    Warm-up rows without full lag/rolling history are dropped.
    """
    out = _compute_features(df)
    out["target_ret"] = df["GLD"].pct_change(1)
    out["GLD_lag1"] = df["GLD"].shift(1)
    out["GLD"] = df["GLD"]
    out["Date"] = df["Date"].values if "Date" in df.columns else pd.NaT
    return out.dropna().reset_index(drop=True)


def build_live_features(history: pd.DataFrame, today: dict):
    """Build one feature row for a live prediction.

    history: recent trading days (oldest first, at least WARMUP_ROWS rows)
             with MARKET_COLUMNS and GLD -- NOT including today.
    today:   mapping of today's SPX/USO/SLV/EUR-USD values.

    Returns (X, gld_lag1) where X is a single-row DataFrame of
    FEATURE_COLUMNS and gld_lag1 is the most recent known GLD close.
    """
    if len(history) < WARMUP_ROWS:
        raise ValueError(
            f"Need at least {WARMUP_ROWS} rows of history, got {len(history)}"
        )
    row = {c: float(today[c]) for c in MARKET_COLUMNS}
    row["GLD"] = np.nan  # unknown -- it is what we are predicting
    frame = pd.concat(
        [history[MARKET_COLUMNS + ["GLD"]], pd.DataFrame([row])],
        ignore_index=True,
    )
    X = _compute_features(frame)[FEATURE_COLUMNS].iloc[[-1]].reset_index(drop=True)
    if X.isna().any().any():
        raise ValueError("History contains gaps; could not compute all features")
    return X, float(history["GLD"].iloc[-1])
