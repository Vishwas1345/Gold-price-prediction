"""App-facing inference helpers.

This is the glue between the UI layer (app/app.py) and the core modules:
market data with a fallback, price prediction with an uncertainty band,
and the per-input sensitivity used by the "what's pulling the prediction"
chart. Lives in src/ so the app layer holds no modeling logic.
"""

import time

import pandas as pd

from .data_loader import load_data
from .features import MARKET_COLUMNS, WARMUP_ROWS, build_live_features
from .model import GoldPriceModel

# Latest market data is cached briefly so UI interactions (tab switches,
# manual-mode edits) don't re-hit Yahoo Finance on every callback.
CACHE_TTL_SECONDS = 15 * 60
_market_cache: dict = {"at": 0.0, "value": None}


def get_market_data(force_refresh: bool = False):
    """Latest values + recent history for a prediction.

    Returns (today, history, is_live):
      today    -- dict of the most recent SPX/USO/SLV/EUR-USD closes
      history  -- prior trading days (oldest first), with GLD
      is_live  -- False when Yahoo Finance was unreachable and the tail of
                  the bundled training CSV was used instead, so the app
                  stays usable offline / rate-limited / in demo settings.
    """
    now = time.monotonic()
    cached = _market_cache["value"]
    if cached is not None and not force_refresh:
        if now - _market_cache["at"] < CACHE_TTL_SECONDS:
            return cached

    try:
        from .live_data import fetch_latest

        today, history = fetch_latest()
        result = (today, history, True)
    except Exception:
        df = load_data()
        latest = df.iloc[-1]
        today = {c: float(latest[c]) for c in MARKET_COLUMNS}
        history = df.iloc[:-1].tail(WARMUP_ROWS + 10).reset_index(drop=True)
        result = (today, history, False)

    _market_cache["at"] = now
    _market_cache["value"] = result
    return result


def predict_with_band(model: GoldPriceModel, history: pd.DataFrame, today: dict):
    """Returns (point_prediction, band_low, band_high, last_known_gld)."""
    X, gld_lag1 = build_live_features(history, today)
    price = float(model.predict_price(X, gld_lag1)[0])
    lo, hi = model.predict_price_interval(X, gld_lag1)
    return price, float(lo[0]), float(hi[0]), gld_lag1


def input_effects(
    model: GoldPriceModel, history: pd.DataFrame, today: dict
) -> pd.DataFrame:
    """Dollar effect of each input's move on the prediction.

    For each market input, freeze it back to its previous close and
    re-predict; the difference is how much that input's move today is
    pulling the predicted gold price. Returns a frame with columns
    Input / effect_usd.
    """
    price, *_ = predict_with_band(model, history, today)
    rows = []
    for name in MARKET_COLUMNS:
        frozen = dict(today)
        frozen[name] = float(history[name].iloc[-1])
        frozen_price, *_ = predict_with_band(model, history, frozen)
        rows.append({"Input": name, "effect_usd": price - frozen_price})
    return pd.DataFrame(rows)
