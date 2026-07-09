"""Tests for feature engineering and the model wrapper."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import (
    FEATURE_COLUMNS,
    MARKET_COLUMNS,
    WARMUP_ROWS,
    build_features,
    build_live_features,
)
from src.model import MODEL_PATH, GoldPriceModel

N_DUMMY = 120


@pytest.fixture
def dummy_prices():
    """Synthetic random-walk price frame shaped like the real dataset."""
    rng = np.random.default_rng(0)
    n = N_DUMMY
    df = pd.DataFrame(
        {
            "Date": pd.bdate_range("2020-01-01", periods=n),
            "SPX": 3000 * np.cumprod(1 + rng.normal(0, 0.01, n)),
            "GLD": 150 * np.cumprod(1 + rng.normal(0, 0.008, n)),
            "USO": 40 * np.cumprod(1 + rng.normal(0, 0.02, n)),
            "SLV": 20 * np.cumprod(1 + rng.normal(0, 0.015, n)),
            "EUR/USD": 1.1 * np.cumprod(1 + rng.normal(0, 0.004, n)),
        }
    )
    return df


def test_build_features_shape_and_columns(dummy_prices):
    features = build_features(dummy_prices)
    for col in FEATURE_COLUMNS + ["target_ret", "GLD_lag1", "GLD", "Date"]:
        assert col in features.columns
    # Warm-up rows (rolling window + longest lag) are dropped, nothing else.
    assert len(features) == N_DUMMY - WARMUP_ROWS
    assert not features[FEATURE_COLUMNS].isna().any().any()


def test_build_live_features_single_row(dummy_prices):
    history = dummy_prices.iloc[:-1]
    today = {c: float(dummy_prices[c].iloc[-1]) for c in MARKET_COLUMNS}
    X, gld_lag1 = build_live_features(history, today)
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == 1
    assert not X.isna().any().any()
    assert gld_lag1 == pytest.approx(float(history["GLD"].iloc[-1]))


def test_build_live_features_rejects_short_history(dummy_prices):
    today = {c: 1.0 for c in MARKET_COLUMNS}
    with pytest.raises(ValueError):
        build_live_features(dummy_prices.iloc[:5], today)


def test_model_train_predict_roundtrip(dummy_prices, tmp_path):
    features = build_features(dummy_prices)
    model = GoldPriceModel(n_estimators=10).train(features, features["target_ret"])

    price = model.predict_price(features.iloc[[0]], features["GLD_lag1"].iloc[0])
    assert price.shape == (1,)
    assert isinstance(float(price[0]), float)
    # Sane: a same-day prediction stays within +/-25% of yesterday's close.
    lag = features["GLD_lag1"].iloc[0]
    assert 0.75 * lag < float(price[0]) < 1.25 * lag

    lo, hi = model.predict_price_interval(
        features.iloc[[0]], features["GLD_lag1"].iloc[0]
    )
    assert float(lo[0]) <= float(price[0]) <= float(hi[0])

    imp = model.feature_importance()
    assert set(imp.index) == set(FEATURE_COLUMNS)
    assert imp.sum() == pytest.approx(1.0)

    path = tmp_path / "model.pkl"
    model.save(path)
    reloaded = GoldPriceModel.load(path)
    assert np.allclose(
        reloaded.predict_return(features), model.predict_return(features)
    )


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="run `python -m src.train` first")
def test_shipped_model_predicts_sane_price():
    from src.data_loader import load_data

    model = GoldPriceModel.load(MODEL_PATH)
    df = load_data()
    history = df.iloc[:-1]
    today = {c: float(df[c].iloc[-1]) for c in MARKET_COLUMNS}
    X, gld_lag1 = build_live_features(history, today)
    price = float(model.predict_price(X, gld_lag1)[0])
    # GLD traded between ~$40 and ~$400 in living memory; anything outside
    # that from in-distribution inputs means the model is broken.
    assert 40 < price < 400
