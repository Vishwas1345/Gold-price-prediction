"""Random Forest wrapper for the gold price nowcast model."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .features import FEATURE_COLUMNS

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.pkl"


class GoldPriceModel:
    """Predicts GLD's same-day return; prices are reconstructed from GLD_lag1.

    The forest is trained on returns, not raw prices -- see src/features.py
    for why that matters for tree models on financial series.
    """

    def __init__(self, n_estimators: int = 200, random_state: int = 42, **kwargs):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            **kwargs,
        )

    def train(self, X: pd.DataFrame, y: pd.Series) -> "GoldPriceModel":
        self.model.fit(X[FEATURE_COLUMNS], y)
        return self

    def predict_return(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[FEATURE_COLUMNS])

    def predict_price(self, X: pd.DataFrame, gld_lag1) -> np.ndarray:
        """Convert predicted returns to prices using the previous GLD close."""
        return np.asarray(gld_lag1) * (1 + self.predict_return(X))

    def predict_price_interval(
        self, X: pd.DataFrame, gld_lag1, lower: float = 5, upper: float = 95
    ):
        """Price band from the spread of per-tree predictions.

        This reflects model (ensemble) uncertainty only -- it is not a
        calibrated statistical confidence interval.
        """
        features = X[FEATURE_COLUMNS].to_numpy()
        per_tree = np.stack([t.predict(features) for t in self.model.estimators_])
        lo_ret = np.percentile(per_tree, lower, axis=0)
        hi_ret = np.percentile(per_tree, upper, axis=0)
        lag = np.asarray(gld_lag1)
        return lag * (1 + lo_ret), lag * (1 + hi_ret)

    def feature_importance(self) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_, index=FEATURE_COLUMNS
        ).sort_values(ascending=False)

    def save(self, path: Path = MODEL_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # compress keeps the pickle small enough to commit (forests are big)
        joblib.dump(self.model, path, compress=3)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "GoldPriceModel":
        instance = cls()
        instance.model = joblib.load(path)
        return instance
