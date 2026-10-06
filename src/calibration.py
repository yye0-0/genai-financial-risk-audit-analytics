from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """Monotonic sigmoid calibration that preserves score ordering."""

    def __init__(self):
        self.model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=300)

    @staticmethod
    def _logit(score: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(score, dtype=float), 1e-8, 1 - 1e-8)
        return np.log(clipped / (1 - clipped)).reshape(-1, 1)

    def fit(self, score: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        self.model.fit(self._logit(score), y)
        return self

    def predict(self, score: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._logit(score))[:, 1]

