from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class SigmoidCalibrator:
    """Map a model's raw probabilities to probabilities learned on validation data."""

    def __init__(self) -> None:
        self._model = LogisticRegression(C=1.0, solver="lbfgs")

    def fit(self, probabilities: np.ndarray, targets: np.ndarray) -> SigmoidCalibrator:
        clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
        logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        self._model.fit(logits, targets)
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
        logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        return self._model.predict_proba(logits)[:, 1]
