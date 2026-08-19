from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credit_risk.config import DEFAULT_LOGISTIC_CONFIG, LogisticConfig


def fit_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: LogisticConfig = DEFAULT_LOGISTIC_CONFIG,
) -> Pipeline:
    # Keep preprocessing inside the fitted pipeline so validation and test data
    # cannot influence the imputation or scaling parameters.
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=config.max_iter,
                    class_weight=config.class_weight,
                    C=config.c,
                ),
            ),
        ]
    ).fit(X_train, y_train)
    