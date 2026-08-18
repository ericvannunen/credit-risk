from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_logistic(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    # Keep preprocessing inside the fitted pipeline so validation and test data
    # cannot influence the imputation or scaling parameters.
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    ).fit(X_train, y_train)
