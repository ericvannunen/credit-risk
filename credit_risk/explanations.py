from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from credit_risk.evaluation import grade_from_probability
from credit_risk.models import TrainedModels


def prediction_explanations(
    models: TrainedModels,
    company_row: np.ndarray,
    top_k: int = 5,
) -> Dict[str, object]:
    if company_row.ndim != 1:
        raise ValueError("company_row must be a 1D array")

    logistic = models.logistic_pipeline
    imputer: SimpleImputer = logistic.named_steps["imputer"]
    scaler: StandardScaler = logistic.named_steps["scaler"]
    classifier: LogisticRegression = logistic.named_steps["model"]

    row_imp = imputer.transform(company_row.reshape(1, -1))
    row_scaled = scaler.transform(row_imp)
    logit_contributions = row_scaled[0] * classifier.coef_[0]
    raw_log_prob = logistic.predict_proba(company_row.reshape(1, -1))[:, 1]
    log_prob = float(
        models.logistic_calibrator.transform(raw_log_prob)[0]
        if models.logistic_calibrator is not None
        else raw_log_prob[0]
    )

    row_nn = models.nn_preprocessor.scaler.transform(
        models.nn_preprocessor.imputer.transform(company_row.reshape(1, -1))
    )
    x_tensor = torch.tensor(row_nn, dtype=torch.float32, requires_grad=True)
    models.nn_model.eval()
    nn_probability_tensor = torch.sigmoid(models.nn_model(x_tensor))
    nn_probability_tensor.backward()
    nn_contributions = (x_tensor.grad.detach().numpy()[0] * row_nn[0]).astype(float)
    raw_nn_prob = np.array([float(nn_probability_tensor.item())])
    nn_prob = float(
        models.nn_calibrator.transform(raw_nn_prob)[0]
        if models.nn_calibrator is not None
        else raw_nn_prob[0]
    )

    def top_features(values: np.ndarray) -> List[Dict[str, float | str]]:
        indexes = np.argsort(np.abs(values))[::-1][:top_k]
        return [
            {"feature": models.feature_names[int(index)], "contribution": float(values[int(index)])}
            for index in indexes
        ]

    return {
        "logistic": {
            "probability": log_prob,
            "risk_grade": grade_from_probability(log_prob),
            "top_factors": top_features(logit_contributions),
        },
        "neural_net": {
            "probability": nn_prob,
            "risk_grade": grade_from_probability(nn_prob),
            "top_factors": top_features(nn_contributions),
        },
    }
