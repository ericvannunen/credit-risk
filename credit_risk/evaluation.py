from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from credit_risk.logistic_model import fit_logistic
from credit_risk.models import TrainedModels
from credit_risk.neural_network import fit_neural_network, nn_predict_proba


@dataclass
class ComparisonResult:
    logistic_pr_auc: float
    logistic_auc: float
    logistic_recall: float
    logistic_brier: float
    logistic_ece: float
    nn_pr_auc: float
    nn_auc: float
    nn_recall: float
    nn_brier: float
    nn_ece: float
    preferred_model: str


def grade_from_probability(probability: float) -> str:
    if probability <= 0.05:
        return "A"
    if probability <= 0.15:
        return "B"
    if probability <= 0.30:
        return "C"
    if probability <= 0.50:
        return "D"
    return "E"


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(len(y_true))
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs >= lo) & (probs < hi if i < bins - 1 else probs <= hi)
        if not np.any(mask):
            continue
        bin_confidence = float(np.mean(probs[mask]))
        bin_accuracy = float(np.mean(y_true[mask]))
        ece += abs(bin_accuracy - bin_confidence) * (float(np.sum(mask)) / total)
    return ece


def compare_models(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    random_state: int = 42,
) -> Tuple[ComparisonResult, TrainedModels, Dict[str, np.ndarray]]:
    # Reserve the test set until the final comparison; validation guides
    # training without leaking information from the test set.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )

    logistic = fit_logistic(X_train, y_train)
    log_probs = logistic.predict_proba(X_test)[:, 1]
    nn_preprocessor, nn_model = fit_neural_network(X_train, y_train, X_val, y_val)
    nn_probs = nn_predict_proba(nn_model, nn_preprocessor, X_test)

    logistic_pr_auc = float(average_precision_score(y_test, log_probs))
    logistic_auc = float(roc_auc_score(y_test, log_probs))
    logistic_recall = float(recall_score(y_test, log_probs >= 0.5, zero_division=0))
    logistic_ece = expected_calibration_error(y_test, log_probs)
    nn_pr_auc = float(average_precision_score(y_test, nn_probs))
    nn_auc = float(roc_auc_score(y_test, nn_probs))
    nn_recall = float(recall_score(y_test, nn_probs >= 0.5, zero_division=0))
    nn_ece = expected_calibration_error(y_test, nn_probs)

    result = ComparisonResult(
        logistic_pr_auc=logistic_pr_auc,
        logistic_auc=logistic_auc,
        logistic_recall=logistic_recall,
        logistic_brier=float(brier_score_loss(y_test, log_probs)),
        logistic_ece=logistic_ece,
        nn_pr_auc=nn_pr_auc,
        nn_auc=nn_auc,
        nn_recall=nn_recall,
        nn_brier=float(brier_score_loss(y_test, nn_probs)),
        nn_ece=nn_ece,
        preferred_model=(
            "neural_net"
            if nn_pr_auc > logistic_pr_auc and nn_ece <= logistic_ece + 0.02
            else "logistic_regression"
        ),
    )
    trained = TrainedModels(logistic, nn_preprocessor, nn_model, feature_names)
    holdout = {"X_test": X_test, "y_test": y_test, "log_probs": log_probs, "nn_probs": nn_probs}
    return result, trained, holdout
