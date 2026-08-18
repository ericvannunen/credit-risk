from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from credit_risk.calibration import SigmoidCalibrator
from credit_risk.config import (
    DEFAULT_LOGISTIC_CONFIG,
    DEFAULT_NEURAL_NETWORK_CONFIG,
    LogisticConfig,
    NeuralNetworkConfig,
)
from credit_risk.logistic_model import fit_logistic
from credit_risk.models import TrainedModels
from credit_risk.neural_network import fit_neural_network, nn_predict_proba

FALSE_NEGATIVE_COST = 5.0
FALSE_POSITIVE_COST = 1.0
TARGET_RECALL = 0.70


@dataclass
class ComparisonResult:
    logistic_pr_auc: float
    logistic_auc: float
    logistic_precision: float
    logistic_recall: float
    logistic_confusion_matrix: list[list[int]]
    logistic_brier: float
    logistic_ece: float
    nn_pr_auc: float
    nn_auc: float
    nn_precision: float
    nn_recall: float
    nn_confusion_matrix: list[list[int]]
    nn_brier: float
    nn_ece: float
    preferred_model: str
    calibration_method: str = "sigmoid_validation"
    logistic_decision_threshold: float = 0.0
    nn_decision_threshold: float = 0.0
    false_negative_cost: float = FALSE_NEGATIVE_COST
    false_positive_cost: float = FALSE_POSITIVE_COST
    target_recall: float = TARGET_RECALL


@dataclass
class ValidationScore:
    model: str
    configuration: dict[str, object]
    pr_auc: float
    precision: float
    recall: float
    decision_threshold: float
    decision_cost: float


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


def select_cost_threshold(
    probabilities: np.ndarray,
    targets: np.ndarray,
    false_negative_cost: float = FALSE_NEGATIVE_COST,
    false_positive_cost: float = FALSE_POSITIVE_COST,
) -> float:
    """Choose the threshold with the lowest validation-set decision cost."""
    if false_negative_cost < 0 or false_positive_cost < 0:
        raise ValueError("Decision costs must be non-negative")

    candidates = np.unique(np.concatenate(([0.0, 1.0], np.clip(probabilities, 0.0, 1.0))))
    best_threshold = 0.0
    best_cost = float("inf")
    for threshold in candidates:
        predictions = probabilities >= threshold
        false_positives = int(np.sum((targets == 0) & predictions))
        false_negatives = int(np.sum((targets == 1) & ~predictions))
        cost = false_positive_cost * false_positives + false_negative_cost * false_negatives
        # On a tie, choose the higher threshold to avoid unnecessary flags.
        if cost < best_cost or (cost == best_cost and threshold > best_threshold):
            best_cost = cost
            best_threshold = float(threshold)
    return best_threshold


def select_recall_threshold(
    probabilities: np.ndarray,
    targets: np.ndarray,
    target_recall: float = TARGET_RECALL,
) -> float:
    """Choose the highest threshold that reaches the requested recall."""
    if not 0.0 <= target_recall <= 1.0:
        raise ValueError("target_recall must be between 0 and 1")

    candidates = np.unique(np.concatenate(([0.0, 1.0], np.clip(probabilities, 0.0, 1.0))))
    eligible_thresholds = []
    for threshold in candidates:
        predictions = probabilities >= threshold
        recall = recall_score(targets, predictions, zero_division=0)
        if recall >= target_recall:
            eligible_thresholds.append(float(threshold))

    # A threshold of zero predicts every company as bankrupt, so an eligible
    # threshold always exists for a valid target recall.
    return max(eligible_thresholds)


def select_constrained_cost_threshold(
    probabilities: np.ndarray,
    targets: np.ndarray,
    minimum_recall: float = TARGET_RECALL,
    false_negative_cost: float = FALSE_NEGATIVE_COST,
    false_positive_cost: float = FALSE_POSITIVE_COST,
) -> float:
    """Minimize cost while requiring at least the minimum recall."""
    if not 0.0 <= minimum_recall <= 1.0:
        raise ValueError("minimum_recall must be between 0 and 1")

    candidates = np.unique(np.concatenate(([0.0, 1.0], np.clip(probabilities, 0.0, 1.0))))
    best_threshold = 0.0
    best_cost = float("inf")
    for threshold in candidates:
        predictions = probabilities >= threshold
        recall = recall_score(targets, predictions, zero_division=0)
        if recall < minimum_recall:
            continue
        false_positives = int(np.sum((targets == 0) & predictions))
        false_negatives = int(np.sum((targets == 1) & ~predictions))
        cost = false_positive_cost * false_positives + false_negative_cost * false_negatives
        if cost < best_cost or (cost == best_cost and threshold > best_threshold):
            best_cost = cost
            best_threshold = float(threshold)
    return best_threshold


def score_configuration_on_validation(
    X: np.ndarray,
    y: np.ndarray,
    logistic_config: LogisticConfig = DEFAULT_LOGISTIC_CONFIG,
    neural_network_config: NeuralNetworkConfig = DEFAULT_NEURAL_NETWORK_CONFIG,
    random_state: int = 42,
    false_negative_cost: float = FALSE_NEGATIVE_COST,
    false_positive_cost: float = FALSE_POSITIVE_COST,
    target_recall: float | None = None,
) -> list[ValidationScore]:
    """Train candidates, calibrate on one split, and score on another."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y
    )
    X_calibration, X_val, y_calibration, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp,
    )

    logistic = fit_logistic(X_train, y_train, config=logistic_config)
    logistic_calibration_raw = logistic.predict_proba(X_calibration)[:, 1]
    logistic_val_raw = logistic.predict_proba(X_val)[:, 1]
    nn_preprocessor, nn_model = fit_neural_network(
        X_train,
        y_train,
        X_calibration,
        y_calibration,
        config=neural_network_config,
    )
    nn_calibration_raw = nn_predict_proba(nn_model, nn_preprocessor, X_calibration)
    nn_val_raw = nn_predict_proba(nn_model, nn_preprocessor, X_val)

    calibrators = [
        (
            "logistic_regression",
            SigmoidCalibrator().fit(logistic_calibration_raw, y_calibration),
            logistic_calibration_raw,
            logistic_val_raw,
            {"c": logistic_config.c, "class_weight": logistic_config.class_weight},
        ),
        (
            "neural_net",
            SigmoidCalibrator().fit(nn_calibration_raw, y_calibration),
            nn_calibration_raw,
            nn_val_raw,
            {
                "hidden_size_1": neural_network_config.hidden_size_1,
                "hidden_size_2": neural_network_config.hidden_size_2,
                "learning_rate": neural_network_config.learning_rate,
                "weight_decay": neural_network_config.weight_decay,
                "pos_weight_multiplier": neural_network_config.pos_weight_multiplier,
            },
        ),
    ]

    scores: list[ValidationScore] = []
    for model_name, calibrator, calibration_raw, validation_raw, configuration in calibrators:
        calibration_probs = calibrator.transform(calibration_raw)
        validation_probs = calibrator.transform(validation_raw)
        if target_recall is None:
            threshold = select_cost_threshold(
                calibration_probs,
                y_calibration,
                false_negative_cost,
                false_positive_cost,
            )
        else:
            threshold = select_constrained_cost_threshold(
                calibration_probs,
                y_calibration,
                minimum_recall=target_recall,
                false_negative_cost=false_negative_cost,
                false_positive_cost=false_positive_cost,
            )
        predictions = validation_probs >= threshold
        false_positives = int(np.sum((y_val == 0) & predictions))
        false_negatives = int(np.sum((y_val == 1) & ~predictions))
        scores.append(
            ValidationScore(
                model=model_name,
                configuration=configuration,
                pr_auc=float(average_precision_score(y_val, validation_probs)),
                precision=float(precision_score(y_val, predictions, zero_division=0)),
                recall=float(recall_score(y_val, predictions, zero_division=0)),
                decision_threshold=threshold,
                decision_cost=(
                    false_positive_cost * false_positives
                    + false_negative_cost * false_negatives
                ),
            )
        )
    return scores


def compare_models(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    random_state: int = 42,
    false_negative_cost: float = FALSE_NEGATIVE_COST,
    false_positive_cost: float = FALSE_POSITIVE_COST,
    target_recall: float | None = TARGET_RECALL,
    logistic_config: LogisticConfig = DEFAULT_LOGISTIC_CONFIG,
    neural_network_config: NeuralNetworkConfig = DEFAULT_NEURAL_NETWORK_CONFIG,
) -> Tuple[ComparisonResult, TrainedModels, Dict[str, np.ndarray]]:
    """Train, calibrate, threshold, and evaluate both configured models."""
    # Reserve the test set until the final comparison; validation guides
    # calibration and threshold selection without leaking test information.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )

    logistic = fit_logistic(X_train, y_train, config=logistic_config)
    logistic_val_probs = logistic.predict_proba(X_val)[:, 1]
    raw_log_probs = logistic.predict_proba(X_test)[:, 1]
    nn_preprocessor, nn_model = fit_neural_network(
        X_train,
        y_train,
        X_val,
        y_val,
        config=neural_network_config,
    )
    nn_val_probs = nn_predict_proba(nn_model, nn_preprocessor, X_val)
    raw_nn_probs = nn_predict_proba(nn_model, nn_preprocessor, X_test)

    # Fit calibration only on validation predictions, then keep the test set
    # untouched until the calibrated final evaluation below.
    logistic_calibrator = SigmoidCalibrator().fit(logistic_val_probs, y_val)
    nn_calibrator = SigmoidCalibrator().fit(nn_val_probs, y_val)
    calibrated_logistic_val_probs = logistic_calibrator.transform(logistic_val_probs)
    calibrated_nn_val_probs = nn_calibrator.transform(nn_val_probs)
    if target_recall is None:
        logistic_threshold = select_cost_threshold(
            calibrated_logistic_val_probs,
            y_val,
            false_negative_cost,
            false_positive_cost,
        )
        nn_threshold = select_cost_threshold(
            calibrated_nn_val_probs,
            y_val,
            false_negative_cost,
            false_positive_cost,
        )
    else:
        logistic_threshold = select_constrained_cost_threshold(
            calibrated_logistic_val_probs,
            y_val,
            minimum_recall=target_recall,
            false_negative_cost=false_negative_cost,
            false_positive_cost=false_positive_cost,
        )
        nn_threshold = select_constrained_cost_threshold(
            calibrated_nn_val_probs,
            y_val,
            minimum_recall=target_recall,
            false_negative_cost=false_negative_cost,
            false_positive_cost=false_positive_cost,
        )
    log_probs = logistic_calibrator.transform(raw_log_probs)
    nn_probs = nn_calibrator.transform(raw_nn_probs)

    logistic_pr_auc = float(average_precision_score(y_test, log_probs))
    logistic_auc = float(roc_auc_score(y_test, log_probs))
    logistic_predictions = log_probs >= logistic_threshold
    logistic_precision = float(precision_score(y_test, logistic_predictions, zero_division=0))
    logistic_recall = float(recall_score(y_test, logistic_predictions, zero_division=0))
    logistic_confusion = confusion_matrix(y_test, logistic_predictions, labels=[0, 1]).tolist()
    logistic_ece = expected_calibration_error(y_test, log_probs)
    nn_pr_auc = float(average_precision_score(y_test, nn_probs))
    nn_auc = float(roc_auc_score(y_test, nn_probs))
    nn_predictions = nn_probs >= nn_threshold
    nn_precision = float(precision_score(y_test, nn_predictions, zero_division=0))
    nn_recall = float(recall_score(y_test, nn_predictions, zero_division=0))
    nn_confusion = confusion_matrix(y_test, nn_predictions, labels=[0, 1]).tolist()
    nn_ece = expected_calibration_error(y_test, nn_probs)

    result = ComparisonResult(
        logistic_pr_auc=logistic_pr_auc,
        logistic_auc=logistic_auc,
        logistic_precision=logistic_precision,
        logistic_recall=logistic_recall,
        logistic_confusion_matrix=logistic_confusion,
        logistic_brier=float(brier_score_loss(y_test, log_probs)),
        logistic_ece=logistic_ece,
        nn_pr_auc=nn_pr_auc,
        nn_auc=nn_auc,
        nn_precision=nn_precision,
        nn_recall=nn_recall,
        nn_confusion_matrix=nn_confusion,
        nn_brier=float(brier_score_loss(y_test, nn_probs)),
        nn_ece=nn_ece,
        preferred_model=(
            "neural_net"
            if nn_pr_auc > logistic_pr_auc and nn_ece <= logistic_ece + 0.02
            else "logistic_regression"
        ),
        logistic_decision_threshold=logistic_threshold,
        nn_decision_threshold=nn_threshold,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
        target_recall=target_recall if target_recall is not None else 0.0,
    )
    trained = TrainedModels(
        logistic,
        nn_preprocessor,
        nn_model,
        feature_names,
        logistic_calibrator,
        nn_calibrator,
    )
    holdout = {
        "X_test": X_test,
        "y_test": y_test,
        "log_probs": log_probs,
        "nn_probs": nn_probs,
        "raw_log_probs": raw_log_probs,
        "raw_nn_probs": raw_nn_probs,
    }
    return result, trained, holdout
