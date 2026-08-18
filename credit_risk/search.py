from __future__ import annotations

import argparse
import json
from pathlib import Path

from credit_risk.config import LogisticConfig, NeuralNetworkConfig
from credit_risk.data import load_polish_bankruptcy_five_year
from credit_risk.evaluation import (
    FALSE_NEGATIVE_COST,
    FALSE_POSITIVE_COST,
    ValidationScore,
    score_configuration_on_validation,
)

TARGET_RECALL = 0.70


LOGISTIC_C_VALUES = (0.1, 1.0, 10.0)
LOGISTIC_CLASS_WEIGHTS = (None, "balanced")
NN_LEARNING_RATES = (5e-4, 1e-3)
NN_WEIGHT_DECAYS = (1e-4, 1e-3)


def run_search(data_path: str | Path) -> dict[str, object]:
    X, y, _ = load_polish_bankruptcy_five_year(data_path=data_path)
    scores: list[ValidationScore] = []

    for c in LOGISTIC_C_VALUES:
        for class_weight in LOGISTIC_CLASS_WEIGHTS:
            logistic_config = LogisticConfig(c=c, class_weight=class_weight)
            scores.append(
                score_configuration_on_validation(
                    X,
                    y,
                    logistic_config=logistic_config,
                    false_negative_cost=FALSE_NEGATIVE_COST,
                    false_positive_cost=FALSE_POSITIVE_COST,
                    target_recall=TARGET_RECALL,
                )[0]
            )

    for learning_rate in NN_LEARNING_RATES:
        for weight_decay in NN_WEIGHT_DECAYS:
            neural_network_config = NeuralNetworkConfig(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            )
            scores.append(
                score_configuration_on_validation(
                    X,
                    y,
                    neural_network_config=neural_network_config,
                    false_negative_cost=FALSE_NEGATIVE_COST,
                    false_positive_cost=FALSE_POSITIVE_COST,
                    target_recall=TARGET_RECALL,
                )[1]
            )

    best_by_model: dict[str, ValidationScore] = {}
    best_by_pr_auc: dict[str, ValidationScore] = {}
    for score in scores:
        current_best = best_by_model.get(score.model)
        if current_best is None or (score.decision_cost, -score.pr_auc) < (
            current_best.decision_cost,
            -current_best.pr_auc,
        ):
            best_by_model[score.model] = score
        current_pr_auc_best = best_by_pr_auc.get(score.model)
        if current_pr_auc_best is None or score.pr_auc > current_pr_auc_best.pr_auc:
            best_by_pr_auc[score.model] = score

    def serialize(score: ValidationScore) -> dict[str, object]:
        return {
            "model": score.model,
            "configuration": score.configuration,
            "validation_pr_auc": score.pr_auc,
            "validation_precision": score.precision,
            "validation_recall": score.recall,
            "validation_threshold": score.decision_threshold,
            "validation_decision_cost": score.decision_cost,
        }

    return {
        "selection_rule": {
            "primary": f"lowest validation decision cost subject to recall >= {TARGET_RECALL:.0%}",
            "secondary": "highest validation PR-AUC",
            "target_recall": TARGET_RECALL,
            "false_negative_cost": FALSE_NEGATIVE_COST,
            "false_positive_cost": FALSE_POSITIVE_COST,
        },
        "best_by_model": {name: serialize(score) for name, score in best_by_model.items()},
        "best_by_pr_auc": {name: serialize(score) for name, score in best_by_pr_auc.items()},
        "all_results": [serialize(score) for score in scores],
        "next_step": "Copy the selected settings into config.py, then run the normal CLI for final test evaluation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Search model settings on the validation split")
    parser.add_argument("--data-path", default="data/5year.arff", help="Path to the 5year ARFF file")
    args = parser.parse_args()
    print(json.dumps(run_search(args.data_path), indent=2))


if __name__ == "__main__":
    main()
