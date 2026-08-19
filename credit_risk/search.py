from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import optuna

from credit_risk.config import LogisticConfig, NeuralNetworkConfig
from credit_risk.data import load_polish_bankruptcy_five_year
from credit_risk.evaluation import (
    FALSE_NEGATIVE_COST,
    FALSE_POSITIVE_COST,
    ValidationScore,
    score_configuration_on_validation,
)

TARGET_RECALL = 0.70
N_TRIALS = 100


def _logistic_objective(
    trial: optuna.Trial,
    X,
    y,
) -> float:
    c = trial.suggest_float("c", 0.01, 100.0, log=True)
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
    logistic_config = LogisticConfig(c=c, class_weight=class_weight)
    scores = score_configuration_on_validation(
        X,
        y,
        logistic_config=logistic_config,
        false_negative_cost=FALSE_NEGATIVE_COST,
        false_positive_cost=FALSE_POSITIVE_COST,
        target_recall=TARGET_RECALL,
    )
    score = scores[0]
    trial.set_user_attr("pr_auc", score.pr_auc)
    trial.set_user_attr("precision", score.precision)
    trial.set_user_attr("recall", score.recall)
    trial.set_user_attr("decision_threshold", score.decision_threshold)
    trial.set_user_attr("decision_cost", score.decision_cost)
    return score.decision_cost


def _nn_objective(
    trial: optuna.Trial,
    X,
    y,
) -> float:
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    pos_weight_multiplier = trial.suggest_float("pos_weight_multiplier", 0.5, 5.0)
    hidden_size_1 = trial.suggest_int("hidden_size_1", 16, 128)
    hidden_size_2 = trial.suggest_int("hidden_size_2", 8, 64)
    neural_network_config = NeuralNetworkConfig(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        pos_weight_multiplier=pos_weight_multiplier,
        hidden_size_1=hidden_size_1,
        hidden_size_2=hidden_size_2,
    )
    scores = score_configuration_on_validation(
        X,
        y,
        neural_network_config=neural_network_config,
        false_negative_cost=FALSE_NEGATIVE_COST,
        false_positive_cost=FALSE_POSITIVE_COST,
        target_recall=TARGET_RECALL,
    )
    score = scores[1]
    trial.set_user_attr("pr_auc", score.pr_auc)
    trial.set_user_attr("precision", score.precision)
    trial.set_user_attr("recall", score.recall)
    trial.set_user_attr("decision_threshold", score.decision_threshold)
    trial.set_user_attr("decision_cost", score.decision_cost)
    return score.decision_cost


def _trial_to_validation_score(trial: optuna.Trial, model_name: str) -> ValidationScore:
    return ValidationScore(
        model=model_name,
        configuration=trial.params,
        pr_auc=trial.user_attrs["pr_auc"],
        precision=trial.user_attrs["precision"],
        recall=trial.user_attrs["recall"],
        decision_threshold=trial.user_attrs["decision_threshold"],
        decision_cost=trial.user_attrs["decision_cost"],
    )


def run_search(data_path: str | Path, n_trials: int = N_TRIALS) -> dict[str, object]:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X, y, _ = load_polish_bankruptcy_five_year(data_path=data_path)

    pruner = optuna.pruners.MedianPruner()

    logistic_study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=pruner,
    )
    logistic_study.optimize(
        lambda trial: _logistic_objective(trial, X, y),
        n_trials=n_trials,
    )

    nn_study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=pruner,
    )
    nn_study.optimize(
        lambda trial: _nn_objective(trial, X, y),
        n_trials=n_trials,
    )

    logistic_scores = [
        _trial_to_validation_score(t, "logistic_regression")
        for t in logistic_study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    nn_scores = [
        _trial_to_validation_score(t, "neural_net")
        for t in nn_study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    all_scores = logistic_scores + nn_scores

    best_by_model: dict[str, ValidationScore] = {}
    best_by_pr_auc: dict[str, ValidationScore] = {}
    for score in all_scores:
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

    logistic_best_trial = logistic_study.best_trial
    nn_best_trial = nn_study.best_trial

    return {
        "selection_rule": {
            "primary": f"lowest validation decision cost subject to recall >= {TARGET_RECALL:.0%}",
            "secondary": "highest validation PR-AUC",
            "target_recall": TARGET_RECALL,
            "false_negative_cost": FALSE_NEGATIVE_COST,
            "false_positive_cost": FALSE_POSITIVE_COST,
        },
        "optimization_metadata": {
            "method": "TPE (Bayesian optimization via Optuna)",
            "n_trials_per_model": n_trials,
            "logistic_best_trial_number": logistic_best_trial.number,
            "nn_best_trial_number": nn_best_trial.number,
            "logistic_completed_trials": len(logistic_scores),
            "nn_completed_trials": len(nn_scores),
        },
        "best_by_model": {name: serialize(score) for name, score in best_by_model.items()},
        "best_by_pr_auc": {name: serialize(score) for name, score in best_by_pr_auc.items()},
        "all_results": [serialize(score) for score in all_scores],
        "next_step": "Copy the selected settings into config.py, then run the normal CLI for final test evaluation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Search model settings on the validation split")
    parser.add_argument("--data-path", default="data/5year.arff", help="Path to the 5year ARFF file")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS, help="Number of Optuna trials per model")
    args = parser.parse_args()
    print(json.dumps(run_search(args.data_path, n_trials=args.n_trials), indent=2))


if __name__ == "__main__":
    main()
