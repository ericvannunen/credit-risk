from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

from credit_risk.data import load_polish_bankruptcy_five_year
from credit_risk.evaluation import ComparisonResult, compare_models, grade_from_probability
from credit_risk.explanations import prediction_explanations
from credit_risk.models import TrainedModels


def run_training(data_dir: str | Path = "data", data_path: str | Path | None = None) -> Dict[str, object]:
    X, y, feature_names = load_polish_bankruptcy_five_year(data_dir=data_dir, data_path=data_path)
    comparison, models, holdout = compare_models(X, y, feature_names)
    sample_explanation = prediction_explanations(models, holdout["X_test"][0])

    return {
        "dataset": {
            "rows": int(X.shape[0]),
            "features": int(X.shape[1]),
            "source_subset": "5year",
            "target": "bankrupt_within_one_year",
        },
        "comparison": comparison.__dict__,
        "sample_prediction_explanation": sample_explanation,
    }


def run_training_json(data_dir: str | Path = "data", data_path: str | Path | None = None) -> str:
    def json_default(value: object) -> object:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    return json.dumps(run_training(data_dir=data_dir, data_path=data_path), indent=2, default=json_default)


__all__ = [
    "ComparisonResult",
    "TrainedModels",
    "compare_models",
    "grade_from_probability",
    "load_polish_bankruptcy_five_year",
    "prediction_explanations",
    "run_training",
    "run_training_json",
]
