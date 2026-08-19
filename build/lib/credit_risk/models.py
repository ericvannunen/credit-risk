from __future__ import annotations

from dataclasses import dataclass

from sklearn.pipeline import Pipeline

from credit_risk.calibration import SigmoidCalibrator
from credit_risk.neural_network import NNPreprocessor, SmallRiskNet


@dataclass
class TrainedModels:
    """The fitted models and preprocessing objects used for explanations."""

    logistic_pipeline: Pipeline
    nn_preprocessor: NNPreprocessor
    nn_model: SmallRiskNet
    feature_names: list[str]
    logistic_calibrator: SigmoidCalibrator | None = None
    nn_calibrator: SigmoidCalibrator | None = None


# Keep these imports available for callers that used credit_risk.models before
# the individual model implementations were split into separate modules.
from credit_risk.logistic_model import fit_logistic
from credit_risk.neural_network import fit_neural_network, nn_predict_proba

__all__ = [
    "NNPreprocessor",
    "SmallRiskNet",
    "TrainedModels",
    "fit_logistic",
    "fit_neural_network",
    "nn_predict_proba",
]
