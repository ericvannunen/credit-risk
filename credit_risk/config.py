from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LogisticConfig:
    """Settings that control logistic-regression training."""

    c: float = 10.0
    class_weight: str | dict[int, float] | None = None
    max_iter: int = 2000


@dataclass(frozen=True)
class NeuralNetworkConfig:
    """Settings that control neural-network architecture and training."""

    hidden_size_1: int = 32
    hidden_size_2: int = 16
    epochs: int = 150
    learning_rate: float = 0.001
    weight_decay: float = 0.001
    pos_weight_multiplier: float = 1.0
    random_seed: int = 42


DEFAULT_LOGISTIC_CONFIG = LogisticConfig()
DEFAULT_NEURAL_NETWORK_CONFIG = NeuralNetworkConfig()
