from __future__ import annotations

import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.error import URLError
from urllib.request import urlretrieve

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required to run the neural-network comparison. Install dependencies from pyproject.toml."
    ) from exc

UCI_DATA_URL = "https://archive.ics.uci.edu/static/public/365/polish+companies+bankruptcy+data.zip"
UCI_ONE_YEAR_FILE = "1year.arff"


@dataclass
class ComparisonResult:
    logistic_auc: float
    logistic_brier: float
    logistic_ece: float
    nn_auc: float
    nn_brier: float
    nn_ece: float
    preferred_model: str


class SmallRiskNet(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


@dataclass
class TrainedModels:
    logistic_pipeline: Pipeline
    nn_scaler: StandardScaler
    nn_model: SmallRiskNet
    feature_names: List[str]


def _download_and_extract(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    arff_path = data_dir / UCI_ONE_YEAR_FILE
    if arff_path.exists():
        return arff_path

    zip_path = data_dir / "polish-bankruptcy.zip"
    try:
        urlretrieve(UCI_DATA_URL, zip_path)
    except URLError as exc:
        raise RuntimeError(
            "Unable to download the UCI dataset in this environment. "
            "Provide a local path to 1year.arff via load_polish_bankruptcy_one_year(data_path=...)."
        ) from exc
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extract(UCI_ONE_YEAR_FILE, path=data_dir)
    return arff_path


def _parse_arff(arff_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    feature_names: List[str] = []
    rows: List[List[float]] = []
    targets: List[int] = []
    in_data = False

    with arff_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue
            lowered = line.lower()
            if lowered.startswith("@attribute") and not in_data:
                parts = line.split()
                if len(parts) >= 2 and parts[1] != "class":
                    feature_names.append(parts[1])
                continue
            if lowered.startswith("@data"):
                in_data = True
                continue
            if not in_data:
                continue

            values = [v.strip() for v in line.split(",")]
            if len(values) < 2:
                continue
            x_vals = [math.nan if v == "?" else float(v) for v in values[:-1]]
            y_val = int(values[-1])
            rows.append(x_vals)
            targets.append(y_val)

    X = np.array(rows, dtype=float)
    y = np.array(targets, dtype=int)
    return X, y, feature_names


def load_polish_bankruptcy_one_year(
    data_dir: str | Path = "data",
    data_path: str | Path | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    arff_path = Path(data_path) if data_path else _download_and_extract(Path(data_dir))
    return _parse_arff(arff_path)


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


def _expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(len(y_true))
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs >= lo) & (probs < hi if i < bins - 1 else probs <= hi)
        if not np.any(mask):
            continue
        bin_conf = float(np.mean(probs[mask]))
        bin_acc = float(np.mean(y_true[mask]))
        ece += abs(bin_acc - bin_conf) * (float(np.sum(mask)) / total)
    return ece


def _fit_logistic(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    ).fit(X_train, y_train)


def _fit_neural_network(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 80,
    learning_rate: float = 1e-3,
) -> Tuple[StandardScaler, SmallRiskNet]:
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train_imp)

    x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.astype(np.float32), dtype=torch.float32)

    model = SmallRiskNet(input_dim=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    positives = float(np.sum(y_train == 1))
    negatives = float(np.sum(y_train == 0))
    pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x_tensor)
        loss = criterion(logits, y_tensor)
        loss.backward()
        optimizer.step()

    scaler._creditrisk_imputer = imputer  # type: ignore[attr-defined]
    return scaler, model


def _nn_predict_proba(model: SmallRiskNet, scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    imputer: SimpleImputer = scaler._creditrisk_imputer  # type: ignore[attr-defined]
    X_imp = imputer.transform(X)
    X_scaled = scaler.transform(X_imp)
    x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def compare_models(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    random_state: int = 42,
) -> Tuple[ComparisonResult, TrainedModels, Dict[str, np.ndarray]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=random_state,
        stratify=y,
    )

    logistic = _fit_logistic(X_train, y_train)
    log_probs = logistic.predict_proba(X_test)[:, 1]

    nn_scaler, nn_model = _fit_neural_network(X_train, y_train)
    nn_probs = _nn_predict_proba(nn_model, nn_scaler, X_test)

    result = ComparisonResult(
        logistic_auc=float(roc_auc_score(y_test, log_probs)),
        logistic_brier=float(brier_score_loss(y_test, log_probs)),
        logistic_ece=float(_expected_calibration_error(y_test, log_probs)),
        nn_auc=float(roc_auc_score(y_test, nn_probs)),
        nn_brier=float(brier_score_loss(y_test, nn_probs)),
        nn_ece=float(_expected_calibration_error(y_test, nn_probs)),
        preferred_model=(
            "neural_net"
            if roc_auc_score(y_test, nn_probs) > roc_auc_score(y_test, log_probs)
            and _expected_calibration_error(y_test, nn_probs)
            <= (_expected_calibration_error(y_test, log_probs) + 0.02)
            else "logistic_regression"
        ),
    )

    trained = TrainedModels(
        logistic_pipeline=logistic,
        nn_scaler=nn_scaler,
        nn_model=nn_model,
        feature_names=feature_names,
    )
    holdout = {"X_test": X_test, "y_test": y_test, "log_probs": log_probs, "nn_probs": nn_probs}
    return result, trained, holdout


def prediction_explanations(models: TrainedModels, company_row: np.ndarray, top_k: int = 5) -> Dict[str, object]:
    if company_row.ndim != 1:
        raise ValueError("company_row must be a 1D array")

    logistic = models.logistic_pipeline
    imputer: SimpleImputer = logistic.named_steps["imputer"]
    scaler: StandardScaler = logistic.named_steps["scaler"]
    clf: LogisticRegression = logistic.named_steps["model"]

    row_imp = imputer.transform(company_row.reshape(1, -1))
    row_scaled = scaler.transform(row_imp)
    logit_contrib = row_scaled[0] * clf.coef_[0]
    log_prob = float(logistic.predict_proba(company_row.reshape(1, -1))[0, 1])

    nn_imputer: SimpleImputer = models.nn_scaler._creditrisk_imputer  # type: ignore[attr-defined]
    row_nn = models.nn_scaler.transform(nn_imputer.transform(company_row.reshape(1, -1)))
    x_tensor = torch.tensor(row_nn, dtype=torch.float32, requires_grad=True)
    models.nn_model.eval()
    nn_prob_tensor = torch.sigmoid(models.nn_model(x_tensor))
    nn_prob_tensor.backward()
    nn_contrib = (x_tensor.grad.detach().numpy()[0] * row_nn[0]).astype(float)
    nn_prob = float(nn_prob_tensor.item())

    def top_features(values: np.ndarray) -> List[Dict[str, float | str]]:
        idxs = np.argsort(np.abs(values))[::-1][:top_k]
        return [
            {"feature": models.feature_names[int(i)], "contribution": float(values[int(i)])}
            for i in idxs
        ]

    return {
        "logistic": {
            "probability": log_prob,
            "risk_grade": grade_from_probability(log_prob),
            "top_factors": top_features(logit_contrib),
        },
        "neural_net": {
            "probability": nn_prob,
            "risk_grade": grade_from_probability(nn_prob),
            "top_factors": top_features(nn_contrib),
        },
    }


def run_training(data_dir: str | Path = "data", data_path: str | Path | None = None) -> Dict[str, object]:
    X, y, feature_names = load_polish_bankruptcy_one_year(data_dir=data_dir, data_path=data_path)
    comparison, models, holdout = compare_models(X, y, feature_names)
    sample_explanation = prediction_explanations(models, holdout["X_test"][0])

    return {
        "dataset": {
            "rows": int(X.shape[0]),
            "features": int(X.shape[1]),
            "target": "bankrupt_within_one_year",
        },
        "comparison": comparison.__dict__,
        "sample_prediction_explanation": sample_explanation,
    }


def run_training_json(data_dir: str | Path = "data", data_path: str | Path | None = None) -> str:
    return json.dumps(run_training(data_dir=data_dir, data_path=data_path), indent=2)
