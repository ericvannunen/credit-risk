from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required to run the neural-network comparison. Install dependencies from pyproject.toml."
    ) from exc


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
class NNPreprocessor:
    imputer: SimpleImputer
    scaler: StandardScaler


def fit_neural_network(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 80,
    learning_rate: float = 1e-3,
) -> Tuple[NNPreprocessor, SmallRiskNet]:
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train_imp)
    X_val_scaled = scaler.transform(imputer.transform(X_val))

    x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.astype(np.float32), dtype=torch.float32)
    x_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.astype(np.float32), dtype=torch.float32)

    model = SmallRiskNet(input_dim=X_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Bankruptcy is the minority class, so missed bankruptcies receive more
    # loss than correct predictions for the majority class.
    positives = float(np.sum(y_train == 1))
    negatives = float(np.sum(y_train == 0))
    pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Keep the validation-best weights rather than the weights from the last
    # epoch, which may have started to overfit the training set.
    best_state = None
    best_val_loss = float("inf")
    stale_epochs = 0
    for _ in range(epochs):
        model.train()
        # One full-batch gradient update per epoch: forward pass, loss,
        # backpropagation, then an optimizer step.
        optimizer.zero_grad()
        logits = model(x_tensor)
        loss = criterion(logits, y_tensor)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val_tensor)
            val_loss = float(criterion(val_logits, y_val_tensor).item())
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= 10:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return NNPreprocessor(imputer=imputer, scaler=scaler), model


def nn_predict_proba(
    model: SmallRiskNet,
    preprocessor: NNPreprocessor,
    X: np.ndarray,
) -> np.ndarray:
    X_imp = preprocessor.imputer.transform(X)
    X_scaled = preprocessor.scaler.transform(X_imp)
    x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        logits = model(x_tensor)
        return torch.sigmoid(logits).cpu().numpy()
