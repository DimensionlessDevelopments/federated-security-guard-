"""Anomaly detector using the autoencoder's reconstruction error."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from federated_ueba.data.generator import NUM_FEATURES
from federated_ueba.detection.scoring import reconstruction_error
from federated_ueba.models import (
    SecurityAutoencoder,
    get_parameters,
    set_parameters,
)


@dataclass
class DetectionResult:
    anomaly_scores: np.ndarray
    predictions: np.ndarray
    threshold: float


class AnomalyDetector:
    """Wraps the autoencoder with training, scoring, and thresholding."""

    def __init__(
        self,
        threshold_percentile: float = 95.0,
        lr: float = 0.001,
        epochs: int = 20,
        batch_size: int = 32,
    ) -> None:
        self.model = SecurityAutoencoder(input_dim=NUM_FEATURES)
        self.threshold_percentile = threshold_percentile
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold: float = 0.5
        self.train_mean: np.ndarray | None = None
        self.train_std: np.ndarray | None = None

    def fit(self, normal_features: np.ndarray) -> dict[str, list[float]]:
        """Train the autoencoder on normal activity and set the threshold."""
        self.train_mean = normal_features.mean(axis=0)
        self.train_std = normal_features.std(axis=0) + 1e-8
        normalized = (normal_features - self.train_mean) / self.train_std

        X = torch.tensor(normalized, dtype=torch.float32)
        dataset = TensorDataset(X, X)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        losses: list[float] = []
        self.model.train()
        for _ in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, _ in loader:
                optimizer.zero_grad()
                recon = self.model(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch_x)
            losses.append(epoch_loss / len(X))

        errors = reconstruction_error(self.model, X).numpy()
        self.threshold = float(np.percentile(errors, self.threshold_percentile))

        return {"losses": losses}

    def score(self, features: np.ndarray) -> np.ndarray:
        """Return per-sample anomaly scores."""
        if self.train_mean is None:
            raise RuntimeError("Detector not fitted -- call fit() first")
        normalized = (features - self.train_mean) / self.train_std
        X = torch.tensor(normalized, dtype=torch.float32)
        return reconstruction_error(self.model, X).numpy()

    def detect(self, features: np.ndarray) -> DetectionResult:
        """Score and classify events as normal or anomalous."""
        scores = self.score(features)
        predictions = (scores > self.threshold).astype(np.int64)
        return DetectionResult(
            anomaly_scores=scores,
            predictions=predictions,
            threshold=self.threshold,
        )

    def get_parameters(self) -> list[np.ndarray]:
        return get_parameters(self.model)

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        set_parameters(self.model, parameters)
