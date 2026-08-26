"""Evaluation utilities for comparing training modes (local vs federated).

Reproduces the federated clients' data exactly (same seeds and
normalization) so a global model artifact can be scored per station on:
  - the station's held-out mixed test set (station-specific attacks)
  - "full" cross-station attacks (tactics first seen at other stations)
"""

from __future__ import annotations

import numpy as np
import torch

from federated_ueba.data.generator import (
    NUM_FEATURES,
    STATION_NAMES,
    generate_normal_events,
)
from federated_ueba.data.preprocessing import normalize_features
from federated_ueba.detection.scoring import reconstruction_error
from federated_ueba.models import SecurityAutoencoder
from federated_ueba.simulation.attack_generator import (
    generate_attack_events,
    generate_mixed_dataset,
)

THRESHOLD_PERCENTILE = 95.0


def station_partition_seed(partition_id: int) -> int:
    """Seed mapping used by the federated clients."""
    return 42 + partition_id * 100


def station_metrics(
    model: SecurityAutoencoder,
    station: str,
    partition_id: int,
) -> dict[str, float]:
    """Score a model on one station's test data, client-style."""
    seed = station_partition_seed(partition_id)

    normal = generate_normal_events(station, n_samples=500, seed=seed)
    _, mean, std = normalize_features(normal.features)

    test_features, test_labels = generate_mixed_dataset(
        station, n_normal=100, n_attack=30, seed=seed + 1000
    )
    test_norm = ((test_features - mean) / std).astype(np.float32)

    errors = reconstruction_error(model, torch.tensor(test_norm)).numpy()
    threshold = float(
        np.percentile(errors[test_labels == 0], THRESHOLD_PERCENTILE)
    )
    preds = (errors > threshold).astype(int)

    tp = int(((preds == 1) & (test_labels == 1)).sum())
    fp = int(((preds == 1) & (test_labels == 0)).sum())
    fn = int(((preds == 0) & (test_labels == 1)).sum())
    tn = int(((preds == 0) & (test_labels == 0)).sum())

    full_attacks, _ = generate_attack_events(
        station, n_samples=200, seed=seed + 2000, attack_type="full"
    )
    full_norm = ((full_attacks - mean) / std).astype(np.float32)
    full_errors = reconstruction_error(model, torch.tensor(full_norm)).numpy()

    return {
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "accuracy": (tp + tn) / len(test_labels),
        "full_attack_recall": float((full_errors > threshold).mean()),
    }


def train_local_baseline(
    station: str,
    partition_id: int,
    epochs: int = 10,
    lr: float = 0.001,
    batch_size: int = 32,
) -> SecurityAutoencoder:
    """Train a local-only model with the same budget as a federated run."""
    seed = station_partition_seed(partition_id)
    torch.manual_seed(partition_id)
    model = SecurityAutoencoder(input_dim=NUM_FEATURES)

    normal = generate_normal_events(station, n_samples=500, seed=seed)
    normalized, _, _ = normalize_features(normal.features)
    X = torch.tensor(normalized)
    dataset = torch.utils.data.TensorDataset(X, X)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        for batch_x, _ in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_x)
            loss.backward()
            optimizer.step()
    return model


def compare_modes(
    global_model: SecurityAutoencoder,
    local_epochs_budget: int = 10,
) -> dict[str, dict[str, dict[str, float]]]:
    """Per-station metrics for the global model vs local-only baselines."""
    results: dict[str, dict[str, dict[str, float]]] = {}
    for pid, station in enumerate(STATION_NAMES):
        local = train_local_baseline(station, pid, epochs=local_epochs_budget)
        results[station] = {
            "local": station_metrics(local, station, pid),
            "federated": station_metrics(global_model, station, pid),
        }
    return results


def mean_metric(
    results: dict[str, dict[str, dict[str, float]]],
    mode: str,
    metric: str,
) -> float:
    return float(
        np.mean([results[s][mode][metric] for s in results])
    )
