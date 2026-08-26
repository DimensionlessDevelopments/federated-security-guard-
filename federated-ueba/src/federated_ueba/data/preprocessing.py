"""Feature preprocessing utilities."""

from __future__ import annotations

import numpy as np


def normalize_features(
    features: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalize features. Returns (normalized, mean, std)."""
    if mean is None:
        mean = features.mean(axis=0)
    if std is None:
        std = features.std(axis=0) + 1e-8
    return ((features - mean) / std).astype(np.float32), mean, std
