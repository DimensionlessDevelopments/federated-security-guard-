"""Anomaly scoring based on autoencoder reconstruction error."""

from __future__ import annotations

import torch
import torch.nn as nn


def reconstruction_error(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Per-sample mean squared reconstruction error."""
    model.eval()
    with torch.no_grad():
        recon = model(x)
        return ((x - recon) ** 2).mean(dim=1)
