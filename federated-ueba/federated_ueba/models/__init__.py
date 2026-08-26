"""Neural network models for behavioural anomaly detection."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from federated_ueba.models.autoencoder import SecurityAutoencoder

__all__ = ["SecurityAutoencoder", "get_parameters", "set_parameters"]


def get_parameters(model: nn.Module) -> list[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
