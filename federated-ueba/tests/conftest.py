"""Shared fixtures for federated UEBA tests."""

import numpy as np
import pytest
import torch

from federated_ueba.data.generator import NUM_FEATURES
from federated_ueba.models import SecurityAutoencoder


@pytest.fixture
def model():
    torch.manual_seed(0)
    return SecurityAutoencoder(input_dim=NUM_FEATURES)


@pytest.fixture
def random_features():
    rng = np.random.default_rng(42)
    return rng.standard_normal((100, NUM_FEATURES)).astype(np.float32)
