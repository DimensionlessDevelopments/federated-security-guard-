"""Tests for the SecurityAutoencoder model."""

import numpy as np
import torch

from federated_ueba.data.generator import FEATURE_NAMES, NUM_FEATURES
from federated_ueba.detection.scoring import reconstruction_error
from federated_ueba.models import (
    SecurityAutoencoder,
    get_parameters,
    set_parameters,
)

LATENT_DIM = 8  # upstream default


class TestSecurityAutoencoder:
    def test_feature_count_matches_names(self):
        assert NUM_FEATURES == len(FEATURE_NAMES) == 11

    def test_forward_preserves_shape(self, model):
        x = torch.randn(16, NUM_FEATURES)
        out = model(x)
        assert out.shape == (16, NUM_FEATURES)

    def test_encoder_bottleneck_dimension(self, model):
        x = torch.randn(8, NUM_FEATURES)
        encoded = model.encoder(x)
        assert encoded.shape == (8, LATENT_DIM)

    def test_custom_dimensions(self):
        model = SecurityAutoencoder(input_dim=10, hidden_dim=16, latent_dim=4)
        x = torch.randn(5, 10)
        assert model(x).shape == (5, 10)
        assert model.encoder(x).shape == (5, 4)

    def test_single_sample(self, model):
        x = torch.randn(1, NUM_FEATURES)
        out = model(x)
        assert out.shape == (1, NUM_FEATURES)


class TestParameterSerialization:
    def test_get_parameters_returns_numpy_arrays(self, model):
        params = get_parameters(model)
        assert all(isinstance(p, np.ndarray) for p in params)

    def test_roundtrip_preserves_weights(self, model):
        params_before = get_parameters(model)
        new_model = SecurityAutoencoder(input_dim=NUM_FEATURES)
        set_parameters(new_model, params_before)
        params_after = get_parameters(new_model)

        for before, after in zip(params_before, params_after):
            np.testing.assert_array_equal(before, after)

    def test_set_parameters_changes_model(self, model):
        original_params = [p.copy() for p in get_parameters(model)]
        noisy = [p + 1.0 for p in original_params]
        set_parameters(model, noisy)
        new_params = get_parameters(model)

        for orig, updated in zip(original_params, new_params):
            np.testing.assert_allclose(updated, orig + 1.0, rtol=1e-5)

    def test_parameter_count(self, model):
        params = get_parameters(model)
        # 4 linear layers (2 encoder + 2 decoder), each with weight + bias = 8
        assert len(params) == 8


class TestReconstructionError:
    def test_returns_per_sample_scores(self, model):
        x = torch.randn(20, NUM_FEATURES)
        errors = reconstruction_error(model, x)
        assert errors.shape == (20,)

    def test_scores_are_non_negative(self, model):
        x = torch.randn(50, NUM_FEATURES)
        errors = reconstruction_error(model, x)
        assert (errors >= 0).all()

    def test_zero_input_produces_nonzero_error(self, model):
        x = torch.zeros(5, NUM_FEATURES)
        errors = reconstruction_error(model, x)
        assert errors.sum() > 0

    def test_model_set_to_eval(self, model):
        model.train()
        x = torch.randn(5, NUM_FEATURES)
        reconstruction_error(model, x)
        assert not model.training
