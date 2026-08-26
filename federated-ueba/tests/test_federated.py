"""Tests for the federated learning client."""

import numpy as np
import pytest
import torch

from federated_ueba.data.generator import NUM_FEATURES, STATION_NAMES
from federated_ueba.federated.client import (
    BATCH_SIZE,
    LOCAL_EPOCHS,
    StationClient,
)
from federated_ueba.models import (
    SecurityAutoencoder,
    get_parameters,
    set_parameters,
)


def fresh_model():
    return SecurityAutoencoder(input_dim=NUM_FEATURES)


class TestStationClientInit:
    def test_creates_model(self):
        client = StationClient("station_alpha", seed=42)
        assert isinstance(client.model, SecurityAutoencoder)

    def test_stores_station_name(self):
        client = StationClient("station_bravo", seed=42)
        assert client.station == "station_bravo"

    def test_generates_training_data(self):
        client = StationClient("station_alpha", seed=42)
        assert client.features.shape == (500, NUM_FEATURES)
        assert client.features_norm.shape == (500, NUM_FEATURES)

    def test_generates_test_data(self):
        client = StationClient("station_alpha", seed=42)
        assert client.test_features_norm.shape[1] == NUM_FEATURES
        assert len(client.test_labels) == 130  # 100 normal + 30 attack

    def test_test_labels_have_both_classes(self):
        client = StationClient("station_alpha", seed=42)
        assert (client.test_labels == 0).sum() == 100
        assert (client.test_labels == 1).sum() == 30

    def test_computes_normalization_stats(self):
        client = StationClient("station_charlie", seed=42)
        assert client.mean.shape == (NUM_FEATURES,)
        assert client.std.shape == (NUM_FEATURES,)

    @pytest.mark.parametrize("station", STATION_NAMES)
    def test_all_stations_initialize(self, station):
        client = StationClient(station, seed=42)
        assert client.station == station
        assert client.features.shape[0] == 500


class TestStationClientGetParameters:
    def test_returns_numpy_arrays(self):
        client = StationClient("station_alpha", seed=42)
        params = client.get_parameters(config={})
        assert all(isinstance(p, np.ndarray) for p in params)

    def test_parameter_count(self):
        client = StationClient("station_alpha", seed=42)
        params = client.get_parameters(config={})
        assert len(params) == 8


class TestStationClientFit:
    @pytest.fixture
    def client(self):
        return StationClient("station_alpha", seed=42)

    def test_returns_updated_parameters(self, client):
        initial_params = get_parameters(fresh_model())
        updated_params, num_examples, metrics = client.fit(
            initial_params, config={}
        )
        assert len(updated_params) == len(initial_params)
        assert num_examples == 500
        assert isinstance(metrics, dict)

    def test_parameters_change_after_fit(self, client):
        initial_params = get_parameters(fresh_model())
        updated_params, _, _ = client.fit(initial_params, config={})
        differs = any(
            not np.array_equal(i, u)
            for i, u in zip(initial_params, updated_params)
        )
        assert differs

    def test_fit_with_received_parameters(self, client):
        model_a = fresh_model()
        params_a = get_parameters(model_a)
        updated, n, _ = client.fit(params_a, config={})
        assert n == 500
        assert len(updated) == 8

    @pytest.mark.parametrize("station", STATION_NAMES)
    def test_fit_all_stations(self, station):
        client = StationClient(station, seed=42)
        params = get_parameters(fresh_model())
        updated, n, _ = client.fit(params, config={})
        assert n == 500
        assert len(updated) == 8


class TestStationClientFitConfig:
    """The server pushes learning-rate and local-epochs via ConfigRecord."""

    @pytest.fixture
    def client(self):
        return StationClient("station_alpha", seed=42)

    def test_zero_epochs_returns_parameters_unchanged(self, client):
        initial = get_parameters(fresh_model())
        updated, n, _ = client.fit(initial, config={"local-epochs": 0})
        assert n == 500
        for i, u in zip(initial, updated):
            np.testing.assert_array_equal(i, u)

    def test_config_epochs_trains(self, client):
        initial = get_parameters(fresh_model())
        updated, _, _ = client.fit(
            initial, config={"local-epochs": 1, "learning-rate": 0.001}
        )
        assert any(not np.array_equal(i, u) for i, u in zip(initial, updated))

    def test_config_values_accept_strings(self, client):
        """Run-config values may arrive as strings; fit coerces them."""
        initial = get_parameters(fresh_model())
        updated, n, _ = client.fit(
            initial, config={"local-epochs": "1", "learning-rate": "0.001"}
        )
        assert n == 500
        assert len(updated) == 8

    def test_empty_config_uses_module_defaults(self, client):
        initial = get_parameters(fresh_model())
        updated, n, _ = client.fit(initial, config={})
        assert n == 500
        assert any(not np.array_equal(i, u) for i, u in zip(initial, updated))


class TestServerApp:
    """Smoke tests for the upstream ServerApp module."""

    def test_module_imports_and_exposes_app(self):
        from flwr.serverapp import ServerApp

        from federated_ueba.federated import server

        assert isinstance(server.app, ServerApp)


class TestConfigAlignment:
    """Server and client must build shape-compatible models from run config."""

    def test_pyproject_config_matches_feature_schema(self):
        import tomllib
        from pathlib import Path

        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        config = tomllib.loads(pyproject.read_text())
        app_config = config["tool"]["flwr"]["app"]["config"]

        assert app_config["input-dim"] == NUM_FEATURES

    def test_pyproject_components_point_at_real_apps(self):
        import importlib
        import tomllib
        from pathlib import Path

        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        config = tomllib.loads(pyproject.read_text())
        components = config["tool"]["flwr"]["app"]["components"]

        for ref in components.values():
            module_path, attr = ref.split(":")
            module = importlib.import_module(module_path)
            assert hasattr(module, attr), f"{ref} does not resolve"

    def test_client_dims_match_server_model(self):
        """A client model must load the server's initial parameters."""
        from federated_ueba.models import SecurityAutoencoder, get_parameters

        server_model = SecurityAutoencoder(
            input_dim=NUM_FEATURES, hidden_dim=32, latent_dim=8
        )
        client = StationClient("station_alpha", seed=42)
        # raises on shape mismatch
        updated, n, _ = client.fit(
            get_parameters(server_model), config={"local-epochs": 0}
        )
        assert len(updated) == 8

    def test_client_custom_dims(self):
        client = StationClient(
            "station_alpha", seed=42, hidden_dim=16, latent_dim=4
        )
        params = client.get_parameters(config={})
        # encoder: 11->16->4, decoder: 4->16->11 (weight+bias each)
        assert params[0].shape == (16, NUM_FEATURES)
        assert params[2].shape == (4, 16)

    def test_make_client_reads_run_config(self):
        from types import SimpleNamespace

        from federated_ueba.federated.client import make_client

        context = SimpleNamespace(
            node_config={"partition-id": 1},
            run_config={"hidden-dim": 16, "latent-dim": 4},
        )
        station_client = make_client(context)
        assert station_client.station == STATION_NAMES[1]
        params = get_parameters(station_client.model)
        assert params[0].shape == (16, NUM_FEATURES)
        assert params[2].shape == (4, 16)

    def test_client_app_uses_message_api(self):
        """The client must speak the same API as the server's strategy.start."""
        from flwr.clientapp import ClientApp

        from federated_ueba.federated import client

        assert isinstance(client.app, ClientApp)


class TestStationClientEvaluate:
    @pytest.fixture
    def trained_client(self):
        client = StationClient("station_alpha", seed=42)
        params = get_parameters(fresh_model())
        client.fit(params, config={})
        return client

    def test_returns_loss_and_metrics(self, trained_client):
        params = trained_client.get_parameters(config={})
        loss, num_examples, metrics = trained_client.evaluate(params, config={})

        assert isinstance(loss, float)
        assert loss >= 0
        assert num_examples == 130

    def test_metrics_contain_required_keys(self, trained_client):
        params = trained_client.get_parameters(config={})
        _, _, metrics = trained_client.evaluate(params, config={})

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "fpr" in metrics
        assert "threshold" in metrics
        assert "station" in metrics

    def test_metrics_in_valid_range(self, trained_client):
        params = trained_client.get_parameters(config={})
        _, _, metrics = trained_client.evaluate(params, config={})

        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["fpr"] <= 1.0
        assert metrics["threshold"] >= 0.0
        assert metrics["station"] == "station_alpha"

    def test_evaluate_with_external_parameters(self, trained_client):
        other = StationClient("station_bravo", seed=99)
        other_params = get_parameters(other.model)
        loss, n, metrics = trained_client.evaluate(other_params, config={})
        assert isinstance(loss, float)
        assert n == 130


class TestDataStaysLocal:
    """The spec's core privacy claim: zero raw event data leaves the client.

    Everything the client returns to the server must be model parameters
    (shapes fixed by the architecture, independent of dataset size) or
    scalar metrics -- never feature rows.
    """

    @pytest.fixture
    def client(self):
        return StationClient("station_alpha", seed=42)

    def test_fit_payload_is_model_shaped_only(self, client):
        model_shapes = [p.shape for p in get_parameters(fresh_model())]
        params, _, metrics = client.fit(
            get_parameters(fresh_model()), config={}
        )

        assert [p.shape for p in params] == model_shapes
        # No returned array matches the local dataset's row count
        for p in params:
            assert client.features.shape[0] not in p.shape

    def test_fit_metrics_contain_no_arrays(self, client):
        _, _, metrics = client.fit(get_parameters(fresh_model()), config={})
        for value in metrics.values():
            assert np.isscalar(value), f"non-scalar in fit metrics: {value!r}"

    def test_evaluate_metrics_are_scalars_only(self, client):
        params = get_parameters(fresh_model())
        client.fit(params, config={})
        _, _, metrics = client.evaluate(
            client.get_parameters(config={}), config={}
        )

        for key, value in metrics.items():
            assert np.isscalar(value), f"non-scalar metric {key}: {value!r}"

    def test_raw_features_never_in_payload(self, client):
        """No parameter array shares memory with, or equals, the raw features."""
        params, _, _ = client.fit(get_parameters(fresh_model()), config={})
        for p in params:
            assert not np.shares_memory(p, client.features)
            assert p.shape != client.features.shape


class TestFederatedRoundtrip:
    """Simulates a simplified federated round without the Flower runtime."""

    def test_two_clients_parameter_exchange(self):
        client_a = StationClient("station_alpha", seed=42)
        client_b = StationClient("station_bravo", seed=99)

        global_params = get_parameters(fresh_model())

        params_a, n_a, _ = client_a.fit(global_params, config={})
        params_b, n_b, _ = client_b.fit(global_params, config={})

        # FedAvg: weighted average
        total = n_a + n_b
        aggregated = [
            (a * n_a + b * n_b) / total for a, b in zip(params_a, params_b)
        ]

        loss_a, _, metrics_a = client_a.evaluate(aggregated, config={})
        loss_b, _, metrics_b = client_b.evaluate(aggregated, config={})

        assert loss_a >= 0
        assert loss_b >= 0
        assert 0 <= metrics_a["accuracy"] <= 1
        assert 0 <= metrics_b["accuracy"] <= 1

    def test_multiple_rounds_reduce_loss(self):
        client = StationClient("station_alpha", seed=42)
        params = get_parameters(fresh_model())

        losses = []
        for _ in range(3):
            params, _, _ = client.fit(params, config={})
            loss, _, _ = client.evaluate(params, config={})
            losses.append(loss)

        assert losses[-1] < losses[0]

    def test_all_stations_converge(self):
        clients = [
            StationClient(station, seed=42 + i * 100)
            for i, station in enumerate(STATION_NAMES)
        ]
        params = get_parameters(fresh_model())

        for round_num in range(3):
            all_params = []
            all_n = []
            for c in clients:
                p, n, _ = c.fit(params, config={})
                all_params.append(p)
                all_n.append(n)

            total = sum(all_n)
            params = [
                sum(p[i] * n for p, n in zip(all_params, all_n)) / total
                for i in range(len(all_params[0]))
            ]

        for c in clients:
            loss, _, metrics = c.evaluate(params, config={})
            assert loss >= 0
            assert 0 <= metrics["accuracy"] <= 1
