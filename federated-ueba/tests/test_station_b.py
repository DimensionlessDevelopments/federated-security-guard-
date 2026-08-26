"""Tests for the non-ML Station B client and its dispatcher wiring.

These verify the two things that make Station B a valid FedAvg participant
without doing any machine learning:
  1. `fit` echoes the received parameters unchanged (shape-compatible no-op).
  2. The information payload is scalar-valued and deterministic.
Plus that the dispatcher routes partitions to the right client.
"""

from __future__ import annotations

import numpy as np
import pytest

from station_b.client_app import StationBClient
from station_b.reporter import summarize
from station_b.synthetic import FIELD_NAMES, generate_events


def _fake_global_params() -> list[np.ndarray]:
    """Params shaped like SecurityAutoencoder(11, 32, 8).state_dict()."""
    return [
        np.random.randn(32, 11).astype(np.float32),
        np.random.randn(32).astype(np.float32),
        np.random.randn(8, 32).astype(np.float32),
        np.random.randn(8).astype(np.float32),
        np.random.randn(32, 8).astype(np.float32),
        np.random.randn(32).astype(np.float32),
        np.random.randn(11, 32).astype(np.float32),
        np.random.randn(11).astype(np.float32),
    ]


def test_synthetic_events_shape_and_determinism():
    a = generate_events(n_events=250, seed=7)
    b = generate_events(n_events=250, seed=7)
    assert a.events.shape == (250, len(FIELD_NAMES))
    assert np.array_equal(a.events, b.events)  # deterministic for a fixed seed


def test_summary_is_scalar_valued():
    info = summarize(generate_events(500, seed=1004), "station_b")
    expected = {
        "station",
        "node_kind",
        "n_events",
        "n_flagged",
        "flagged_rate",
        "mean_failed_logins",
        "max_records_accessed",
    }
    assert set(info) == expected
    # Flower metrics must be scalars (int/float/str/bool).
    assert all(isinstance(v, (int, float, str, bool)) for v in info.values())
    assert info["n_events"] == 500
    assert 0.0 <= info["flagged_rate"] <= 1.0


def test_fit_echoes_parameters_unchanged():
    """The core FedAvg contract: no training, identical shapes back."""
    client = StationBClient("station_b", seed=1004, n_events=500)
    params = _fake_global_params()
    returned, num_examples, metrics = client.fit(params, {})

    assert [p.shape for p in returned] == [p.shape for p in params]
    assert all(np.array_equal(a, b) for a, b in zip(returned, params))
    assert num_examples == 500
    assert metrics["station"] == "station_b"
    assert metrics["node_kind"] == "station_b_lite"


def test_evaluate_returns_bounded_scalar_loss():
    client = StationBClient("station_b", seed=1004, n_events=500)
    loss, num_examples, metrics = client.evaluate(_fake_global_params(), {})
    assert 0.0 <= loss <= 1.0
    assert num_examples == 500
    assert metrics["n_events"] == 500


def test_no_ml_dependencies():
    """Station B must not *import* from the ML package `federated_ueba`."""
    import ast

    import station_b.client_app as ca
    import station_b.reporter as rp
    import station_b.synthetic as sy

    for mod in (ca, rp, sy):
        tree = ast.parse(open(mod.__file__).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("federated_ueba"), (
                    f"{mod.__file__} imports from federated_ueba"
                )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("federated_ueba"), (
                        f"{mod.__file__} imports federated_ueba"
                    )


def test_dispatcher_routes_by_partition():
    from types import SimpleNamespace

    from federated_ueba.data.generator import STATION_NAMES
    from federated_ueba.federated.client_dispatch import client_fn

    def ctx(pid: int):
        return SimpleNamespace(
            node_config={"partition-id": pid},
            run_config={"hidden-dim": 32, "latent-dim": 8, "station-b-events": 100},
        )

    n = len(STATION_NAMES)
    # Real stations -> ML client; extra partition -> Station B.
    for pid in range(n):
        client = client_fn(ctx(pid))
        inner = getattr(client, "numpy_client", client)
        assert type(inner).__name__ == "StationClient"

    sb = client_fn(ctx(n))
    inner = getattr(sb, "numpy_client", sb)
    assert type(inner).__name__ == "StationBClient"
