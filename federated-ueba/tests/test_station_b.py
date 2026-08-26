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


def _ctx(pid: int):
    from types import SimpleNamespace

    return SimpleNamespace(
        node_config={"partition-id": pid},
        run_config={"hidden-dim": 32, "latent-dim": 8, "station-b-events": 100},
    )


def test_dispatcher_routes_by_partition():
    """Partitions 0..N-1 are ML stations; N and beyond are Station B."""
    from federated_ueba.data.generator import STATION_NAMES
    from federated_ueba.federated.client_dispatch import is_station_b

    n = len(STATION_NAMES)
    for pid in range(n):
        assert is_station_b(_ctx(pid)) is False
    assert is_station_b(_ctx(n)) is True
    assert is_station_b(_ctx(n + 3)) is True


def test_make_station_b_client_reads_context():
    from station_b.client_app import StationBClient, make_station_b_client

    client = make_station_b_client(_ctx(4))
    assert isinstance(client, StationBClient)
    assert client.log.n_events == 100  # from run_config station-b-events


def test_train_handler_echoes_arrays_and_reports_metrics():
    """End-to-end via the Flower message API: the dispatcher routes partition
    4 to Station B, which echoes the arrays and reports information metrics."""
    from flwr.app import ArrayRecord, Message, RecordDict

    from federated_ueba.federated.client_dispatch import train
    from federated_ueba.models import SecurityAutoencoder

    model = SecurityAutoencoder(input_dim=11, hidden_dim=32, latent_dim=8)
    arrays = ArrayRecord(model.state_dict())
    incoming = Message(
        RecordDict({"arrays": arrays}), dst_node_id=0, message_type="train"
    )

    reply = train(incoming, _ctx(4))  # partition 4 -> Station B

    echoed = reply.content["arrays"].to_numpy_ndarrays()
    original = arrays.to_numpy_ndarrays()
    assert [e.shape for e in echoed] == [o.shape for o in original]
    assert all(np.array_equal(e, o) for e, o in zip(echoed, original))
    metrics = reply.content["metrics"]
    assert metrics["num-examples"] == 100
    assert "n_flagged" in metrics
