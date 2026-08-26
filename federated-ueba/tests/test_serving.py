"""Tests for the serving plane: event stream, SQLite store, and serving loop."""

from __future__ import annotations

import numpy as np

from federated_ueba.data.generator import NUM_FEATURES, STATION_NAMES
from federated_ueba.models import SecurityAutoencoder
from federated_ueba.serving.loop import StationScorer, run_serving
from federated_ueba.serving.store import Detection, DetectionStore
from federated_ueba.simulation.event_stream import stream_events


# -- event stream ---------------------------------------------------------
def test_stream_shapes_and_determinism():
    kw = dict(stations=["station_alpha"], ticks=3, events_per_tick=10, seed=1)
    a = list(stream_events(**kw))
    b = list(stream_events(**kw))
    assert len(a) == 3  # one batch per tick for one station
    for batch in a:
        assert batch.features.shape == (10, NUM_FEATURES)
        assert batch.is_attack.shape == (10,)
        assert batch.is_attack.dtype == bool
    # deterministic for a fixed seed
    assert all(np.array_equal(x.features, y.features) for x, y in zip(a, b))


def test_stream_injects_attacks_only_from_attack_at():
    batches = list(
        stream_events(
            stations=["station_alpha"],
            ticks=4,
            events_per_tick=10,
            attack_at=2,
            attack_fraction=0.5,
            seed=1,
        )
    )
    by_tick = {b.tick: b for b in batches}
    assert by_tick[0].is_attack.sum() == 0
    assert by_tick[1].is_attack.sum() == 0
    assert by_tick[2].is_attack.sum() == 5  # 50% of 10
    assert by_tick[3].is_attack.sum() == 5


def test_stream_attacked_stations_filter():
    batches = list(
        stream_events(
            stations=["station_alpha", "station_bravo"],
            ticks=2,
            events_per_tick=10,
            attack_at=0,
            attacked_stations=["station_bravo"],
            attack_fraction=0.5,
            seed=1,
        )
    )
    for b in batches:
        if b.station == "station_alpha":
            assert b.is_attack.sum() == 0
        else:
            assert b.is_attack.sum() == 5


# -- store ----------------------------------------------------------------
def test_store_record_and_query():
    store = DetectionStore(":memory:")
    store.record_many(
        [
            Detection("station_alpha", 0.9, 0.5, True, {"x": 1.0}, tick=0, is_attack=True),
            Detection("station_alpha", 0.1, 0.5, False, {"x": 2.0}, tick=0, is_attack=False),
            Detection("station_bravo", 0.7, 0.4, True, {"x": 3.0}, tick=1, is_attack=False),
        ]
    )
    assert store.count() == 3

    recent = store.recent(limit=2)
    assert len(recent) == 2
    assert recent[0]["station"] == "station_bravo"  # newest first
    assert recent[0]["features"] == {"x": 3.0}
    assert recent[0]["is_anomaly"] is True

    alpha = store.recent(station="station_alpha")
    assert len(alpha) == 2

    summary = {r["station"]: r for r in store.summary()}
    assert summary["station_alpha"]["n_events"] == 2
    assert summary["station_alpha"]["n_flagged"] == 1
    assert summary["station_alpha"]["n_attack"] == 1
    store.close()


# -- scorer + loop --------------------------------------------------------
def test_station_scorer_calibrates_threshold():
    model = SecurityAutoencoder(input_dim=NUM_FEATURES)
    scorer = StationScorer(model, "station_alpha")
    assert scorer.threshold > 0
    # scoring returns one score per event
    from federated_ueba.data.generator import generate_normal_events

    feats = generate_normal_events("station_alpha", n_samples=5, seed=99).features
    scores = scorer.score(feats)
    assert scores.shape == (5,)


def test_run_serving_writes_detections_and_flags_attacks():
    model = SecurityAutoencoder(input_dim=NUM_FEATURES)
    store = DetectionStore(":memory:")
    written = run_serving(
        store,
        model=model,  # in-memory model, no artifact needed
        stations=["station_alpha"],
        ticks=4,
        events_per_tick=25,
        attack_at=2,
        attacked_stations=["station_alpha"],
        attack_fraction=0.6,
        seed=3,
    )
    assert written == 4 * 25
    assert store.count() == written

    # Attacks should be flagged at a higher rate than normal traffic. Compare
    # flag rate among attack rows vs. non-attack rows.
    rows = store.recent(limit=written)
    attack_flag = [r["is_anomaly"] for r in rows if r["is_attack"]]
    normal_flag = [r["is_anomaly"] for r in rows if not r["is_attack"]]
    assert attack_flag, "expected some attack events"
    attack_rate = sum(attack_flag) / len(attack_flag)
    normal_rate = sum(normal_flag) / len(normal_flag)
    assert attack_rate > normal_rate
    store.close()
