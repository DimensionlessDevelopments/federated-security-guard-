"""Tests for the dashboard FastAPI backend.

The API reads its DB path from the SERVING_DB env var at import time, so we set
it and populate a temp store before importing the app.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

# Make the repo root importable so `apps.api.main` resolves (pytest only adds src).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from federated_ueba.data.generator import FEATURE_NAMES  # noqa: E402
from federated_ueba.serving.store import Detection, DetectionStore  # noqa: E402


def _features(**overrides) -> dict[str, float]:
    feats = {name: 0.0 for name in FEATURE_NAMES}
    feats.update(overrides)
    return feats


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    db = tmp_path / "serving.db"
    store = DetectionStore(str(db))
    store.record_many(
        [
            # station_alpha: two flagged, one a confirmed attack
            Detection("station_alpha", 3.0, 1.5, True,
                      _features(priv_escalation_attempts=0.4), tick=0, is_attack=True),
            Detection("station_alpha", 2.0, 1.5, True,
                      _features(hour_of_day=3.0), tick=0, is_attack=False),
            Detection("station_alpha", 0.2, 1.5, False, _features(), tick=0, is_attack=False),
            # station_bravo: all normal
            Detection("station_bravo", 0.3, 1.4, False, _features(), tick=0, is_attack=False),
        ]
    )
    store.close()

    monkeypatch.setenv("SERVING_DB", str(db))
    import apps.api.main as main
    importlib.reload(main)  # pick up SERVING_DB
    return TestClient(main.app)


def test_summary(client):
    s = client.get("/api/summary").json()
    assert s["events_monitored"] == 4
    assert s["needs_a_look"] == 2  # two flagged
    assert s["confirmed_issues"] == 1  # flagged AND attack
    assert s["raw_data_shared"] == "None"
    # All known stations are reported as online in the baseline.
    assert s["stations_participating"] == s["stations_total"]


def test_stations(client):
    rows = {r["id"]: r for r in client.get("/api/stations").json()}
    # All known stations are always listed (green baseline for unseeded ones).
    assert len(rows) == s_total(client)
    assert rows["station_alpha"]["n_flagged"] == 2
    assert rows["station_alpha"]["status"] == "Needs attention"  # 2/3 flagged
    assert rows["station_bravo"]["status"] == "Healthy"
    # A station with no detections shows a healthy baseline.
    assert rows["station_charlie"]["n_events"] == 0
    assert rows["station_charlie"]["status"] == "Healthy"
    assert rows["station_charlie"]["health"] == 100


def s_total(client) -> int:
    return client.get("/api/summary").json()["stations_total"]


def test_empty_store_is_all_green(tmp_path, monkeypatch):
    """With no detections, the dashboard baseline is all-healthy, no alerts."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("SERVING_DB", str(tmp_path / "empty.db"))
    import apps.api.main as main

    importlib.reload(main)
    c = TestClient(main.app)

    assert c.get("/api/summary").json()["needs_a_look"] == 0
    assert c.get("/api/alerts").json() == []
    stations = c.get("/api/stations").json()
    assert stations and all(s["status"] == "Healthy" for s in stations)


def test_post_incident_populates_store(tmp_path, monkeypatch):
    """POST /api/incident injects an attack so the feed/stats populate."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("SERVING_DB", str(tmp_path / "pop.db"))
    import apps.api.main as main

    importlib.reload(main)
    c = TestClient(main.app)

    assert c.get("/api/summary").json()["events_monitored"] == 0  # green start
    rep = c.post("/api/incident?station=station_alpha").json()
    assert rep["station"] == "station_alpha"
    # the injected attack now shows up in the store-backed views
    assert c.get("/api/summary").json()["events_monitored"] > 0
    assert len(c.get("/api/alerts").json()) > 0
    rows = {r["id"]: r for r in c.get("/api/stations").json()}
    assert rows["station_alpha"]["n_flagged"] > 0


def test_alerts_are_flagged_only_with_explanations(client):
    alerts = client.get("/api/alerts?limit=10").json()
    assert len(alerts) == 2  # only flagged rows
    for a in alerts:
        assert a["title"]
        assert "baseline" in a["detail"]
        assert 1 <= a["unusualness"] <= 100
    # the confirmed attack should surface as an attack alert
    assert any(a["is_attack"] for a in alerts)


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Federated Security Guard" in r.text


def test_incident_endpoint(client):
    # Explicit attack fraction so the incident triggers regardless of the store.
    r = client.get("/api/incident?station=station_alpha&attack_fraction=0.3")
    assert r.status_code == 200
    d = r.json()
    assert d["station"] == "station_alpha"
    assert d["triggered"] is True
    assert d["feature_deviations"]  # names the driving features
    assert d["prompt"]  # analyst prompt present on a triggered incident
    assert d["model"] in ("federated", "local-fallback")
