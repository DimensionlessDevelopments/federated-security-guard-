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
    assert s["stations_participating"] == 2


def test_stations(client):
    rows = {r["id"]: r for r in client.get("/api/stations").json()}
    assert rows["station_alpha"]["n_flagged"] == 2
    assert rows["station_alpha"]["status"] == "Needs attention"  # 2/3 flagged
    assert rows["station_bravo"]["status"] == "Healthy"
    assert 0 <= rows["station_alpha"]["health"] <= 100


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
