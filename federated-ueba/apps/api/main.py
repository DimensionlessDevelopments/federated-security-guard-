"""FastAPI backend for the Federated Security Guard dashboard.

Serves JSON derived from the serving plane's SQLite detection store (plus the
global-model artifact status) and the static single-page frontend. Read-only:
it never writes to the store.

Run:
    uv run uvicorn apps.api.main:app --reload
    # then open http://127.0.0.1:8000
Environment:
    SERVING_DB    path to the detections SQLite db (default artifacts/serving.db)
    GLOBAL_MODEL  path to the global model artifact (default artifacts/global_model.pt)
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from federated_ueba.agent.incident import (
    build_insights_prompt,
    format_incident_report,
    gather_incident_data,
    load_global_model,
)
from federated_ueba.data.generator import (
    FEATURE_NAMES,
    STATION_NAMES,
    generate_normal_events,
)
from federated_ueba.data.preprocessing import normalize_features
from federated_ueba.serving.store import DetectionStore

DB_PATH = os.environ.get("SERVING_DB", "artifacts/serving.db")
MODEL_PATH = os.environ.get("GLOBAL_MODEL", "artifacts/global_model.pt")
STATIC_DIR = Path(__file__).parent / "static"

# Presentational metadata (names/locations are display-only).
STATION_META: dict[str, dict[str, str]] = {
    "station_alpha": {"name": "Station Alpha", "location": "Austin, US"},
    "station_bravo": {"name": "Station Bravo", "location": "Berlin, DE"},
    "station_charlie": {"name": "Station Charlie", "location": "Tokyo, JP"},
    "central_helpdesk": {"name": "Central Helpdesk", "location": "London, UK"},
}

# Plain-language phrasing for the behavioural feature that drove an alert.
FEATURE_TITLES: dict[str, str] = {
    "hour_of_day": "Activity at an unusual hour",
    "day_of_week": "Activity on an unusual day",
    "device_id": "Sign-in from a new device",
    "user_role": "Unusual role activity",
    "failed_login_count": "Repeated failed logins",
    "password_reset_freq": "Password-reset burst",
    "priv_escalation_attempts": "Privilege escalation attempt",
    "session_duration": "Unusual session length",
    "sensitive_record_access": "Unusual sensitive-data access",
    "network_risk_score": "Connection to a risky destination",
    "command_sequence_score": "Unusual command sequence",
}
FEATURE_PHRASES: dict[str, str] = {
    "hour_of_day": "activity at an unusual hour",
    "day_of_week": "activity on an unusual day",
    "device_id": "a new or unfamiliar device",
    "user_role": "unusual role activity",
    "failed_login_count": "repeated failed logins",
    "password_reset_freq": "a burst of password resets",
    "priv_escalation_attempts": "privilege-escalation attempts",
    "session_duration": "an unusual session length",
    "sensitive_record_access": "a large volume of sensitive records accessed",
    "network_risk_score": "a connection to a risky destination",
    "command_sequence_score": "an unusual command sequence",
}

app = FastAPI(title="Federated Security Guard")


@lru_cache(maxsize=None)
def _baseline(station: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-station baseline mean/std (matches the serving scorer calibration)."""
    idx = STATION_NAMES.index(station) if station in STATION_NAMES else 0
    base = generate_normal_events(station, n_samples=500, seed=42 + idx * 100)
    _, mean, std = normalize_features(base.features)
    return mean, std


def _store() -> DetectionStore:
    return DetectionStore(DB_PATH)


def _relative(ts_iso: str) -> str:
    try:
        t = datetime.fromisoformat(ts_iso)
    except (ValueError, TypeError):
        return ""
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    secs = int((datetime.now(timezone.utc) - t).total_seconds())
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{secs // 60} min ago"
    if secs < 86400:
        return f"{secs // 3600} hours ago"
    return f"{secs // 86400} days ago"


def _top_feature(station: str, features: dict[str, float]) -> tuple[str, float]:
    """Return (feature_name, signed sigma) of the most deviant feature."""
    mean, std = _baseline(station)
    vals = np.array([features.get(f, 0.0) for f in FEATURE_NAMES], dtype=float)
    sigma = (vals - mean) / std
    j = int(np.argmax(np.abs(sigma)))
    return FEATURE_NAMES[j], float(sigma[j])


def _station_name(station: str) -> str:
    return STATION_META.get(station, {}).get("name", station)


@app.get("/api/summary")
def summary() -> dict:
    store = _store()
    try:
        rows = store.summary()
        total = sum(r["n_events"] for r in rows)
        flagged = sum((r["n_flagged"] or 0) for r in rows)
        confirmed = int(
            store.conn.execute(
                "SELECT COUNT(*) FROM detections "
                "WHERE is_anomaly = 1 AND is_attack = 1"
            ).fetchone()[0]
        )
    finally:
        store.close()

    model = Path(MODEL_PATH)
    if model.exists():
        updated = _relative(
            datetime.fromtimestamp(
                model.stat().st_mtime, tz=timezone.utc
            ).isoformat()
        )
        model_status = {"ready": True, "updated": updated}
    else:
        model_status = {"ready": False, "updated": None}

    return {
        "events_monitored": total,
        "needs_a_look": flagged,
        "confirmed_issues": confirmed,
        "raw_data_shared": "None",
        "model": model_status,
        # All known stations are online in the baseline.
        "stations_participating": len(STATION_NAMES),
        "stations_total": len(STATION_NAMES),
    }


@app.get("/api/stations")
def stations() -> list[dict]:
    store = _store()
    try:
        rows = {r["station"]: r for r in store.summary()}
    finally:
        store.close()

    total_events = sum(r["n_events"] for r in rows.values()) or 1
    out = []
    # Always list all known stations, so an empty store shows a green baseline.
    for station in STATION_NAMES:
        r = rows.get(station)
        n_events = r["n_events"] if r else 0
        n_flagged = (r["n_flagged"] or 0) if r else 0
        flag_rate = n_flagged / n_events if n_events else 0.0
        health = max(0, min(100, round(100 * (1 - flag_rate))))
        meta = STATION_META.get(station, {"name": station, "location": ""})
        out.append(
            {
                "id": station,
                "name": meta["name"],
                "location": meta.get("location", ""),
                "n_events": n_events,
                "n_flagged": n_flagged,
                "health": health,
                "status": "Healthy" if flag_rate < 0.10 else "Needs attention",
                "contribution": round(100 * n_events / total_events),
            }
        )
    return out


@app.get("/api/alerts")
def alerts(limit: int = 20) -> list[dict]:
    store = _store()
    try:
        # Recent flagged detections, newest first.
        raw = store.conn.execute(
            "SELECT * FROM detections WHERE is_anomaly = 1 "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        rows = [DetectionStore._row_to_dict(r) for r in raw]
    finally:
        store.close()

    out = []
    for r in rows:
        station = r["station"]
        feat, sigma = _top_feature(station, r["features"])
        ratio = r["anomaly_score"] / r["threshold"] if r["threshold"] else 0.0
        unusualness = int(min(100, max(1, round(50 * ratio))))
        direction = "above" if sigma > 0 else "below"
        out.append(
            {
                "id": r["id"],
                "station": station,
                "station_name": _station_name(station),
                "title": FEATURE_TITLES.get(feat, "Unusual behaviour"),
                "detail": (
                    f"Behaviour shows {FEATURE_PHRASES.get(feat, feat)} — "
                    f"{abs(sigma):.1f}σ {direction} this site's baseline."
                ),
                "score": round(r["anomaly_score"], 3),
                "threshold": round(r["threshold"], 3),
                "ratio": round(ratio, 1),
                "unusualness": unusualness,
                "is_attack": r["is_attack"],
                "relative": _relative(r["ts"]),
            }
        )
    return out


@lru_cache(maxsize=None)
def _incident_model(station: str):
    """The model the incident agent scores with: the federated global model if
    it exists, else a per-station fallback (trained once, then cached)."""
    if Path(MODEL_PATH).exists():
        return load_global_model(MODEL_PATH)
    from federated_ueba.agent.__main__ import train_fallback_model

    return train_fallback_model(station, epochs=10)


def _pick_station(station: str | None, rows: list[dict]) -> str:
    """Resolve the target station: given, else the hottest, else the default."""
    if station and station in STATION_NAMES:
        return station
    if rows:
        return max(
            rows, key=lambda r: (r["n_flagged"] or 0) / max(r["n_events"], 1)
        )["station"]
    return STATION_NAMES[0]


def _run_incident(station: str, attack_fraction: float) -> dict:
    model = _incident_model(station)
    report = gather_incident_data(model, station, attack_fraction=attack_fraction)
    return {
        "station": station,
        "station_name": _station_name(station),
        "triggered": report.triggered,
        "n_events": report.n_events,
        "n_flagged": report.n_flagged,
        "flagged_rate": round(report.flagged_rate, 3),
        "threshold": round(report.threshold, 4),
        "feature_deviations": report.feature_deviations,
        "report_text": format_incident_report(report),
        "prompt": build_insights_prompt(report) if report.triggered else None,
        "attack_fraction": round(attack_fraction, 3),
        "model": "federated" if Path(MODEL_PATH).exists() else "local-fallback",
    }


@app.get("/api/incident")
def incident(station: str | None = None, attack_fraction: float = 0.25) -> dict:
    """Read-only: run the incident agent for a station and return its analysis.

    Does not modify the store. With no station, analyses the hottest one.
    """
    store = _store()
    try:
        rows = store.summary()
    finally:
        store.close()
    return _run_incident(_pick_station(station, rows), attack_fraction)


@app.post("/api/incident")
def report_incident(station: str | None = None) -> dict:
    """Simulate + report an incident: inject an attack for the target station
    into the detection store (so the live dashboard populates), then return the
    agent's analysis. This is what the "Report Incident" button calls.
    """
    from federated_ueba.serving.loop import run_serving

    store = _store()
    try:
        target = _pick_station(station, store.summary())
        # Inject a short burst of attack traffic for this station.
        run_serving(
            store,
            model=_incident_model(target),
            stations=[target],
            ticks=3,
            events_per_tick=20,
            attack_at=0,
            attacked_stations=[target],
            seed=int(datetime.now(timezone.utc).timestamp()),
        )
    finally:
        store.close()

    return _run_incident(target, attack_fraction=0.3)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/", StaticFiles(directory=str(STATIC_DIR)), name="static")
