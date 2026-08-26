"""SQLite detection store for the serving plane.

A single ``detections`` table holds one row per scored event. WAL mode is
enabled so a frontend (or the correlation agent) can read the feed while the
serving loop writes to it. All access goes through DetectionStore; the schema
is created on first use.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT    NOT NULL,
    tick          INTEGER,
    station       TEXT    NOT NULL,
    anomaly_score REAL    NOT NULL,
    threshold     REAL    NOT NULL,
    is_anomaly    INTEGER NOT NULL,
    is_attack     INTEGER,
    features      TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_detections_station ON detections(station);
CREATE INDEX IF NOT EXISTS idx_detections_ts ON detections(ts);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Detection:
    """One scored event, ready to persist."""

    station: str
    anomaly_score: float
    threshold: float
    is_anomaly: bool
    features: dict[str, float]
    tick: int | None = None
    is_attack: bool | None = None
    ts: str = field(default_factory=_now_iso)


class DetectionStore:
    """Thin SQLite wrapper around the detections feed."""

    def __init__(self, db_path: str | Path = "artifacts/serving.db") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False so a frontend thread can read the connection;
        # writes here are single-threaded from the serving loop.
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # -- writes -----------------------------------------------------------
    def record(self, detection: Detection) -> None:
        self.record_many([detection])

    def record_many(self, detections: list[Detection]) -> None:
        self.conn.executemany(
            """
            INSERT INTO detections
                (ts, tick, station, anomaly_score, threshold,
                 is_anomaly, is_attack, features)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    d.ts,
                    d.tick,
                    d.station,
                    float(d.anomaly_score),
                    float(d.threshold),
                    int(d.is_anomaly),
                    None if d.is_attack is None else int(d.is_attack),
                    json.dumps(d.features),
                )
                for d in detections
            ],
        )
        self.conn.commit()

    # -- reads ------------------------------------------------------------
    def recent(
        self, limit: int = 50, station: str | None = None
    ) -> list[dict]:
        """Most recent detections, newest first."""
        sql = "SELECT * FROM detections"
        params: list = []
        if station is not None:
            sql += " WHERE station = ?"
            params.append(station)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def summary(self) -> list[dict]:
        """Per-station counts: total events and flagged anomalies."""
        rows = self.conn.execute(
            """
            SELECT station,
                   COUNT(*)              AS n_events,
                   SUM(is_anomaly)       AS n_flagged,
                   SUM(COALESCE(is_attack, 0)) AS n_attack
            FROM detections
            GROUP BY station
            ORDER BY station
            """
        ).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0])

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["features"] = json.loads(d["features"])
        d["is_anomaly"] = bool(d["is_anomaly"])
        if d["is_attack"] is not None:
            d["is_attack"] = bool(d["is_attack"])
        return d

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "DetectionStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
