"""Turn Station B's synthetic events into the information it reports.

No machine learning: deterministic summary statistics plus simple threshold
rules. Every returned value is a Flower-compatible scalar (int/float/str) so
it can travel in an FL metrics dict.
"""

from __future__ import annotations

import numpy as np

from station_b.synthetic import EventLog

# Rule thresholds. Deterministic, auditable, no learning involved.
RESET_BURST_THRESHOLD = 5      # more than this many resets in the batch window
OFF_HOURS = {0, 1, 2, 3, 4}    # small hours


def _flag(row: np.ndarray) -> bool:
    """A row is flagged if it trips any deterministic rule."""
    hour, _failed, resets, escalation, _records = row
    reset_burst = resets > RESET_BURST_THRESHOLD
    off_hours_escalation = int(hour) in OFF_HOURS and escalation > 0
    return bool(reset_burst or off_hours_escalation)


def summarize(log: EventLog, station: str) -> dict[str, float | int | str]:
    """Produce the information dict Station B sends to the server.

    Keys are chosen to sit alongside the ML stations' metrics in the server
    logs. All values are scalars.
    """
    events = log.events
    n_flagged = int(sum(_flag(row) for row in events))

    return {
        "station": station,
        "node_kind": "station_b_lite",
        "n_events": log.n_events,
        "n_flagged": n_flagged,
        "flagged_rate": float(n_flagged / log.n_events) if log.n_events else 0.0,
        "mean_failed_logins": float(log.column("failed_login_count").mean()),
        "max_records_accessed": float(log.column("sensitive_records_accessed").max()),
    }
