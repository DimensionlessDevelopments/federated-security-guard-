"""Synthetic event generation for Station B.

Self-contained: depends only on numpy. Station B observes its own local
security events. These are *not* tied to the 11-feature UEBA schema used by
the ML stations -- they are just the raw signals this node happens to collect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Fields Station B records for each event. Kept small and human-readable.
FIELD_NAMES = [
    "hour_of_day",
    "failed_login_count",
    "password_reset_count",
    "priv_escalation_attempts",
    "sensitive_records_accessed",
]


@dataclass
class EventLog:
    """A batch of synthetic events as a (n_events, n_fields) float array."""

    events: np.ndarray  # shape (n_events, len(FIELD_NAMES))

    @property
    def n_events(self) -> int:
        return int(self.events.shape[0])

    def column(self, name: str) -> np.ndarray:
        return self.events[:, FIELD_NAMES.index(name)]


def generate_events(n_events: int = 500, seed: int = 0) -> EventLog:
    """Generate a deterministic batch of synthetic Station B events.

    The distribution is benign by default (a working station), with a small
    fraction of naturally noisy rows so the rule-based flags are non-trivial.
    """
    rng = np.random.default_rng(seed)

    hour = rng.integers(0, 24, size=n_events).astype(np.float32)
    failed_logins = rng.poisson(0.3, size=n_events).astype(np.float32)
    password_resets = rng.poisson(0.5, size=n_events).astype(np.float32)
    priv_escalation = (rng.random(n_events) < 0.02).astype(np.float32)
    records_accessed = rng.gamma(shape=2.0, scale=20.0, size=n_events).astype(
        np.float32
    )

    events = np.column_stack(
        [
            hour,
            failed_logins,
            password_resets,
            priv_escalation,
            records_accessed,
        ]
    ).astype(np.float32)

    return EventLog(events=events)
