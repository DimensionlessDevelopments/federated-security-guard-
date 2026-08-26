"""Synthetic event stream for the serving loop.

Stands in for live station telemetry: yields batches of behavioural events per
tick, per station, with optional attack injection from a given tick onward.
Deterministic for a fixed seed so runs are reproducible and testable.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from federated_ueba.data.generator import (
    NUM_FEATURES,
    STATION_NAMES,
    generate_normal_events,
)
from federated_ueba.simulation.attack_generator import generate_attack_events


@dataclass
class EventBatch:
    """One tick's worth of events observed at a station."""

    tick: int
    station: str
    features: np.ndarray  # (n_events, NUM_FEATURES), float32
    is_attack: np.ndarray  # (n_events,), bool -- ground-truth label

    @property
    def n_events(self) -> int:
        return int(self.features.shape[0])


def _batch_seed(seed: int, tick: int, station: str) -> int:
    """Stable per-(tick, station) seed derived from the run seed."""
    idx = STATION_NAMES.index(station) if station in STATION_NAMES else 0
    return seed + tick * 10_007 + idx * 101


def stream_events(
    stations: list[str] | None = None,
    ticks: int = 10,
    events_per_tick: int = 20,
    attack_at: int | None = None,
    attacked_stations: list[str] | None = None,
    attack_fraction: float = 0.4,
    seed: int = 0,
) -> Iterator[EventBatch]:
    """Yield EventBatch objects tick by tick, station by station.

    attack_at: the first tick (0-indexed) at which attacks are injected; None
    for an all-normal stream. attacked_stations: which stations are under
    attack (default: all). attack_fraction: fraction of a tick's events that
    are attack events while under attack.
    """
    stations = list(stations) if stations else list(STATION_NAMES)
    attacked = set(attacked_stations) if attacked_stations else set(stations)

    for tick in range(ticks):
        for station in stations:
            batch_seed = _batch_seed(seed, tick, station)

            under_attack = (
                attack_at is not None
                and tick >= attack_at
                and station in attacked
            )
            n_attack = (
                int(round(events_per_tick * attack_fraction))
                if under_attack
                else 0
            )
            n_normal = events_per_tick - n_attack

            normal = generate_normal_events(
                station, n_samples=n_normal, seed=batch_seed
            ).features

            if n_attack > 0:
                attack_features, _ = generate_attack_events(
                    station, n_samples=n_attack, seed=batch_seed + 500
                )
                features = np.vstack([normal, attack_features])
                labels = np.concatenate(
                    [
                        np.zeros(n_normal, dtype=bool),
                        np.ones(n_attack, dtype=bool),
                    ]
                )
                # Shuffle so attacks are interleaved with normal events.
                order = np.random.default_rng(batch_seed + 999).permutation(
                    events_per_tick
                )
                features, labels = features[order], labels[order]
            else:
                features = normal
                labels = np.zeros(n_normal, dtype=bool)

            yield EventBatch(
                tick=tick,
                station=station,
                features=features.astype(np.float32).reshape(-1, NUM_FEATURES),
                is_attack=labels,
            )
