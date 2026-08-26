"""Serving loop: score a live event stream with the trained global model.

Loads the aggregated global model produced by the ServerApp, calibrates a
per-station anomaly threshold against that station's normal baseline (exactly
as training/incident scoring does), then scores an event stream tick by tick
and writes each detection to the store.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from federated_ueba.data.generator import (
    FEATURE_NAMES,
    NUM_FEATURES,
    STATION_NAMES,
    generate_normal_events,
)
from federated_ueba.data.preprocessing import normalize_features
from federated_ueba.detection.scoring import reconstruction_error
from federated_ueba.models import SecurityAutoencoder
from federated_ueba.serving.store import Detection, DetectionStore
from federated_ueba.simulation.event_stream import EventBatch, stream_events

BASELINE_SAMPLES = 500
THRESHOLD_PERCENTILE = 95.0


def station_seed(station: str) -> int:
    """Baseline seed matching the federated clients' partition order."""
    idx = STATION_NAMES.index(station) if station in STATION_NAMES else 0
    return 42 + idx * 100


def load_global_model(
    path: str | Path,
    hidden_dim: int = 32,
    latent_dim: int = 8,
) -> SecurityAutoencoder:
    """Load the aggregated global model saved by the ServerApp."""
    state_dict = torch.load(Path(path), weights_only=True)
    model = SecurityAutoencoder(
        input_dim=NUM_FEATURES, hidden_dim=hidden_dim, latent_dim=latent_dim
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


class StationScorer:
    """Scores events for one station against its calibrated baseline."""

    def __init__(
        self,
        model: SecurityAutoencoder,
        station: str,
        baseline_samples: int = BASELINE_SAMPLES,
        threshold_percentile: float = THRESHOLD_PERCENTILE,
    ) -> None:
        self.model = model
        self.station = station

        baseline = generate_normal_events(
            station, n_samples=baseline_samples, seed=station_seed(station)
        )
        _, self.mean, self.std = normalize_features(baseline.features)
        baseline_norm = ((baseline.features - self.mean) / self.std).astype(
            np.float32
        )
        errors = reconstruction_error(
            self.model, torch.tensor(baseline_norm)
        ).numpy()
        self.threshold = float(np.percentile(errors, threshold_percentile))

    def score(self, features: np.ndarray) -> np.ndarray:
        """Per-event reconstruction-error anomaly scores."""
        norm = ((features - self.mean) / self.std).astype(np.float32)
        return reconstruction_error(self.model, torch.tensor(norm)).numpy()


def _batch_to_detections(
    batch: EventBatch, scorer: StationScorer
) -> list[Detection]:
    scores = scorer.score(batch.features)
    flagged = scores > scorer.threshold
    detections = []
    for i in range(batch.n_events):
        detections.append(
            Detection(
                station=batch.station,
                tick=batch.tick,
                anomaly_score=float(scores[i]),
                threshold=scorer.threshold,
                is_anomaly=bool(flagged[i]),
                is_attack=bool(batch.is_attack[i]),
                features={
                    name: float(batch.features[i, j])
                    for j, name in enumerate(FEATURE_NAMES)
                },
            )
        )
    return detections


def run_serving(
    store: DetectionStore,
    model_path: str | Path = "artifacts/global_model.pt",
    stations: list[str] | None = None,
    ticks: int = 10,
    events_per_tick: int = 20,
    attack_at: int | None = None,
    attacked_stations: list[str] | None = None,
    attack_fraction: float = 0.4,
    interval_s: float = 0.0,
    hidden_dim: int = 32,
    latent_dim: int = 8,
    seed: int = 0,
    model: SecurityAutoencoder | None = None,
) -> int:
    """Run the serving loop, writing detections to ``store``.

    Pass ``model`` to score with an in-memory model (tests); otherwise the
    aggregated global model is loaded from ``model_path``. Returns the number
    of detections written.
    """
    stations = list(stations) if stations else list(STATION_NAMES)
    if model is None:
        model = load_global_model(model_path, hidden_dim, latent_dim)

    scorers = {st: StationScorer(model, st) for st in stations}

    written = 0
    last_tick = -1
    for batch in stream_events(
        stations=stations,
        ticks=ticks,
        events_per_tick=events_per_tick,
        attack_at=attack_at,
        attacked_stations=attacked_stations,
        attack_fraction=attack_fraction,
        seed=seed,
    ):
        detections = _batch_to_detections(batch, scorers[batch.station])
        store.record_many(detections)
        written += len(detections)

        # Sleep once per tick boundary to pace a live-ish feed.
        if interval_s > 0 and batch.tick != last_tick:
            last_tick = batch.tick
            time.sleep(interval_s)

    return written
