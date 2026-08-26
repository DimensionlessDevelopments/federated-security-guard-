"""Synthetic behavioural event generator for transport infrastructure stations.

Each station has a distinct normal operating profile. The generator produces
events as dicts with the 11 behavioural features, plus metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import TensorDataset

FEATURE_NAMES = [
    "hour_of_day",
    "day_of_week",
    "device_id",
    "user_role",
    "failed_login_count",
    "password_reset_freq",
    "priv_escalation_attempts",
    "session_duration",
    "sensitive_record_access",
    "network_risk_score",
    "command_sequence_score",
]

NUM_FEATURES = len(FEATURE_NAMES)

STATION_PROFILES: dict[str, dict[str, tuple[float, float]]] = {
    "station_alpha": {
        "hour_of_day": (10.0, 2.0),
        "day_of_week": (2.5, 1.5),
        "device_id": (3.0, 1.0),
        "user_role": (1.0, 0.3),
        "failed_login_count": (0.1, 0.2),
        "password_reset_freq": (0.05, 0.05),
        "priv_escalation_attempts": (0.0, 0.01),
        "session_duration": (45.0, 15.0),
        "sensitive_record_access": (2.0, 1.5),
        "network_risk_score": (0.1, 0.05),
        "command_sequence_score": (0.15, 0.08),
    },
    "station_bravo": {
        "hour_of_day": (14.0, 3.0),
        "day_of_week": (3.0, 1.8),
        "device_id": (5.0, 2.0),
        "user_role": (1.5, 0.5),
        "failed_login_count": (0.2, 0.3),
        "password_reset_freq": (0.08, 0.06),
        "priv_escalation_attempts": (0.0, 0.02),
        "session_duration": (60.0, 20.0),
        "sensitive_record_access": (3.0, 2.0),
        "network_risk_score": (0.12, 0.06),
        "command_sequence_score": (0.18, 0.1),
    },
    "station_charlie": {
        "hour_of_day": (6.0, 2.5),
        "day_of_week": (2.0, 1.2),
        "device_id": (2.0, 0.8),
        "user_role": (1.0, 0.2),
        "failed_login_count": (0.05, 0.1),
        "password_reset_freq": (0.02, 0.03),
        "priv_escalation_attempts": (0.0, 0.01),
        "session_duration": (30.0, 10.0),
        "sensitive_record_access": (1.0, 0.8),
        "network_risk_score": (0.08, 0.04),
        "command_sequence_score": (0.1, 0.06),
    },
    "central_helpdesk": {
        "hour_of_day": (12.0, 4.0),
        "day_of_week": (3.0, 2.0),
        "device_id": (8.0, 3.0),
        "user_role": (2.0, 0.8),
        "failed_login_count": (0.3, 0.4),
        "password_reset_freq": (0.15, 0.1),
        "priv_escalation_attempts": (0.01, 0.02),
        "session_duration": (90.0, 30.0),
        "sensitive_record_access": (5.0, 3.0),
        "network_risk_score": (0.15, 0.08),
        "command_sequence_score": (0.2, 0.12),
    },
}

STATION_NAMES = list(STATION_PROFILES.keys())


@dataclass
class GeneratedData:
    events: list[dict[str, float]]
    features: np.ndarray
    labels: np.ndarray
    station: str
    dataset: TensorDataset = field(init=False)

    def __post_init__(self) -> None:
        self.dataset = TensorDataset(
            torch.tensor(self.features, dtype=torch.float32),
            torch.tensor(self.labels, dtype=torch.float32),
        )


def generate_normal_events(
    station: str,
    n_samples: int = 500,
    seed: int = 42,
) -> GeneratedData:
    """Generate normal behavioural events for a station."""
    rng = np.random.default_rng(seed)
    profile = STATION_PROFILES[station]
    samples = np.zeros((n_samples, NUM_FEATURES))

    for i, feat_name in enumerate(FEATURE_NAMES):
        mean, std = profile[feat_name]
        samples[:, i] = rng.normal(mean, std, n_samples)

    samples = np.clip(samples, 0, None)

    events = []
    for row in samples:
        events.append({name: float(row[i]) for i, name in enumerate(FEATURE_NAMES)})

    return GeneratedData(
        events=events,
        features=samples.astype(np.float32),
        labels=np.zeros(n_samples, dtype=np.int64),
        station=station,
    )
