"""Synthetic compromised-credential attack generator.

Attack variants are station-specific -- each station sees a different
subset of attack tactics, which creates the need for federated learning:
a model trained only on station_alpha's attacks won't recognise the
tactics first observed at station_bravo.
"""

from __future__ import annotations

import numpy as np

from federated_ueba.data.generator import (
    FEATURE_NAMES,
    NUM_FEATURES,
    STATION_PROFILES,
    generate_normal_events,
)

STATION_ATTACK_TACTICS: dict[str, dict[str, tuple[float, float]]] = {
    "station_alpha": {
        "hour_of_day": (5.0, 1.5),
        "priv_escalation_attempts": (0.08, 0.03),
        "command_sequence_score": (0.35, 0.08),
    },
    "station_bravo": {
        "failed_login_count": (1.2, 0.4),
        "password_reset_freq": (0.35, 0.1),
        "device_id": (10.0, 2.0),
    },
    "station_charlie": {
        "session_duration": (55.0, 10.0),
        "sensitive_record_access": (3.5, 1.0),
        "network_risk_score": (0.22, 0.05),
    },
    "central_helpdesk": {
        "hour_of_day": (4.0, 2.0),
        "failed_login_count": (1.0, 0.3),
        "sensitive_record_access": (12.0, 2.5),
    },
}

FULL_ATTACK_TACTICS: dict[str, tuple[float, float]] = {
    "hour_of_day": (4.5, 1.5),
    "failed_login_count": (1.0, 0.4),
    "password_reset_freq": (0.3, 0.1),
    "priv_escalation_attempts": (0.06, 0.03),
    "session_duration": (50.0, 10.0),
    "sensitive_record_access": (10.0, 2.0),
    "network_risk_score": (0.2, 0.05),
    "command_sequence_score": (0.35, 0.08),
    "device_id": (9.0, 2.0),
}


def generate_attack_events(
    station: str,
    n_samples: int = 50,
    seed: int = 1337,
    attack_type: str = "station_specific",
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Generate attack events for a station."""
    rng = np.random.default_rng(seed)
    profile = STATION_PROFILES[station]

    if attack_type == "full":
        tactics = FULL_ATTACK_TACTICS
    else:
        tactics = STATION_ATTACK_TACTICS.get(station, {})

    samples = np.zeros((n_samples, NUM_FEATURES))
    for i, feat_name in enumerate(FEATURE_NAMES):
        if feat_name in tactics:
            mean, std = tactics[feat_name]
        else:
            mean, std = profile[feat_name]
        samples[:, i] = rng.normal(mean, std, n_samples)
    samples = np.clip(samples, 0, None)

    events = []
    for row in samples:
        event = {name: float(row[i]) for i, name in enumerate(FEATURE_NAMES)}
        event["is_attack"] = True
        event["attack_type"] = attack_type
        events.append(event)

    return samples.astype(np.float32), events


def generate_mixed_dataset(
    station: str,
    n_normal: int = 500,
    n_attack: int = 50,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a dataset with both normal and attack events."""
    normal = generate_normal_events(station, n_normal, seed)
    attack_features, _ = generate_attack_events(station, n_attack, seed + 100)

    features = np.vstack([normal.features, attack_features])
    labels = np.concatenate([
        np.zeros(n_normal, dtype=np.int64),
        np.ones(n_attack, dtype=np.int64),
    ])

    rng = np.random.default_rng(seed)
    shuffle = rng.permutation(len(labels))
    return features[shuffle], labels[shuffle]
