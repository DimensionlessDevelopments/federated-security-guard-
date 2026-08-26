"""Incident detection and reporting against the federated global model.

LLM-free core of the agent: loads the trained global model, gathers recent
events from the local environment (synthetic stand-in for real telemetry),
scores them, and -- when the anomaly rate exceeds policy -- assembles an
incident report with per-feature deviation insights for the analyst LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
from federated_ueba.simulation.attack_generator import generate_mixed_dataset

THRESHOLD_PERCENTILE = 95.0
# An incident is declared when the flagged fraction exceeds this rate --
# twice the expected false-positive rate of the 95th-percentile threshold.
TRIGGER_RATE = 0.10
TOP_EVENTS = 3
TOP_FEATURES = 5


def station_seed(station: str) -> int:
    """Same seed mapping the federated clients use (client partition order)."""
    return 42 + STATION_NAMES.index(station) * 100


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


@dataclass
class IncidentReport:
    station: str
    n_events: int
    n_flagged: int
    flagged_rate: float
    threshold: float
    triggered: bool
    top_events: list[dict] = field(default_factory=list)
    feature_deviations: list[dict] = field(default_factory=list)


def gather_incident_data(
    model: SecurityAutoencoder,
    station: str,
    n_events: int = 200,
    attack_fraction: float = 0.15,
    seed: int | None = None,
    trigger_rate: float = TRIGGER_RATE,
) -> IncidentReport:
    """Gather recent events from the local environment and score them.

    The synthetic mixed dataset stands in for live telemetry collection.
    Set attack_fraction=0 to model a quiet period.
    """
    if seed is None:
        seed = station_seed(station)

    # Calibrate against the station's normal baseline, exactly like training
    baseline = generate_normal_events(station, n_samples=500, seed=seed)
    _, mean, std = normalize_features(baseline.features)
    baseline_norm = ((baseline.features - mean) / std).astype(np.float32)
    baseline_errors = reconstruction_error(
        model, torch.tensor(baseline_norm)
    ).numpy()
    threshold = float(np.percentile(baseline_errors, THRESHOLD_PERCENTILE))

    # "Recent" events observed at the station
    n_attack = int(round(n_events * attack_fraction))
    n_normal = n_events - n_attack
    if n_attack > 0:
        events, _ = generate_mixed_dataset(
            station, n_normal=n_normal, n_attack=n_attack, seed=seed + 7000
        )
    else:
        events = generate_normal_events(
            station, n_samples=n_normal, seed=seed + 7000
        ).features

    events_norm = ((events - mean) / std).astype(np.float32)
    scores = reconstruction_error(model, torch.tensor(events_norm)).numpy()
    flagged = scores > threshold
    n_flagged = int(flagged.sum())
    flagged_rate = n_flagged / len(events)

    report = IncidentReport(
        station=station,
        n_events=len(events),
        n_flagged=n_flagged,
        flagged_rate=flagged_rate,
        threshold=threshold,
        triggered=flagged_rate >= trigger_rate,
    )

    if n_flagged == 0:
        return report

    # Highest-scoring events, with raw feature values for the analyst
    top_idx = np.argsort(scores)[::-1][:TOP_EVENTS]
    for i in top_idx:
        report.top_events.append(
            {
                "score": float(scores[i]),
                "features": {
                    name: float(events[i, j])
                    for j, name in enumerate(FEATURE_NAMES)
                },
            }
        )

    # Which behavioural features drove the incident: mean deviation of the
    # flagged events from the station's normal baseline, in sigma units.
    flagged_sigma = events_norm[flagged].mean(axis=0)
    order = np.argsort(np.abs(flagged_sigma))[::-1][:TOP_FEATURES]
    for j in order:
        report.feature_deviations.append(
            {
                "feature": FEATURE_NAMES[j],
                "mean_sigma": float(flagged_sigma[j]),
                "direction": "above" if flagged_sigma[j] > 0 else "below",
            }
        )

    return report


def format_incident_report(report: IncidentReport) -> str:
    """Render the report as text for the console and the analyst LLM."""
    lines = [
        f"INCIDENT REPORT -- {report.station}",
        f"Status: {'TRIGGERED' if report.triggered else 'no incident'}",
        f"Events observed: {report.n_events}",
        f"Events flagged anomalous: {report.n_flagged}"
        f" ({report.flagged_rate:.1%})",
        f"Anomaly threshold (reconstruction error): {report.threshold:.4f}",
    ]
    if report.feature_deviations:
        lines.append("Behavioural features driving the anomaly:")
        for dev in report.feature_deviations:
            lines.append(
                f"  - {dev['feature']}: {abs(dev['mean_sigma']):.1f} sigma"
                f" {dev['direction']} station baseline"
            )
    if report.top_events:
        lines.append("Highest-scoring events:")
        for event in report.top_events:
            feats = ", ".join(
                f"{k}={v:.2f}" for k, v in event["features"].items()
            )
            lines.append(f"  - score {event['score']:.3f}: {feats}")
    return "\n".join(lines)


def build_insights_prompt(report: IncidentReport) -> str:
    """The analyst prompt sent to the LLM when an incident triggers."""
    return (
        "You are a security analyst for a distributed transport network. "
        "A behavioural anomaly detector (autoencoder trained via federated "
        "learning on normal station activity) has flagged the following "
        "incident. Assess the likely attack scenario (e.g. compromised "
        "credentials, insider misuse), explain which behavioural deviations "
        "support that assessment, and recommend immediate response actions "
        "for the station operator.\n\n" + format_incident_report(report)
    )
