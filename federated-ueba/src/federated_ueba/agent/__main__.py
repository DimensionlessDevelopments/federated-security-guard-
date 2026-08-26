"""Testing entrypoint for the incident-report agent.

Runs the full incident flow -- connect to the global model, gather local
events, score, report -- without the Flower runtime or an LLM key:

    python -m federated_ueba.agent
    python -m federated_ueba.agent --station station_charlie --attack-fraction 0
    python -m federated_ueba.agent --model-path artifacts/global_model.pt

If the global model artifact does not exist yet (no federated run so far),
a local fallback model is trained on the station's normal baseline so the
entrypoint always works on a fresh clone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from federated_ueba.agent.incident import (
    build_insights_prompt,
    format_incident_report,
    gather_incident_data,
    load_global_model,
    station_seed,
)
from federated_ueba.data.generator import (
    NUM_FEATURES,
    STATION_NAMES,
    generate_normal_events,
)
from federated_ueba.data.preprocessing import normalize_features
from federated_ueba.models import SecurityAutoencoder


def train_fallback_model(station: str, epochs: int) -> SecurityAutoencoder:
    """Local stand-in when no federated global model has been produced yet."""
    torch.manual_seed(0)
    model = SecurityAutoencoder(input_dim=NUM_FEATURES)
    normal = generate_normal_events(
        station, n_samples=500, seed=station_seed(station)
    )
    normalized, _, _ = normalize_features(normal.features)
    X = torch.tensor(normalized)
    dataset = torch.utils.data.TensorDataset(X, X)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for batch_x, _ in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_x)
            loss.backward()
            optimizer.step()
    return model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m federated_ueba.agent",
        description="Run the incident-report flow against the global model.",
    )
    parser.add_argument(
        "--station",
        default="central_helpdesk",
        choices=STATION_NAMES,
        help="station whose local environment is inspected",
    )
    parser.add_argument(
        "--attack-fraction",
        type=float,
        default=0.15,
        help="fraction of gathered events that are attacks (0 = quiet period)",
    )
    parser.add_argument(
        "--model-path",
        default="artifacts/global_model.pt",
        help="path to the aggregated global model saved by the ServerApp",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="override the station seed"
    )
    parser.add_argument(
        "--fallback-epochs",
        type=int,
        default=10,
        help="training epochs for the local fallback model",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="also print the analyst-LLM prompt a triggered incident sends",
    )
    args = parser.parse_args(argv)

    model_path = Path(args.model_path)
    if model_path.exists():
        print(f"Using federated global model: {model_path}")
        model = load_global_model(model_path)
    else:
        print(
            f"No global model at {model_path} -- training local fallback "
            f"({args.fallback_epochs} epochs). Run `flwr run . local-sim` "
            "to produce the federated model."
        )
        model = train_fallback_model(args.station, args.fallback_epochs)

    report = gather_incident_data(
        model,
        args.station,
        attack_fraction=args.attack_fraction,
        seed=args.seed,
    )
    print()
    print(format_incident_report(report))

    if report.triggered and args.show_prompt:
        print()
        print("--- analyst LLM prompt ---")
        print(build_insights_prompt(report))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
