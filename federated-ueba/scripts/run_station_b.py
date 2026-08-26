"""Run Station B on its own, without the full Flower runtime.

This drives the exact client logic Station B executes inside a FedAvg round:
each round it receives the global model parameters, echoes them back unchanged
(no ML), and reports its synthetic information to the "server".

Usage:
    uv run python scripts/run_station_b.py
    uv run python scripts/run_station_b.py --rounds 5 --events 800

This is a standalone demo/verification. The full federation (all 5 nodes) runs
via `uv run flwr run . local-sim` once the flwr packaging issue is resolved.
"""

from __future__ import annotations

import argparse

import numpy as np

from federated_ueba.models import SecurityAutoencoder, get_parameters
from station_b.client_app import StationBClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Station B client.")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--events", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1004)
    args = parser.parse_args()

    # The server seeds every round from its global model. We build the same
    # model here to produce correctly-shaped global parameters to hand out.
    model = SecurityAutoencoder(input_dim=11, hidden_dim=32, latent_dim=8)
    global_params = get_parameters(model)

    client = StationBClient("station_b", seed=args.seed, n_events=args.events)

    print("=" * 64)
    print("Station B — non-ML federated participant")
    print("=" * 64)
    print(f"rounds={args.rounds}  events/round={args.events}  seed={args.seed}")
    print(f"global model params: {len(global_params)} tensors "
          f"{[p.shape for p in global_params]}")
    print("-" * 64)

    for rnd in range(1, args.rounds + 1):
        # fit(): echo the global params, report information via metrics.
        returned, num_examples, info = client.fit(global_params, {})

        # Prove the FedAvg parameter contract: identical shapes, unchanged.
        shapes_ok = [p.shape for p in returned] == [p.shape for p in global_params]
        echoed = all(np.array_equal(a, b) for a, b in zip(returned, global_params))

        print(f"round {rnd}:")
        print(f"  params echoed unchanged: {echoed}  shapes preserved: {shapes_ok}")
        print(f"  num_examples (FedAvg weight): {num_examples}")
        print(f"  info -> flagged={info['n_flagged']} "
              f"rate={info['flagged_rate']:.3f} "
              f"mean_failed_logins={info['mean_failed_logins']:.2f} "
              f"max_records={info['max_records_accessed']:.0f}")

    print("-" * 64)
    print("Station B reported information each round; raw events never left the node.")


if __name__ == "__main__":
    main()
