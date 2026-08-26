"""Run the serving loop: score a live event stream and write detections.

Loads the aggregated global model and scores synthetic station telemetry,
writing each detection to a SQLite store a frontend / the correlation agent
can read.

Usage:
    # against the trained global model produced by federated training
    uv run python scripts/run_serving.py --ticks 10 --attack-at 5 \
        --attacked-stations station_alpha

    # demo without a trained model (uses a fresh untrained autoencoder;
    # per-station thresholds are still calibrated, so detection works)
    uv run python scripts/run_serving.py --untrained-demo --attack-at 5

The store defaults to artifacts/serving.db.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from federated_ueba.data.generator import NUM_FEATURES
from federated_ueba.models import SecurityAutoencoder
from federated_ueba.serving.loop import run_serving
from federated_ueba.serving.store import DetectionStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the serving loop.")
    parser.add_argument("--model-path", default="artifacts/global_model.pt")
    parser.add_argument("--db-path", default="artifacts/serving.db")
    parser.add_argument("--ticks", type=int, default=10)
    parser.add_argument("--events-per-tick", type=int, default=20)
    parser.add_argument("--attack-at", type=int, default=None,
                        help="first tick to inject attacks (default: none)")
    parser.add_argument("--attacked-stations", nargs="*", default=None,
                        help="stations under attack (default: all)")
    parser.add_argument("--interval", type=float, default=0.0,
                        help="seconds to sleep between ticks (live pacing)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--untrained-demo", action="store_true",
                        help="score with a fresh untrained model when no "
                             "trained global model is available")
    args = parser.parse_args()

    model = None
    if args.untrained_demo:
        model = SecurityAutoencoder(input_dim=NUM_FEATURES)
        print("Using a fresh UNTRAINED model (demo mode).")
    elif not Path(args.model_path).exists():
        raise SystemExit(
            f"No global model at {args.model_path}. Run federated training "
            f"first, or pass --untrained-demo for a model-free demo."
        )

    store = DetectionStore(args.db_path)
    print(f"Serving -> {args.db_path}")
    written = run_serving(
        store,
        model_path=args.model_path,
        ticks=args.ticks,
        events_per_tick=args.events_per_tick,
        attack_at=args.attack_at,
        attacked_stations=args.attacked_stations,
        interval_s=args.interval,
        seed=args.seed,
        model=model,
    )

    print(f"\nWrote {written} detections.")
    print("-" * 60)
    print(f"{'station':<20}{'events':>8}{'flagged':>9}{'attacks':>9}")
    for row in store.summary():
        print(f"{row['station']:<20}{row['n_events']:>8}"
              f"{row['n_flagged'] or 0:>9}{row['n_attack'] or 0:>9}")
    store.close()


if __name__ == "__main__":
    main()
