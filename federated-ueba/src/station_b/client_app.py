"""Station B Flower client -- a non-ML participant in the FedAvg run.

How it satisfies FedAvg without doing machine learning:

- ``fit`` echoes the received global parameters back unchanged. This is
  model-agnostic (Station B never needs to know the parameter shapes) and is a
  safe no-op contribution to the weighted average.
- The node's real payload -- summary stats and rule-based flags over its own
  synthetic events -- travels in the metrics dict, not in the parameters.
- ``num_examples`` is Station B's synthetic event count, the weight FedAvg
  uses (``weighted_by_key="num-examples"`` in the server strategy).

The package imports nothing from ``federated_ueba``; it is wired into the run
through a thin dispatcher (see ``federated_ueba.federated.client_dispatch``).
"""

from __future__ import annotations

import numpy as np
from flwr.app import Context
from flwr.client import ClientApp, NumPyClient

from station_b.reporter import summarize
from station_b.synthetic import generate_events


class StationBClient(NumPyClient):
    """A lightweight, no-ML station that reports information each round."""

    def __init__(self, station: str, seed: int, n_events: int = 500) -> None:
        self.station = station
        self.log = generate_events(n_events=n_events, seed=seed)
        self.info = summarize(self.log, station)

    def get_parameters(self, config):
        # The server seeds the model from its own ``initial_arrays``, so this
        # is not used for initialisation. Return an empty list defensively.
        return []

    def fit(self, parameters, config):
        # Echo the global parameters back unchanged -- no training. Station B's
        # information rides in the metrics dict.
        return parameters, self.log.n_events, dict(self.info)

    def evaluate(self, parameters, config):
        # Report a benign scalar for the aggregated eval loss. We use the
        # flagged rate: it is bounded [0, 1] and meaningful for this node.
        loss = float(self.info["flagged_rate"])
        return loss, self.log.n_events, dict(self.info)


def client_fn(context: Context) -> "StationBClient":
    """Standalone entry point (lets Station B run on its own if desired)."""
    partition_id = int(context.node_config.get("partition-id", 0))
    n_events = int(context.run_config.get("station-b-events", 500))
    seed = 1000 + partition_id
    return StationBClient("station_b", seed=seed, n_events=n_events).to_client()


app = ClientApp(client_fn=client_fn)
