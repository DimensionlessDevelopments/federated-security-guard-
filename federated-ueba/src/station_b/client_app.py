"""Station B client -- a non-ML participant in the FedAvg run.

Uses the Flower message API (matching the ServerApp's `strategy.start` and the
ML `StationClient`): the server sends ArrayRecord/ConfigRecord messages, and
Station B replies with the **same arrays echoed back unchanged** plus scalar
information metrics. It does no machine learning:

- Echoing the received ArrayRecord makes Station B shape-compatible with FedAvg
  without ever knowing the model's parameter shapes.
- Its real payload (event counts, rule flags, feature stats) rides in the
  metrics, weighted by ``num-examples`` (the server's FedAvg weight key).

The package imports nothing from ``federated_ueba``; it is wired into the run
through a thin dispatcher (see ``federated_ueba.federated.client_dispatch``).
"""

from __future__ import annotations

from flwr.app import Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from station_b.reporter import summarize
from station_b.synthetic import generate_events


class StationBClient:
    """Non-ML station. Transport-agnostic: plain fit/evaluate over ndarray
    lists so it can run with or without the Flower runtime."""

    def __init__(self, station: str, seed: int, n_events: int = 500) -> None:
        self.station = station
        self.log = generate_events(n_events=n_events, seed=seed)
        self.info = summarize(self.log, station)

    def numeric_info(self) -> dict[str, float | int]:
        """Information metrics restricted to numeric values (for MetricRecord)."""
        return {
            k: v for k, v in self.info.items() if isinstance(v, (int, float))
        }

    def fit(self, parameters, config):
        # Echo the global parameters back unchanged -- no training. Station B's
        # information rides in the metrics dict.
        return parameters, self.log.n_events, dict(self.info)

    def evaluate(self, parameters, config):
        # Report a benign scalar for the aggregated eval loss. We use the
        # flagged rate: it is bounded [0, 1] and meaningful for this node.
        loss = float(self.info["flagged_rate"])
        return loss, self.log.n_events, dict(self.info)


def make_station_b_client(context: Context) -> StationBClient:
    """Build the Station B client for this node from the run context."""
    partition_id = int(context.node_config.get("partition-id", 0))
    n_events = int(context.run_config.get("station-b-events", 500))
    seed = 1000 + partition_id
    return StationBClient("station_b", seed=seed, n_events=n_events)


def station_b_train(msg: Message, context: Context) -> Message:
    """Train handler: echo the arrays, report information via metrics."""
    client = make_station_b_client(context)
    content = RecordDict(
        {
            # Echo the received ArrayRecord verbatim -- preserves keys/shapes,
            # so FedAvg aggregation is a safe no-op contribution.
            "arrays": msg.content["arrays"],
            "metrics": MetricRecord(
                {"num-examples": client.log.n_events, **client.numeric_info()}
            ),
        }
    )
    return Message(content=content, reply_to=msg)


def station_b_evaluate(msg: Message, context: Context) -> Message:
    """Evaluate handler: report a benign scalar loss plus information metrics."""
    client = make_station_b_client(context)
    parameters = msg.content["arrays"].to_numpy_ndarrays()
    loss, num_examples, _info = client.evaluate(parameters, {})
    content = RecordDict(
        {
            "metrics": MetricRecord(
                {
                    "eval_loss": loss,
                    "num-examples": num_examples,
                    **client.numeric_info(),
                }
            )
        }
    )
    return Message(content=content, reply_to=msg)


# Standalone ClientApp: lets Station B run on its own if desired.
app = ClientApp()
app.train()(station_b_train)
app.evaluate()(station_b_evaluate)
