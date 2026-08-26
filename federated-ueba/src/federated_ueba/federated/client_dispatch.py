"""Dispatcher ClientApp: routes each SuperNode to the right client by partition.

Partitions ``0 .. len(STATION_NAMES)-1`` run the existing ML client's message
handlers (unchanged). Any higher partition runs the independent, non-ML
Station B handlers from the ``station_b`` package.

This is the single point of coupling between the ML pipeline and Station B.
``server.py`` and ``client.py`` are not modified; register this module as the
run's ``clientapp`` in ``pyproject.toml``.
"""

from __future__ import annotations

from flwr.app import Context, Message
from flwr.clientapp import ClientApp

from federated_ueba.data.generator import STATION_NAMES
from federated_ueba.federated.client import evaluate as ml_evaluate
from federated_ueba.federated.client import train as ml_train
from station_b.client_app import station_b_evaluate, station_b_train


def is_station_b(context: Context) -> bool:
    """Extra partitions beyond the real stations are Station B nodes."""
    partition_id = int(context.node_config.get("partition-id", 0))
    return partition_id >= len(STATION_NAMES)


app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    if is_station_b(context):
        return station_b_train(msg, context)
    return ml_train(msg, context)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    if is_station_b(context):
        return station_b_evaluate(msg, context)
    return ml_evaluate(msg, context)
