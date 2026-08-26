"""Dispatcher ClientApp: routes each SuperNode to the right client by partition.

Partitions ``0 .. len(STATION_NAMES)-1`` run the existing ML ``StationClient``
via its own ``client_fn`` (unchanged). Any higher partition runs the
independent, non-ML ``StationBClient`` from the ``station_b`` package.

This is the single point of coupling between the ML pipeline and Station B.
``server.py`` and ``client.py`` are not modified; register this module as the
run's ``clientapp`` in ``pyproject.toml``.
"""

from __future__ import annotations

from flwr.app import Context
from flwr.client import ClientApp

from federated_ueba.data.generator import STATION_NAMES
from federated_ueba.federated.client import client_fn as ueba_client_fn
from station_b.client_app import StationBClient


def client_fn(context: Context):
    partition_id = int(context.node_config.get("partition-id", 0))

    if partition_id >= len(STATION_NAMES):
        # Extra partitions beyond the real stations are Station B nodes.
        n_events = int(context.run_config.get("station-b-events", 500))
        seed = 1000 + partition_id
        return StationBClient(
            "station_b", seed=seed, n_events=n_events
        ).to_client()

    # Real stations: delegate to the existing, unmodified ML client.
    return ueba_client_fn(context)


app = ClientApp(client_fn=client_fn)
