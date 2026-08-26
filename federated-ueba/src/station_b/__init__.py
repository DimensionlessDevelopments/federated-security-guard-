"""Station B: a minimal, self-contained Flower client.

This package is deliberately independent of ``federated_ueba``. It does no
machine learning: it observes its own synthetic event stream and reports
summary information to the Flower ServerApp via the FL metrics channel, while
satisfying FedAvg's parameter contract by echoing the global model unchanged.

See ``docs/station-b.md`` for the design rationale.
"""

from __future__ import annotations

from station_b.client_app import StationBClient, app, client_fn

__all__ = ["StationBClient", "app", "client_fn"]
