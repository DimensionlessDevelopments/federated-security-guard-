"""Serving plane: score live events with the trained global model.

Decoupled from the Flower training plane. The serving loop loads the global
model artifact produced by the ServerApp, scores an event stream against a
per-station baseline, and writes detections to a store that a frontend or the
correlation agent can read.
"""

from __future__ import annotations

from federated_ueba.serving.loop import StationScorer, run_serving
from federated_ueba.serving.store import Detection, DetectionStore

__all__ = ["DetectionStore", "Detection", "StationScorer", "run_serving"]
