"""Tests for the mixed-federation FedAvg strategy.

FedAvg raises `InconsistentMessageReplies` when replies in a round carry
differing MetricRecord keys. In a mixed federation the ML stations report
detection metrics while a non-ML node (Station B) reports its own keys, so
the strategy normalises replies before aggregation.
"""

from __future__ import annotations

import numpy as np
import pytest
from flwr.app import ArrayRecord, Message, MetricRecord, RecordDict

from federated_ueba.federated.client import (
    EVAL_METRIC_KEYS,
    TRAIN_METRIC_KEYS,
)
from federated_ueba.federated.strategy import MixedFederationFedAvg
from federated_ueba.models import SecurityAutoencoder


def _arrays() -> ArrayRecord:
    model = SecurityAutoencoder(input_dim=11, hidden_dim=32, latent_dim=8)
    return ArrayRecord(model.state_dict())


def _reply(metrics: dict, with_arrays: bool = False) -> Message:
    records: dict = {"metrics": MetricRecord(metrics)}
    if with_arrays:
        records["arrays"] = _arrays()
    incoming = Message(
        RecordDict({"arrays": _arrays()}), dst_node_id=0, message_type="train"
    )
    return Message(content=RecordDict(records), reply_to=incoming)


def _strategy() -> MixedFederationFedAvg:
    return MixedFederationFedAvg(
        weighted_by_key="num-examples",
        train_metric_keys=TRAIN_METRIC_KEYS,
        evaluate_metric_keys=EVAL_METRIC_KEYS,
    )


def _ml_eval_metrics() -> dict:
    return {
        "eval_loss": 1.5,
        "num-examples": 130,
        "accuracy": 0.93,
        "precision": 0.84,
        "recall": 0.86,
        "fpr": 0.05,
        "threshold": 1.2,
    }


def _station_b_eval_metrics() -> dict:
    """Station B's own keys -- deliberately disjoint from the ML set."""
    return {
        "eval_loss": 0.004,
        "num-examples": 500,
        "n_events": 500,
        "n_flagged": 2,
        "flagged_rate": 0.004,
        "mean_failed_logins": 0.37,
        "max_records_accessed": 167.0,
    }


class TestNormalization:
    def test_evaluate_keys_unified(self):
        strategy = _strategy()
        replies = [
            _reply(_ml_eval_metrics()),
            _reply(_station_b_eval_metrics()),
        ]
        normalized = strategy._normalize(
            replies, EVAL_METRIC_KEYS, zero_weight_padded=True
        )
        key_sets = [
            set(msg.content["metrics"].keys()) for msg in normalized
        ]
        assert key_sets[0] == key_sets[1] == set(EVAL_METRIC_KEYS)

    def test_ml_reply_values_preserved(self):
        strategy = _strategy()
        original = _ml_eval_metrics()
        normalized = strategy._normalize(
            [_reply(original)], EVAL_METRIC_KEYS, zero_weight_padded=True
        )
        metrics = normalized[0].content["metrics"]
        for key, value in original.items():
            assert metrics[key] == pytest.approx(value)

    def test_padded_reply_zero_weighted_on_evaluate(self):
        """Station B's incomparable loss must not skew reported metrics."""
        strategy = _strategy()
        normalized = strategy._normalize(
            [_reply(_station_b_eval_metrics())],
            EVAL_METRIC_KEYS,
            zero_weight_padded=True,
        )
        assert normalized[0].content["metrics"]["num-examples"] == 0

    def test_train_weight_preserved(self):
        """On train, a non-ML node keeps its event count as FedAvg weight."""
        strategy = _strategy()
        normalized = strategy._normalize(
            [_reply({"num-examples": 500, "n_flagged": 2}, with_arrays=True)],
            TRAIN_METRIC_KEYS,
            zero_weight_padded=False,
        )
        metrics = normalized[0].content["metrics"]
        assert metrics["num-examples"] == 500
        assert set(metrics.keys()) == set(TRAIN_METRIC_KEYS)

    def test_extra_keys_dropped_from_metrics(self):
        strategy = _strategy()
        normalized = strategy._normalize(
            [_reply(_station_b_eval_metrics())],
            EVAL_METRIC_KEYS,
            zero_weight_padded=True,
        )
        metrics = normalized[0].content["metrics"]
        for key in ("n_flagged", "flagged_rate", "max_records_accessed"):
            assert key not in metrics


class TestAggregation:
    def test_mixed_replies_aggregate_without_error(self):
        """The regression guard: real FedAvg validation must accept the round."""
        strategy = _strategy()
        replies = [
            _reply(_ml_eval_metrics()),
            _reply(_station_b_eval_metrics()),
        ]
        result = strategy.aggregate_evaluate(1, replies)
        metrics = result[0] if isinstance(result, tuple) else result
        assert metrics is not None

    def test_zero_weighted_node_does_not_skew_loss(self):
        """Aggregate loss must equal the ML station's, not an average with
        Station B's 0.004 flagged-rate."""
        strategy = _strategy()
        replies = [
            _reply(_ml_eval_metrics()),
            _reply(_station_b_eval_metrics()),
        ]
        result = strategy.aggregate_evaluate(1, replies)
        metrics = result[0] if isinstance(result, tuple) else result
        assert metrics["eval_loss"] == pytest.approx(1.5, abs=1e-6)

    def test_train_aggregation_accepts_mixed_replies(self):
        strategy = _strategy()
        replies = [
            _reply({"num-examples": 500}, with_arrays=True),
            _reply({"num-examples": 500, "n_flagged": 2}, with_arrays=True),
        ]
        arrays, _ = strategy.aggregate_train(1, replies)
        assert arrays is not None
