"""FedAvg variant tolerating non-ML nodes in the federation.

Flower's FedAvg refuses to aggregate a round whose replies carry differing
MetricRecord keys (``InconsistentMessageReplies``). That is a problem for a
mixed federation: the ML stations report detection metrics
(``accuracy``/``precision``/...), while a non-ML node such as Station B
reports its own information keys (``n_flagged``/``flagged_rate``/...).

This strategy normalises replies before aggregation, without either node type
having to know about the other:

- Every reply's MetricRecord is restricted to the ML stations' canonical key
  set, padding missing keys with 0.0.
- Keys outside that set (a non-ML node's own report) are logged verbatim per
  round, so the information is surfaced rather than averaged into nonsense.
- During evaluation, a reply that had to be padded is zero-weighted, so a
  non-ML node's incomparable loss scale never skews the reported detection
  metrics. Training weights are left untouched, so such a node still
  contributes its event count to the FedAvg average as intended.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from logging import INFO

from flwr.app import ArrayRecord, Message, MetricRecord
from flwr.common.logger import log
from flwr.serverapp.strategy import FedAvg

WEIGHT_KEY = "num-examples"


class MixedFederationFedAvg(FedAvg):
    """FedAvg that accepts replies from non-ML nodes.

    Parameters
    ----------
    train_metric_keys, evaluate_metric_keys :
        The canonical MetricRecord keys the ML clients reply with. Every
        reply is coerced to exactly these keys before aggregation.
    """

    def __init__(
        self,
        *args,
        train_metric_keys: Sequence[str] = (WEIGHT_KEY,),
        evaluate_metric_keys: Sequence[str] = ("eval_loss", WEIGHT_KEY),
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.train_metric_keys = tuple(train_metric_keys)
        self.evaluate_metric_keys = tuple(evaluate_metric_keys)

    def _normalize(
        self,
        replies: Iterable[Message],
        canonical: Sequence[str],
        zero_weight_padded: bool,
    ) -> list[Message]:
        """Coerce every reply's MetricRecord to the canonical key set."""
        materialized = list(replies)
        canonical_set = set(canonical)

        for msg in materialized:
            if msg.has_error() or not msg.has_content():
                continue
            records = msg.content.metric_records
            if not records:
                continue
            record_key = next(iter(records.keys()))
            source = dict(records[record_key])

            extras = {k: v for k, v in source.items() if k not in canonical_set}
            missing = canonical_set - source.keys()

            if extras:
                log(
                    INFO,
                    "\t> Report from non-ML node %d: %s",
                    msg.metadata.src_node_id,
                    ", ".join(f"{k}={v}" for k, v in sorted(extras.items())),
                )

            normalized = {
                key: source.get(key, 0.0) for key in canonical
            }
            # A padded reply came from a node that does not compute the ML
            # metrics; during evaluation its scales are not comparable.
            if missing and zero_weight_padded and WEIGHT_KEY in normalized:
                normalized[WEIGHT_KEY] = 0
            msg.content[record_key] = MetricRecord(normalized)

        return materialized

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        return super().aggregate_train(
            server_round,
            self._normalize(
                replies, self.train_metric_keys, zero_weight_padded=False
            ),
        )

    def aggregate_evaluate(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[MetricRecord | None]:
        return super().aggregate_evaluate(
            server_round,
            self._normalize(
                replies, self.evaluate_metric_keys, zero_weight_padded=True
            ),
        )
