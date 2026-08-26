# Station B — a non-ML Flower client

Station B is a minimal, self-contained participant in the federated run. It does
**no machine learning**: it observes its own synthetic event stream and reports
summary information to the Flower `ServerApp`. It exists to show that a node can
join the same FedAvg run and contribute *information* without training a model —
and to serve as a template for wiring additional independent nodes.

See [`architecture.md`](architecture.md) and
[`federated-learning.md`](federated-learning.md) for the wider system.

---

## Design constraints

1. **Entirely independent.** The `station_b` package imports nothing from
   `federated_ueba`. Its logic (synthetic data, rules, reporting) stands alone.
2. **No machine learning.** No model, no training, no gradients.
3. **Joins the *same* FedAvg run.** It participates alongside the ML stations
   against the existing `server.py`, which is **not modified**.

## How it satisfies FedAvg without doing ML

FedAvg imposes one hard contract: every client's `fit()` must return
`(parameters, num_examples, metrics)` where `parameters` has the **same shapes**
as every other client's, because the strategy computes a weighted element-wise
average of them.

Station B satisfies this by **echoing the global parameters back unchanged**:

```
fit(parameters, config) -> (parameters, n_events, info)   # identity on params
```

This is **model-agnostic** — Station B never needs to know the parameter shapes,
it just returns the list it was handed. It is a safe no-op contribution to the
average (at most a mild dilution toward the previous global, weighted by
`num_examples`).

The node's real payload travels in the **metrics dict**, which FedAvg leaves
free-form:

```json
{
  "station": "station_b",
  "node_kind": "station_b_lite",
  "n_events": 500,
  "n_flagged": 2,
  "flagged_rate": 0.004,
  "mean_failed_logins": 0.37,
  "max_records_accessed": 167.4
}
```

`num_examples` is Station B's synthetic event count — the weight key the server
strategy uses (`weighted_by_key="num-examples"`).

`evaluate()` returns the `flagged_rate` as its scalar loss (bounded `[0, 1]`).
Note: the server aggregates eval loss weighted by `num_examples`, so this scalar
does colour the aggregated eval loss. That is an accepted trade-off of joining
the run without modifying `server.py`.

## Package layout

```
station_b/
  synthetic.py    # numpy-only synthetic event generation (EventLog)
  reporter.py     # deterministic summary stats + threshold rules -> info dict
  client_app.py   # StationBClient(NumPyClient) + standalone ClientApp
```

Rules in `reporter.py` are deterministic and auditable (e.g. reset bursts,
off-hours privilege escalation) — no learning involved.

## Wiring into the run

A single thin dispatcher is the only coupling point:

```
federated_ueba/federated/client_dispatch.py
```

It routes by `partition-id`: partitions `0..3` run the existing, unmodified ML
`StationClient`; partition `4` runs `StationBClient`. `pyproject.toml` registers
the dispatcher as the run's `clientapp` and sets the federation to 5 supernodes:

```toml
[tool.flwr.app.components]
serverapp = "federated_ueba.federated.server:app"
clientapp = "federated_ueba.federated.client_dispatch:app"

[tool.flwr.federations.local-sim]
options.num-supernodes = 5   # 4 ML stations + 1 Station B
```

`server.py` and `client.py` are untouched.

## Run it

```bash
uv sync --extra dev
uv run flwr run .
```

Station B's `info` dict appears in the per-round client metrics in the server
output. To run Station B on its own, `station_b.client_app:app` is a valid
standalone `ClientApp`.

## Extending

To add another independent non-ML node, copy `station_b/`, give it its own
synthetic source and rules, add a partition branch in the dispatcher, and bump
`num-supernodes`.
