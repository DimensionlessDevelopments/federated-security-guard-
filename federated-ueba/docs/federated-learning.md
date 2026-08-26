# Federated Learning with Flower

How the collaborative-learning layer works, why it uses Flower, and how it is
implemented in this project. Read [`architecture.md`](architecture.md) first for
the system-level picture.

---

## 1. Why federated learning here

The scenario is distributed by nature: helpdesk terminals and station
controllers each observe their own security telemetry. Centralising raw
behavioural logs is exactly the concentration risk the project is trying to
remove. Federated learning lets every site contribute to a shared behavioural
model while its **raw event logs never leave the site** — only model parameter
updates are exchanged.

The value we demonstrate: a pattern first seen at one site (an attack) improves
the global model, so **other sites recognise related patterns after subsequent
rounds** without ever seeing the first site's data.

## 2. Why Flower specifically

- **Python-native, framework-agnostic** — wraps our existing PyTorch autoencoder
  with minimal glue.
- **Simulation Engine** — run many clients on one machine, so we can model
  Alpha / Bravo / Charlie / Helpdesk without edge hardware. The same
  `ClientApp`/`ServerApp` code later runs in deployment mode unchanged.
- **Strategy API** — lets us extend FedAvg to weight high-risk nodes more
  heavily (§5).
- **Built-in SecAgg / SecAgg+** — a credible privacy story without homomorphic
  encryption (§7).

Flower is a **primary objective** of the hackathon, but by design it sits behind
the `LearningCoordinator` abstraction so the security demo survives without it.

## 3. The FedAvg loop

```
        ┌──────────────────────── round r ────────────────────────┐
        │                                                          │
   global model  ──►  client selection  ──►  local fit on          │
    (params θ_r)                              each site's logs      │
        ▲                                          │                │
        │                                   parameter updates θ_r^k │
        │                                          ▼                │
        └──────────────  strategy aggregates  ◄────────────────────┘
                         θ_{r+1} = Σ_k (n_k / n) · θ_r^k
```

Each client `k`:

1. receives global parameters `θ_r`;
2. trains the autoencoder locally on its own event history;
3. returns updated parameters `θ_r^k` (plus `n_k`, its example count);
4. the server aggregates (weighted mean) into `θ_{r+1}`;
5. the new global model is redistributed.

No raw logs are transferred at any step — only parameters and scalar metrics.

## 4. Non-IID data — the point of the exercise

If every site had the same distribution, federation would be pointless. We
deliberately generate **non-identically distributed** partitions so the shared
model has to reconcile genuinely different baselines.

| Site | Character | Normal baseline |
|---|---|---|
| Alpha | Daytime station | 09:00–18:00 activity, few admin actions, few resets |
| Bravo | 24-hour operations | round-the-clock logins, more privileged users |
| Charlie | High passenger volume | large auth volume, different traffic mix |
| Helpdesk | Support desk | reset- and escalation-heavy (the breach surface) |

Partitioning lives in `src/federated_ueba/data/partition.py`; generation in
`data/generator.py`. Each site's slice is written to `data/synthetic/`.

Because baselines differ, a naive global average must still keep each site's
false-positive rate acceptable — which is exactly what the local-vs-federated
comparison measures.

## 5. Aggregation strategy

Default: **FedAvg** (example-count weighted mean).

Extension in `src/federated_ueba/federated/strategy.py`: subclass Flower's
strategy to **up-weight updates from higher-risk nodes**, so a site actively
seeing attack behaviour contributes more strongly to the round. This realises
the plan's "weigh security alerts from high-risk nodes more heavily" idea through
the Strategy API rather than as a bespoke protocol.

```python
class RiskWeightedFedAvg(FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        # scale each client's weight by a risk factor reported in its metrics
        ...
```

Keep the risk factor bounded and auditable — it is a demo lever, not a security
control on its own.

## 6. Implementation map

| Piece | File | Responsibility |
|---|---|---|
| Client | `federated/client.py` | Flower `ClientApp`: `get_parameters`, `fit`, `evaluate` over the local autoencoder |
| Server | `federated/server.py` | Flower `ServerApp`: wires the strategy, sets number of rounds |
| Strategy | `federated/strategy.py` | `RiskWeightedFedAvg` (FedAvg + risk weighting) |
| Simulation | `federated/simulation.py` | Spins up N virtual clients on one machine |
| Coordinator | `federated/` via `training/` | `FlowerCoordinator` implementing `LearningCoordinator.train()` |
| Config | `configs/federated.yaml` | rounds, clients, local epochs, strategy params |
| Entry point | `scripts/train_federated.py` | Run a federated training session end to end |

The client `fit()` wraps the exact same PyTorch training step used by
`training/local.py`, so local and federated modes train an identical model —
only the coordination differs.

## 7. Privacy add-ons

Additive levels (see [`architecture.md`](architecture.md) §8), sequenced so they
never block the core demo:

- **Secure Aggregation (SecAgg / SecAgg+).** The server only ever sees the
  *aggregate* of updates, not any individual client's update. Configured via the
  Flower strategy. This is the recommended privacy feature to demonstrate.
- **Differential Privacy (Opacus).** Wrap the local PyTorch training with a
  `PrivacyEngine` for gradient clipping + noise, bounding what any single
  update can reveal about a specific user.

SecAgg and DP are **complementary**: SecAgg hides individual updates in transit /
at the server; DP bounds information leakage from the updates themselves.

## 8. What FL does *not* do

- It is **not** a real-time propagation mechanism. Signature sharing happens over
  training rounds, not milliseconds. Immediate blocking is deterministic and
  local (see [`threat-model.md`](threat-model.md)).
- It does **not** guarantee zero leakage. Model updates can leak; the
  coordination layer still needs authentication, transport security, and the
  privacy layers above.

## 9. Metrics to report

Measured from the experiment, never hard-coded:

- Detection rate / recall on injected attack variants, **per site**, per mode.
- False-positive rate on normal traffic per site (fairness across non-IID sites).
- Reconstruction-error distributions: normal vs. attack.
- Global model version progression across rounds.
- "Raw logs transferred: 0" — a headline, provable from the design.

The core comparison table (local-only vs. federated) is produced by
`training/evaluate.py` and surfaced on the dashboard.
