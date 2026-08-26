# Architecture — Federated UEBA

A zero-trust distributed behavioural defence platform for transport infrastructure.

Security telemetry stays local to each operational environment. Local anomaly
detectors provide **immediate enforcement**, while [Flower](https://flower.ai)
federated learning lets participating environments **collaboratively improve a
shared behavioural model without centralising the underlying raw logs**.

Flower is a primary objective of this project — but the security functionality
does **not** depend on federation. The system runs in three interchangeable
learning modes (local, centralised, federated), so the demo degrades gracefully
if the federated component becomes unstable during the hackathon.

---

## 1. Design principles

1. **Federation is a switch, not a foundation.** The application talks to a
   `LearningCoordinator` abstraction. Local-only, centralised, and Flower-backed
   coordinators are drop-in implementations. Everything above the coordinator is
   identical across modes.
2. **Detection is decoupled from learning.** Immediate blocking uses
   deterministic policy plus the current local model. Federated learning is a
   slower background loop that improves the model over time. FL is *not* claimed
   to be a real-time propagation mechanism.
3. **Start simple.** A small PyTorch autoencoder over tabular behavioural
   features is enough to tell the story. No LLM, no Kubernetes, no homomorphic
   encryption in the MVP.
4. **Privacy is layered and optional.** Secure aggregation and differential
   privacy are additive modes, sequenced so they can never block the core demo.

---

## 2. System overview

```
                         ┌─────────────────────┐
                         │  Demo Dashboard      │  apps/dashboard  (Streamlit)
                         │  anomalies · rounds  │
                         │  local vs federated  │
                         └──────────┬───────────┘
                                    │ HTTP
                         ┌──────────▼───────────┐
                         │  API                 │  apps/api  (FastAPI)
                         └──────────┬───────────┘
                                    │
                         ┌──────────▼───────────┐
                         │  LearningCoordinator │  training/ + federated/
                         │  local│central│flower│
                         └───┬───────────┬──────┘
                             │           │
              ┌──────────────┘           └──────────────┐
     ┌────────▼────────┐              ┌─────────▼────────┐
     │ Station Alpha   │              │ Station Bravo    │
     │ Flower Client   │              │ Flower Client    │
     │ local event log │     ...      │ local event log  │
     │ autoencoder     │              │ autoencoder      │
     │ anomaly scoring │              │ anomaly scoring  │
     └────────┬────────┘              └─────────┬────────┘
              │                                 │
     ┌────────▼────────┐              ┌─────────▼────────┐
     │ Station Charlie │              │ Central Helpdesk │
     │ local-only data │              │ local-only data  │
     └─────────────────┘              └──────────────────┘
```

Each site holds its own logs, trains locally, and (in federated mode) sends only
model parameter updates to the Flower `ServerApp`, which aggregates them
(FedAvg by default) and redistributes a new global model.

---

## 3. Component map

The architecture maps directly onto the repository layout.

| Concern | Module | Role |
|---|---|---|
| Synthetic data | `federated_ueba/data/generator.py` | Generate per-site behavioural event logs |
| | `federated_ueba/data/partition.py` | Produce **non-IID** partitions across sites |
| | `federated_ueba/data/preprocessing.py`, `dataset.py` | Feature encoding, PyTorch datasets |
| Model | `federated_ueba/models/autoencoder.py` | Small reconstruction autoencoder |
| Detection | `federated_ueba/detection/anomaly_detector.py` | Wraps model, produces anomaly scores |
| | `federated_ueba/detection/scoring.py`, `thresholds.py` | Reconstruction error → score → decision |
| Immediate response | `federated_ueba/security/risk_engine.py` | Combine detector + deterministic rules |
| | `federated_ueba/security/policy.py`, `response.py` | Block / re-auth / alert actions |
| Training modes | `federated_ueba/training/local.py` | Local-only training |
| | `federated_ueba/training/centralized.py` | Pooled baseline |
| | `federated_ueba/training/evaluate.py` | Shared eval / metrics |
| Federation | `federated_ueba/federated/client.py` | Flower `ClientApp` (local fit/evaluate) |
| | `federated_ueba/federated/server.py` | Flower `ServerApp` |
| | `federated_ueba/federated/strategy.py` | Aggregation strategy (FedAvg + custom weighting) |
| | `federated_ueba/federated/simulation.py` | Multi-client simulation harness |
| Attack story | `federated_ueba/simulation/attack_generator.py` | Inject social-engineering / lateral-movement sequences |
| | `federated_ueba/simulation/scenarios.py`, `event_stream.py` | Scripted demo scenarios |
| Entry points | `scripts/*.py` | Generate data, train (local/central/federated), run demo |
| Surfaces | `apps/api/main.py`, `apps/dashboard/app.py` | API + dashboard |
| Config | `configs/{base,local,centralized,federated}.yaml` | Per-mode configuration |

---

## 4. The ML problem

**Behavioural anomaly detection**, not phishing or LLMs. Each event is a row of
tabular features observed at a site:

```
timestamp, hour_of_day, day_of_week, device_id, user_role,
failed_login_count, password_reset_count, privilege_escalation_attempt,
session_duration, sensitive_records_accessed,
network_destination_risk, command_sequence_score
```

| Feature | Normal | Suspicious |
|---|---|---|
| Login time | 09:00–18:00 | 02:37 |
| Password resets | 1–3 / day | 25 in 5 min |
| Privilege escalation | rare | repeated |
| Sensitive records | typical volume | large spike |
| Device / location | known | new / unusual |
| Command sequence | typical workflow | recon / lateral movement |

An autoencoder learns to reconstruct *normal* activity. High reconstruction
error = anomaly:

```
input → encoder → latent → decoder → reconstruction
                                          │
                                   reconstruction error
                                          ├─ low  → normal
                                          └─ high → anomaly → alert / block
```

A small PyTorch network is sufficient; Isolation Forest is a viable
lighter-weight alternative for the baseline.

---

## 5. How Flower fits

Each simulated site receives a **different data distribution** — this is what
makes it a genuine federated (non-IID) problem, not just parallel training.

- **Alpha** — mostly daytime users, few admin actions, few resets
- **Bravo** — 24-hour operations, more privileged users
- **Charlie** — high passenger-service auth volume, different baseline
- **Helpdesk** — reset/escalation-heavy, the original breach surface

The FedAvg loop per round:

```
global model ──► each client fits on its local logs ──► send parameter updates
      ▲                                                        │
      └──────────────── strategy aggregates ◄──────────────────┘
```

`strategy.py` extends FedAvg to **weight updates from higher-risk nodes more
heavily**, matching the "high-risk node" idea in the plan via Flower's Strategy
API.

Development uses the **Flower Simulation Engine** (many clients on one machine);
the same `ClientApp`/`ServerApp` code runs in deployment mode later.

---

## 6. Detection vs. learning — two loops

The proposal's "within minutes every node updates" claim is softened into a
defensible two-loop design:

```
              suspicious event
                     │
                     ▼
              local detector
                     │
             ┌───────┴────────┐
             ▼                ▼
    immediate protection   learning signal
    (deterministic +          │
     current model)           ▼
             │            FL training round
             ▼                │
    block / re-auth / alert   ▼
                          updated global model
```

- **Immediate protection** is deterministic and local — no federation on the
  hot path:

  ```
  IF unusual_login AND privilege_escalation AND sensitive_data_access
  THEN terminate session, require re-auth, raise alert
  ```

- **Longer-term intelligence** is federated: Flower improves the shared model so
  other sites recognise related patterns *after* subsequent rounds — without
  ever seeing Alpha's raw logs.

---

## 7. Three learning modes

The core abstraction:

```python
class LearningCoordinator:
    def train(self, clients): ...

class LocalCoordinator(LearningCoordinator): ...        # training/local.py
class CentralizedCoordinator(LearningCoordinator): ...  # training/centralized.py
class FlowerCoordinator(LearningCoordinator): ...       # federated/
```

The dashboard exposes a toggle: **Local · Centralised · Federated**. Running the
same attack scenario across modes produces the headline result:

```
                   LOCAL ONLY     FEDERATED
Alpha attack        DETECTED       DETECTED
Bravo variant       MISSED         DETECTED
Charlie variant     MISSED         DETECTED
```

> Actual numbers are **measured and reported from the experiment**, never
> hard-coded into the presentation.

This also provides the fallback: if Flower breaks before the demo, Local mode
still delivers a working decentralised UEBA story.

---

## 8. Privacy layers (additive, sequenced)

Ordered so privacy engineering can never block the demo:

| Level | Capability | Where |
|---|---|---|
| 1 | Local anomaly detection | `detection/` |
| 2 | Federated learning (FedAvg) | `federated/` |
| 3 | Secure aggregation (SecAgg / SecAgg+) | `federated/strategy.py`, `security/` |
| 4 | Differential privacy (Opacus) | `training/` |
| 5 | Real distributed deployment (Docker) | `docker-compose.yml` |

**Secure aggregation** and **differential privacy** are complementary, not
interchangeable: SecAgg hides *individual* updates from the server; DP bounds
what any update can leak about a specific user.

---

## 9. Explicitly out of scope for the MVP

Deferred to keep the demo robust — extensible later, not dependencies now:

- **Kubernetes / KubeEdge** — Docker Compose is enough for 3–10 simulated clients.
- **Ray** — the Flower Simulation Engine already covers multi-client simulation.
- **TenSEAL / homomorphic encryption** — expensive; SecAgg is the better privacy
  demo.
- **LLM / FedPhishLLM** — phishing becomes an optional second modular detector
  feeding the risk engine, never a core dependency.

---

## 10. Build order

```
Phase 1  generate synthetic logs · train autoencoder · demo local anomaly detection
Phase 2  3 non-IID datasets · Flower simulation · FedAvg aggregation
Phase 3  attack simulator · dashboard · compare local vs federated
Optional secure aggregation · Opacus DP · Docker deployment
```

See [`federated-learning.md`](federated-learning.md) and [`threat-model.md`](threat-model.md)
for the detail behind each area.
