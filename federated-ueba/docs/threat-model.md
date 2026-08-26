# Threat Model

What this system defends against, how, and — just as important — what it does
**not** claim. Read [`architecture.md`](architecture.md) for the component
picture and [`federated-learning.md`](federated-learning.md) for the learning
layer.

---

## 1. Reference incident

The motivating scenario is a social-engineering breach of a transport
operator's IT helpdesk: an attacker tricks a staff member, obtains **valid
credentials**, logs in, and moves laterally to exfiltrate data over a period of
days. Traditional access control fails because the credentials are technically
valid.

```
Traditional:  phish → valid login → "authenticated" → lateral movement → exfiltration
This system:  phish → valid login → behaviour mismatch → session terminated + alert
```

The defence is **behavioural**, not credential-based. We assume the attacker
*will* sometimes get in; the goal is to remove their ability to *act* once they
do.

## 2. Assets

- Commuter / user behavioural data (sensitive; must not be centralised in raw form).
- Per-site security telemetry (logs of logins, resets, escalations, access).
- The global behavioural model and its updates.
- The central coordination / aggregation layer.

## 3. Adversaries and goals

| Adversary | Capability assumed | Goal |
|---|---|---|
| Social engineer | Tricks a human into surrendering valid credentials | Log in, move laterally, exfiltrate |
| Insider / compromised endpoint | Operates a legitimate session abnormally | Escalate privilege, bulk-access data |
| Central-server attacker | Breaches the aggregator / orchestration layer | Learn per-site data or vulnerabilities |
| Network attacker | Observes / tampers with traffic | Read updates, inject poisoned updates |

## 4. What the system defends — and how

### 4.1 Post-breach behavioural blocking (primary)

Even with valid credentials, the attacker's *behaviour* deviates from the
learned local baseline: unusual time, new device, password-reset bursts,
privilege escalation, abnormal sensitive-data volume, recon/lateral command
sequences.

Two loops (see [`architecture.md`](architecture.md) §6):

```
suspicious event
       │
       ▼
 local detector (autoencoder score) + deterministic rules
       │
   ┌───┴────────────────────────┐
   ▼                            ▼
immediate protection        learning signal
block / re-auth / alert     → FL round → better global model
```

- **Immediate protection is deterministic and local.** No federation on the hot
  path, so blocking does not wait for a training round:

  ```
  IF unusual_login AND privilege_escalation AND sensitive_data_access
  THEN terminate session; require re-auth; raise alert
  ```

  Implemented in `security/risk_engine.py`, `security/policy.py`,
  `security/response.py`.

- **Longer-term intelligence is federated.** Flower improves the shared model so
  other sites recognise related patterns in later rounds — without receiving the
  first site's raw logs.

### 4.2 Reduced central-data concentration

Because raw logs stay on-site, a breach of the central coordination layer does
**not** hand the attacker a central repository of commuter data. With secure
aggregation, the server sees only aggregated updates, not per-site updates —
limiting what a server-side attacker learns about specific stations'
vulnerabilities.

### 4.3 Privacy of model updates

- **Secure Aggregation (SecAgg/SecAgg+)** — server sees only the aggregate.
- **Differential Privacy (Opacus)** — bounds what any single update reveals
  about a specific user, so a user's behavioural profile cannot be
  reverse-engineered from global updates.

## 5. Explicit non-claims

Overclaiming is itself a risk. This system does **not** assert:

- ❌ "A breach of the central helpdesk yields *no* usable commuter data."
  Model updates can leak; the coordination layer still needs securing.
- ❌ "Within minutes every node updates to block a new signature."
  FL shares signal over training rounds, not in real time. Real-time blocking is
  the *local* deterministic loop, not FL.
- ❌ "FL alone guarantees privacy." SecAgg and DP are separate, complementary
  layers that must be added on top.
- ❌ "FL stops the human from being tricked." It removes the attacker's ability
  to *leverage* the trick; it does not prevent the initial compromise.

Defensible framing:

> The architecture reduces the concentration of raw behavioural data in a central
> training repository. Combined with secure aggregation, differential privacy,
> authentication, and endpoint controls, this reduces the *impact* of a
> compromise of the central coordination layer.

## 6. Threats introduced by federation itself

Federation adds its own attack surface, acknowledged here:

| Threat | Mitigation (MVP → later) |
|---|---|
| Model-update leakage | SecAgg + DP |
| Poisoning (malicious client sends bad updates) | Robust/bounded aggregation; risk weighting must be capped; anomaly checks on updates |
| Model/data inversion from the global model | DP; limit rounds/precision exposed |
| Central-server compromise | SecAgg (no per-client visibility); authN/Z on the aggregator |
| Transport interception / tampering | TLS between clients and server; authenticated channels |
| Sybil / rogue client | Client authentication / enrolment (deployment mode) |

The **risk-weighting strategy** (`federated/strategy.py`) is a double-edged
lever: it must be bounded and auditable, or it becomes a poisoning amplifier.

## 7. Out of scope for the MVP

- Endpoint hardening, EDR, and OS-level controls (assumed to exist alongside).
- Identity provider / MFA integration (the "require re-auth" action is
  simulated).
- Production key management (Vault) and full K8s/KubeEdge deployment.
- Phishing/social-engineering *content* detection — an optional future modular
  detector feeding the risk engine, not a core dependency.

## 8. Trust boundaries

```
┌── site (trusted-local) ───────────────┐        ┌── central (semi-trusted) ──┐
│ raw logs · local model · detector     │        │ aggregator · global model  │
│ deterministic response                │        │ (sees only aggregates       │
│                                       │        │  under SecAgg)              │
└──────────────┬────────────────────────┘        └──────────────┬─────────────┘
               │  parameter updates only (TLS, authenticated)    │
               └─────────────────────────────────────────────────┘
                        raw logs never cross this boundary
```

The single most important invariant: **raw event logs never cross the
site→central boundary.** Everything else is defence in depth around that line.
