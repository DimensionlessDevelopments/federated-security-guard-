# Federated Security Guard

Federated behavioural anomaly detection (UEBA) for distributed transport
infrastructure, built on [Flower](https://flower.ai). Four simulated
operational environments -- `station_alpha`, `station_bravo`,
`station_charlie`, `central_helpdesk` -- each train a small autoencoder on
their own normal activity. Flower aggregates the model updates; raw
security telemetry never leaves a station. An LLM agent connects to the
trained global model, gathers local events, and raises incident reports
with analyst insights.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

All commands in this README are run from **this directory**
(`federated-ueba/`, the one containing `pyproject.toml`):

```bash
cd federated-ueba
uv sync
```

`uv sync` creates `.venv` and installs the project plus all dependencies
from `uv.lock`. (Plain `pip install -e ".[dev]"` also works if you prefer
managing the environment yourself.)

## Layout

```
federated_ueba/
  models/autoencoder.py        SecurityAutoencoder (input_dim -> hidden -> latent)
  data/generator.py            11-feature UEBA schema + per-station normal profiles
  data/preprocessing.py        z-score normalization
  simulation/attack_generator.py  station-specific + full attack tactics
  detection/anomaly_detector.py   train/score/threshold pipeline
  detection/scoring.py         reconstruction-error anomaly score
  federated/client.py          Flower ClientApp (message API) -- StationClient
  federated/server.py          Flower ServerApp -- FedAvg, saves global model
  training/evaluate.py         local-vs-federated statistical comparison
  agent/incident.py            gather -> score -> incident report (LLM-free)
  agent/agent_app.py           Flower AgentApp -- LLM insights on incidents
station_b/                     non-ML station (own package, no torch)
  client_app.py                StationBClient -- rule-based, echoes the model
tests/                         unit suite + e2e simulation suite
```

Both packages sit at the app root, which is the layout Flower expects: a
FAB installed on a SuperNode puts the app directory itself on the import
path, so `federated_ueba` and `station_b` resolve with no `PYTHONPATH`
shim. See [Packaging for Flower Hub](#packaging-for-flower-hub).

## Running the tests

Fast unit suite (excludes e2e by default):

```bash
uv run pytest
```

Full end-to-end simulation test -- launches the real `flwr run` pipeline
(server + 4 clients + 5 FedAvg rounds), verifies every round succeeds, and
statistically validates the produced global model against local-only
baselines (recall floor, FPR policy, cross-station generalisation,
federated >= local):

```bash
uv run pytest -m e2e
```

The e2e suite needs the `flwr` CLI on PATH and takes a few minutes; it
skips itself with a reason when the CLI is missing.

## Full run guide

### Step 1 -- Federated training (ServerApp + ClientApps)

One command runs the whole training pipeline: Flower auto-starts a local
SuperLink daemon, the ServerApp orchestrates FedAvg, and 4 simulated
ClientApps (one per station) train locally and return model updates:

```bash
uv run flwr run . local-sim --stream
```

- All 5 rounds stream to the terminal (`[ROUND n/5]` + aggregated metrics).
- The server saves the aggregated model to `artifacts/global_model.pt`.
- The local SuperLink keeps running afterwards (inspect runs with
  `uv run flwr ls`; stop the daemon with `pkill -f flower-superlink`).
- `uv run` is only about using the project's `.venv`; the app modules
  themselves need no path setup (root-package layout). Any environment
  with the dependencies installed can run `flwr run .` directly.

### Step 2 -- Incident agent on the trained model

```bash
uv run python -m federated_ueba.agent
```

Loads `artifacts/global_model.pt`, gathers recent events at the station,
scores them, and prints the incident report (see options below).

### Step 3 (optional) -- Agent as a Flower AgentApp / `flwr chat`

Flower decides what `flwr run` executes by the FAB's components: **if an
`agentapp` component is registered, it runs the AgentApp instead of the
training pipeline.** The `agentapp` line in `pyproject.toml` is therefore
commented out by default. To run the agent through Flower:

1. Uncomment `agentapp = "federated_ueba.agent.agent_app:app"` in
   `[tool.flwr.app.components]`.
2. `uv run flwr run . local-sim --stream` now executes the agent: it
   waits for `artifacts/global_model.pt` (train first, Step 1), gathers
   events, and reports. Without LLM runtime credentials it prints the
   analyst prompt instead of calling the model.
3. For LLM insights and `uv run flwr chat`, authenticate first with
   `uv run flwr login` -- chat sessions run against Flower's hosted
   SuperLink (`supergrid.flower.ai`), which injects the LLM credentials.
4. Re-comment the `agentapp` line before running training again.

Training configuration lives in `pyproject.toml` under
`[tool.flwr.app.config]`:

| key | default | meaning |
|---|---|---|
| `input-dim` | 11 | must match the 11-feature UEBA schema |
| `hidden-dim` / `latent-dim` | 32 / 8 | autoencoder dimensions (server and clients build from the same values) |
| `num-server-rounds` | 5 | FedAvg rounds |
| `fraction-train` / `fraction-evaluate` | 1.0 | client sampling fractions |
| `learning-rate` / `local-epochs` | 0.001 / 2 | pushed to clients via ConfigRecord each round |
| `global-model-path` | `artifacts/global_model.pt` | where the server saves the aggregated model |
| `agent.model` | `openai/gpt-5.6-sol` | analyst LLM for the AgentApp |
| `agent.station` | `central_helpdesk` | station the agent inspects |

## Packaging for Flower Hub

Flower distributes an app as a **FAB** (Flower App Bundle): a zip of the
app source plus its `pyproject.toml`, which a SuperNode installs and puts
on the import path before a run.

```bash
uv run flwr build
```

produces `stevenrce0.federated-ueba.0-1-0.<hash>.fab`:

```
pyproject.toml     app metadata, components, [tool.flwr.app.config] defaults
LICENSE  README.md
federated_ueba/    ServerApp, ClientApp dispatcher, AgentApp, model, detection
station_b/         non-ML ClientApp
```

`fab-include` in `pyproject.toml` pins that list, so tests, docs and
`docker-compose.yml` stay out of the bundle; `artifacts/` and `secrets/`
are gitignored and never reach it either.

Verify the bundle the way a node consumes it -- install it and import the
entrypoints from the installed copy, with the source tree off the path:

```bash
uv run flwr install stevenrce0.federated-ueba.0-1-0.<hash>.fab
```

It unpacks to `~/.flwr/apps/stevenrce0.federated-ueba.0.1.0.<hash>/`, where
`federated_ueba.federated.server:app`,
`federated_ueba.federated.client_dispatch:app` and
`federated_ueba.agent.agent_app:app` all resolve from the app directory
alone. That is the reason for the root-package layout: with a `src/`
layout the FAB root holds `src/federated_ueba/`, and nothing on a
SuperNode adds `src/` to `sys.path`.

Publishing to the Hub uploads the project *source* rather than the FAB
(73 files / ~180 KB here, against limits of 1 MB per file and 10 MB
total). The file set is `.gitignore` minus an extension allowlist --
`.py`, `.md`, `.toml`, `.yaml`/`.yml`, `.json`, plus root `LICENSE` --
which is why the trained model (`artifacts/*.pt`) and the SuperNode SSH
keys (`secrets/`) cannot be swept in.

```bash
uv run flwr login          # account on supergrid.flower.ai
uv run flwr app publish .
```

Hub versions are immutable: re-publishing an existing `[project].version`
returns `409 this version already exists`, so bump the version to ship a
change.

Publish-time requirements, all met by this app: `publisher` under
`[tool.flwr.app]` must equal the authenticated Hub account (`stevenrce0`)
or the upload is rejected with `403 publisher does not match
authenticated user`; the app directory name (`federated-ueba`) must start
with a letter and hold only letters, digits and hyphens (≤32 chars);
`[project].description` must be non-empty and ≤200 characters; and
`[project].license.file`, if declared, must be `LICENSE` or `LICENSE.md`
and present in the upload.

## Incident agent

Standalone testing entrypoint (no Flower runtime, no LLM key):

```bash
uv run python -m federated_ueba.agent
```

```bash
uv run python -m federated_ueba.agent --station station_bravo --attack-fraction 0.15 --show-prompt
```

Options: `--station`, `--attack-fraction` (0 = quiet period),
`--model-path`, `--seed`, `--fallback-epochs`, `--show-prompt`. If no
global model artifact exists yet, a local fallback model is trained so the
entrypoint works on a fresh clone.

As a Flower AgentApp (Step 3 above), the agent waits for the global model
artifact, gathers recent events at its station, and -- when the anomaly
rate exceeds policy (10%) -- sends the incident report to the analyst LLM
for an attack assessment and response actions. Without LLM runtime
credentials (`FLWR_RUNTIME_BASE_URL` / `FLWR_RUNTIME_API_KEY`, injected by
the Flower runtime) it prints the analyst prompt instead of calling the
model.

## Measured results

From a full `flwr run` (5 rounds, identical training budget for the
local-only baselines), station-specific attack recall at a fixed ~5% FPR:

| Station | local-only | federated |
|---|---|---|
| station_alpha | 96.7% | 96.7% |
| station_bravo | 96.7% | 96.7% |
| station_charlie | 46.7% | **90.0%** |
| central_helpdesk | 46.7% | **60.0%** |
| mean cross-station ("full") attack recall | 91.4% | **96.3%** |

Federation helps most where the local signal is weakest, at no
false-positive cost, with zero raw data exchanged.

## Known caveats

- `flwr run` executes EITHER training OR the agent, depending on whether
  the `agentapp` component is registered -- see Step 3 of the run guide.
- `ray` is a direct dependency because Flower's simulation engine needs it.
- The LLM leg of the AgentApp only runs under a Flower runtime that
  injects LLM credentials; everything up to that boundary is deterministic
  and covered by the unit suite.
