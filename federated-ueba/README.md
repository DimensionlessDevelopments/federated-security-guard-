# federated-ueba

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
src/federated_ueba/
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
  simulation/event_stream.py   synthetic per-tick event stream (serving input)
  serving/loop.py              score the stream with the global model
  serving/store.py             SQLite detection store (feed for the dashboard)
apps/api/main.py               FastAPI backend + static dashboard
apps/api/static/index.html     single-page dashboard (polls the API)
tests/                         unit suite + e2e simulation suite
```

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
- `uv run` matters here: it puts the project package on the path for
  Flower's tooling (the repo uses a `src/` layout). Outside uv, prefix
  with `PYTHONPATH=src` instead.

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

## Serving loop and dashboard

A separate **serving plane** turns the trained model into a live detector: it
scores an event stream and writes detections to SQLite that a frontend (and,
later, the correlation agent) reads. Decoupled from the Flower training plane
-- it imports nothing from `agent/`, and `server.py`/`client.py` are untouched.

### Step 4 -- Serving loop

```bash
# against the trained global model from Step 1
uv run python scripts/run_serving.py --ticks 10 --attack-at 5 --attacked-stations station_alpha

# demo without a trained model (per-station thresholds still calibrate)
uv run python scripts/run_serving.py --untrained-demo --attack-at 5 --attacked-stations station_alpha
```

Options: `--model-path` (default `artifacts/global_model.pt`), `--db-path`
(default `artifacts/serving.db`), `--ticks`, `--events-per-tick`,
`--attack-at` (first tick to inject attacks), `--attacked-stations`,
`--interval` (seconds between ticks for live pacing), `--untrained-demo`.
Detections land in `artifacts/serving.db` (table `detections`).

### Step 5 -- Dashboard

```bash
uv run uvicorn apps.api.main:app --reload
# open http://127.0.0.1:8000
```

The FastAPI backend serves read-only JSON from the detection store
(`/api/summary`, `/api/stations`, `/api/alerts`) plus the single-page
dashboard, which polls every 5s. Point it at a different store with the
`SERVING_DB` env var. `fastapi`/`uvicorn` come in via `uv sync` (transitive
through `flwr`), so no extra install is needed.

The **Report Incident** button runs the incident agent (`agent/incident.py`)
via `GET /api/incident` and shows the report in-page: by default it analyses
the station with the highest live flag rate (the one under attack), naming the
behavioural features that drove the anomaly and the analyst-LLM prompt. It uses
the federated global model if present, otherwise a local fallback.

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

- Run `flwr run` through `uv run` (or with `PYTHONPATH=src`) until the
  src-layout vs root-package question is settled.
- `flwr run` executes EITHER training OR the agent, depending on whether
  the `agentapp` component is registered -- see Step 3 of the run guide.
- `ray` is a direct dependency because Flower's simulation engine needs it.
- The LLM leg of the AgentApp only runs under a Flower runtime that
  injects LLM credentials; everything up to that boundary is deterministic
  and covered by the unit suite.
