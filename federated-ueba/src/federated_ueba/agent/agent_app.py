"""Flower AgentApp: incident reporting with LLM insights.

Connects to the trained global model, gathers recent events from the local
environment, and -- when the anomaly rate exceeds policy -- triggers an
incident report and asks the analyst LLM for insights and response actions.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from flwr.agentapp import AgentApp, AgentSession
from flwr.app import Context
from openai import OpenAI

from federated_ueba.agent.incident import (
    build_insights_prompt,
    format_incident_report,
    gather_incident_data,
    load_global_model,
)

DEFAULT_STATION = "central_helpdesk"
DEFAULT_MODEL_PATH = "artifacts/global_model.pt"
# The agent starts alongside federated training; wait for the server to
# finish its rounds and save the aggregated global model.
MODEL_WAIT_TIMEOUT_S = 300.0
MODEL_POLL_INTERVAL_S = 2.0

app = AgentApp()


def configured_string(context: Context, name: str, default: str) -> str:
    value = context.run_config.get(name, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def wait_for_model(
    path: str | Path,
    timeout_s: float = MODEL_WAIT_TIMEOUT_S,
    poll_interval_s: float = MODEL_POLL_INTERVAL_S,
) -> bool:
    """Wait until the global model artifact exists. False on timeout."""
    deadline = time.monotonic() + timeout_s
    path = Path(path)
    while not path.exists():
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll_interval_s)
    return True


@app.main()
def main(agent: AgentSession, context: Context) -> None:
    """Gather local data, score it with the global model, report insights."""
    llm_model = configured_string(context, "agent.model", "openai/gpt-5.6-sol")
    station = configured_string(context, "agent.station", DEFAULT_STATION)
    model_path = configured_string(
        context, "global-model-path", DEFAULT_MODEL_PATH
    )
    hidden_dim = int(context.run_config.get("hidden-dim", 32))
    latent_dim = int(context.run_config.get("latent-dim", 8))

    # Step 1: connect to the federated global model (produced by the
    # ServerApp at the end of training -- wait for it, don't crash the run)
    print(f"Agent: waiting for global model at {model_path} ...")
    if not wait_for_model(model_path):
        print(
            f"Agent: no global model appeared at {model_path} within "
            f"{MODEL_WAIT_TIMEOUT_S:.0f}s -- skipping incident analysis."
        )
        return
    model = load_global_model(
        model_path, hidden_dim=hidden_dim, latent_dim=latent_dim
    )

    # Step 2: gather recent events from the local environment and score them
    report = gather_incident_data(model, station)
    print(format_incident_report(report))

    if not report.triggered:
        print("Anomaly rate within policy -- no incident report raised.")
        return

    # Step 3: incident triggered -- ask the analyst LLM for insights
    base_url = os.environ.get("FLWR_RUNTIME_BASE_URL")
    api_key = os.environ.get("FLWR_RUNTIME_API_KEY")
    if not base_url or not api_key:
        print(
            "Agent: LLM runtime credentials not available -- "
            "printing the analyst prompt instead of calling the model."
        )
        print(build_insights_prompt(report))
        return

    client = OpenAI(base_url=base_url, api_key=api_key)

    stream = client.responses.create(
        model=llm_model,
        input=[
            {
                "type": "message",
                "role": "user",
                "content": build_insights_prompt(report),
            }
        ],
        instructions=(
            "Respond with a concise incident assessment: likely attack "
            "scenario, supporting behavioural evidence, and recommended "
            "immediate actions."
        ),
        stream=True,
    )

    output_text: list[str] = []
    for event in stream:
        agent.events.emit(event.to_dict())
        if event.type in {"error", "response.failed"}:
            raise RuntimeError(f"Model response failed: {event}")
        if event.type == "response.output_text.delta":
            output_text.append(event.delta)

    print("".join(output_text))
