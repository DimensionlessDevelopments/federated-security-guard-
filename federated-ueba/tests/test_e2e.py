"""End-to-end test of the federated simulation.

Launches the real `flwr run . local-sim` pipeline (server + 4 station
clients + FedAvg rounds), then statistically validates the produced global
model against local-only baselines.

Excluded from the default test run (it takes minutes and needs the flwr
CLI + local SuperLink). Entrypoint:

    pytest -m e2e
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from federated_ueba.agent.incident import load_global_model
from federated_ueba.data.generator import NUM_FEATURES
from federated_ueba.models import SecurityAutoencoder
from federated_ueba.training.evaluate import compare_modes, mean_metric

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        shutil.which("flwr") is None, reason="flwr CLI not on PATH"
    ),
]

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "artifacts" / "global_model.pt"
NUM_ROUNDS = 5
NUM_NODES = 5  # 4 ML stations + 1 non-ML Station B node
RUN_TIMEOUT_S = 600


@pytest.fixture(scope="module")
def simulation_output() -> str:
    """Run the full federated simulation once; return its combined output."""
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()  # the run must recreate it

    # No PYTHONPATH shim: the app uses the root-package layout, so the
    # modules import from the app directory itself -- the same way a FAB
    # installed from Flower Hub is loaded on a SuperNode.
    result = subprocess.run(
        ["flwr", "run", ".", "local-sim", "--stream"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_S,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"flwr run failed:\n{output[-3000:]}"
    return output


class TestSimulationRun:
    def test_all_rounds_complete(self, simulation_output):
        for r in range(1, NUM_ROUNDS + 1):
            assert f"[ROUND {r}/{NUM_ROUNDS}]" in simulation_output

    def test_no_client_failures(self, simulation_output):
        received = re.findall(
            r"Received (\d+) results and (\d+) failures", simulation_output
        )
        # one train + one evaluate aggregation per round
        assert len(received) == 2 * NUM_ROUNDS
        for results, failures in received:
            assert int(results) == NUM_NODES
            assert int(failures) == 0

    def test_evaluate_metrics_reported(self, simulation_output):
        assert "eval_loss" in simulation_output
        assert "recall" in simulation_output

    def test_training_completes_and_saves_model(self, simulation_output):
        assert "Federated training complete" in simulation_output
        assert MODEL_PATH.exists()


class TestGlobalModelStatistics:
    """The federated model must hold up statistically, per the spec metrics."""

    @pytest.fixture(scope="class")
    def results(self, simulation_output):
        global_model = load_global_model(MODEL_PATH)
        return compare_modes(global_model, local_epochs_budget=10)

    def test_model_loads_with_expected_architecture(self, simulation_output):
        model = load_global_model(MODEL_PATH)
        expected = sum(
            p.numel()
            for p in SecurityAutoencoder(input_dim=NUM_FEATURES).parameters()
        )
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params == expected

    def test_mean_recall_meets_floor(self, results):
        assert mean_metric(results, "federated", "recall") >= 0.65

    def test_fpr_within_policy(self, results):
        assert mean_metric(results, "federated", "fpr") <= 0.10

    def test_cross_station_generalisation(self, results):
        assert mean_metric(results, "federated", "full_attack_recall") >= 0.85

    def test_federated_not_worse_than_local(self, results):
        """The core claim: federation matches or beats local-only training."""
        fed = mean_metric(results, "federated", "recall")
        local = mean_metric(results, "local", "recall")
        assert fed >= local - 0.05, (
            f"federated mean recall {fed:.3f} below local {local:.3f}"
        )
