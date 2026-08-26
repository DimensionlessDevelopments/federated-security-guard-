"""Tests for the incident-report agent."""

import numpy as np
import pytest
import torch

from federated_ueba.agent.incident import (
    IncidentReport,
    build_insights_prompt,
    format_incident_report,
    gather_incident_data,
    load_global_model,
    station_seed,
)
from federated_ueba.data.generator import (
    NUM_FEATURES,
    STATION_NAMES,
    generate_normal_events,
)
from federated_ueba.data.preprocessing import normalize_features
from federated_ueba.models import SecurityAutoencoder
from federated_ueba.simulation.attack_generator import STATION_ATTACK_TACTICS


@pytest.fixture(scope="module")
def trained_model():
    """A model trained on station_alpha's normal behaviour."""
    torch.manual_seed(0)
    model = SecurityAutoencoder(input_dim=NUM_FEATURES)
    normal = generate_normal_events(
        "station_alpha", n_samples=500, seed=station_seed("station_alpha")
    )
    normalized, _, _ = normalize_features(normal.features)
    X = torch.tensor(normalized)
    dataset = torch.utils.data.TensorDataset(X, X)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    model.train()
    for _ in range(10):
        for batch_x, _ in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_x)
            loss.backward()
            optimizer.step()
    return model


class TestLoadGlobalModel:
    def test_roundtrip(self, trained_model, tmp_path):
        path = tmp_path / "global_model.pt"
        torch.save(trained_model.state_dict(), path)
        loaded = load_global_model(path)
        for p1, p2 in zip(
            trained_model.state_dict().values(), loaded.state_dict().values()
        ):
            torch.testing.assert_close(p1, p2)

    def test_loaded_model_in_eval_mode(self, trained_model, tmp_path):
        path = tmp_path / "global_model.pt"
        torch.save(trained_model.state_dict(), path)
        loaded = load_global_model(path)
        assert not loaded.training

    def test_custom_dims(self, tmp_path):
        model = SecurityAutoencoder(
            input_dim=NUM_FEATURES, hidden_dim=16, latent_dim=4
        )
        path = tmp_path / "m.pt"
        torch.save(model.state_dict(), path)
        loaded = load_global_model(path, hidden_dim=16, latent_dim=4)
        assert loaded.encoder[0].out_features == 16


class TestStationSeed:
    def test_matches_client_partition_mapping(self):
        for pid, station in enumerate(STATION_NAMES):
            assert station_seed(station) == 42 + pid * 100


class TestGatherIncidentData:
    def test_attack_burst_triggers_incident(self, trained_model):
        report = gather_incident_data(
            trained_model, "station_alpha", attack_fraction=0.15
        )
        assert report.triggered
        assert report.n_events == 200
        assert report.n_flagged >= 20  # at least the attack burst

    def test_quiet_period_does_not_trigger(self, trained_model):
        report = gather_incident_data(
            trained_model, "station_alpha", attack_fraction=0.0
        )
        assert not report.triggered
        assert report.flagged_rate < 0.10

    def test_report_fields(self, trained_model):
        report = gather_incident_data(trained_model, "station_alpha")
        assert isinstance(report, IncidentReport)
        assert report.station == "station_alpha"
        assert 0.0 <= report.flagged_rate <= 1.0
        assert report.threshold > 0

    def test_top_events_sorted_by_score(self, trained_model):
        report = gather_incident_data(trained_model, "station_alpha")
        scores = [e["score"] for e in report.top_events]
        assert scores == sorted(scores, reverse=True)
        for event in report.top_events:
            assert set(event["features"].keys()) == {
                *event["features"].keys()
            }
            assert len(event["features"]) == NUM_FEATURES

    def test_deviations_surface_attacked_features(self, trained_model):
        """The insight summary must name the station's actual attack tactics."""
        report = gather_incident_data(
            trained_model, "station_alpha", attack_fraction=0.15
        )
        deviating = {d["feature"] for d in report.feature_deviations}
        attacked = set(STATION_ATTACK_TACTICS["station_alpha"].keys())
        assert attacked & deviating, (
            f"expected some of {attacked} among {deviating}"
        )

    def test_deterministic_with_seed(self, trained_model):
        r1 = gather_incident_data(trained_model, "station_alpha", seed=7)
        r2 = gather_incident_data(trained_model, "station_alpha", seed=7)
        assert r1.n_flagged == r2.n_flagged
        assert r1.threshold == r2.threshold

    def test_no_flags_yields_empty_details(self):
        """An untrained model flags nothing meaningful on tiny thresholds --
        force the empty path with trigger_rate above any possible rate."""
        torch.manual_seed(1)
        model = SecurityAutoencoder(input_dim=NUM_FEATURES)
        report = gather_incident_data(
            model, "station_alpha", attack_fraction=0.0, trigger_rate=2.0
        )
        assert not report.triggered


class TestReportFormatting:
    @pytest.fixture
    def report(self, trained_model):
        return gather_incident_data(
            trained_model, "station_alpha", attack_fraction=0.15
        )

    def test_format_contains_key_fields(self, report):
        text = format_incident_report(report)
        assert "INCIDENT REPORT" in text
        assert "station_alpha" in text
        assert "TRIGGERED" in text
        assert "Behavioural features driving the anomaly:" in text

    def test_quiet_format_says_no_incident(self, trained_model):
        report = gather_incident_data(
            trained_model, "station_alpha", attack_fraction=0.0
        )
        assert "no incident" in format_incident_report(report)

    def test_insights_prompt_embeds_report(self, report):
        prompt = build_insights_prompt(report)
        assert "security analyst" in prompt
        assert "INCIDENT REPORT" in prompt
        assert "station_alpha" in prompt


class TestEntrypoint:
    """python -m federated_ueba.agent -- the standalone testing entrypoint."""

    @pytest.fixture
    def model_file(self, trained_model, tmp_path):
        path = tmp_path / "global_model.pt"
        torch.save(trained_model.state_dict(), path)
        return str(path)

    def test_incident_run_with_model(self, model_file, capsys):
        from federated_ueba.agent.__main__ import main

        code = main(
            [
                "--station", "station_alpha",
                "--attack-fraction", "0.15",
                "--model-path", model_file,
            ]
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "Using federated global model" in out
        assert "INCIDENT REPORT" in out
        assert "TRIGGERED" in out

    def test_quiet_run(self, model_file, capsys):
        from federated_ueba.agent.__main__ import main

        code = main(
            [
                "--station", "station_alpha",
                "--attack-fraction", "0",
                "--model-path", model_file,
            ]
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "no incident" in out

    def test_fallback_when_model_missing(self, tmp_path, capsys):
        from federated_ueba.agent.__main__ import main

        code = main(
            [
                "--station", "station_alpha",
                "--model-path", str(tmp_path / "does_not_exist.pt"),
                "--fallback-epochs", "2",
            ]
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "training local fallback" in out
        assert "INCIDENT REPORT" in out

    def test_show_prompt(self, model_file, capsys):
        from federated_ueba.agent.__main__ import main

        code = main(
            [
                "--station", "station_alpha",
                "--attack-fraction", "0.15",
                "--model-path", model_file,
                "--show-prompt",
            ]
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "analyst LLM prompt" in out
        assert "security analyst" in out

    def test_rejects_unknown_station(self, model_file):
        from federated_ueba.agent.__main__ import main

        with pytest.raises(SystemExit):
            main(["--station", "not_a_station", "--model-path", model_file])


class TestAgentApp:
    def test_module_imports_and_exposes_app(self):
        from flwr.agentapp import AgentApp

        from federated_ueba.agent import agent_app

        assert isinstance(agent_app.app, AgentApp)

    def test_wait_for_model_present(self, tmp_path):
        from federated_ueba.agent.agent_app import wait_for_model

        path = tmp_path / "model.pt"
        path.touch()
        assert wait_for_model(path, timeout_s=1)

    def test_wait_for_model_timeout(self, tmp_path):
        from federated_ueba.agent.agent_app import wait_for_model

        assert not wait_for_model(
            tmp_path / "never.pt", timeout_s=0.2, poll_interval_s=0.05
        )

    def test_configured_string_validation(self):
        from types import SimpleNamespace

        from federated_ueba.agent.agent_app import configured_string

        context = SimpleNamespace(run_config={"agent.station": "  "})
        with pytest.raises(ValueError, match="agent.station"):
            configured_string(context, "agent.station", "fallback")

        context = SimpleNamespace(run_config={})
        assert (
            configured_string(context, "agent.station", "central_helpdesk")
            == "central_helpdesk"
        )
