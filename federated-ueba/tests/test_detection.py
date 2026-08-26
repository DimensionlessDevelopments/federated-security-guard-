"""Tests for the anomaly detection pipeline."""

import numpy as np
import pytest

from federated_ueba.data.generator import generate_normal_events
from federated_ueba.simulation.attack_generator import (
    generate_attack_events,
    generate_mixed_dataset,
    STATION_ATTACK_TACTICS,
    FULL_ATTACK_TACTICS,
)
from federated_ueba.data.generator import NUM_FEATURES
from federated_ueba.detection.anomaly_detector import AnomalyDetector, DetectionResult


class TestAttackGenerator:
    def test_output_shape(self):
        features, events = generate_attack_events("station_alpha", n_samples=20)
        assert features.shape == (20, NUM_FEATURES)
        assert len(events) == 20

    def test_events_marked_as_attack(self):
        _, events = generate_attack_events("station_bravo", n_samples=5)
        for event in events:
            assert event["is_attack"] is True

    def test_station_specific_type(self):
        _, events = generate_attack_events("station_alpha", attack_type="station_specific")
        assert all(e["attack_type"] == "station_specific" for e in events)

    def test_full_attack_type(self):
        _, events = generate_attack_events("station_alpha", attack_type="full")
        assert all(e["attack_type"] == "full" for e in events)

    def test_deterministic(self):
        f1, _ = generate_attack_events("station_alpha", n_samples=10, seed=42)
        f2, _ = generate_attack_events("station_alpha", n_samples=10, seed=42)
        np.testing.assert_array_equal(f1, f2)

    def test_all_stations_have_tactics(self):
        for station in STATION_ATTACK_TACTICS:
            features, _ = generate_attack_events(station, n_samples=10)
            assert features.shape == (10, NUM_FEATURES)

    def test_features_non_negative(self):
        features, _ = generate_attack_events("station_charlie", n_samples=50)
        assert (features >= 0).all()


class TestMixedDataset:
    def test_shape(self):
        features, labels = generate_mixed_dataset("station_alpha", n_normal=50, n_attack=10)
        assert features.shape == (60, NUM_FEATURES)
        assert labels.shape == (60,)

    def test_label_counts(self):
        _, labels = generate_mixed_dataset("station_bravo", n_normal=80, n_attack=20)
        assert (labels == 0).sum() == 80
        assert (labels == 1).sum() == 20

    def test_shuffled(self):
        _, labels = generate_mixed_dataset("station_alpha", n_normal=50, n_attack=50)
        # with 50/50 split and shuffling, first half shouldn't be all one class
        assert not (labels[:50] == 0).all()


class TestAnomalyDetector:
    @pytest.fixture
    def fitted_detector(self):
        detector = AnomalyDetector(epochs=5, threshold_percentile=95.0)
        normal = generate_normal_events("station_alpha", n_samples=200, seed=42)
        detector.fit(normal.features)
        return detector

    def test_fit_returns_losses(self):
        detector = AnomalyDetector(epochs=3)
        normal = generate_normal_events("station_alpha", n_samples=50, seed=0)
        result = detector.fit(normal.features)
        assert "losses" in result
        assert len(result["losses"]) == 3

    def test_losses_decrease(self):
        detector = AnomalyDetector(epochs=10)
        normal = generate_normal_events("station_alpha", n_samples=200, seed=0)
        result = detector.fit(normal.features)
        assert result["losses"][-1] < result["losses"][0]

    def test_threshold_set_after_fit(self, fitted_detector):
        assert fitted_detector.threshold > 0

    def test_score_before_fit_raises(self):
        detector = AnomalyDetector()
        features = np.random.randn(10, NUM_FEATURES).astype(np.float32)
        with pytest.raises(RuntimeError, match="not fitted"):
            detector.score(features)

    def test_score_shape(self, fitted_detector):
        features = generate_normal_events("station_alpha", n_samples=20, seed=99).features
        scores = fitted_detector.score(features)
        assert scores.shape == (20,)

    def test_detect_returns_result(self, fitted_detector):
        features = generate_normal_events("station_alpha", n_samples=20, seed=99).features
        result = fitted_detector.detect(features)
        assert isinstance(result, DetectionResult)
        assert result.predictions.shape == (20,)
        assert result.anomaly_scores.shape == (20,)
        assert result.threshold == fitted_detector.threshold

    def test_normal_events_mostly_not_anomalous(self, fitted_detector):
        normal = generate_normal_events("station_alpha", n_samples=100, seed=999)
        result = fitted_detector.detect(normal.features)
        normal_rate = (result.predictions == 0).mean()
        assert normal_rate > 0.8

    def test_attack_events_score_higher(self, fitted_detector):
        normal = generate_normal_events("station_alpha", n_samples=100, seed=50)
        attack, _ = generate_attack_events("station_alpha", n_samples=100, seed=50)
        normal_scores = fitted_detector.score(normal.features)
        attack_scores = fitted_detector.score(attack)
        assert attack_scores.mean() > normal_scores.mean()

    def test_parameter_roundtrip(self, fitted_detector):
        params = fitted_detector.get_parameters()
        new_detector = AnomalyDetector()
        new_detector.set_parameters(params)
        new_params = new_detector.get_parameters()
        for p1, p2 in zip(params, new_params):
            np.testing.assert_array_equal(p1, p2)
