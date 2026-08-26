"""Tests for data generation and preprocessing."""

import numpy as np

from federated_ueba.data.generator import (
    FEATURE_NAMES,
    NUM_FEATURES,
    STATION_NAMES,
    STATION_PROFILES,
    GeneratedData,
    generate_normal_events,
)
from federated_ueba.data.preprocessing import normalize_features


class TestStationProfiles:
    def test_all_stations_have_all_features(self):
        for station, profile in STATION_PROFILES.items():
            assert set(profile.keys()) == set(FEATURE_NAMES), (
                f"{station} missing features"
            )

    def test_four_stations_defined(self):
        assert len(STATION_NAMES) == 4

    def test_all_std_positive(self):
        for station, profile in STATION_PROFILES.items():
            for feat, (mean, std) in profile.items():
                assert std > 0, f"{station}/{feat} has non-positive std"


class TestGenerateNormalEvents:
    def test_output_shape(self):
        data = generate_normal_events("station_alpha", n_samples=50, seed=0)
        assert data.features.shape == (50, NUM_FEATURES)
        assert data.labels.shape == (50,)
        assert len(data.events) == 50

    def test_labels_all_zero(self):
        data = generate_normal_events("station_bravo", n_samples=20, seed=1)
        assert (data.labels == 0).all()

    def test_features_non_negative(self):
        data = generate_normal_events("station_charlie", n_samples=200, seed=2)
        assert (data.features >= 0).all()

    def test_deterministic_with_seed(self):
        d1 = generate_normal_events("station_alpha", n_samples=30, seed=99)
        d2 = generate_normal_events("station_alpha", n_samples=30, seed=99)
        np.testing.assert_array_equal(d1.features, d2.features)

    def test_different_seeds_differ(self):
        d1 = generate_normal_events("station_alpha", n_samples=30, seed=1)
        d2 = generate_normal_events("station_alpha", n_samples=30, seed=2)
        assert not np.array_equal(d1.features, d2.features)

    def test_events_match_features(self):
        data = generate_normal_events("station_alpha", n_samples=10, seed=0)
        for i, event in enumerate(data.events):
            for j, name in enumerate(FEATURE_NAMES):
                np.testing.assert_allclose(event[name], data.features[i, j], rtol=1e-6)

    def test_station_metadata(self):
        data = generate_normal_events("central_helpdesk", n_samples=5, seed=0)
        assert data.station == "central_helpdesk"

    def test_dataset_created(self):
        data = generate_normal_events("station_alpha", n_samples=10, seed=0)
        assert len(data.dataset) == 10


class TestNormalizeFeatures:
    def test_output_shape(self, random_features):
        norm, mean, std = normalize_features(random_features)
        assert norm.shape == random_features.shape
        assert mean.shape == (NUM_FEATURES,)
        assert std.shape == (NUM_FEATURES,)

    def test_normalized_mean_near_zero(self, random_features):
        norm, _, _ = normalize_features(random_features)
        np.testing.assert_allclose(norm.mean(axis=0), 0, atol=0.1)

    def test_normalized_std_near_one(self, random_features):
        norm, _, _ = normalize_features(random_features)
        np.testing.assert_allclose(norm.std(axis=0), 1, atol=0.15)

    def test_custom_mean_std(self, random_features):
        mean = np.zeros(NUM_FEATURES)
        std = np.ones(NUM_FEATURES)
        norm, m, s = normalize_features(random_features, mean=mean, std=std)
        np.testing.assert_array_equal(m, mean)
        np.testing.assert_array_equal(s, std)
        np.testing.assert_allclose(norm, random_features, atol=1e-6)

    def test_output_dtype_float32(self, random_features):
        norm, _, _ = normalize_features(random_features)
        assert norm.dtype == np.float32
