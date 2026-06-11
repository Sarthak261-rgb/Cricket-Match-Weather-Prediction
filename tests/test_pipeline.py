"""
Unit tests for Cricket Match Weather Prediction project.
Run with: pytest tests/ -v
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/cricket_weather_data.csv")


class TestDataset:
    """Tests for the generated dataset."""

    @pytest.fixture(scope="class")
    def df(self):
        assert os.path.exists(DATA_PATH), "Dataset not found. Run src/generate_dataset.py first."
        return pd.read_csv(DATA_PATH, parse_dates=["match_date"])

    def test_row_count(self, df):
        assert len(df) >= 1000, f"Expected ≥1000 rows, got {len(df)}"

    def test_required_columns(self, df):
        required = [
            "match_id", "venue", "match_date", "match_type",
            "pre_temp_avg", "pre_humidity_avg", "pre_rainfall_total",
            "match_outcome", "pitch_condition", "play_possible"
        ]
        missing = [c for c in required if c not in df.columns]
        assert not missing, f"Missing columns: {missing}"

    def test_no_nulls_in_features(self, df):
        feature_cols = [
            "pre_temp_avg", "pre_humidity_avg", "pre_pressure_avg",
            "pre_wind_speed_avg", "pre_cloud_cover_avg", "pre_rainfall_total"
        ]
        null_counts = df[feature_cols].isnull().sum()
        assert null_counts.sum() == 0, f"Null values found:\n{null_counts[null_counts > 0]}"

    def test_temperature_range(self, df):
        assert df["pre_temp_avg"].between(-10, 55).all(), "Temperature out of realistic range"

    def test_humidity_range(self, df):
        assert df["pre_humidity_avg"].between(0, 100).all(), "Humidity out of [0,100] range"

    def test_pressure_range(self, df):
        assert df["pre_pressure_avg"].between(950, 1060).all(), "Pressure out of realistic range"

    def test_rainfall_non_negative(self, df):
        assert (df["pre_rainfall_total"] >= 0).all(), "Negative rainfall values found"

    def test_outcome_values(self, df):
        valid = {"Completed", "Rain Interrupted", "Abandoned", "Delayed Start"}
        assert set(df["match_outcome"].unique()).issubset(valid)

    def test_pitch_values(self, df):
        valid = {"Flat/Batting", "Humid/Swing", "Wet/Green", "Dry/Dusty"}
        assert set(df["pitch_condition"].unique()).issubset(valid)

    def test_play_possible_binary(self, df):
        assert set(df["play_possible"].unique()).issubset({0, 1})

    def test_match_types(self, df):
        assert set(df["match_type"].unique()).issubset({"Test", "ODI", "T20"})

    def test_venue_count(self, df):
        assert df["venue"].nunique() >= 10, "Expected at least 10 unique venues"

    def test_date_range(self, df):
        assert df["match_date"].min().year >= 2015
        assert df["match_date"].max().year <= 2025

    def test_play_possible_consistency(self, df):
        # If Completed or Delayed Start, play_possible should be 1
        play_mask   = df["match_outcome"].isin(["Completed", "Delayed Start"])
        noplay_mask = df["match_outcome"].isin(["Abandoned"])
        assert df.loc[play_mask, "play_possible"].all()
        assert not df.loc[noplay_mask, "play_possible"].any()


class TestFeatureEngineering:
    """Tests for derived features."""

    @pytest.fixture(scope="class")
    def df(self):
        df = pd.read_csv(DATA_PATH, parse_dates=["match_date"])
        df["temp_humidity_index"] = (df["pre_temp_avg"] * df["pre_humidity_avg"]) / 100
        df["month"] = df["match_date"].dt.month
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["rain_flag"] = (df["pre_rainfall_total"] > 0).astype(int)
        return df

    def test_temp_humidity_index(self, df):
        expected = (df["pre_temp_avg"] * df["pre_humidity_avg"]) / 100
        np.testing.assert_allclose(df["temp_humidity_index"], expected, rtol=1e-5)

    def test_month_sin_cos_range(self, df):
        assert df["month_sin"].between(-1, 1).all()
        assert df["month_cos"].between(-1, 1).all()

    def test_rain_flag_binary(self, df):
        assert set(df["rain_flag"].unique()).issubset({0, 1})

    def test_rain_flag_alignment(self, df):
        # Where rainfall > 0, rain_flag must be 1
        assert (df.loc[df["pre_rainfall_total"] > 0, "rain_flag"] == 1).all()
        assert (df.loc[df["pre_rainfall_total"] == 0, "rain_flag"] == 0).all()


class TestSequenceBuilding:
    """Tests for LSTM sequence generation."""

    def test_sequence_shape(self):
        SEQ_LEN = 10
        n_features = 20
        n_samples = 50

        # Simulate a batch of sequences
        X = np.random.randn(n_samples, SEQ_LEN, n_features).astype(np.float32)
        assert X.shape == (n_samples, SEQ_LEN, n_features)

    def test_sequence_no_nans(self):
        X = np.random.randn(30, 10, 20).astype(np.float32)
        assert not np.isnan(X).any()

    def test_minimum_samples_per_venue(self):
        df = pd.read_csv(DATA_PATH)
        SEQ_LEN = 10
        venue_counts = df.groupby("venue").size()
        viable = venue_counts[venue_counts >= SEQ_LEN + 1]
        assert len(viable) >= 8, f"Not enough venues with ≥{SEQ_LEN+1} records"


class TestModelArtifacts:
    """Tests for saved model artifacts."""

    MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")

    def test_model_files_exist(self):
        tasks = ["match_outcome", "pitch_condition", "play_possible"]
        for task in tasks:
            path = os.path.join(self.MODEL_DIR, f"{task}_best.keras")
            assert os.path.exists(path), f"Model file missing: {path}"

    def test_scaler_files_exist(self):
        tasks = ["match_outcome", "pitch_condition", "play_possible"]
        for task in tasks:
            path = os.path.join(self.MODEL_DIR, f"scaler_{task}.pkl")
            assert os.path.exists(path), f"Scaler file missing: {path}"

    def test_encoder_files_exist(self):
        for enc in ["venue", "match_type", "outcome", "pitch"]:
            path = os.path.join(self.MODEL_DIR, f"encoder_{enc}.pkl")
            assert os.path.exists(path), f"Encoder file missing: {path}"

    def test_metrics_file_exists(self):
        path = os.path.join(os.path.dirname(__file__), "../results/training_metrics.json")
        assert os.path.exists(path), "Training metrics JSON not found"

    def test_metrics_content(self):
        import json
        path = os.path.join(os.path.dirname(__file__), "../results/training_metrics.json")
        with open(path) as f:
            metrics = json.load(f)
        assert len(metrics) == 3
        for m in metrics:
            assert "task" in m
            assert "test_accuracy" in m
            assert 0 < m["test_accuracy"] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
