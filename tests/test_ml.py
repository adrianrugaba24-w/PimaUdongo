"""
tests/test_ml.py - Unit & Pipeline Tests for ml/train_soc.py
Validates Spatial Block CV, Random CV, haversine distance to Mukono,
conformal uncertainty interval calculation, and epistemic honesty metadata.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.train_soc import (
    haversine_distance,
    compute_distances_to_mukono,
    generate_synthetic_african_soil_data,
    run_training_pipeline,
    MUKONO_LAT,
    MUKONO_LON,
    DEFAULT_CONFORMAL_HALF_WIDTH
)


class TestSpatialMath:
    def test_haversine_zero_distance_at_same_point(self):
        dist = haversine_distance(MUKONO_LAT, MUKONO_LON, MUKONO_LAT, MUKONO_LON)
        assert dist == pytest.approx(0.0, abs=1e-3)

    def test_haversine_known_distance(self):
        # Distance between Mukono, Uganda (0.3547 N, 32.7475 E) and Nairobi, Kenya (~-1.286389, 36.817223)
        # Known straight-line distance is ~500 km
        dist = haversine_distance(MUKONO_LAT, MUKONO_LON, -1.286389, 36.817223)
        assert 450 < dist < 550

    def test_compute_distances_to_mukono(self):
        lats = np.array([0.3547, 6.5244])    # Mukono, Lagos Nigeria
        lons = np.array([32.7475, 3.3792])
        res = compute_distances_to_mukono(lats, lons)
        assert res["min_km"] == pytest.approx(0.0, abs=0.1)
        # Lagos is ~3,300 km from Mukono
        assert res["max_km"] > 3000


class TestSyntheticDataGenerator:
    def test_generator_structure(self):
        df = generate_synthetic_african_soil_data(n_samples=50, seed=123)
        assert len(df) == 50
        expected_cols = {"SSN", "Latitude", "Longitude", "Country", "pH_H2O", "EC", "Moisture", "Temperature", "OC"}
        assert expected_cols.issubset(set(df.columns))

    def test_data_ranges_realistic(self):
        df = generate_synthetic_african_soil_data(n_samples=60, seed=456)
        assert (df["pH_H2O"] >= 3.5).all() and (df["pH_H2O"] <= 9.5).all()
        assert (df["EC"] >= 0.01).all()
        assert (df["OC"] > 0).all()


class TestTrainingPipelineExecution:
    def test_training_pipeline_smoke(self, tmp_path):
        df = generate_synthetic_african_soil_data(n_samples=80, seed=42)
        out = run_training_pipeline(
            df=df,
            blocks=3,
            repeats=2,
            export_dir=str(tmp_path)
        )
        meta = out["metadata"]
        assert meta["project"] == "PimaUdongo"
        assert "spatial_distance_to_mukono" in meta
        assert "conformal_interval" in meta
        assert meta["conformal_interval"]["nominal_reported_half_width"] == DEFAULT_CONFORMAL_HALF_WIDTH
        assert "epistemic_honesty_statement" in meta
        assert "regional continental prior extrapolation" in meta["epistemic_honesty_statement"].lower()

        # Check export files were created
        assert (tmp_path / "model_ridge.joblib").exists()
        assert (tmp_path / "model_rf.joblib").exists()
        assert (tmp_path / "soc_metadata.json").exists()
        assert (tmp_path / "cv_summary.csv").exists()
