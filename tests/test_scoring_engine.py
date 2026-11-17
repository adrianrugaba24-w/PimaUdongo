"""
tests/test_scoring_engine.py - Unit & Integration Tests for PimaUdongo 3-Pillar Scoring Engine
Validates agronomic thresholds, phosphorus lock-up detection, Liebig's Law penalty,
conformal uncertainty bounds (+-0.41% SOC), and dual-role epistemological translations.
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.scoring_engine import (
    ScoringEngine,
    SoilTelemetry,
    MUKONO_ISDASOIL_PRIOR_SOC,
    CONFORMAL_HALF_WIDTH_NOMINAL,
    MUKONO_LATITUDE,
    MUKONO_LONGITUDE
)


@pytest.fixture
def engine():
    return ScoringEngine()


# -----------------------------------------------------------------------------
# 1. Telemetry Validation Tests
# -----------------------------------------------------------------------------
class TestTelemetryValidation:
    def test_valid_telemetry_no_warnings(self):
        t = SoilTelemetry(moisture=24.0, temperature=25.0, ph=6.2, ec=0.25)
        assert t.validate() == []

    def test_extreme_out_of_bounds_warnings(self):
        t = SoilTelemetry(moisture=120.0, temperature=-25.0, ph=15.0, ec=15.0)
        warnings = t.validate()
        assert len(warnings) == 4
        assert any("Moisture" in w for w in warnings)
        assert any("Temperature" in w for w in warnings)
        assert any("pH" in w for w in warnings)
        assert any("EC" in w for w in warnings)


# -----------------------------------------------------------------------------
# 2. Physical Pillar Tests
# -----------------------------------------------------------------------------
class TestPhysicalPillar:
    def test_optimal_moisture_and_temp(self, engine):
        t = SoilTelemetry(moisture=26.0, temperature=24.0, ph=6.5, ec=0.25)
        score = engine.evaluate_physical_pillar(t)
        assert score.score == 100.0
        assert score.status == "Optimal"
        assert score.limiting_factor is None

    def test_critical_dry_wilting_point(self, engine):
        t = SoilTelemetry(moisture=8.0, temperature=35.0, ph=6.5, ec=0.25)
        score = engine.evaluate_physical_pillar(t)
        assert score.score < 50.0
        assert "Dry" in score.details["moisture_band"]
        assert score.limiting_factor is not None

    def test_waterlogged_anoxic_condition(self, engine):
        t = SoilTelemetry(moisture=52.0, temperature=24.0, ph=6.5, ec=0.25)
        score = engine.evaluate_physical_pillar(t)
        assert score.score < 80.0
        assert "Waterlogged" in score.details["moisture_band"]

    def test_heat_stress_temperature(self, engine):
        t = SoilTelemetry(moisture=25.0, temperature=38.0, ph=6.5, ec=0.25)
        score = engine.evaluate_physical_pillar(t)
        assert "Heat Stress" in score.details["temperature_band"]


# -----------------------------------------------------------------------------
# 3. Chemical Pillar & Phosphorus Lock-up Tests
# -----------------------------------------------------------------------------
class TestChemicalPillar:
    def test_optimal_neutral_ph_and_ec(self, engine):
        t = SoilTelemetry(moisture=25.0, temperature=25.0, ph=6.5, ec=0.30)
        score = engine.evaluate_chemical_pillar(t)
        assert score.score == 100.0
        assert score.status == "Optimal"
        assert score.details["acid_hazard"] is False
        assert score.details["lime_recommended"] is False

    def test_critical_acidic_phosphorus_lock_trigger(self, engine):
        """Crucial test: pH < 5.5 must trigger phosphorus lock-up warning and lime recommendation."""
        t = SoilTelemetry(moisture=25.0, temperature=25.0, ph=4.8, ec=0.20)
        score = engine.evaluate_chemical_pillar(t)
        assert score.score < 50.0
        assert score.status == "Critical-Acidic"
        assert score.details["acid_hazard"] is True
        assert score.details["lime_recommended"] is True
        assert "Phosphorus" in score.limiting_factor or "Acidity" in score.limiting_factor

    def test_alkaline_micronutrient_lockout(self, engine):
        t = SoilTelemetry(moisture=25.0, temperature=25.0, ph=8.4, ec=0.20)
        score = engine.evaluate_chemical_pillar(t)
        assert "Alkaline" in score.details["ph_band"]
        assert score.details["acid_hazard"] is False
        assert score.details["lime_recommended"] is False

    def test_nutrient_leached_low_ec(self, engine):
        t = SoilTelemetry(moisture=25.0, temperature=25.0, ph=6.5, ec=0.04)
        score = engine.evaluate_chemical_pillar(t)
        assert "Leached" in score.details["ec_band"]
        assert score.score < 70.0

    def test_saline_high_ec(self, engine):
        t = SoilTelemetry(moisture=25.0, temperature=25.0, ph=6.5, ec=1.80)
        score = engine.evaluate_chemical_pillar(t)
        assert "Saline" in score.details["ec_band"]


# -----------------------------------------------------------------------------
# 4. Biological Pillar & Conformal Bounds Tests
# -----------------------------------------------------------------------------
class TestBiologicalPillar:
    def test_conformal_uncertainty_interval(self, engine):
        t = SoilTelemetry(moisture=25.0, temperature=24.0, ph=6.2, ec=0.25)
        score = engine.evaluate_biological_pillar(t)
        details = score.details
        soc = details["predicted_soc_pct"]
        conf = details["conformal_interval"]
        
        assert isinstance(soc, float)
        assert soc > 0.0
        assert conf["half_width"] == CONFORMAL_HALF_WIDTH_NOMINAL
        assert conf["low"] == pytest.approx(max(0.1, round(soc - 0.41, 2)))
        assert conf["high"] == pytest.approx(round(soc + 0.41, 2))
        assert conf["low"] <= soc <= conf["high"]

    def test_depleted_soc_scoring(self, engine):
        # Extreme drought, heat, acidity leads to low predicted SOC
        t = SoilTelemetry(moisture=5.0, temperature=40.0, ph=3.9, ec=0.03)
        score = engine.evaluate_biological_pillar(t)
        assert score.details["predicted_soc_pct"] < 2.0
        assert score.score < 60.0


# -----------------------------------------------------------------------------
# 5. Composite Index & Liebig's Law Tests
# -----------------------------------------------------------------------------
class TestCompositeHolisticIndex:
    def test_balanced_soil_high_score(self, engine):
        t = SoilTelemetry(moisture=26.0, temperature=24.0, ph=6.4, ec=0.28)
        res = engine.score_telemetry(t)
        assert res["composite_index"] >= 80.0
        assert res["overall_status"] == "Optimal"

    def test_liebigs_law_penalty_on_acidic_failure(self, engine):
        """Even with ideal moisture, severe acidity (pH 4.4) must drag down composite score."""
        t = SoilTelemetry(moisture=26.0, temperature=24.0, ph=4.4, ec=0.28)
        res = engine.score_telemetry(t)
        assert res["composite_index"] < 60.0
        assert "Acidic" in res["overall_status"]
        assert res["limiting_factor"] is not None


# -----------------------------------------------------------------------------
# 6. Epistemological Dual-Role Translations
# -----------------------------------------------------------------------------
class TestDualRoleEpistemology:
    def test_farmer_view_strips_raw_numeric_units(self, engine):
        t = SoilTelemetry(moisture=25.0, temperature=24.0, ph=5.0, ec=0.15)
        res = engine.score_telemetry(t)
        farmer = res["farmer_view"]

        # Check that high-level qualitative bands and directives are present
        assert "overall_status" in farmer
        assert "immediate_actions" in farmer
        assert len(farmer["immediate_actions"]) > 0

        # Verify lime directive is given for pH 5.0
        lime_directive = any("lime" in act.lower() for act in farmer["immediate_actions"])
        assert lime_directive is True

        # Check qualitative status descriptions don't expose raw math formulas
        assert "regression" not in farmer["acidity_status"].lower()
        assert "groupkfold" not in farmer["water_status"].lower()

    def test_researcher_view_exposes_raw_and_epistemic_metadata(self, engine):
        t = SoilTelemetry(moisture=25.0, temperature=24.0, ph=5.0, ec=0.15)
        res = engine.score_telemetry(t)
        researcher = res["researcher_view"]

        # Raw telemetry present with exact channels
        raw = researcher["raw_telemetry"]
        assert raw["moisture_vwc_percent"] == 25.0
        assert raw["temperature_celsius"] == 24.0
        assert raw["soil_ph_h2o"] == 5.0
        assert raw["electrical_conductivity_ds_m"] == 0.15

        # Conformal bounds present
        soc_info = researcher["soft_sensing_soc"]
        assert "conformal_interval" in soc_info
        assert soc_info["conformal_interval"]["half_width"] == 0.41

        # Epistemic Honesty metadata verified
        epistemic = researcher["epistemic_honesty"]
        assert "Mukono" in epistemic["spatial_anchor"]["district"]
        assert "2,533 km" in epistemic["regional_extrapolation_notice"]
        assert "no physical npk" in epistemic["sensor_telemetry_disclosure"].lower()

        # Spatial CV vs Random CV metrics verified
        val = researcher["validation_performance"]
        assert val["spatial_block_cv_rmse"] > val["random_kfold_cv_rmse"]
        assert val["spatial_optimism_gap_rmse"] > 0


# -----------------------------------------------------------------------------
# 7. Parameterized Matrix Tests across Mukono Agronomic Ranges
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("m,t,ph,ec,expected_status", [
    (25.0, 24.0, 6.5, 0.25, "Optimal"),
    (8.0, 36.0, 6.2, 0.20, "Critical"),
    (24.0, 24.0, 4.5, 0.18, "Critical-Acidic"),
    (22.0, 24.0, 5.7, 0.15, "Moderate"),
    (48.0, 24.0, 6.2, 0.20, "Moderate"),
])
def test_matrix_evaluation(engine, m, t, ph, ec, expected_status):
    tel = SoilTelemetry(moisture=m, temperature=t, ph=ph, ec=ec)
    res = engine.score_telemetry(tel)
    assert expected_status in res["overall_status"]
