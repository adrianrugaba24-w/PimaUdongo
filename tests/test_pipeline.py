"""
tests/test_pipeline.py  –  unit tests for data validity and validation logic.

Synthetic stand-in only.  These tests check that the machinery does what the
proposal says it does: an audited cleaning pass, folds that respect sites,
metrics on the original target scale, and no carbon-derived feature.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import afsis_data as A
import afsis_experiment as E
import afsis_validation as V
import synthetic_afsis as S
from afsis_data import find


# ---------------------------------------------------------------------------
# 1.  Loading and merge
# ---------------------------------------------------------------------------

class TestLoad:
    def test_load_returns_dataframe_and_manifest(self, raw_frame):
        df, manifest = raw_frame
        assert isinstance(df, pd.DataFrame)
        assert manifest and all(len(m["sha256"]) == 64 for m in manifest)

    def test_merge_has_both_labs(self, raw_frame):
        df, _ = raw_frame
        assert set(df["lab"].unique()) == set(S.LABS)

    def test_expected_columns_present(self, raw_frame):
        df, _ = raw_frame
        for col in ("SSN", "Latitude", "Longitude", "Country", "Profile_ID",
                    "Depth_From_cm", "Depth_To_cm", "SOC", "pH_H2O", "EC"):
            assert col in df.columns, f"Missing column: {col}"

    def test_coordinates_only_come_from_georeferences(self, raw_frame):
        """A merged table must not grow Latitude_x / Latitude_y columns."""
        df, _ = raw_frame
        assert not [c for c in df.columns if c.endswith(("_x", "_y"))]

    def test_load_raw_does_not_clean(self, raw_frame, cleaned):
        """Raw keeps depth increments and duplicates; cleaning is a separate step."""
        df, _ = raw_frame
        assert len(df) > len(cleaned[0])
        assert df["SSN"].duplicated().any()


# ---------------------------------------------------------------------------
# 2.  Column detection
# ---------------------------------------------------------------------------

class TestSchema:
    def test_find_helper(self):
        cols = ["SSN", "pH_H2O", "EC", "Clay percent", "SOC"]
        assert find(cols, r"(^|[^a-z])ph([^a-z]|$)") == "pH_H2O"
        assert find(cols, r"^ec$") == "EC"
        assert find(["SSN", "SOC"], r"^ec$") is None

    def test_target_and_features_detected(self, schema):
        assert schema.target == "SOC"
        assert set(schema.features) == {"pH_H2O", "EC", "Clay percent",
                                        "Sand percent", "Silt percent"}

    def test_no_carbon_derived_feature(self, schema):
        """Total C / Inorganic C would be target-derived leakage."""
        assert not {"Total C", "Inorganic C", "SOC"} & set(schema.features)

    def test_carbon_feature_is_blocked_even_when_requested(self, raw_frame):
        """An override cannot smuggle a target-derived column into the model."""
        df, _ = raw_frame
        s = A.detect_schema(df, overrides={"features": ["pH_H2O", "Total C", "SOC"]})
        assert s.features == ["pH_H2O"]
        assert set(s.unmatched) == {"Total C", "SOC"}

    def test_group_key_prefers_profile(self, schema):
        assert schema.group_key() == "Profile_ID"

    def test_missing_override_column_raises(self, raw_frame):
        df, _ = raw_frame
        with pytest.raises(ValueError, match="not in the data"):
            A.detect_schema(df, overrides={"target": "Not_A_Column"})


# ---------------------------------------------------------------------------
# 3.  Units
# ---------------------------------------------------------------------------

class TestUnits:
    def test_percent_when_small(self):
        u, how = A.infer_target_units(pd.Series([1.2, 2.0, 3.4]))
        assert u.startswith("percent") and "median" in how

    def test_grams_per_kilogram_when_large(self):
        u, _ = A.infer_target_units(pd.Series([25.0, 31.0, 40.0]))
        assert u == "g/kg"

    def test_units_recorded_in_results(self, meta):
        assert meta["target"]["units"] == meta["data"]["units"]
        assert meta["data"]["units_basis"]


# ---------------------------------------------------------------------------
# 4.  Cleaning ledger: a count and a reason for every removal
# ---------------------------------------------------------------------------

class TestCleaningLedger:
    def test_every_step_has_a_reason_and_counts(self, cleaned):
        _, log = cleaned
        for entry in log:
            assert {"step", "n_before", "n_removed", "n_after", "reason"} <= set(entry)
            assert str(entry["reason"]).strip(), f"{entry['step']} has no reason"

    def test_row_counts_are_arithmetic(self, cleaned):
        """Removals must add up; a silent drop cannot hide inside the ledger."""
        d, log = cleaned
        row_steps = [e for e in log
                     if e["step"] not in ("impossible_texture_sum",)
                     and not e["step"].startswith("impossible_value:")]
        for e in row_steps:
            assert e["n_before"] - e["n_removed"] == e["n_after"], e
        assert row_steps[0]["n_before"] > len(d)
        assert row_steps[-1]["n_after"] == len(d)

    def test_required_removal_reasons_are_logged(self, cleaned):
        _, log = cleaned
        steps = {e["step"] for e in log}
        for required in ("exact_duplicate_rows",       # duplicates
                         "missing_coordinates", "impossible_coordinates",
                         "missing_identifier",          # missing / broken coords
                         "depth_selection", "multi_lab_replicates",
                         "missing_target", "impossible_target"):
            assert required in steps, f"ledger never reports {required}"

    def test_injected_duplicates_are_removed(self, raw_frame, cleaned):
        """The stand-in injects a known count, so the ledger can be checked."""
        _, log = cleaned
        entry = next(e for e in log if e["step"] == "exact_duplicate_rows")
        assert entry["n_removed"] == S.N_DUPLICATE_ROWS

    def test_impossible_values_become_nan_not_silent_rows(self, pipeline_inputs):
        d, schema = pipeline_inputs["d"], pipeline_inputs["schema"]
        assert (d[schema.target] > 0).all()
        assert (d[schema.target] <= 100).all()
        assert d["pH_H2O"].dropna().ge(0).all()
        assert d["Clay percent"].dropna().le(100).all()
        ok = d["Clay percent"].notna() & d["Sand percent"].notna()
        assert (d.loc[ok, "Clay percent"] + d.loc[ok, "Sand percent"]).le(100.5).all()

    def test_surface_depth_only(self, pipeline_inputs):
        d, schema = pipeline_inputs["d"], pipeline_inputs["schema"]
        assert d[schema.depth_to].max() <= A.DEPTH_TARGET_CM + 0.5
        assert d[schema.depth_from].min() == 0

    def test_one_row_per_site_after_cleaning(self, pipeline_inputs):
        d, schema = pipeline_inputs["d"], pipeline_inputs["schema"]
        assert d[schema.group_key()].is_unique


# ---------------------------------------------------------------------------
# 5.  Uganda subset reported separately from the African set
# ---------------------------------------------------------------------------

class TestDataCard:
    def test_card_counts_and_subset(self, pipeline_inputs, cleaned):
        d, log = pipeline_inputs["d"], cleaned[1]
        card = A.data_card(d, pipeline_inputs["schema"], log, "Uganda")
        assert card["n_rows"] == len(d)
        assert card["countries"]["Uganda"] == card["focus_subset"]["n_rows"]
        assert 0 < card["focus_subset"]["share_of_rows"] < 1.0
        assert card["focus_subset"]["n_sites"] == card["focus_subset"]["n_rows"]

    def test_absent_country_is_reported_as_absent_not_zero_division(self, pipeline_inputs,
                                                                   cleaned):
        card = A.data_card(pipeline_inputs["d"], pipeline_inputs["schema"], cleaned[1],
                           "Atlantis")
        assert card["focus_subset"]["n_rows"] == 0
        assert "not present" in card["focus_subset"]["note"]

    def test_provenance_written_to_results(self, meta, raw_frame):
        _, manifest = raw_frame
        assert meta["data"]["files"] == manifest
        assert meta["data"]["dataset_version"].startswith("afsis-")
        assert meta["package_versions"]["scikit-learn"]

    def test_dataset_version_is_deterministic(self, synthetic_afsis_root, raw_frame):
        _, m1 = raw_frame
        # Same files -> same version, however many times the manifest is rebuilt.
        assert A.dataset_version(m1) == A.dataset_version(A.file_manifest(synthetic_afsis_root))
        touched = [dict(m) for m in m1]
        touched[0]["sha256"] = "0" * 64
        assert A.dataset_version(m1) != A.dataset_version(touched)


# ---------------------------------------------------------------------------
# 5a.  Uncertainty reporting: a 0.000 SD must never be an unexplained zero
# ---------------------------------------------------------------------------

class TestUncertaintyReporting:
    def test_partition_uncertainty_is_measured_not_assumed(self, meta):
        ps = meta["results"]["partition_sensitivity"]
        assert ps, "spatial CV SDs are zero by design, so partition uncertainty must be run"
        scope = meta["validation"]["experiment_scope"]
        assert {p["model"] for p in ps} == set(scope["partition_models"])
        for p in ps:
            assert p["n_partitions"] == len(p["partition_seeds"]) == scope["partition_seeds"]
            assert p["r2_min"] <= p["r2_mean"] <= p["r2_max"]
            # one learner seed for every redraw, otherwise model randomness is
            # mixed into what should be a partition-only spread
            assert p["learner_seed"] == meta["validation"]["seeds"][0]
            assert 1 <= p["distinct_partitions"] <= p["n_partitions"]
            if p["n_partitions"] > 1:
                assert isinstance(p["r2_sd"], float) and p["r2_sd"] >= 0.0
            else:
                assert p["r2_sd"] is None, "an SD from one partition would be invented"

    def test_each_partition_row_carries_the_seed_that_built_it(self, meta):
        rows = meta["results"]["partition_rows"]
        seeds = sorted({r["block_seed"] for r in rows})
        assert len(seeds) == meta["validation"]["experiment_scope"]["partition_seeds"]
        assert all(r["cv"] == "spatial" and r["experiment"] == "partition_sensitivity"
                   for r in rows)

    def test_zero_sd_is_explained_in_the_record(self, meta):
        note = meta["validation"]["spatial_partition_note"]
        assert "partition_sensitivity" in note
        zero = [r for r in meta["results"]["main"] if r["cv"] == "spatial"
                and r["n_seeds"] > 1 and r["r2_sd"] == 0.0]
        if zero:                      # the reason must travel with the number
            assert "0.000" in note or "learner randomness" in note

    def test_a_different_blocking_seed_can_move_the_blocks(self, pipeline_inputs):
        p = pipeline_inputs
        g0, d0 = V.spatial_groups(p["coords"], p["ids"], 6, 0)
        g1, d1 = V.spatial_groups(p["coords"], p["ids"], 6, 987654321)
        assert d0["blocks"] == d1["blocks"] >= 2
        assert len(np.unique(g0)) == len(np.unique(g1)) == d0["blocks"]
        assert not np.array_equal(g0, g1), "the blocking seed had no effect at all"


# ---------------------------------------------------------------------------
# 5b.  Data labelling: a stand-in may never be reported as real AfSIS data
# ---------------------------------------------------------------------------

class TestDataSourceLabelling:
    """
    Guards the bug that was found during the audit: passing --root was taken as
    proof of real data, so results.json described synthetic files with the real
    AfSIS licence.  The label now comes from evidence on disk.
    """

    def test_generated_stand_in_identifies_itself(self, synthetic_afsis_root):
        assert S.is_stand_in(synthetic_afsis_root) is True
        text = (synthetic_afsis_root / S.MARKER_NAME).read_text(encoding="utf-8")
        assert "NOT" in text.upper() and "AfSIS" in text

    def test_marker_decides_even_without_the_flag(self, synthetic_afsis_root):
        assert E.classify_data_source(synthetic_afsis_root, False) == (
            "synthetic_stand_in", False, True)

    def test_unmarked_directory_is_trusted_only_when_not_declared_fake(self, tmp_path):
        # No marker: nothing on disk contradicts the user, so --root is believed.
        assert S.is_stand_in(tmp_path) is False
        assert E.classify_data_source(tmp_path, False) == ("afsis_real", True, False)
        # --synthetic can only ever move the label towards "not citable".
        kind, real, marker = E.classify_data_source(tmp_path, True)
        assert (kind, real, marker) == ("synthetic_stand_in", False, False)

    def test_no_root_is_never_real(self):
        assert E.classify_data_source(None, False)[1] is False

    def test_export_carries_the_stand_in_label(self, meta):
        d = meta["data"]
        assert d["kind"] == "synthetic_stand_in"
        assert d["is_real_data"] is False
        assert d["stand_in_marker"] is True
        assert d["synthetic_specification"]["real_data"] is False
        # the licence must describe these files, not the upstream dataset
        assert "No third-party data" in d["licence"]

    def test_booleans_are_written_as_booleans(self, synthetic_export):
        """np.bool_/bool must not silently become 1/0 in results.json."""
        raw = (synthetic_export / "results.json").read_text(encoding="utf-8")
        for key in ("is_real_data", "stand_in_marker"):
            assert f'"{key}": 0' not in raw and f'"{key}": 1' not in raw, key
            assert f'"{key}": true' in raw or f'"{key}": false' in raw, key
        assert json.loads(raw)["data"]["is_real_data"] is False


# ---------------------------------------------------------------------------
# 6.  Spatial blocking
# ---------------------------------------------------------------------------

class TestSpatialBlocking:
    def test_one_site_never_splits_across_groups(self, dup_pipeline_inputs):
        p = dup_pipeline_inputs
        if p["d"][p["schema"].group_key()].is_unique:
            pytest.skip("fixture has no repeated sites to test")
        seen = {}
        for g, i in zip(p["groups"], p["ids"]):
            assert seen.setdefault(i, g) == g, f"site {i} appears in two blocks"

    def test_group_count_matches_blocks(self, pipeline_inputs):
        assert len(set(pipeline_inputs["groups"])) <= pipeline_inputs["diag"]["blocks"]

    def test_folds_are_shared_by_every_model(self, pipeline_inputs):
        """Folds depend on (scheme, seed, groups) only - never on the model."""
        p = pipeline_inputs
        a = V.outer_splits("spatial", p["X"], p["y"], p["groups"], 1)
        b = V.outer_splits("spatial", p["X"], p["y"], p["groups"], 99)
        assert [t.tolist() for _, t in a] == [t.tolist() for _, t in b]

    def test_random_scheme_does_reuse_sites_across_folds(self, dup_pipeline_inputs):
        """
        Documents WHY blocking exists: with replicate rows kept, an unblocked
        row-wise split puts the same site in train and test.
        """
        p = dup_pipeline_inputs
        overlaps = 0
        for seed in range(3):
            for tr, te in V.outer_splits("random", p["X"], p["y"], p["groups"], seed):
                overlaps += len(set(p["ids"][tr]) & set(p["ids"][te]))
        assert overlaps > 0, "random CV unexpectedly found no shared site"


# ---------------------------------------------------------------------------
# 7.  Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_perfect_prediction(self):
        y = [1.0, 2.0, 3.0, 4.0]
        m = V.metrics(y, y)
        assert m["r2"] == pytest.approx(1.0)
        assert m["rmse"] == pytest.approx(0.0)
        assert m["mae"] == pytest.approx(0.0)

    def test_mean_prediction_has_zero_r2(self):
        y, p = [1.0, 2.0, 3.0, 4.0], [2.5] * 4
        m = V.metrics(y, p)
        assert m["r2"] == pytest.approx(0.0)
        assert m["rmse"] == pytest.approx(np.sqrt(5 / 4))

    def test_rpiq_is_iqr_over_rmse(self):
        y, p = [1.0, 2.0, 3.0, 4.0], [1.5, 2.5, 3.5, 4.5]
        m = V.metrics(y, p)
        iqr = float(np.subtract(*np.percentile(y, [75, 25])))
        assert m["rpiq"] == pytest.approx(iqr / m["rmse"])

    def test_metrics_ignore_non_finite(self):
        m = V.metrics([1.0, 2.0, np.nan, 3.0], [1.0, 2.0, 9.0, 3.0])
        assert m["n"] == 3 and m["r2"] == pytest.approx(1.0)

    def test_metrics_are_on_the_original_scale_not_log(self, pipeline_inputs):
        """A log-target model must still be scored in percent (or g/kg)."""
        p = pipeline_inputs
        r = V.run_cv(p["X"], p["y"], p["groups"], V.model_specs()["ridge"],
                     "spatial", 0, log_target=p["log_target"], tune=False)
        assert r["metrics"]["rmse"] < float(np.std(p["y"]))     # better than the mean
        assert r["metrics"]["rmse"] > 0.01, "log-scale metric would be far too small"


# ---------------------------------------------------------------------------
# 8.  Pipeline construction
# ---------------------------------------------------------------------------

class TestPipelineConstruction:
    def test_baseline_has_no_target_transform(self):
        spec = V.model_specs()["mean_baseline"]
        pipe = V.build_pipeline(spec, log_target=True, seed=0)
        assert not hasattr(pipe.named_steps["model"], "regressor")

    def test_log_target_predicts_in_original_units(self, pipeline_inputs):
        p = pipeline_inputs
        pipe = V.build_pipeline(V.model_specs()["ridge"], log_target=True, seed=0)
        pipe.fit(p["X"], p["y"])
        pred = pipe.predict(p["X"][:5])
        assert pred.min() > 0 and pred.max() < float(np.nanmax(p["y"])) * 3

    def test_seed_is_pinned_on_stochastic_learners(self):
        spec = V.model_specs()["random_forest"]
        a = V.build_pipeline(spec, False, seed=7).named_steps["model"]
        b = V.build_pipeline(spec, False, seed=7).named_steps["model"]
        assert a.random_state == 7 and b.random_state == 7

    def test_grid_keys_match_the_pipeline(self):
        specs = V.model_specs()
        for name, spec in specs.items():
            if not spec.grid:
                continue
            pipe = V.build_pipeline(spec, log_target=True, seed=0)
            for key in V.tunable_grid(spec, pipe):
                assert key in pipe.get_params(), f"{name}: bad grid key {key}"

    def test_models_list_matches_the_proposal(self):
        assert set(V.model_specs()) == {"mean_baseline", "ridge", "pls",
                                         "random_forest", "gradient_boosting"}

    def test_noise_uses_training_fold_sd_only(self, pipeline_inputs, monkeypatch):
        """Test-fold spread must not set the noise scale."""
        p = pipeline_inputs
        seen = []
        orig = V._nan_sd

        def spy(X):
            seen.append(X.shape[0])
            return orig(X)

        monkeypatch.setattr(V, "_nan_sd", spy)
        V.run_cv(p["X"], p["y"], p["groups"], V.model_specs()["ridge"], "spatial",
                 0, p["log_target"], tune=False, noise_level=0.5, arm="train_noisy")
        assert seen and max(seen) < p["X"].shape[0]

    def test_noise_leaves_missing_values_missing(self):
        X = np.array([[1.0, np.nan], [2.0, 3.0]])
        out = V._add_noise(X, np.array([1.0, 1.0]), 0.5, np.random.default_rng(0))
        assert np.isnan(out[0, 1]) and np.isfinite(out[0, 0])
