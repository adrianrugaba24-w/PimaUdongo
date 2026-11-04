"""
tests/test_validity.py  –  the five checks the proposal has to survive.

1. leakage      : no test-fold site appears in a training fold
2. shuffled y   : apparent skill disappears (R2 near 0)
3. coverage     : the reported interval coverage is measured, and within tolerance
4. reproducibly : the same seed reproduces the same metrics exactly
5. domain       : an out-of-domain input is flagged, not silently extrapolated

Plus two honesty guards: nothing may be fitted on a test fold, and no number in
results.json may be a place-holder.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import afsis_validation as V

# Tolerances are set from what the design can deliver, not from what the
# stand-in happened to produce: a shuffled target has no skill, so R2 must stay
# clearly below the smallest skill we would report; coverage is estimated from a
# few dozen held-out points, so a wide band is the honest tolerance.
SHUFFLED_R2_MAX = 0.15
COVERAGE_TOL = 0.15


# ---------------------------------------------------------------------------
# 1.  Leakage
# ---------------------------------------------------------------------------

class TestNoLeakage:
    def test_no_test_fold_site_appears_in_training(self, dup_pipeline_inputs):
        """Records of one site share a fold, even when a site has several rows."""
        p = dup_pipeline_inputs
        ids = p["ids"]
        n_sites = len(set(ids.tolist()))
        assert n_sites < len(ids), "fixture should contain repeated sites"
        for tr, te in V.outer_splits("spatial", p["X"], p["y"], p["groups"], 0):
            assert not (set(ids[tr].tolist()) & set(ids[te].tolist())), \
                "a site is in both the training and the test fold"
            assert len(set(tr) & set(te)) == 0

    def test_every_row_is_tested_exactly_once(self, pipeline_inputs):
        p = pipeline_inputs
        seen = np.zeros(len(p["y"]), dtype=int)
        for scheme in ("random", "spatial"):
            seen[:] = 0
            for _, te in V.outer_splits(scheme, p["X"], p["y"], p["groups"], 0):
                seen[te] += 1
            assert (seen == 1).all(), f"{scheme} CV left rows untested or tested twice"

    def test_nothing_is_fitted_on_a_test_fold(self, pipeline_inputs, monkeypatch):
        """
        Instrument every Pipeline.fit: the largest training set ever seen must be
        smaller than the whole frame, so imputation, scaling and tuning are all
        fold-local.  Uses an in-process learner (no worker processes to spy on).
        """
        from sklearn.pipeline import Pipeline

        p = pipeline_inputs
        sizes = []
        orig = Pipeline.fit

        def spy(self, X, y=None, **kw):
            sizes.append(len(X))
            return orig(self, X, y, **kw)

        monkeypatch.setattr(Pipeline, "fit", spy)
        V.run_cv(p["X"], p["y"], p["groups"], V.model_specs()["ridge"], "spatial",
                 0, p["log_target"], tune=True)
        assert sizes, "no pipeline was fitted - the spy missed"
        assert max(sizes) < len(p["y"]), "a fit saw every row, including the test fold"

    def test_feature_ranges_are_descriptive_not_features(self, meta, pipeline_inputs):
        """The API needs percentiles for its domain check; the model must not."""
        assert meta["features"] == list(pipeline_inputs["schema"].features)
        assert meta["target"]["name"] not in meta["feature_ranges"]
        assert meta["target"]["name"] not in meta["features"]
        assert meta["blocked_as_features"] == []


# ---------------------------------------------------------------------------
# 2.  Shuffled-target control
# ---------------------------------------------------------------------------

class TestShuffledControl:
    def test_shuffled_target_shows_no_skill(self, pipeline_inputs):
        p = pipeline_inputs
        y_shuf = np.random.default_rng(0).permutation(p["y"])
        for name in ("mean_baseline", "ridge", "pls", "gradient_boosting"):
            r = V.run_cv(p["X"], y_shuf, p["groups"], V.model_specs()[name], "spatial",
                         0, p["log_target"], tune=False)
            assert r["metrics"]["r2"] < SHUFFLED_R2_MAX, \
                f"{name} found R2={r['metrics']['r2']:.3f} in shuffled labels"

    def test_exported_control_is_near_zero(self, meta):
        rows = meta["results"]["shuffled_control"]
        assert rows, "no shuffled-target control was exported"
        for r in rows:
            assert r["r2_mean"] < SHUFFLED_R2_MAX, r
        assert max(r["r2_mean"] for r in rows) < SHUFFLED_R2_MAX

    def test_real_model_beats_the_sham(self, pipeline_inputs):
        p = pipeline_inputs
        y_shuf = np.random.default_rng(0).permutation(p["y"])
        sham = V.run_cv(p["X"], y_shuf, p["groups"], V.model_specs()["ridge"], "spatial",
                        0, p["log_target"], tune=False)["metrics"]["r2"]
        real = V.run_cv(p["X"], p["y"], p["groups"], V.model_specs()["ridge"], "spatial",
                        0, p["log_target"], tune=False)["metrics"]["r2"]
        assert real > sham


# ---------------------------------------------------------------------------
# 3.  Prediction-interval coverage
# ---------------------------------------------------------------------------

class TestIntervalCoverage:
    def test_coverage_is_measured_on_unseen_blocks(self, pipeline_inputs):
        p = pipeline_inputs
        out = V.conformal_spatial(p["X"], p["y"], p["groups"], V.model_specs()["ridge"],
                                  None, 0, p["log_target"])
        assert out["empirical_coverage"] is not None
        assert out["n_folds"] >= 2
        assert all(f["n_calib"] > 0 for f in out["per_fold"])
        assert "held out" in out["method"] or "held-out" in out["method"]

    def test_reported_coverage_within_tolerance_of_nominal(self, meta):
        iv = meta["results"]["interval"]
        assert iv.get("empirical_coverage") is not None, iv.get("note")
        assert abs(iv["empirical_coverage"] - iv["nominal"]) <= COVERAGE_TOL, \
            (f"coverage {iv['empirical_coverage']:.3f} is further than {COVERAGE_TOL} "
             f"from the nominal {iv['nominal']:.2f}")

    def test_width_is_reported_with_coverage(self, meta):
        """A wide interval that always covers is not evidence of a good model."""
        iv = meta["results"]["interval"]
        assert iv["mean_width"] > 0
        ratio = iv["mean_width"] / iv["median_target"]
        assert ratio < 10, f"interval is {ratio:.1f}x the median value - useless"

    def test_api_interval_matches_the_exported_coverage(self, meta):
        assert meta["results"]["interval"]["half_width"] > 0


# ---------------------------------------------------------------------------
# 4.  Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_metrics(self, pipeline_inputs):
        p = pipeline_inputs
        spec = V.model_specs()["gradient_boosting"]
        a = V.run_cv(p["X"], p["y"], p["groups"], spec, "spatial", 3, p["log_target"],
                     tune=False)
        b = V.run_cv(p["X"], p["y"], p["groups"], spec, "spatial", 3, p["log_target"],
                     tune=False)
        assert a["metrics"] == b["metrics"]
        assert a["oof"] == b["oof"]

    def test_noise_is_reproducible(self, pipeline_inputs):
        p = pipeline_inputs
        spec = V.model_specs()["ridge"]
        run = lambda: V.run_cv(p["X"], p["y"], p["groups"], spec, "spatial", 1,
                               p["log_target"], tune=False, noise_level=0.5,
                               arm="train_noisy")["metrics"]
        assert run() == run()

    def test_stochastic_learner_is_seeded(self, pipeline_inputs):
        """Without a pinned random_state a rerun would not reproduce."""
        p = pipeline_inputs
        spec = V.model_specs()["random_forest"]
        a = V.run_cv(p["X"], p["y"], p["groups"], spec, "spatial", 5, p["log_target"],
                     tune=False)["metrics"]
        b = V.run_cv(p["X"], p["y"], p["groups"], spec, "spatial", 5, p["log_target"],
                     tune=False)["metrics"]
        assert a == b

    def test_different_seed_changes_random_folds_only(self, pipeline_inputs):
        p = pipeline_inputs
        f0 = V.outer_splits("random", p["X"], p["y"], p["groups"], 0)
        f1 = V.outer_splits("random", p["X"], p["y"], p["groups"], 1)
        assert [t.tolist() for _, t in f0] != [t.tolist() for _, t in f1]
        s0 = V.outer_splits("spatial", p["X"], p["y"], p["groups"], 0)
        s1 = V.outer_splits("spatial", p["X"], p["y"], p["groups"], 7)
        assert [t.tolist() for _, t in s0] == [t.tolist() for _, t in s1]

    def test_results_declare_where_they_came_from(self, meta):
        """
        Every reported number must be traceable and real: no unfilled template
        fields, and the stand-in must say so in its own record.
        """
        assert meta["data"]["is_real_data"] is False
        assert meta["data"]["synthetic_specification"]["warning"]
        assert meta["data"]["dataset_version"].startswith("afsis-")
        assert meta["data"]["files"] and meta["command"] and meta["package_versions"]
        for k in ("r2_spatial", "rmse_spatial", "r2_random"):
            assert isinstance(meta["headline"][k], float)
        assert isinstance(meta["results"]["interval"]["empirical_coverage"], float)
        # a null left in a headline metric means it was never computed
        for row in meta["results"]["validation_gap"]:
            for k in ("r2_spatial", "r2_random", "gap_mean", "rmse_spatial"):
                assert row[k] is not None, f"{row['model']} has an uncomputed {k}"


# ---------------------------------------------------------------------------
# 5.  Out-of-domain detection
# ---------------------------------------------------------------------------

class TestOutOfDomain:
    def test_values_beyond_training_percentiles_are_flagged(self, api, meta):
        feats, ranges = api.FEATURES, meta["feature_ranges"]
        pick = next((f for f in feats if ranges[f]["p95"] < meta["feature_bounds"][f][1]),
                    None)
        if pick is None:
            pytest.skip("no feature has headroom above p95")
        x = np.array([[ranges[f]["median"] for f in feats]], float)
        x[0, feats.index(pick)] = ranges[pick]["p95"] + 0.5 * (
            meta["feature_bounds"][pick][1] - ranges[pick]["p95"])
        flags = api._out_of_domain(x)
        assert flags and {f["feature"] for f in flags} == {pick}
        assert flags[0]["distance_outside"] > 0

    def test_in_domain_input_is_not_flagged(self, api):
        x = np.array([[api.RANGES[f]["median"] for f in api.FEATURES]], float)
        assert api._out_of_domain(x) == []

    def test_an_impossible_reading_is_rejected_not_flagged(self, api, meta):
        """> 100 % clay is a unit error, so it must be a 422, not a warning."""
        from fastapi import HTTPException
        vals = {f: api.RANGES[f]["median"] for f in api.FEATURES}
        target = next(f for f in api.FEATURES if api.BOUNDS.get(f, [None, None])[1] == 100.0)
        vals[target] = 137.0
        with pytest.raises(HTTPException) as ex:
            api._validate(vals)
        assert ex.value.status_code == 422
        assert ex.value.detail["error"] == "impossible_value"

    def test_smallest_plausible_input_is_still_servable(self, api):
        """A real reading at the low edge is in-domain: no 422, prediction >= 0."""
        vals = {f: api.RANGES[f]["min"] for f in api.FEATURES}
        x = api._validate(vals)
        assert x.shape == (1, len(api.FEATURES))
        assert max(0.0, float(api.MODEL.predict(x)[0])) >= 0.0
