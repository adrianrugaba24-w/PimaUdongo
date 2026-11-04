"""
tests/test_api.py  –  FastAPI integration tests using httpx + TestClient.

Everything here runs against the SYNTHETIC export produced by the session
fixture (a real `afsis_experiment.py --quick` run).  No real AfSIS data and no
fabricated numbers: if a value is asserted, it is asserted against
results.json, never against a literal copied from a screenshot.
"""
import importlib
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_app(export_dir: Path):
    """
    Import soil_api with MODEL_DIR pointed at `export_dir`.

    The bundle is verified at import time on purpose, so a broken export raises
    BundleError here instead of serving predictions from a stale model.
    """
    os.environ["MODEL_DIR"] = str(export_dir)
    if "soil_api" in sys.modules:
        del sys.modules["soil_api"]
    import soil_api
    return soil_api.app


@pytest.fixture
def restore_app(synthetic_export):
    """Any test that reloads the module must leave a working app behind."""
    yield
    if "soil_api" in sys.modules:
        del sys.modules["soil_api"]
    make_app(synthetic_export)


def median_payload(meta, override=None):
    """A valid request built from the training medians recorded in results.json."""
    vals = {f: meta["feature_ranges"][f]["median"] for f in meta["features"]}
    if override:
        vals.update(override)
    return {"values": vals}


# ---------------------------------------------------------------------------
# 1.  Results endpoints
# ---------------------------------------------------------------------------

class TestResultsEndpoint:
    def test_results_200(self, client):
        assert client.get("/results").status_code == 200

    def test_results_has_features(self, client, meta):
        assert client.get("/results").json()["features"] == meta["features"]

    def test_results_has_cv_summary(self, client):
        """The CV tables the web page draws its numbers from."""
        body = client.get("/results").json()
        assert body["results"]["main"], "no CV rows exported"
        assert body["results"]["validation_gap"], "no validation-gap rows exported"

    def test_results_json_route_matches(self, client):
        assert client.get("/results.json").json() == client.get("/results").json()

    def test_both_validation_schemes_present(self, client, meta):
        schemes = {r["cv"] for r in client.get("/results").json()["results"]["main"]}
        assert {"random", "spatial"} <= schemes


# ---------------------------------------------------------------------------
# 2.  /predict
# ---------------------------------------------------------------------------

class TestPredictEndpoint:
    def test_predict_200(self, client, meta):
        assert client.post("/predict", json=median_payload(meta)).status_code == 200

    def test_predict_returns_prediction(self, client, meta):
        body = client.post("/predict", json=median_payload(meta)).json()
        assert isinstance(body["prediction"], (int, float))
        assert body["prediction"] >= 0.0      # SOC cannot be negative

    def test_predict_returns_interval(self, client, meta):
        body = client.post("/predict", json=median_payload(meta)).json()
        assert body["low"] is not None and body["high"] is not None
        assert body["low"] <= body["prediction"] <= body["high"]

    def test_predict_reports_units_and_versions(self, client, meta):
        body = client.post("/predict", json=median_payload(meta)).json()
        assert body["units"] == meta["target"]["units"]
        assert body["model_version"] == meta["model_version"]
        assert body["training_data_version"] == meta["data"]["dataset_version"]
        assert body["interval_empirical_coverage"] is not None

    def test_predict_numbers_come_from_results_json(self, client, meta):
        """No hard-coded metric may sit in the API."""
        body = client.post("/predict", json=median_payload(meta)).json()
        iv = meta["results"]["interval"]
        assert body["interval_empirical_coverage"] == iv["empirical_coverage"]
        assert body["interval_level"] == iv["nominal"]
        assert body["high"] - body["low"] == pytest.approx(iv["mean_width"], rel=1e-6)

    def test_predict_missing_feature_422(self, client, meta):
        first = meta["features"][0]
        r = client.post("/predict", json={"values": {first: 6.0}})
        assert r.status_code == 422
        assert r.json()["detail"]["error"] == "missing_features"

    def test_predict_empty_values_422(self, client):
        assert client.post("/predict", json={"values": {}}).status_code == 422

    def test_predict_unknown_feature_422(self, client, meta):
        p = median_payload(meta)
        p["values"]["Clay_of_the_Sea"] = 30.0
        r = client.post("/predict", json=p)
        assert r.status_code == 422
        assert "unknown_features" in r.json()["detail"]["error"]

    def test_predict_non_numeric_422(self, client, meta):
        p = median_payload(meta, {meta["features"][0]: "six point two"})
        assert client.post("/predict", json=p).status_code == 422

    def test_predict_impossible_value_422(self, client, meta):
        """clay > 100 % cannot exist: reject rather than extrapolate."""
        clay = next(f for f in meta["features"]
                    if "clay" in f.lower() and meta["feature_bounds"][f][1] == 100.0)
        p = median_payload(meta, {clay: 140.0})
        r = client.post("/predict", json=p)
        assert r.status_code == 422
        assert r.json()["detail"]["error"] == "impossible_value"

    def test_predict_outside_range_flag(self, client, meta):
        """Physically valid but outside the training 5-95 % band -> 200 + flag."""
        feat, value = _ood_case(meta)
        r = client.post("/predict", json=median_payload(meta, {feat: value}))
        assert r.status_code == 200
        body = r.json()
        assert body["out_of_domain"] is True
        assert {o["feature"] for o in body["out_of_domain_features"]} == {feat}


def _ood_case(meta):
    """Pick a feature with room between its training p95 and its physical max."""
    for f in meta["features"]:
        hi = meta["feature_bounds"][f][1]
        p95 = meta["feature_ranges"][f]["p95"]
        if hi is not None and p95 < hi:
            return f, p95 + 0.6 * (hi - p95)
    pytest.skip("no feature has a value above its training p95 but inside physical bounds")


# ---------------------------------------------------------------------------
# 3.  /health and /model-info
# ---------------------------------------------------------------------------

class TestHealthAndInfo:
    def test_health_200(self, client, meta):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["training_data_version"] == meta["data"]["dataset_version"]

    def test_model_info_200(self, client):
        assert client.get("/model-info").status_code == 200

    def test_model_info_puts_spatial_cv_first(self, client, meta):
        body = client.get("/model-info").json()
        rows = body["metrics"]["spatial_cv_first"]
        assert rows and all(r["r2_spatial"] is not None for r in rows)
        # listed by spatial R2, descending
        assert rows == sorted(rows, key=lambda r: -r["r2_spatial"])
        assert meta["headline"]["r2_spatial"] == pytest.approx(rows[0]["r2_spatial"])

    def test_model_info_carries_limitations(self, client, meta):
        body = client.get("/model-info").json()
        assert body["limitations"] == meta["limitations"]
        assert len(body["limitations"]) >= 3

    def test_model_info_reports_uncertainty_without_inventing_it(self, client, meta):
        """The API must relay the design note and the measured partition spread."""
        body = client.get("/model-info").json()
        assert body["validation_design"]["spatial_partition_note"] == \
            meta["validation"]["spatial_partition_note"]
        assert body["partition_sensitivity"] == meta["results"]["partition_sensitivity"]
        assert body["validation_design"]["experiment_scope"] == \
            meta["validation"]["experiment_scope"]

    def test_disclaimer_exact(self, client):
        assert client.get("/model-info").json()["disclaimer"] == (
            "Research prototype. Not validated on field sensors or Ugandan farms. "
            "Not for agronomic decisions.")

    def test_soc_is_never_called_soil_health(self, client, meta):
        """The output is predicted SOC, not a 'soil health' score."""
        blob = json.dumps(client.get("/model-info").json()).lower()
        assert "soil health" not in blob
        assert "organic carbon" in meta["target"]["name"].lower() or \
               "soc" in meta["target"]["name"].lower()

    def test_data_is_marked_synthetic_when_it_is(self, client, meta):
        body = client.get("/model-info").json()
        assert body["data_is_real"] == meta["data"]["is_real_data"]
        if not meta["data"]["is_real_data"]:
            assert body["data_kind"] == "synthetic_stand_in"


# ---------------------------------------------------------------------------
# 4.  Bundle integrity: the service must refuse to start
# ---------------------------------------------------------------------------

def _copy_bundle(src: Path, dst: Path) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("model.joblib", "model_meta.json", "results.json"):
        shutil.copy(src / f, dst / f)
    return dst


class TestBundleRefusal:
    def test_missing_everything(self, tmp_path):
        import soil_api
        with pytest.raises(soil_api.BundleError, match="missing"):
            soil_api.load_bundle(tmp_path)

    def test_missing_meta(self, synthetic_export, tmp_path):
        import soil_api
        d = _copy_bundle(synthetic_export, tmp_path / "a")
        (d / "model_meta.json").unlink()
        with pytest.raises(soil_api.BundleError, match="model_meta.json"):
            soil_api.load_bundle(d)

    def test_edited_results_json_is_detected(self, synthetic_export, tmp_path):
        """A hand-patched metric must not be servable."""
        import soil_api
        d = _copy_bundle(synthetic_export, tmp_path / "b")
        r = json.loads((d / "results.json").read_text())
        r["headline"]["r2_spatial"] = 0.99
        (d / "results.json").write_text(json.dumps(r))
        with pytest.raises(soil_api.BundleError, match="results.json does not match"):
            soil_api.load_bundle(d)

    def test_swapped_model_is_detected(self, synthetic_export, tmp_path):
        import soil_api
        d = _copy_bundle(synthetic_export, tmp_path / "c")
        d.joinpath("model.joblib").write_bytes(b"not a model")
        with pytest.raises(Exception):     # joblib cannot read it at all
            soil_api.load_bundle(d)

    def test_incomplete_metadata_is_detected(self, synthetic_export, tmp_path):
        import soil_api
        d = _copy_bundle(synthetic_export, tmp_path / "d")
        m = json.loads((d / "model_meta.json").read_text())
        del m["training_data_version"]
        (d / "model_meta.json").write_text(json.dumps(m))
        with pytest.raises(soil_api.BundleError, match="training_data_version"):
            soil_api.load_bundle(d)

    def test_app_import_fails_on_a_broken_dir(self, tmp_path, synthetic_export,
                                              restore_app):
        with pytest.raises(Exception):
            make_app(tmp_path)          # empty dir -> BundleError at import
        # restore_app re-loads a working app for the tests that follow


# ---------------------------------------------------------------------------
# 5.  Web page
# ---------------------------------------------------------------------------

class TestIndexPage:
    def test_index_200(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_index_carries_the_banner_and_no_baked_numbers(self, client):
        html = client.get("/").text
        assert "Not validated on field sensors or Ugandan farms" in html
        assert "soil health" not in html.lower()
        # the page must fetch its numbers; it may not contain them
        assert "fetch(" in html
