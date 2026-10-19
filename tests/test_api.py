"""
tests/test_api.py  –  FastAPI integration tests using httpx + TestClient.

All tests use the SYNTHETIC export fixture.  No real model or AfSIS data.
"""
import sys
import os
import json
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_app(export_dir: Path):
    """
    Import soil_api with MODEL_DIR pointed at the synthetic export.
    We reload the module each time so different export_dirs can be tested.
    """
    import importlib
    os.environ["MODEL_DIR"] = str(export_dir)

    # Force re-import so module-level load() picks up the new env var
    if "soil_api" in sys.modules:
        del sys.modules["soil_api"]

    import soil_api
    importlib.reload(soil_api)
    return soil_api.app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client(synthetic_export):
    from fastapi.testclient import TestClient
    app = make_app(synthetic_export)
    return TestClient(app)


@pytest.fixture(scope="module")
def meta(synthetic_export):
    return json.loads((synthetic_export / "results.json").read_text())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestResultsEndpoint:
    def test_results_200(self, client):
        r = client.get("/results")
        assert r.status_code == 200

    def test_results_has_features(self, client, meta):
        r = client.get("/results")
        data = r.json()
        assert "features" in data
        assert data["features"] == meta["features"]

    def test_results_has_cv_summary(self, client):
        r = client.get("/results")
        assert "cv_summary" in r.json()


class TestPredictEndpoint:
    def _payload(self, meta, override=None):
        """Build a valid payload using feature medians from meta."""
        vals = {f: meta["feature_ranges"][f][2] for f in meta["features"]}
        if override:
            vals.update(override)
        return {"values": vals}

    def test_predict_200(self, client, meta):
        r = client.post("/predict", json=self._payload(meta))
        assert r.status_code == 200

    def test_predict_returns_prediction(self, client, meta):
        r = client.post("/predict", json=self._payload(meta))
        body = r.json()
        assert "prediction" in body
        assert isinstance(body["prediction"], (int, float))
        assert body["prediction"] > 0   # OC should be positive

    def test_predict_returns_interval(self, client, meta):
        r = client.post("/predict", json=self._payload(meta))
        body = r.json()
        assert "low" in body and "high" in body
        assert body["low"] <= body["prediction"] <= body["high"]

    def test_predict_missing_feature_422(self, client, meta):
        """Sending only the first feature should trigger 422."""
        first = meta["features"][0]
        r = client.post("/predict", json={"values": {first: 6.0}})
        assert r.status_code == 422

    def test_predict_empty_values_422(self, client):
        r = client.post("/predict", json={"values": {}})
        assert r.status_code == 422

    def test_predict_outside_range_flag(self, client, meta):
        """A value way outside the training range should trigger the warning."""
        first_feat = meta["features"][0]
        lo, hi, _ = meta["feature_ranges"][first_feat]
        extreme = hi * 100      # far outside
        r = client.post("/predict", json=self._payload(meta, {first_feat: extreme}))
        assert r.status_code == 200
        assert first_feat in r.json()["outside_training_range"]


class TestIndexPage:
    def test_index_200(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
