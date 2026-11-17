"""
tests/test_backend.py - Integration Tests for PimaUdongo FastAPI Backend Service
Validates telemetry ingestion, token authentication, simulation endpoints,
dual-role routes, and offline fallback resilience.
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.main import app, INGESTION_API_KEY, TELEMETRY_HISTORY


@pytest.fixture
def client():
    # Clear telemetry history before tests
    TELEMETRY_HISTORY.clear()
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_200(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "online"
        assert "Mukono" in data["location"]


class TestTelemetryIngestion:
    def test_ingestion_unauthorized_without_key(self, client):
        payload = {
            "moisture": 24.0,
            "temperature": 25.0,
            "ph": 6.2,
            "ec": 0.20
        }
        r = client.post("/api/telemetry", json=payload)
        assert r.status_code == 401

    def test_ingestion_unauthorized_with_wrong_key(self, client):
        payload = {
            "moisture": 24.0,
            "temperature": 25.0,
            "ph": 6.2,
            "ec": 0.20
        }
        r = client.post("/api/telemetry", json=payload, headers={"X-API-Key": "wrong-key"})
        assert r.status_code == 401

    def test_ingestion_success_with_header_key(self, client):
        payload = {
            "moisture": 24.5,
            "temperature": 25.2,
            "ph": 5.2,
            "ec": 0.18,
            "node_id": "esp32-node-mukono",
            "battery_mv": 4020
        }
        r = client.post("/api/telemetry", json=payload, headers={"X-API-Key": INGESTION_API_KEY})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "success"
        assert data["node_id"] == "esp32-node-mukono"
        assert "soil_health_index" in data

    def test_ingestion_success_with_query_param(self, client):
        payload = {
            "moisture": 22.0,
            "temperature": 24.0,
            "ph": 6.1,
            "ec": 0.22
        }
        r = client.post(f"/api/telemetry?token={INGESTION_API_KEY}", json=payload)
        assert r.status_code == 200


class TestDualRoleRoutes:
    def test_farmer_route_structure(self, client):
        r = client.get("/api/health/farmer")
        assert r.status_code == 200
        data = r.json()
        assert "farmer_view" in data
        assert "overall_status" in data
        fv = data["farmer_view"]
        assert "headline" in fv
        assert "immediate_actions" in fv
        assert len(fv["immediate_actions"]) > 0

    def test_researcher_route_structure(self, client):
        r = client.get("/api/health/researcher")
        assert r.status_code == 200
        data = r.json()
        assert "researcher_view" in data
        rv = data["researcher_view"]
        assert "raw_telemetry" in rv
        assert "soft_sensing_soc" in rv
        assert "epistemic_honesty" in rv
        assert "validation_performance" in rv

    def test_latest_complete_state(self, client):
        r = client.get("/api/latest")
        assert r.status_code == 200
        data = r.json()
        assert "composite_index" in data
        assert "farmer_view" in data
        assert "researcher_view" in data
        assert "pillars" in data


class TestSimulationEndpoints:
    def test_simulate_scenario(self, client):
        payload = {
            "moisture": 26.0,
            "temperature": 24.0,
            "ph": 6.4,
            "ec": 0.25
        }
        r = client.post("/api/simulate", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert "composite_index" in data
        assert data["overall_status"] == "Optimal"
        assert "farmer_view" in data
        assert "researcher_view" in data

    def test_simulation_presets(self, client):
        r = client.get("/api/simulation/presets")
        assert r.status_code == 200
        presets = r.json()
        assert len(presets) >= 4
        ids = [p["id"] for p in presets]
        assert "mukono_typical_ferralsol" in ids
        assert "acidic_degraded_phosphorus_locked" in ids

    def test_history_endpoint(self, client):
        # Push 2 readings
        for m in (20.0, 25.0):
            client.post(
                "/api/telemetry",
                json={"moisture": m, "temperature": 24.0, "ph": 6.0, "ec": 0.2},
                headers={"X-API-Key": INGESTION_API_KEY}
            )
        r = client.get("/api/history")
        assert r.status_code == 200
        hist = r.json()
        assert hist["count"] == 2


class TestOfflineResilience:
    def test_offline_fallback_on_empty_history(self, client):
        TELEMETRY_HISTORY.clear()
        r = client.get("/api/health/farmer")
        assert r.status_code == 200
        assert r.json()["source"] == "simulated_fallback"


class TestDashboardEndpoint:
    def test_root_index_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "PimaUdongo" in r.text
