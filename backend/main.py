"""
backend/main.py - FastAPI Application for PimaUdongo IoT Soft-Sensing & Health Scoring
Mukono, Uganda Ground Truth Context (0.3547° N, 32.7475° E).

Features:
  - Token-authenticated Arduino / ESP32 JSON telemetry ingestion (/api/telemetry).
  - Dual-Role Epistemological Routes:
      * /api/health/farmer: High-contrast qualitative status bands and agronomic action directives.
      * /api/health/researcher: Raw channels, Ridge conformal bounds (+-0.41%), spatial CV metrics.
  - Interactive "What-If" Simulation workbench endpoints (/api/simulate, /api/simulation/presets).
  - Offline field resilience: automatic demo-proof simulation fallback if hardware node is inactive.
  - Clean additive architecture alongside legacy soil_api.py.
"""

from collections import deque
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader, HTTPBearer
from pydantic import BaseModel, Field

from backend.scoring_engine import (
    CONFORMAL_HALF_WIDTH_NOMINAL,
    MUKONO_ISDASOIL_PRIOR_SOC,
    MUKONO_LATITUDE,
    MUKONO_LONGITUDE,
    FarmerView,
    ResearcherView,
    ScoringEngine,
    SoilTelemetry,
)

# Authentication Configuration
INGESTION_API_KEY = os.environ.get("PIMAUDONGO_API_KEY", "pimaudongo-mukono-key-2026")
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Directory Paths
BASE_DIR = Path(__file__).parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
ML_EXPORT_DIR = BASE_DIR / "ml_export"

# Application Initialization
app = FastAPI(
    title="PimaUdongo API - African Soil Health & IoT Soft-Sensing",
    description="Agricultural soft-sensing prototype for Mukono, Uganda (Pulungan et al. 2026).",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize scoring engine
scoring_engine = ScoringEngine(ml_export_dir=ML_EXPORT_DIR)

# Ring buffer for recent field telemetry (keeps last 50 readings)
TELEMETRY_HISTORY: deque = deque(maxlen=50)

# Default Simulation / Offline Fallback Telemetry (Mukono Ferralsol baseline)
DEFAULT_FALLBACK_TELEMETRY = SoilTelemetry(
    moisture=24.2,
    temperature=25.4,
    ph=5.2,
    ec=0.18,
    node_id="pimaudongo-mukono-demo",
    timestamp=time.time(),
    battery_mv=3950,
    is_simulated=True
)


# -----------------------------------------------------------------------------
# Request & Response Schemas
# -----------------------------------------------------------------------------
class TelemetryPayload(BaseModel):
    """Schema for ESP32/Arduino JSON packets."""
    moisture: float = Field(..., ge=0.0, le=100.0, description="Volumetric water content (%)")
    temperature: float = Field(..., ge=-10.0, le=60.0, description="Soil temperature in °C")
    ph: float = Field(..., ge=1.0, le=14.0, description="Soil pH value")
    ec: float = Field(..., ge=0.0, le=10.0, description="Electrical conductivity in dS/m")
    node_id: Optional[str] = Field("node-mukono-01", description="Identifier of the field node")
    timestamp: Optional[float] = Field(None, description="Unix timestamp of sensor reading")
    battery_mv: Optional[int] = Field(3950, description="Battery voltage in millivolts")
    nitrogen_index: Optional[float] = Field(180.0, ge=0.0, le=2999.0, description="Relative Nitrogen index")
    phosphorus_index: Optional[float] = Field(25.0, ge=0.0, le=2999.0, description="Relative Phosphorus index")
    potassium_index: Optional[float] = Field(120.0, ge=0.0, le=2999.0, description="Relative Potassium index")


class SimulationPayload(BaseModel):
    """Schema for interactive What-If simulation slider adjustments."""
    moisture: float = Field(22.0, ge=0.0, le=100.0)
    temperature: float = Field(25.0, ge=0.0, le=60.0)
    ph: float = Field(6.0, ge=2.0, le=12.0)
    ec: float = Field(0.20, ge=0.0, le=5.0)
    nitrogen_index: Optional[float] = Field(180.0, ge=0.0, le=2999.0)
    phosphorus_index: Optional[float] = Field(25.0, ge=0.0, le=2999.0)
    potassium_index: Optional[float] = Field(120.0, ge=0.0, le=2999.0)


# -----------------------------------------------------------------------------
# Authentication Dependency
# -----------------------------------------------------------------------------
async def verify_telemetry_token(
    x_api_key: Optional[str] = Security(API_KEY_HEADER),
    token: Optional[str] = Query(None)
):
    """
    Validates token via X-API-Key header or URL query parameter ?token=.
    Protects live sensor ingestion against unauthorized writes.
    """
    provided_key = x_api_key or token
    if not provided_key or provided_key != INGESTION_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing PimaUdongo telemetry API key."
        )
    return provided_key


# -----------------------------------------------------------------------------
# Helper: Retrieve Latest Telemetry with Offline Fallback
# -----------------------------------------------------------------------------
def get_current_telemetry() -> SoilTelemetry:
    """Returns the most recent live telemetry or a realistic Mukono simulation fallback."""
    if TELEMETRY_HISTORY:
        # Check if the most recent reading is reasonably fresh (< 240 seconds)
        latest = TELEMETRY_HISTORY[-1]
        if latest.timestamp and (time.time() - latest.timestamp) < 240:
            return latest
    # Return offline fallback
    fallback = SoilTelemetry(
        moisture=DEFAULT_FALLBACK_TELEMETRY.moisture,
        temperature=DEFAULT_FALLBACK_TELEMETRY.temperature,
        ph=DEFAULT_FALLBACK_TELEMETRY.ph,
        ec=DEFAULT_FALLBACK_TELEMETRY.ec,
        node_id="pimaudongo-mukono-offline-fallback",
        timestamp=time.time(),
        battery_mv=3950,
        is_simulated=True
    )
    return fallback


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/health")
def health_check():
    """System health check endpoint."""
    return {
        "status": "online",
        "service": "PimaUdongo Backend",
        "location": "Mukono, Uganda (0.3547° N, 32.7475° E)",
        "cached_telemetry_count": len(TELEMETRY_HISTORY),
        "models_loaded": scoring_engine.ridge_model is not None or scoring_engine.rf_model is not None,
        "continental_rf_model": scoring_engine.continental_rf_model is not None
    }


@app.post("/api/telemetry")
def ingest_telemetry(payload: TelemetryPayload, _auth=Depends(verify_telemetry_token)):
    """
    Token-authenticated ingestion endpoint for ESP32/Arduino field nodes.
    Validates sensor telemetry, updates ring buffer, and calculates instant health score.
    """
    ts = payload.timestamp or time.time()
    telemetry = SoilTelemetry(
        moisture=payload.moisture,
        temperature=payload.temperature,
        ph=payload.ph,
        ec=payload.ec,
        node_id=payload.node_id or "node-mukono-01",
        timestamp=ts,
        battery_mv=payload.battery_mv,
        is_simulated=False,
        nitrogen_index=payload.nitrogen_index or 180.0,
        phosphorus_index=payload.phosphorus_index or 25.0,
        potassium_index=payload.potassium_index or 120.0
    )

    warnings = telemetry.validate()
    TELEMETRY_HISTORY.append(telemetry)
    evaluated = scoring_engine.score_telemetry(telemetry)

    return {
        "status": "success",
        "message": "Telemetry received and scored successfully.",
        "node_id": telemetry.node_id,
        "timestamp": ts,
        "warnings": warnings,
        "soil_health_index": evaluated["composite_index"],
        "overall_status": evaluated["overall_status"]
    }


@app.get("/api/health/farmer")
def get_farmer_view():
    """
    The Farmer View:
    Strips raw numeric units (%, dS/m, °C) and regression math.
    Translates telemetry into high-contrast qualitative status bands and
    concrete agronomic action directives (e.g. agricultural lime for phosphorus lock-up).
    """
    telemetry = get_current_telemetry()
    scored = scoring_engine.score_telemetry(telemetry)
    return {
        "farmer_view": scored["farmer_view"],
        "overall_status": scored["overall_status"],
        "limiting_factor": scored["limiting_factor"],
        "source": "simulated_fallback" if telemetry.is_simulated else "live_node",
        "node_id": telemetry.node_id
    }


@app.get("/api/health/researcher")
def get_researcher_view():
    """
    The Researcher View:
    Exposes raw channel readings, Ridge regression conformal intervals (+-0.41% SOC),
    spatial block CV vs random CV metrics, and explicit epistemic honesty metadata.
    """
    telemetry = get_current_telemetry()
    scored = scoring_engine.score_telemetry(telemetry)
    return {
        "researcher_view": scored["researcher_view"],
        "pillars": scored["pillars"],
        "composite_index": scored["composite_index"],
        "source": "simulated_fallback" if telemetry.is_simulated else "live_node",
        "node_id": telemetry.node_id
    }


@app.get("/api/latest")
def get_latest_complete_state():
    """Returns combined payload featuring both views and system connectivity state."""
    telemetry = get_current_telemetry()
    scored = scoring_engine.score_telemetry(telemetry)
    return {
        "node_id": telemetry.node_id,
        "timestamp": telemetry.timestamp,
        "is_simulated": telemetry.is_simulated,
        "battery_mv": telemetry.battery_mv,
        "composite_index": scored["composite_index"],
        "overall_status": scored["overall_status"],
        "limiting_factor": scored["limiting_factor"],
        "farmer_view": scored["farmer_view"],
        "researcher_view": scored["researcher_view"],
        "pillars": scored["pillars"]
    }


@app.get("/api/history")
def get_telemetry_history():
    """Returns recent telemetry buffer for graphing."""
    data = []
    for item in list(TELEMETRY_HISTORY):
        data.append({
            "timestamp": item.timestamp,
            "moisture": item.moisture,
            "temperature": item.temperature,
            "ph": item.ph,
            "ec": item.ec,
            "battery_mv": item.battery_mv,
            "is_simulated": item.is_simulated
        })
    return {"count": len(data), "history": data}


@app.post("/api/simulate")
def simulate_scenario(payload: SimulationPayload):
    """
    Interactive 'What-If' Simulation endpoint:
    Enables farmers and researchers to slide hypothetical sensor parameters
    and immediately view the 3-pillar response, conformal bounds, and agronomic directives.
    """
    telemetry = SoilTelemetry(
        moisture=payload.moisture,
        temperature=payload.temperature,
        ph=payload.ph,
        ec=payload.ec,
        node_id="workbench-simulator",
        timestamp=time.time(),
        battery_mv=4100,
        is_simulated=True,
        nitrogen_index=payload.nitrogen_index or 180.0,
        phosphorus_index=payload.phosphorus_index or 25.0,
        potassium_index=payload.potassium_index or 120.0
    )
    scored = scoring_engine.score_telemetry(telemetry)
    return scored


@app.get("/api/simulation/presets")
def get_simulation_presets():
    """
    Realistic agroecological presets for Mukono, Uganda soils:
    Demonstrates diverse soil health scenarios including phosphorus lock-up,
    optimal composted fertility, and drought heat stress.
    """
    return [
        {
            "id": "mukono_typical_ferralsol",
            "name": "Mukono Typical Ferralsol",
            "description": "Moderately weathered red sandy clay loam typical of Mukono smallholder maize/bean plots.",
            "values": {"moisture": 23.5, "temperature": 24.8, "ph": 5.8, "ec": 0.17}
        },
        {
            "id": "acidic_degraded_phosphorus_locked",
            "name": "Degraded Acidic Soil (Phosphorus Locked)",
            "description": "Severely leached, low pH soil where iron/aluminum oxides lock phosphorus from root absorption.",
            "values": {"moisture": 14.5, "temperature": 28.2, "ph": 4.6, "ec": 0.07}
        },
        {
            "id": "composted_organic_plot",
            "name": "Composted Organic Farm Plot",
            "description": "Soil enriched with seasoned manure, mulch, and balanced lime, high organic sponge capacity.",
            "values": {"moisture": 28.5, "temperature": 23.0, "ph": 6.5, "ec": 0.38}
        },
        {
            "id": "dry_season_heatwave",
            "name": "Dry Season Heatwave / Water-Stressed",
            "description": "Severe equatorial dry season sun; soil baked hard with accelerated organic carbon oxidation.",
            "values": {"moisture": 8.0, "temperature": 35.5, "ph": 5.4, "ec": 0.24}
        }
    ]


@app.get("/api/metadata")
def get_epistemic_metadata():
    """
    Returns complete epistemic provenance, spatial block CV evaluation,
    and continental regional prior extrapolation details.
    """
    return {
        "spatial_anchor": {
            "name": "Mukono District, Central Region, Uganda",
            "latitude": MUKONO_LATITUDE,
            "longitude": MUKONO_LONGITUDE,
            "median_distance_to_training_cluster_km": 2533.0
        },
        "epistemic_honesty": (
            "Regional Continental Prior Extrapolation: The model trains on open regional "
            "African wet-chem data (AfSIS/iSDAsoil) centered in West Africa (~2,533 km from Mukono). "
            "It functions as a continental regional prior extrapolation, NOT a hyper-local Mukono fit. "
            "All telemetry represents low-cost proximal proxy readings; no raw physical NPK sensors are used."
        ),
        "conformal_interval_nominal_half_width": CONFORMAL_HALF_WIDTH_NOMINAL,
        "isdasoil_prior_soc": MUKONO_ISDASOIL_PRIOR_SOC
    }


# -----------------------------------------------------------------------------
# Static Frontend Serving
# -----------------------------------------------------------------------------
@app.get("/")
def serve_dashboard():
    """Serves the responsive zero-dependency Tailwind + Alpine dashboard."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    # Fallback to legacy web/index.html if frontend is not yet placed
    legacy_file = BASE_DIR / "web" / "index.html"
    if legacy_file.exists():
        return FileResponse(legacy_file)
    return JSONResponse({"message": "PimaUdongo API is online."})
