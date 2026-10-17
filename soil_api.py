"""
Prediction service for the Path A prototype.
Run:  MODEL_DIR=export uvicorn soil_api:app --reload     (then open http://127.0.0.1:8000/)
Needs: pip install fastapi uvicorn scikit-learn joblib numpy
"""
import json, os
from pathlib import Path
import joblib, numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "export"))
WEB_DIR = Path(__file__).parent / "web"
app = FastAPI(title="Soil carbon prototype (Path A)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = joblib.load(MODEL_DIR / "model.joblib")
meta = json.loads((MODEL_DIR / "results.json").read_text())


class Query(BaseModel):
    values: dict[str, float]


@app.get("/results")
def results():
    return meta


@app.post("/predict")
def predict(q: Query):
    missing = [f for f in meta["features"] if f not in q.values]
    if missing:
        raise HTTPException(422, f"Missing features: {missing}")
    x = np.array([[q.values[f] for f in meta["features"]]], float)
    p = float(model.predict(x)[0])
    if meta["log_target"]:
        p = float(np.expm1(p))
    half = meta["interval"]["half_width"]
    warn = [f for f in meta["features"] if not (meta["feature_ranges"][f][0] <= q.values[f] <= meta["feature_ranges"][f][1])]
    return {"target": meta["target"], "prediction": p, "low": max(0.0, p - half), "high": p + half,
            "interval_level": meta["interval"]["level"], "interval_source": meta["interval"]["source"],
            "outside_training_range": warn}


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/results.json")
def results_json():
    return meta
