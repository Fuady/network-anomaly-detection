"""
src/api/app.py
───────────────
FastAPI REST API for network anomaly detection.

Endpoints:
  POST /detect           — Score a batch of KPI readings
  GET  /anomalies/active — Current active anomaly zones (GeoJSON)
  GET  /alerts/history   — Historical alert log
  POST /marketing/trigger— Manually trigger marketing for a zone
  GET  /sites            — List all monitored sites
  GET  /health           — Health check
  GET  /metrics          — Prometheus metrics

Usage:
    uvicorn src.api.app:app --reload --port 8000
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.api.schemas import (
    KPIReading, KPIBatch, DetectionResult, BatchDetectionResponse,
    MarketingTriggerRequest, MarketingTriggerResponse, HealthResponse,
)
from src.api.model_loader import ModelLoader

app = FastAPI(
    title="Network Anomaly Detection API",
    description="Real-time network KPI anomaly detection and proactive marketing trigger.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── Prometheus metrics ────────────────────────────────────────────────────────
anomaly_counter    = Counter("anomalies_detected_total", "Total anomalies detected", ["severity"])
request_latency    = Histogram("api_request_seconds", "API request latency", ["endpoint"])
active_anomalies   = Gauge("active_anomaly_zones", "Number of currently active anomaly zones")
marketing_triggers = Counter("marketing_triggers_total", "Total marketing triggers fired", ["tier"])

loader = ModelLoader()


@app.on_event("startup")
async def startup():
    loader.load()
    logger.info("API ready.")


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(
        status="healthy" if loader.is_loaded() else "degraded",
        models_loaded=loader.is_loaded(),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/metrics", tags=["System"])
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ── Detection ──────────────────────────────────────────────────────────────────
@app.post("/detect", response_model=DetectionResult, tags=["Detection"])
async def detect_single(reading: KPIReading):
    """Score a single KPI reading for network anomaly."""
    if not loader.is_loaded():
        raise HTTPException(503, "Models not loaded")
    try:
        with request_latency.labels("detect").time():
            result = loader.score_event(reading.model_dump())
        if result.get("is_anomaly"):
            anomaly_counter.labels(severity=result.get("severity", "unknown")).inc()
        return DetectionResult(**result)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/detect/batch", response_model=BatchDetectionResponse, tags=["Detection"])
async def detect_batch(batch: KPIBatch):
    """Score a batch of KPI readings (max 500)."""
    if not loader.is_loaded():
        raise HTTPException(503, "Models not loaded")
    if len(batch.readings) > 500:
        raise HTTPException(400, "Batch too large (max 500)")
    try:
        results = [loader.score_event(r.model_dump()) for r in batch.readings]
        n_anomalies = sum(1 for r in results if r.get("is_anomaly"))
        return BatchDetectionResponse(
            results=results,
            total=len(results),
            n_anomalies=n_anomalies,
        )
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Anomaly zones ──────────────────────────────────────────────────────────────
@app.get("/anomalies/active", tags=["Anomalies"])
async def active_anomaly_zones():
    """Return current anomaly zones as GeoJSON (from latest batch scoring)."""
    geojson_path = Path("data/processed/anomaly_zones.geojson")
    if not geojson_path.exists():
        raise HTTPException(404, "No anomaly map. Run: python src/models/geo_impact_map.py")
    with open(geojson_path) as f:
        geojson = json.load(f)
    alert_features = [
        f for f in geojson["features"]
        if f["properties"].get("is_alert", 0)
    ]
    active_anomalies.set(len(alert_features))
    return {"type": "FeatureCollection", "features": alert_features}


@app.get("/anomalies/zones", tags=["Anomalies"])
async def all_anomaly_zones():
    """Return all H3 zones with anomaly scores."""
    geojson_path = Path("data/processed/anomaly_zones.geojson")
    if not geojson_path.exists():
        raise HTTPException(404, "No anomaly map.")
    with open(geojson_path) as f:
        return json.load(f)


# ── Alerts ─────────────────────────────────────────────────────────────────────
@app.get("/alerts/history", tags=["Alerts"])
async def alert_history(limit: int = 100):
    """Return historical alert records."""
    alerts_path = Path("data/processed/alerts.parquet")
    if not alerts_path.exists():
        return {"alerts": [], "total": 0}
    alerts = pd.read_parquet(alerts_path)
    alerts = alerts.sort_values("timestamp", ascending=False).head(limit)
    return {
        "alerts": alerts.to_dict(orient="records"),
        "total": len(alerts),
    }


# ── Marketing triggers ─────────────────────────────────────────────────────────
@app.post("/marketing/trigger", response_model=MarketingTriggerResponse, tags=["Marketing"])
async def manual_marketing_trigger(request: MarketingTriggerRequest):
    """Manually trigger a marketing compensation for a specific zone."""
    offer_map = {
        "mild":     {"offer_id": "data_1gb",  "name": "1GB Data Bonus"},
        "moderate": {"offer_id": "data_5gb",  "name": "5GB Data Bonus"},
        "severe":   {"offer_id": "day_pass",  "name": "Free Day Pass"},
        "critical": {"offer_id": "week_free", "name": "1 Week Free"},
    }
    offer = offer_map.get(request.severity, offer_map["mild"])
    marketing_triggers.labels(tier=request.severity).inc()

    return MarketingTriggerResponse(
        zone_id=request.zone_id,
        severity=request.severity,
        offer_id=offer["offer_id"],
        offer_name=offer["name"],
        triggered_at=datetime.now(timezone.utc).isoformat(),
        estimated_affected_subscribers=request.estimated_affected,
    )


# ── Sites ──────────────────────────────────────────────────────────────────────
@app.get("/sites", tags=["Sites"])
async def list_sites(limit: int = 50):
    """List monitored cell sites."""
    sites_path = Path("data/raw/sites.parquet")
    if not sites_path.exists():
        raise HTTPException(404, "Site data not found")
    sites = pd.read_parquet(sites_path).head(limit)
    return sites.to_dict(orient="records")


@app.get("/", tags=["System"])
async def root():
    return {"name": "Network Anomaly Detection API", "version": "1.0.0",
            "docs": "/docs", "health": "/health"}
