"""
src/api/schemas.py  +  src/api/model_loader.py combined
"""
# schemas.py ──────────────────────────────────────────────────────────────────
from typing import Optional, List
from pydantic import BaseModel, Field


class KPIReading(BaseModel):
    site_id:          str   = Field(..., example="SITE_0001")
    timestamp:        str   = Field(..., example="2024-01-15T08:30:00Z")
    rsrq_avg:         float = Field(-11.0, example=-18.5, description="dB, Good >-10, Poor <-15")
    rsrp_avg:         float = Field(-95.0, example=-105.0, description="dBm")
    throughput_mbps:  float = Field(25.0,  ge=0, example=4.2)
    latency_ms:       float = Field(20.0,  ge=0, example=85.0)
    packet_loss_pct:  float = Field(0.5,   ge=0, le=100, example=3.2)
    connected_users:  int   = Field(50,    ge=0, example=145)
    prb_utilization:  float = Field(0.4,   ge=0, le=1, example=0.85)
    sinr_avg:         float = Field(12.0,  example=3.5)
    latitude:         Optional[float] = Field(None, example=-6.2088)
    longitude:        Optional[float] = Field(None, example=106.8456)

    class Config:
        json_schema_extra = {"example": {
            "site_id": "SITE_0001", "timestamp": "2024-01-15T08:30:00Z",
            "rsrq_avg": -18.5, "rsrp_avg": -105.0, "throughput_mbps": 4.2,
            "latency_ms": 85.0, "packet_loss_pct": 3.2, "connected_users": 145,
            "prb_utilization": 0.85, "sinr_avg": 3.5,
            "latitude": -6.2088, "longitude": 106.8456,
        }}


class KPIBatch(BaseModel):
    readings: List[KPIReading] = Field(..., max_length=500)


class DetectionResult(BaseModel):
    site_id:          str
    timestamp:        str
    anomaly_score:    float
    is_anomaly:       int
    severity:         str
    offer_id:         Optional[str] = None
    confidence:       float = 0.0
    scored_at:        str


class BatchDetectionResponse(BaseModel):
    results:     List[dict]
    total:       int
    n_anomalies: int


class MarketingTriggerRequest(BaseModel):
    zone_id:              str
    severity:             str = Field("moderate", description="mild|moderate|severe|critical")
    estimated_affected:   int = Field(100, ge=0)
    notes:                Optional[str] = None


class MarketingTriggerResponse(BaseModel):
    zone_id:                      str
    severity:                     str
    offer_id:                     str
    offer_name:                   str
    triggered_at:                 str
    estimated_affected_subscribers: int


class HealthResponse(BaseModel):
    status:        str
    models_loaded: bool
    timestamp:     str
