"""
src/api/model_loader.py
────────────────────────
Loads trained anomaly detection artifacts and runs online inference.
"""

import sys
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import h3
    H3_AVAILABLE = True
except ImportError:
    H3_AVAILABLE = False


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_alert_rules(path="configs/alert_rules.yaml"):
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}


class SiteBuffer:
    """Per-site rolling window of recent KPI readings for online feature computation."""
    KPI_COLS = [
        "rsrq_avg", "rsrp_avg", "throughput_mbps", "latency_ms",
        "packet_loss_pct", "connected_users", "prb_utilization", "sinr_avg",
    ]

    def __init__(self, maxlen: int = 12):
        self.buffer: deque = deque(maxlen=maxlen)

    def push(self, event: dict) -> None:
        self.buffer.append({k: float(event.get(k, 0)) for k in self.KPI_COLS})

    def get_features(self) -> dict:
        if not self.buffer:
            return {}
        df = pd.DataFrame(list(self.buffer))
        features = {}
        for col in self.KPI_COLS:
            if col not in df.columns:
                continue
            vals = df[col].values
            features[col]              = float(vals[-1])
            features[f"{col}_rmean_30"]= float(np.mean(vals))
            features[f"{col}_rstd_30"] = float(np.std(vals) if len(vals) > 1 else 0)
            features[f"{col}_diff_1"]  = float(vals[-1] - vals[-2]) if len(vals) > 1 else 0.0
        return features


class ModelLoader:
    """Loads all anomaly detection artifacts and provides scoring interface."""

    def __init__(self):
        self.if_artifact   = None
        self.config        = {}
        self.alert_rules   = {}
        self._site_buffers = defaultdict(lambda: SiteBuffer(maxlen=12))
        self._site_anomaly_streaks: dict[str, int] = defaultdict(int)

    def load(self, models_dir: Path = Path("data/models")) -> None:
        try:
            self.config      = load_config()
            self.alert_rules = load_alert_rules()
        except FileNotFoundError:
            self.config      = {}
            self.alert_rules = {}

        if_path = models_dir / "isolation_forest.pkl"
        if if_path.exists():
            self.if_artifact = joblib.load(if_path)
            logger.success(f"Isolation Forest loaded from {if_path}")
        else:
            logger.warning(f"No IF model at {if_path} — run batch training first")

    def is_loaded(self) -> bool:
        return self.if_artifact is not None

    def score_event(self, event: dict) -> dict:
        """Score one KPI event. Maintains per-site rolling buffer."""
        site_id   = event.get("site_id", "unknown")
        timestamp = event.get("timestamp", "")
        buf       = self._site_buffers[site_id]
        buf.push(event)

        score      = 0.0
        confidence = 0.5

        if self.if_artifact:
            features  = buf.get_features()
            feat_cols = self.if_artifact.get("feature_cols", [])
            avail     = [c for c in feat_cols if c in features]
            if avail:
                X = np.array([[features.get(c, 0) for c in feat_cols]])
                try:
                    X_sc  = self.if_artifact["scaler"].transform(X)
                    raw   = self.if_artifact["model"].decision_function(X_sc)[0]
                    score = float(np.clip(-raw / 0.5, 0, 1))
                    confidence = float(np.clip(abs(raw) / 0.3, 0, 1))
                except Exception:
                    pass

        threshold = self.config.get("anomaly_detection", {}).get("threshold", 0.65)
        is_anomaly = int(score >= threshold)

        # Update streak counter (consecutive anomalous readings)
        if is_anomaly:
            self._site_anomaly_streaks[site_id] += 1
        else:
            self._site_anomaly_streaks[site_id] = 0
        streak = self._site_anomaly_streaks[site_id]

        severity = self._get_severity(score)
        offer_id = self._get_offer(severity) if is_anomaly and streak >= 3 else None

        # H3 cell
        h3_cell = None
        if H3_AVAILABLE and event.get("latitude") and event.get("longitude"):
            try:
                h3_cell = h3.geo_to_h3(event["latitude"], event["longitude"], 8)
            except Exception:
                pass

        return {
            "site_id":       site_id,
            "timestamp":     timestamp,
            "anomaly_score": round(score, 4),
            "is_anomaly":    is_anomaly,
            "severity":      severity,
            "offer_id":      offer_id,
            "confidence":    round(confidence, 3),
            "streak":        streak,
            "h3_cell":       h3_cell,
            "scored_at":     datetime.now(timezone.utc).isoformat(),
        }

    def _get_severity(self, score: float) -> str:
        tiers = self.alert_rules.get("severity_tiers", {})
        if tiers:
            for tier, cfg in sorted(tiers.items(), key=lambda x: x[1]["min_score"], reverse=True):
                if score >= cfg["min_score"]:
                    return tier
        if score < 0.20:   return "normal"
        elif score < 0.40: return "mild"
        elif score < 0.65: return "moderate"
        elif score < 0.85: return "severe"
        return "critical"

    def _get_offer(self, severity: str) -> str:
        offers_map = self.config.get("marketing", {}).get("offers", {})
        tier = offers_map.get(severity, {})
        return tier.get("offer_id", "data_1gb")
