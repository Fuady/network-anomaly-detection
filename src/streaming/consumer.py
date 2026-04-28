"""
src/streaming/consumer.py
──────────────────────────
Real-time Kafka KPI stream consumer with online anomaly detection.

Consumes KPI events from Kafka, applies the ensemble detector,
publishes anomaly alerts, and triggers marketing compensation offers.

Maintains a per-site rolling buffer (last 60 min of readings) for
online feature computation — no need to query historical data.

Usage:
    # With Kafka:
    python src/streaming/consumer.py --live

    # Demo mode (no Kafka, uses simulated events):
    python src/streaming/consumer.py --demo
"""

import sys
import json
import time
import threading
import signal
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from loguru import logger
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_config(path="configs/config.yaml"):
    import yaml
    with open(path) as f: return yaml.safe_load(f)


def load_alert_rules(path="configs/alert_rules.yaml"):
    import yaml
    try:
        with open(path) as f: return yaml.safe_load(f)
    except FileNotFoundError:
        return {}


# ── Online feature computer ───────────────────────────────────────────────────
class SiteBuffer:
    """Rolling buffer of recent KPI readings for one site."""

    KPI_COLS = [
        "rsrq_avg", "rsrp_avg", "throughput_mbps", "latency_ms",
        "packet_loss_pct", "connected_users", "prb_utilization", "sinr_avg"
    ]

    def __init__(self, maxlen: int = 60):
        self.buffer: deque = deque(maxlen=maxlen)
        self.maxlen = maxlen

    def push(self, event: dict) -> None:
        self.buffer.append({k: event.get(k, 0) for k in self.KPI_COLS})

    def get_features(self) -> dict:
        """Compute rolling features from buffer."""
        if len(self.buffer) == 0:
            return {}
        df = pd.DataFrame(list(self.buffer))
        features = {}
        for col in self.KPI_COLS:
            if col not in df.columns:
                continue
            vals = df[col].values.astype(float)
            features[col]            = float(vals[-1])
            features[f"{col}_mean"]  = float(np.mean(vals))
            features[f"{col}_std"]   = float(np.std(vals))
            features[f"{col}_min"]   = float(np.min(vals))
            features[f"{col}_max"]   = float(np.max(vals))
            if len(vals) >= 2:
                features[f"{col}_diff"] = float(vals[-1] - vals[-2])
        return features

    def __len__(self):
        return len(self.buffer)


# ── Streaming anomaly detector ────────────────────────────────────────────────
class StreamingAnomalyDetector:
    """
    Online anomaly detector using pre-trained Isolation Forest.
    (LSTM and Prophet require batch retraining; they score in batch mode.)
    """

    def __init__(self, models_dir: Path = Path("data/models")):
        self.if_artifact = None
        self.site_buffers: dict[str, SiteBuffer] = defaultdict(lambda: SiteBuffer(maxlen=60))
        self._load_model(models_dir)

    def _load_model(self, models_dir: Path) -> None:
        if_path = models_dir / "isolation_forest.pkl"
        if if_path.exists():
            self.if_artifact = joblib.load(if_path)
            logger.success("Isolation Forest model loaded for streaming")
        else:
            logger.warning(f"No IF model at {if_path} — run batch training first")

    def process_event(self, event: dict) -> dict:
        """Process one KPI event and return anomaly score."""
        site_id = event.get("site_id", "unknown")
        buf     = self.site_buffers[site_id]
        buf.push(event)

        score = 0.0
        if self.if_artifact and len(buf) >= 3:
            features = buf.get_features()
            feat_cols = self.if_artifact.get("feature_cols", [])
            X = np.array([[features.get(c, 0) for c in feat_cols]])

            try:
                X_scaled = self.if_artifact["scaler"].transform(X)
                raw      = self.if_artifact["model"].decision_function(X_scaled)[0]
                score    = float(np.clip(-raw / 0.5, 0, 1))
            except Exception as e:
                logger.debug(f"Scoring error for {site_id}: {e}")

        return {
            "site_id":         site_id,
            "timestamp":       event.get("timestamp", ""),
            "anomaly_score":   round(score, 4),
            "is_anomaly":      int(score >= 0.65),
            "buffer_length":   len(buf),
            "scored_at":       datetime.now(timezone.utc).isoformat(),
        }


# ── Marketing trigger ─────────────────────────────────────────────────────────
class MarketingTrigger:
    """Evaluates anomaly results and fires compensation offers."""

    def __init__(self, alert_rules: dict, config: dict):
        self.alert_rules = alert_rules
        self.config      = config
        self._sent_today: dict[str, int] = defaultdict(int)
        self._alerts_log: list[dict] = []

    def should_trigger(self, site_id: str, score: float, duration_min: float) -> bool:
        """Decide if a marketing offer should be sent."""
        rules = self.alert_rules
        min_dur = self.config["marketing"]["min_impact_minutes"]
        max_per_day = rules.get("rate_limit", {}).get("max_offers_per_day", 1)
        if duration_min < min_dur:
            return False
        if self._sent_today.get(site_id, 0) >= max_per_day:
            return False
        return score >= 0.20

    def get_offer(self, score: float) -> dict:
        """Select the appropriate offer tier based on severity score."""
        offers = self.config["marketing"]["offers"]
        for tier, cfg in sorted(offers.items(),
                                key=lambda x: x[1]["threshold"], reverse=True):
            if score >= cfg["threshold"]:
                return {"tier": tier, **cfg}
        return {"tier": "mild", "offer_id": "data_1gb", "message": "1GB bonus data"}

    def trigger(self, site_id: str, score: float, timestamp: str) -> dict | None:
        """Fire a marketing trigger for the affected zone."""
        offer = self.get_offer(score)
        self._sent_today[site_id] += 1
        alert = {
            "site_id":    site_id,
            "timestamp":  timestamp,
            "score":      score,
            "offer_id":   offer.get("offer_id", "data_1gb"),
            "tier":       offer.get("tier", "mild"),
            "message":    offer.get("message", "Bonus data added"),
            "triggered_at": datetime.now(timezone.utc).isoformat(),
        }
        self._alerts_log.append(alert)
        logger.info(
            f"🔔 MARKETING TRIGGER | site={site_id} | "
            f"score={score:.2f} | offer={offer['offer_id']}"
        )
        return alert

    @property
    def alerts_log(self) -> list:
        return self._alerts_log


# ── Kafka consumer ────────────────────────────────────────────────────────────
class KPIConsumer:
    """Consumes KPI events from Kafka and runs real-time anomaly detection."""

    def __init__(self, config: dict, alert_rules: dict):
        self.config       = config
        self.detector     = StreamingAnomalyDetector()
        self.marketing    = MarketingTrigger(alert_rules, config)
        self._running     = threading.Event()
        self._running.set()
        self._stats       = {"processed": 0, "alerts": 0, "errors": 0}
        self._site_alert_duration: dict[str, float] = defaultdict(float)

    def start_kafka(self) -> None:
        """Start consuming from Kafka topic."""
        try:
            from kafka import KafkaConsumer
            from kafka import KafkaProducer
        except ImportError:
            logger.error("kafka-python not installed")
            sys.exit(1)

        consumer = KafkaConsumer(
            self.config["kafka"]["kpi_topic"],
            bootstrap_servers=self.config["kafka"]["bootstrap_servers"],
            group_id=self.config["kafka"]["group_id"],
            auto_offset_reset="latest",
            enable_auto_commit=True,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            consumer_timeout_ms=1000,
        )
        producer = KafkaProducer(
            bootstrap_servers=self.config["kafka"]["bootstrap_servers"],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )

        logger.success(f"Consuming from '{self.config['kafka']['kpi_topic']}'...")
        while self._running.is_set():
            for message in consumer:
                if not self._running.is_set():
                    break
                try:
                    result = self.detector.process_event(message.value)
                    self._stats["processed"] += 1

                    if result["is_anomaly"]:
                        site_id = result["site_id"]
                        self._site_alert_duration[site_id] += 5  # 5-min intervals
                        alert = self.marketing.trigger(
                            site_id, result["anomaly_score"], result["timestamp"]
                        )
                        if alert:
                            producer.send(self.config["kafka"]["alert_topic"], value=alert)
                            producer.send(self.config["kafka"]["marketing_topic"], value=alert)
                            self._stats["alerts"] += 1
                    else:
                        site_id = result["site_id"]
                        self._site_alert_duration[site_id] = 0

                except Exception as e:
                    logger.error(f"Processing error: {e}")
                    self._stats["errors"] += 1

        consumer.close()
        producer.close()

    def run_demo(self, events: list) -> list:
        """Process a list of events without Kafka (demo mode)."""
        from src.streaming.producer import run_demo as gen_demo
        if events is None:
            events = gen_demo(n_events=200)

        results = []
        for event in events:
            result = self.detector.process_event(event)
            self._stats["processed"] += 1
            if result["is_anomaly"]:
                site_id = result["site_id"]
                self._site_alert_duration[site_id] += 5
                alert = self.marketing.trigger(
                    site_id, result["anomaly_score"],
                    result["timestamp"]
                )
                if alert:
                    results.append(alert)
                    self._stats["alerts"] += 1

        return results

    def shutdown(self):
        self._running.clear()


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Real-time KPI anomaly consumer")
    parser.add_argument("--demo",  action="store_true", help="Demo mode (no Kafka)")
    parser.add_argument("--live",  action="store_true", help="Connect to Kafka")
    args = parser.parse_args()

    config      = load_config()
    alert_rules = load_alert_rules()
    consumer    = KPIConsumer(config, alert_rules)

    if args.demo:
        logger.info("Running streaming demo (no Kafka required)...")
        from src.streaming.producer import run_demo as gen_events
        events  = gen_events(n_events=300)
        alerts  = consumer.run_demo(events)

        print("\n" + "=" * 60)
        print("STREAMING DEMO RESULTS")
        print("=" * 60)
        print(f"  Events processed  : {consumer._stats['processed']:,}")
        print(f"  Anomaly alerts    : {consumer._stats['alerts']}")
        print(f"  Processing errors : {consumer._stats['errors']}")
        if alerts:
            print(f"\n  Last alert:")
            for k, v in alerts[-1].items():
                print(f"    {k}: {v}")
        print("=" * 60)

    elif args.live:
        def handler(sig, frame):
            logger.info("Shutting down...")
            consumer.shutdown()
            sys.exit(0)
        signal.signal(signal.SIGINT, handler)
        consumer.start_kafka()
    else:
        parser.print_help()


import argparse
if __name__ == "__main__":
    main()
