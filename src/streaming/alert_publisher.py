"""
src/streaming/alert_publisher.py
──────────────────────────────────
Publishes anomaly alerts and marketing triggers to Kafka topics.
Handles deduplication, rate limiting, and alert enrichment.

Usage:
    from src.streaming.alert_publisher import AlertPublisher
    pub = AlertPublisher(config)
    pub.publish_anomaly(site_id, score, severity, timestamp)
"""

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_alert_rules(path="configs/alert_rules.yaml"):
    import yaml
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}


class AlertPublisher:
    """
    Publishes structured anomaly alerts with deduplication and rate limiting.
    Works in both Kafka mode and local log mode (no Kafka needed).
    """

    def __init__(self, config: dict, kafka_mode: bool = False):
        self.config      = config
        self.kafka_mode  = kafka_mode
        self.alert_rules = load_alert_rules()
        self._producer   = None
        self._sent_log: list[dict]  = []
        self._site_last_alert: dict = {}  # site_id → last alert datetime
        self._cooldown_hours = (
            self.alert_rules.get("rate_limit", {}).get("cooldown_hours", 24)
        )

        if kafka_mode:
            self._init_kafka()

    def _init_kafka(self) -> None:
        try:
            from kafka import KafkaProducer
            self._producer = KafkaProducer(
                bootstrap_servers=self.config.get("kafka", {}).get(
                    "bootstrap_servers", "localhost:9092"
                ),
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                acks="all",
                retries=3,
            )
            logger.success("AlertPublisher connected to Kafka")
        except Exception as e:
            logger.warning(f"Kafka unavailable — falling back to local log: {e}")
            self.kafka_mode = False

    def _is_rate_limited(self, site_id: str) -> bool:
        """Check if this site is within the alert cooldown window."""
        last = self._site_last_alert.get(site_id)
        if last is None:
            return False
        elapsed = (datetime.now(timezone.utc) - last).total_seconds() / 3600
        return elapsed < self._cooldown_hours

    def _get_offer_details(self, severity: str) -> dict:
        """Look up offer details for a given severity tier."""
        tiers = self.alert_rules.get("severity_tiers", {})
        tier  = tiers.get(severity, {})
        return {
            "offer_id":   tier.get("offer_id",   "data_1gb"),
            "offer_name": tier.get("offer_name", "1GB Data Bonus"),
            "channel":    tier.get("channel",    "push"),
        }

    def publish_anomaly(
        self,
        site_id: str,
        score: float,
        severity: str,
        timestamp: str,
        extra: dict = None,
    ) -> dict | None:
        """
        Publish an anomaly alert.
        Returns the alert dict if published, None if rate-limited.
        """
        if self._is_rate_limited(site_id):
            logger.debug(f"Rate-limited: {site_id} — cooldown active")
            return None

        offer  = self._get_offer_details(severity)
        alert  = {
            "alert_id":       f"ALERT_{site_id}_{int(datetime.now().timestamp())}",
            "site_id":        site_id,
            "timestamp":      timestamp,
            "anomaly_score":  round(score, 4),
            "severity":       severity,
            "offer_id":       offer["offer_id"],
            "offer_name":     offer["offer_name"],
            "channel":        offer["channel"],
            "published_at":   datetime.now(timezone.utc).isoformat(),
            **(extra or {}),
        }

        # Update rate limit tracker
        self._site_last_alert[site_id] = datetime.now(timezone.utc)
        self._sent_log.append(alert)

        if self.kafka_mode and self._producer:
            try:
                alert_topic    = self.config["kafka"]["alert_topic"]
                marketing_topic = self.config["kafka"]["marketing_topic"]
                self._producer.send(alert_topic, value=alert)
                self._producer.send(marketing_topic, value=alert)
                self._producer.flush()
            except Exception as e:
                logger.error(f"Kafka publish error: {e}")
        else:
            logger.info(
                f"🚨 ALERT | {site_id} | score={score:.2f} | "
                f"severity={severity} | offer={offer['offer_id']}"
            )

        return alert

    def publish_zone_alert(
        self,
        h3_zone: str,
        affected_sites: list[str],
        avg_score: float,
        severity: str,
        timestamp: str,
    ) -> dict:
        """Publish a zone-level alert (multiple sites affected)."""
        offer = self._get_offer_details(severity)
        zone_alert = {
            "alert_id":        f"ZONE_{h3_zone}_{int(datetime.now().timestamp())}",
            "alert_type":      "zone",
            "h3_zone":         h3_zone,
            "affected_sites":  affected_sites,
            "n_affected_sites":len(affected_sites),
            "avg_anomaly_score": round(avg_score, 4),
            "severity":        severity,
            "offer_id":        offer["offer_id"],
            "offer_name":      offer["offer_name"],
            "channel":         offer["channel"],
            "published_at":    datetime.now(timezone.utc).isoformat(),
            "timestamp":       timestamp,
        }

        self._sent_log.append(zone_alert)
        logger.info(
            f"🗺️  ZONE ALERT | {h3_zone} | "
            f"{len(affected_sites)} sites | severity={severity}"
        )

        if self.kafka_mode and self._producer:
            try:
                self._producer.send(
                    self.config["kafka"]["alert_topic"],
                    value=zone_alert
                )
                self._producer.flush()
            except Exception as e:
                logger.error(f"Zone alert Kafka error: {e}")

        return zone_alert

    def save_log(self, output_path: str = "data/processed/alerts_stream.parquet") -> None:
        """Save the in-memory alert log to Parquet."""
        if not self._sent_log:
            return
        import pandas as pd
        pd.DataFrame(self._sent_log).to_parquet(output_path, index=False)
        logger.success(f"Alert log saved ({len(self._sent_log)} alerts) → {output_path}")

    def get_summary(self) -> dict:
        """Return alert statistics."""
        by_severity = {}
        for a in self._sent_log:
            sev = a.get("severity", "unknown")
            by_severity[sev] = by_severity.get(sev, 0) + 1
        return {
            "total_alerts": len(self._sent_log),
            "by_severity":  by_severity,
            "unique_sites": len(set(a["site_id"] for a in self._sent_log
                                    if "site_id" in a)),
        }

    def close(self) -> None:
        if self._producer:
            self._producer.close()
