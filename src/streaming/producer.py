"""
src/streaming/producer.py
──────────────────────────
Kafka KPI event producer.

Reads historical KPI data and publishes to Kafka topic,
simulating a real-time stream at configurable speed.

Also supports standalone demo mode (no Kafka needed) for local testing.

Usage:
    # With Kafka:
    python src/streaming/producer.py --live
    python src/streaming/producer.py --speed 10   # 10× real-time

    # Demo mode (no Kafka):
    python src/streaming/producer.py --demo
"""

import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from loguru import logger
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_config(path="configs/config.yaml"):
    import yaml
    with open(path) as f: return yaml.safe_load(f)


def make_kpi_event(row: pd.Series) -> dict:
    """Serialise one KPI reading to a Kafka message dict."""
    return {
        "site_id":          str(row["site_id"]),
        "timestamp":        str(row["timestamp"]),
        "rsrq_avg":         round(float(row.get("rsrq_avg", -11.0)), 3),
        "rsrp_avg":         round(float(row.get("rsrp_avg", -95.0)), 3),
        "throughput_mbps":  round(float(row.get("throughput_mbps", 25.0)), 3),
        "latency_ms":       round(float(row.get("latency_ms", 20.0)), 3),
        "packet_loss_pct":  round(float(row.get("packet_loss_pct", 0.5)), 4),
        "connected_users":  int(row.get("connected_users", 50)),
        "prb_utilization":  round(float(row.get("prb_utilization", 0.4)), 4),
        "sinr_avg":         round(float(row.get("sinr_avg", 12.0)), 3),
        "published_at":     datetime.now(timezone.utc).isoformat(),
    }


def run_kafka_producer(config: dict, speed_factor: float = 1.0) -> None:
    """Stream KPI data to Kafka at configurable speed."""
    try:
        from kafka import KafkaProducer
        from kafka.errors import NoBrokersAvailable
    except ImportError:
        logger.error("kafka-python not installed: pip install kafka-python")
        sys.exit(1)

    kpi_path = Path("data/raw/network_kpis.parquet")
    if not kpi_path.exists():
        logger.error("KPI data not found. Run: python src/data_engineering/generate_data.py")
        sys.exit(1)

    df = pd.read_parquet(kpi_path).sort_values(["timestamp", "site_id"])
    topic = config["kafka"]["kpi_topic"]
    interval_sec = config["data_generation"]["interval_min"] * 60 / speed_factor

    logger.info(f"Connecting to Kafka: {config['kafka']['bootstrap_servers']}")
    try:
        producer = KafkaProducer(
            bootstrap_servers=config["kafka"]["bootstrap_servers"],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks="all",
            retries=3,
        )
    except NoBrokersAvailable:
        logger.error("Kafka broker not available. Start Kafka first: make docker-up")
        sys.exit(1)

    logger.success(f"Streaming to topic '{topic}' at {speed_factor}× speed...")
    timestamps = sorted(df["timestamp"].unique())
    sent = 0

    for ts in timestamps:
        batch = df[df["timestamp"] == ts]
        for _, row in batch.iterrows():
            event = make_kpi_event(row)
            producer.send(topic, value=event)
            sent += 1

        if sent % (len(batch) * 12) == 0:  # log every ~hour of simulated data
            logger.info(f"  Sent {sent:,} events up to {ts}")

        producer.flush()
        time.sleep(interval_sec)

    producer.close()
    logger.success(f"Streaming complete. Sent {sent:,} events.")


def run_demo(n_events: int = 200) -> list:
    """Generate synthetic KPI events without Kafka (for local testing)."""
    import yaml
    config = load_config()
    rng = np.random.default_rng(42)

    events = []
    sites = [f"SITE_{i:04d}" for i in range(10)]
    now = pd.Timestamp.now()

    for i in range(n_events):
        site = rng.choice(sites)
        ts   = now - pd.Timedelta(minutes=5 * (n_events - i))

        # Inject a simulated anomaly every ~50 events
        is_anomaly = (i % 47 < 5) and i > 20
        rsrq = float(rng.normal(-11, 1.5) + (-8 if is_anomaly else 0))
        tput = float(rng.lognormal(3.2, 0.5) * (0.1 if is_anomaly else 1.0))

        event = {
            "site_id":         site,
            "timestamp":       str(ts),
            "rsrq_avg":        round(max(-25, min(-3, rsrq)), 3),
            "rsrp_avg":        round(float(rng.normal(-95, 5)), 3),
            "throughput_mbps": round(max(0.1, tput), 3),
            "latency_ms":      round(float(rng.normal(20, 5) * (4 if is_anomaly else 1)), 3),
            "packet_loss_pct": round(float(abs(rng.normal(0.3, 0.2)) + (5 if is_anomaly else 0)), 4),
            "connected_users": int(rng.integers(20, 150)),
            "prb_utilization": round(float(rng.uniform(0.2, 0.9)), 4),
            "sinr_avg":        round(float(rng.normal(12, 3)), 3),
            "published_at":    datetime.now(timezone.utc).isoformat(),
            "_is_anomaly":     int(is_anomaly),  # hidden label for evaluation
        }
        events.append(event)

    return events


def main():
    parser = argparse.ArgumentParser(description="Network KPI stream producer")
    parser.add_argument("--demo",  action="store_true", help="Demo mode (no Kafka)")
    parser.add_argument("--live",  action="store_true", help="Stream to Kafka")
    parser.add_argument("--speed", type=float, default=60.0,
                        help="Speed factor (60 = 60× real time)")
    args = parser.parse_args()

    if args.demo:
        logger.info("Running demo producer (no Kafka required)...")
        events = run_demo(n_events=100)
        print(f"\nGenerated {len(events)} demo KPI events.")
        anomalies = sum(1 for e in events if e.get("_is_anomaly", 0))
        print(f"Anomalous events: {anomalies} ({anomalies/len(events):.0%})")
        print("\nSample event:")
        import json
        print(json.dumps({k: v for k, v in events[50].items()
                          if not k.startswith("_")}, indent=2))
    elif args.live:
        config = load_config()
        run_kafka_producer(config, speed_factor=args.speed)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
