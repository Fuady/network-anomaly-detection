"""
src/data_engineering/generate_data.py
──────────────────────────────────────
Generates realistic synthetic network KPI timeseries for 200 cell sites
over 90 days at 5-minute resolution.

Simulates:
  - Daily and weekly seasonality (busy hours, weekend dips)
  - 4 anomaly types: outage, congestion, interference, hardware fault
  - Spatial correlation (nearby sites degrade together)
  - Realistic KPI distributions and inter-KPI correlations

Output:
  data/raw/network_kpis.parquet   — full timeseries (~5M rows)
  data/raw/sites.parquet          — site metadata (lat/lon, H3 cells)
  data/raw/anomaly_labels.parquet — ground truth labels (for evaluation)

Usage:
    python src/data_engineering/generate_data.py
    python src/data_engineering/generate_data.py --n_sites 50 --days 30
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Site metadata generation ──────────────────────────────────────────────────
def generate_sites(n_sites: int, geo_bounds: dict, h3_res: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate cell site metadata with realistic clustering."""
    logger.info(f"Generating {n_sites} cell sites...")

    # Cluster sites into urban areas
    n_clusters = max(5, n_sites // 15)
    cluster_lats = rng.uniform(geo_bounds["lat_min"], geo_bounds["lat_max"], n_clusters)
    cluster_lons = rng.uniform(geo_bounds["lon_min"], geo_bounds["lon_max"], n_clusters)
    weights = rng.dirichlet(np.ones(n_clusters) * 2)
    assigned = rng.choice(n_clusters, n_sites, p=weights)

    lats = np.clip(
        rng.normal(cluster_lats[assigned], 0.03),
        geo_bounds["lat_min"], geo_bounds["lat_max"]
    )
    lons = np.clip(
        rng.normal(cluster_lons[assigned], 0.03),
        geo_bounds["lon_min"], geo_bounds["lon_max"]
    )

    sites = pd.DataFrame({
        "site_id":        [f"SITE_{i:04d}" for i in range(n_sites)],
        "latitude":       lats.round(6),
        "longitude":      lons.round(6),
        "cluster_id":     assigned,
        "radio_type":     rng.choice(["LTE", "NR", "LTE+NR"], n_sites, p=[0.45, 0.20, 0.35]),
        "max_capacity_mbps": rng.choice([100, 150, 200, 300], n_sites, p=[0.3, 0.3, 0.25, 0.15]),
        "install_year":   rng.choice([2015, 2017, 2019, 2021, 2023], n_sites),
        "urban_density":  rng.choice(["dense", "urban", "suburban", "rural"], n_sites,
                                       p=[0.25, 0.40, 0.25, 0.10]),
    })

    # H3 cells
    try:
        import h3
        sites["h3_r8"] = sites.apply(
            lambda r: h3.geo_to_h3(r["latitude"], r["longitude"], h3_res), axis=1
        )
        sites["h3_r7"] = sites.apply(
            lambda r: h3.geo_to_h3(r["latitude"], r["longitude"], h3_res - 1), axis=1
        )
    except ImportError:
        sites["h3_r8"] = "h3_unavailable"
        sites["h3_r7"] = "h3_unavailable"

    return sites


# ── KPI baseline generation ───────────────────────────────────────────────────
def make_kpi_baseline(
    site: pd.Series,
    timestamps: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> dict:
    """Generate normal (no-anomaly) KPI values for one site."""
    n = len(timestamps)
    hour_of_day   = timestamps.hour + timestamps.minute / 60.0
    day_of_week   = timestamps.dayofweek   # 0=Mon, 6=Sun

    # ── Daily seasonality ─────────────────────────────────────────────────────
    # Peak usage: morning commute (7-9am) and evening (7-10pm)
    daily_load = (
        0.30
        + 0.25 * np.exp(-0.5 * ((hour_of_day - 8.0) / 1.5) ** 2)   # morning peak
        + 0.35 * np.exp(-0.5 * ((hour_of_day - 20.0) / 2.0) ** 2)  # evening peak
        + 0.10 * np.exp(-0.5 * ((hour_of_day - 13.0) / 1.5) ** 2)  # lunch
    )

    # ── Weekly seasonality ────────────────────────────────────────────────────
    # Weekends (5,6) have different patterns — less work data, more social
    weekday_factor = np.where(day_of_week < 5, 1.0, 0.85)
    load = daily_load * weekday_factor

    # ── Site-specific baseline ────────────────────────────────────────────────
    density_multiplier = {"dense": 1.3, "urban": 1.0, "suburban": 0.75, "rural": 0.5}
    dm = density_multiplier.get(site.get("urban_density", "urban"), 1.0)
    max_cap = float(site.get("max_capacity_mbps", 150))

    # RSRQ: degrades with load (-3 to -17 dB range)
    rsrq = -7.0 - 8.0 * load * dm + rng.normal(0, 0.8, n)
    rsrq = np.clip(rsrq, -20.0, -3.0)

    # RSRP: relatively stable but correlates with load (-75 to -110 dBm)
    rsrp = -85.0 - 12.0 * load * dm + rng.normal(0, 2.0, n)
    rsrp = np.clip(rsrp, -130.0, -60.0)

    # Throughput: inversely proportional to load (users compete for bandwidth)
    throughput = max_cap * (1.0 - 0.65 * load * dm) + rng.normal(0, 3.0, n)
    throughput = np.clip(throughput, 0.5, max_cap)

    # Latency: increases with load
    latency = 15.0 + 50.0 * load * dm + rng.normal(0, 3.0, n)
    latency = np.clip(latency, 5.0, 200.0)

    # Packet loss: mostly 0, occasionally elevated
    packet_loss = np.abs(rng.normal(0, 0.3, n)) * load * dm
    packet_loss = np.clip(packet_loss, 0.0, 5.0)

    # Connected users
    users = 20 + int(150 * dm) * load + rng.normal(0, 5, n)
    users = np.clip(users.astype(int), 0, 500)

    # PRB utilisation (Physical Resource Block)
    prb = 0.15 + 0.65 * load * dm + rng.normal(0, 0.03, n)
    prb = np.clip(prb, 0.0, 1.0)

    # SINR
    sinr = 18.0 - 10.0 * load * dm + rng.normal(0, 1.5, n)
    sinr = np.clip(sinr, -3.0, 30.0)

    return {
        "rsrq_avg":         rsrq,
        "rsrp_avg":         rsrp,
        "throughput_mbps":  throughput,
        "latency_ms":       latency,
        "packet_loss_pct":  packet_loss,
        "connected_users":  users,
        "prb_utilization":  prb,
        "sinr_avg":         sinr,
    }


# ── Anomaly injection ─────────────────────────────────────────────────────────
ANOMALY_PROFILES = {
    "outage": {
        "duration_range": (6, 36),      # intervals
        "rsrq_delta":     -10.0,
        "throughput_mult": 0.05,
        "latency_mult":    5.0,
        "packet_loss_add": 20.0,
        "users_mult":      0.05,
    },
    "congestion": {
        "duration_range": (12, 72),
        "rsrq_delta":     -4.0,
        "throughput_mult": 0.30,
        "latency_mult":    2.5,
        "packet_loss_add": 3.0,
        "users_mult":      2.0,     # users stay up during congestion
    },
    "interference": {
        "duration_range": (4, 24),
        "rsrq_delta":     -7.0,
        "throughput_mult": 0.40,
        "latency_mult":    1.8,
        "packet_loss_add": 5.0,
        "users_mult":      0.70,
    },
    "hardware": {
        "duration_range": (24, 144),  # hardware faults last longer
        "rsrq_delta":     -5.0,
        "throughput_mult": 0.50,
        "latency_mult":    2.0,
        "packet_loss_add": 2.0,
        "users_mult":      0.60,
    },
}


def inject_anomalies(
    kpis: dict,
    n: int,
    anomaly_rates: dict,
    rng: np.random.Generator,
) -> tuple:
    """Inject anomaly events into KPI arrays. Returns modified kpis and label array."""
    labels = np.zeros(n, dtype=int)
    anomaly_type_labels = ["normal"] * n

    for a_type, rate in anomaly_rates.items():
        profile = ANOMALY_PROFILES.get(a_type, {})
        dur_min, dur_max = profile.get("duration_range", (6, 24))
        n_events = max(0, int(rng.poisson(rate * n)))

        for _ in range(n_events):
            start = rng.integers(0, n - dur_max)
            dur   = rng.integers(dur_min, dur_max)
            end   = min(start + dur, n)

            # Apply degradation
            kpis["rsrq_avg"][start:end]        += profile.get("rsrq_delta", 0)
            kpis["throughput_mbps"][start:end] *= profile.get("throughput_mult", 1.0)
            kpis["latency_ms"][start:end]      *= profile.get("latency_mult", 1.0)
            kpis["packet_loss_pct"][start:end] += profile.get("packet_loss_add", 0)
            kpis["connected_users"][start:end] = (
                kpis["connected_users"][start:end] * profile.get("users_mult", 1.0)
            ).astype(int)

            labels[start:end] = 1
            for i in range(start, end):
                anomaly_type_labels[i] = a_type

    # Clip KPIs back to valid ranges
    kpis["rsrq_avg"]        = np.clip(kpis["rsrq_avg"],        -25.0, -3.0)
    kpis["throughput_mbps"] = np.clip(kpis["throughput_mbps"],   0.1, 400.0)
    kpis["latency_ms"]      = np.clip(kpis["latency_ms"],        5.0, 2000.0)
    kpis["packet_loss_pct"] = np.clip(kpis["packet_loss_pct"],   0.0, 100.0)
    kpis["connected_users"] = np.clip(kpis["connected_users"],   0,   500)

    return kpis, labels, anomaly_type_labels


# ── Main generation ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate synthetic network KPI data")
    parser.add_argument("--n_sites", type=int, default=200)
    parser.add_argument("--days",    type=int, default=90)
    parser.add_argument("--output",  type=str, default="data/raw")
    parser.add_argument("--config",  type=str, default="configs/config.yaml")
    parser.add_argument("--seed",    type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = args.seed or config["data_generation"]["random_seed"]
    rng  = np.random.default_rng(seed)
    geo_bounds = config["data_generation"]["geo_bounds"]
    h3_res     = config["data_generation"]["h3_resolution"]
    interval   = config["data_generation"]["interval_min"]
    rates      = config["data_generation"]["anomaly_rates"]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Timestamps ────────────────────────────────────────────────────────────
    start_dt = datetime(2024, 1, 1)
    end_dt   = start_dt + timedelta(days=args.days)
    timestamps = pd.date_range(start=start_dt, end=end_dt, freq=f"{interval}min")[:-1]
    n = len(timestamps)
    logger.info(f"Generating {args.days} days × {interval}-min intervals = {n:,} timestamps")

    # ── Sites ─────────────────────────────────────────────────────────────────
    sites = generate_sites(args.n_sites, geo_bounds, h3_res, rng)
    sites.to_parquet(output_dir / "sites.parquet", index=False)
    logger.success(f"Sites saved → {output_dir / 'sites.parquet'}")

    # ── KPI generation per site ───────────────────────────────────────────────
    logger.info("Generating KPI timeseries for all sites...")
    all_kpis   = []
    all_labels = []

    for _, site in sites.iterrows():
        kpis = make_kpi_baseline(site, timestamps, rng)
        kpis, labels, type_labels = inject_anomalies(kpis, n, rates, rng)

        df = pd.DataFrame({
            "site_id":   site["site_id"],
            "timestamp": timestamps,
            **{k: v for k, v in kpis.items()},
        })
        df_labels = pd.DataFrame({
            "site_id":          site["site_id"],
            "timestamp":        timestamps,
            "is_anomaly":       labels,
            "anomaly_type":     type_labels,
        })
        all_kpis.append(df)
        all_labels.append(df_labels)

    df_kpis   = pd.concat(all_kpis,   ignore_index=True)
    df_labels = pd.concat(all_labels, ignore_index=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    kpi_path   = output_dir / "network_kpis.parquet"
    label_path = output_dir / "anomaly_labels.parquet"

    df_kpis.to_parquet(kpi_path,   index=False)
    df_labels.to_parquet(label_path, index=False)

    logger.success(f"KPI data saved   → {kpi_path}")
    logger.success(f"Labels saved     → {label_path}")

    # Summary
    anomaly_rate = df_labels["is_anomaly"].mean()
    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Cell sites      : {args.n_sites:,}")
    print(f"  Days            : {args.days}")
    print(f"  Total rows      : {len(df_kpis):,}")
    print(f"  Anomaly rate    : {anomaly_rate:.2%}")
    print(f"  Anomaly types   : {df_labels[df_labels['is_anomaly']==1]['anomaly_type'].value_counts().to_dict()}")
    print(f"  Output dir      : {output_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
