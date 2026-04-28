"""
src/models/geo_impact_map.py
──────────────────────────────
Aggregates site-level anomaly scores to H3 hex zones.
Produces geographic risk maps showing which areas are experiencing
network degradation and the affected subscriber count.

Usage:
    python src/models/geo_impact_map.py
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_config(path="configs/config.yaml"):
    with open(path) as f: return yaml.safe_load(f)


SEVERITY_COLORS = {
    "normal":   "#2ecc71",
    "mild":     "#f1c40f",
    "moderate": "#e67e22",
    "severe":   "#e74c3c",
    "critical": "#8e44ad",
}


def aggregate_to_h3(
    df_scores: pd.DataFrame,
    sites_df: pd.DataFrame,
    h3_col: str = "h3_r8",
    threshold: float = 0.55,
    min_sites: int = 1,
) -> pd.DataFrame:
    """Aggregate per-site anomaly scores to H3 zone level."""
    # Merge site H3 assignments
    if h3_col not in df_scores.columns:
        if sites_df is not None and h3_col in sites_df.columns:
            df_scores = df_scores.merge(
                sites_df[["site_id", h3_col]], on="site_id", how="left"
            )
        else:
            logger.warning(f"Column {h3_col} not found — skipping geo aggregation")
            return pd.DataFrame()

    # Latest snapshot per site
    latest = df_scores.sort_values("timestamp").groupby("site_id").last().reset_index()

    # H3 aggregation
    agg = latest.groupby(h3_col).agg(
        n_sites=("site_id", "count"),
        avg_anomaly_score=("ensemble_score", "mean"),
        max_anomaly_score=("ensemble_score", "max"),
        n_anomalous_sites=("ensemble_score", lambda x: (x >= threshold).sum()),
    ).reset_index()

    agg = agg[agg["n_sites"] >= min_sites].copy()
    agg["zone_anomaly_rate"] = (agg["n_anomalous_sites"] / agg["n_sites"]).round(3)
    agg["avg_anomaly_score"] = agg["avg_anomaly_score"].round(4)
    agg["max_anomaly_score"] = agg["max_anomaly_score"].round(4)

    # Severity tier
    def severity(score):
        if score < 0.20:   return "normal"
        elif score < 0.40: return "mild"
        elif score < 0.65: return "moderate"
        elif score < 0.85: return "severe"
        return "critical"

    agg["severity"]  = agg["avg_anomaly_score"].apply(severity)
    agg["color"]     = agg["severity"].map(SEVERITY_COLORS)
    agg["is_alert"]  = (agg["avg_anomaly_score"] >= threshold).astype(int)

    return agg


def build_geojson(
    zone_df: pd.DataFrame,
    h3_col: str = "h3_r8",
) -> dict:
    """Convert H3 zone DataFrame to GeoJSON FeatureCollection."""
    try:
        import h3 as h3lib
    except ImportError:
        logger.error("h3 not installed. pip install h3")
        return {"type": "FeatureCollection", "features": []}

    features = []
    for _, row in zone_df.iterrows():
        cell = row[h3_col]
        try:
            boundary = h3lib.h3_to_geo_boundary(cell, geo_json=True)
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [boundary]},
                "properties": {
                    "h3_cell":              cell,
                    "n_sites":              int(row["n_sites"]),
                    "avg_anomaly_score":    float(row["avg_anomaly_score"]),
                    "max_anomaly_score":    float(row["max_anomaly_score"]),
                    "n_anomalous_sites":    int(row["n_anomalous_sites"]),
                    "zone_anomaly_rate":    float(row["zone_anomaly_rate"]),
                    "severity":             row["severity"],
                    "color":                row["color"],
                    "is_alert":             int(row["is_alert"]),
                },
            })
        except Exception:
            pass

    return {"type": "FeatureCollection", "features": features}


def create_folium_map(
    geojson: dict,
    sites_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Build interactive Folium anomaly map."""
    try:
        import folium
    except ImportError:
        logger.warning("folium not installed — skipping interactive map")
        return

    # Compute centre
    if sites_df is not None and "latitude" in sites_df.columns:
        centre = [sites_df["latitude"].mean(), sites_df["longitude"].mean()]
    else:
        centre = [-6.2, 106.85]

    m = folium.Map(location=centre, zoom_start=10, tiles="CartoDB positron")

    # H3 zone polygons
    for feat in geojson["features"]:
        p    = feat["properties"]
        col  = p.get("color", "#888")
        sev  = p.get("severity", "normal")
        n_s  = p.get("n_sites", 0)
        score= p.get("avg_anomaly_score", 0)
        rate = p.get("zone_anomaly_rate", 0)

        folium.GeoJson(
            feat,
            style_function=lambda f, c=col: {
                "fillColor":   c,
                "color":       "#333",
                "weight":      0.4,
                "fillOpacity": 0.65,
            },
            tooltip=folium.Tooltip(
                f"<b>Severity: {sev.upper()}</b><br>"
                f"Avg score: {score:.2f}<br>"
                f"Sites: {n_s} ({rate:.0%} anomalous)"
            ),
        ).add_to(m)

    # Site markers
    if sites_df is not None:
        for _, row in sites_df.iterrows():
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=3, color="#555",
                fill=True, fill_opacity=0.6, weight=0.5,
                tooltip=row["site_id"],
            ).add_to(m)

    # Legend
    legend_html = "".join([
        f'<tr><td style="background:{c};width:14px;height:12px;border-radius:3px;"></td>'
        f'<td style="padding-left:6px;font-size:12px;">{s.title()}</td></tr>'
        for s, c in SEVERITY_COLORS.items()
    ])
    m.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;bottom:25px;left:25px;z-index:1000;'
        f'background:#fff;padding:10px;border-radius:8px;border:1px solid #ccc;">'
        f'<b>Anomaly Severity</b><table style="margin-top:4px;">{legend_html}</table></div>'
    ))

    m.save(str(output_path))
    logger.success(f"Interactive map → {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/config.yaml")
    parser.add_argument("--output",  default="data/processed/anomaly_zones.geojson")
    args = parser.parse_args()

    config    = load_config(args.config)
    threshold = config["geo_impact"]["zone_alert_threshold"]
    h3_col    = f"h3_r{config['data_generation']['h3_resolution']}"

    # Load scores and sites
    scores_path = Path("data/processed/ensemble_scores.parquet")
    sites_path  = Path("data/raw/sites.parquet")

    if not scores_path.exists():
        logger.error("Ensemble scores not found. Run ensemble_detector.py --score")
        sys.exit(1)

    df_scores = pd.read_parquet(scores_path)
    sites_df  = pd.read_parquet(sites_path) if sites_path.exists() else None

    # Aggregate
    zone_df = aggregate_to_h3(df_scores, sites_df, h3_col=h3_col,
                               threshold=threshold, min_sites=config["geo_impact"]["min_sites_per_cell"])

    # Save outputs
    geojson = build_geojson(zone_df, h3_col=h3_col)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(geojson, f)
    logger.success(f"GeoJSON → {args.output}")

    parquet_path = Path(args.output).with_suffix(".parquet")
    zone_df.to_parquet(parquet_path, index=False)

    map_path = Path("data/processed/anomaly_map.html")
    create_folium_map(geojson, sites_df, map_path)

    # Summary
    n_alert = zone_df["is_alert"].sum()
    print(f"\nGeo Impact Map: {len(zone_df)} H3 zones | {n_alert} alert zones")
    print(zone_df["severity"].value_counts().to_string())


if __name__ == "__main__":
    main()
