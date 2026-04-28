"""
src/visualization/geo_plots.py
────────────────────────────────
Geospatial visualization for anomaly zone maps.
Builds interactive Folium maps showing network degradation zones.
"""

from pathlib import Path
from typing import Optional
import json

import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster
from loguru import logger

SEVERITY_COLORS = {
    "normal":   "#2ecc71",
    "mild":     "#f1c40f",
    "moderate": "#e67e22",
    "severe":   "#e74c3c",
    "critical": "#8e44ad",
}


def make_base_map(lat: float = -6.2, lon: float = 106.85, zoom: int = 10) -> folium.Map:
    return folium.Map(location=[lat, lon], zoom_start=zoom, tiles="CartoDB positron")


def add_anomaly_zone_layer(m: folium.Map, geojson: dict) -> folium.Map:
    """Add H3 hex anomaly zones coloured by severity."""
    for feat in geojson.get("features", []):
        p   = feat["properties"]
        col = p.get("color", "#888")
        sev = p.get("severity", "normal")
        n   = p.get("n_sites", 0)
        sc  = p.get("avg_anomaly_score", 0)

        try:
            folium.GeoJson(
                feat,
                style_function=lambda f, c=col: {
                    "fillColor":   c,
                    "color":       "#444",
                    "weight":      0.4,
                    "fillOpacity": 0.65,
                },
                tooltip=folium.Tooltip(
                    f"<b>Severity: {sev.upper()}</b><br>"
                    f"Avg score: {sc:.2f}<br>"
                    f"Sites: {n}"
                ),
            ).add_to(m)
        except Exception:
            pass
    return m


def add_site_markers(
    m: folium.Map,
    sites_df: pd.DataFrame,
    scores_df: Optional[pd.DataFrame] = None,
    threshold: float = 0.65,
) -> folium.Map:
    """Add cell site markers, coloured by anomaly status."""
    if scores_df is not None and "ensemble_score" in scores_df.columns:
        latest = scores_df.sort_values("timestamp").groupby("site_id")["ensemble_score"].last().reset_index()
        sites_df = sites_df.merge(latest, on="site_id", how="left")
        sites_df["ensemble_score"] = sites_df["ensemble_score"].fillna(0)
    else:
        sites_df["ensemble_score"] = 0

    for _, row in sites_df.iterrows():
        score = row.get("ensemble_score", 0)
        if score >= threshold:
            color = "#e74c3c"
            radius = 5
        elif score >= 0.40:
            color = "#f39c12"
            radius = 4
        else:
            color = "#2ecc71"
            radius = 3

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=radius,
            color=color, fill=True, fill_color=color,
            fill_opacity=0.7, weight=0.5,
            tooltip=f"{row['site_id']} | score={score:.2f}",
        ).add_to(m)

    return m


def add_kpi_heatmap(
    m: folium.Map,
    df: pd.DataFrame,
    score_col: str = "ensemble_score",
) -> folium.Map:
    """Add KPI anomaly score heatmap layer."""
    if "latitude" not in df.columns:
        return m

    max_val = df[score_col].quantile(0.95) if score_col in df.columns else 1.0
    heat_data = [
        [row["latitude"], row["longitude"], row.get(score_col, 0) / max(max_val, 1e-6)]
        for _, row in df.iterrows()
        if pd.notna(row.get("latitude")) and pd.notna(row.get("longitude"))
    ]
    if heat_data:
        HeatMap(
            heat_data, radius=12, blur=8, max_zoom=13,
            gradient={"0.0": "#2ecc71", "0.4": "#f39c12",
                      "0.7": "#e74c3c", "1.0": "#8e44ad"},
            name="Anomaly Heatmap",
        ).add_to(m)
    return m


def add_severity_legend(m: folium.Map) -> folium.Map:
    rows = "".join([
        f'<tr><td style="background:{c};width:14px;height:12px;border-radius:3px;"></td>'
        f'<td style="padding-left:6px;font-size:12px;">{s.title()}</td></tr>'
        for s, c in SEVERITY_COLORS.items()
    ])
    m.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;bottom:25px;left:25px;z-index:1000;'
        f'background:#fff;padding:10px;border-radius:8px;border:1px solid #ccc;">'
        f'<b>Anomaly Severity</b>'
        f'<table style="margin-top:4px;">{rows}</table></div>'
    ))
    return m


def create_anomaly_map(
    geojson: dict,
    sites_df: Optional[pd.DataFrame] = None,
    scores_df: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
    include_heatmap: bool = False,
) -> folium.Map:
    """Build a complete layered anomaly map."""
    lats, lons = [], []
    for feat in geojson.get("features", [])[:30]:
        for coord in feat["geometry"]["coordinates"][0]:
            lons.append(coord[0]); lats.append(coord[1])
    centre = [np.mean(lats) if lats else -6.2, np.mean(lons) if lons else 106.85]
    m = make_base_map(centre[0], centre[1])

    m = add_anomaly_zone_layer(m, geojson)
    if sites_df is not None and "latitude" in sites_df.columns:
        m = add_site_markers(m, sites_df.copy(), scores_df)
    if include_heatmap and sites_df is not None and scores_df is not None:
        merged = sites_df.merge(
            scores_df.sort_values("timestamp").groupby("site_id")["ensemble_score"].last().reset_index(),
            on="site_id", how="left"
        )
        m = add_kpi_heatmap(m, merged)

    m = add_severity_legend(m)
    folium.LayerControl(collapsed=False).add_to(m)

    if output_path:
        m.save(str(output_path))
        logger.success(f"Map saved → {output_path}")
    return m
