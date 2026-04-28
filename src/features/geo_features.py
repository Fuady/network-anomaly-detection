"""
src/features/geo_features.py  &  feature_pipeline.py
──────────────────────────────
Geospatial features: H3 neighbour aggregation, tower density, spatial anomaly context.
"""

import numpy as np
import pandas as pd
from loguru import logger


def add_h3_neighbour_features(
    df: pd.DataFrame,
    h3_col: str = "h3_r8",
    kpi_cols: list = None,
) -> pd.DataFrame:
    """
    For each site, compute the avg KPI of its H3 neighbours.
    Helps detect zone-wide degradation vs. single-site issues.
    """
    if h3_col not in df.columns:
        return df

    if kpi_cols is None:
        kpi_cols = ["rsrq_avg", "throughput_mbps", "latency_ms"]

    df = df.copy()
    # H3-cell-level aggregates (proxy for neighbourhood average)
    for col in kpi_cols:
        if col not in df.columns:
            continue
        cell_avg = df.groupby([h3_col, "timestamp"])[col].mean().reset_index()
        cell_avg = cell_avg.rename(columns={col: f"{col}_h3_avg"})
        df = df.merge(cell_avg, on=[h3_col, "timestamp"], how="left")
        df[f"{col}_vs_h3"] = (df[col] - df[f"{col}_h3_avg"]).round(4)

    return df


def add_site_metadata_features(
    df: pd.DataFrame,
    sites_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge site metadata (urban density, max capacity, radio type) into KPI df."""
    if sites_df is None or len(sites_df) == 0:
        return df
    meta_cols = ["site_id", "urban_density", "max_capacity_mbps",
                 "radio_type", "install_year"]
    meta_cols = [c for c in meta_cols if c in sites_df.columns]
    sites_meta = sites_df[meta_cols].copy()

    # Encode categoricals
    if "urban_density" in sites_meta.columns:
        density_map = {"dense": 4, "urban": 3, "suburban": 2, "rural": 1}
        sites_meta["density_score"] = sites_meta["urban_density"].map(density_map).fillna(2)
    if "radio_type" in sites_meta.columns:
        radio_map = {"NR": 3, "LTE+NR": 2, "LTE": 1, "UMTS": 0}
        sites_meta["radio_score"] = sites_meta["radio_type"].map(radio_map).fillna(1)

    drop_cols = ["urban_density", "radio_type"]
    sites_meta = sites_meta.drop(columns=[c for c in drop_cols if c in sites_meta.columns])
    return df.merge(sites_meta, on="site_id", how="left")
