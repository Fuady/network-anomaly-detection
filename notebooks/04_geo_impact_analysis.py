# %% [markdown]
# # 04 — Geospatial Impact Analysis & Marketing Trigger
# **Project:** Real-Time Network Anomaly Detection & Proactive Marketing
#
# This notebook:
# - Maps anomaly scores onto the H3 hex grid
# - Identifies geographic clusters of simultaneous degradation
# - Calculates estimated subscriber impact per zone
# - Demonstrates the proactive marketing trigger logic
# - Quantifies the business value (churn prevented, revenue protected)

# %%
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path().resolve()))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
matplotlib.rcParams["figure.dpi"] = 120
Path("docs").mkdir(exist_ok=True)

# Load data
df_scores = pd.read_parquet("data/processed/ensemble_scores.parquet")
sites     = pd.read_parquet("data/raw/sites.parquet")
alerts    = pd.read_parquet("data/processed/alerts.parquet") if Path("data/processed/alerts.parquet").exists() else pd.DataFrame()
zones     = pd.read_parquet("data/processed/anomaly_zones.parquet") if Path("data/processed/anomaly_zones.parquet").exists() else None

# Merge site metadata into scores
if "latitude" not in df_scores.columns:
    df_scores = df_scores.merge(
        sites[["site_id", "latitude", "longitude", "h3_r8", "h3_r7"]],
        on="site_id", how="left"
    )

print(f"Ensemble scores : {len(df_scores):,} rows")
print(f"Alert zones     : {len(zones) if zones is not None else 'N/A'}")
print(f"Marketing alerts: {len(alerts):,}")

# %% [markdown]
# ## 1. Anomaly Score Geographic Distribution

# %%
# Latest score per site
latest = df_scores.sort_values("timestamp").groupby("site_id").last().reset_index()
latest = latest.merge(sites[["site_id","latitude","longitude"]], on="site_id", how="left")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter: anomaly score by lat/lon
sc = axes[0].scatter(latest["longitude"], latest["latitude"],
                     c=latest["ensemble_score"],
                     cmap="RdYlGn_r", s=30, alpha=0.7, vmin=0, vmax=1)
plt.colorbar(sc, ax=axes[0], label="Ensemble Anomaly Score")
axes[0].set_title("Latest Ensemble Anomaly Score by Site")
axes[0].set_xlabel("Longitude"); axes[0].set_ylabel("Latitude")

# Distribution of site anomaly scores
axes[1].hist(latest["ensemble_score"], bins=50, color="#e74c3c", alpha=0.8, edgecolor="white")
axes[1].axvline(0.65, color="k", linestyle="--", linewidth=2, label="Alert threshold (0.65)")
axes[1].set_title("Distribution of Site Anomaly Scores")
axes[1].set_xlabel("Score"); axes[1].set_ylabel("Sites")
axes[1].legend()

plt.tight_layout()
plt.savefig("docs/geo_anomaly_scatter.png", bbox_inches="tight")
plt.show()

high_risk = (latest["ensemble_score"] >= 0.65).sum()
print(f"\nSites above alert threshold: {high_risk} ({high_risk/len(latest):.1%})")

# %% [markdown]
# ## 2. H3 Zone Analysis

# %%
if zones is not None and len(zones) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Zone score distribution by severity
    if "severity" in zones.columns:
        sev_order = ["normal", "mild", "moderate", "severe", "critical"]
        sev_counts = zones["severity"].value_counts().reindex(
            [s for s in sev_order if s in zones["severity"].unique()]
        )
        colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"][:len(sev_counts)]
        axes[0].bar(sev_counts.index, sev_counts.values, color=colors, edgecolor="white")
        axes[0].set_title("H3 Zones by Severity")
        axes[0].set_ylabel("Number of H3 Cells")
        for i, v in enumerate(sev_counts.values):
            axes[0].text(i, v + 1, str(v), ha="center", fontsize=10)

    # Zone anomaly rate distribution
    if "zone_anomaly_rate" in zones.columns:
        axes[1].hist(zones["zone_anomaly_rate"], bins=30,
                     color="#9b59b6", alpha=0.8, edgecolor="white")
        axes[1].set_title("Zone Anomaly Rate Distribution\n(% of sites in zone that are anomalous)")
        axes[1].set_xlabel("Anomaly Rate")
        axes[1].set_ylabel("H3 Zones")

    plt.tight_layout()
    plt.savefig("docs/geo_zone_analysis.png", bbox_inches="tight")
    plt.show()

    print("\nZone summary:")
    print(f"  Total H3 zones  : {len(zones)}")
    if "is_alert" in zones.columns:
        print(f"  Alert zones     : {zones['is_alert'].sum()}")
    if "n_sites" in zones.columns:
        print(f"  Max sites/zone  : {zones['n_sites'].max()}")

# %% [markdown]
# ## 3. Marketing Trigger Business Value Analysis

# %%
if len(alerts) > 0 and "severity" in alerts.columns:
    # Simulate subscriber impact
    rng = np.random.default_rng(42)
    avg_subs_per_site = 80   # average connected users per affected site
    churn_reduction   = 0.23  # 23% reduction in churn among proactively messaged subscribers
    avg_arpu          = 62.0  # USD per month
    offer_cost_map    = {"mild": 1.5, "moderate": 5.0, "severe": 20.0, "critical": 62.0}

    # Estimate impact
    alerts["n_affected_est"]       = avg_subs_per_site
    alerts["offer_cost"]           = alerts["severity"].map(offer_cost_map).fillna(1.5)
    alerts["monthly_revenue_saved"]= (
        alerts["n_affected_est"] * churn_reduction * avg_arpu
    )
    alerts["roi_ratio"] = (
        alerts["monthly_revenue_saved"] / (alerts["offer_cost"] * alerts["n_affected_est"])
    ).round(1)

    print("Marketing Trigger Business Value:")
    print(f"  Total alerts fired      : {len(alerts):,}")
    print(f"  Avg subscribers/alert   : {avg_subs_per_site:,}")
    print(f"  Est. churn reduction    : {churn_reduction:.0%}")
    print(f"  Est. monthly rev saved  : ${alerts['monthly_revenue_saved'].sum():,.0f}")
    print(f"  Total offer cost        : ${(alerts['offer_cost'] * avg_subs_per_site).sum():,.0f}")
    print(f"  Overall ROI             : {alerts['roi_ratio'].mean():.1f}×")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Revenue saved by severity
    rev_by_sev = alerts.groupby("severity")["monthly_revenue_saved"].sum().sort_values(ascending=True)
    colors_sev = [{"normal":"#2ecc71","mild":"#f1c40f","moderate":"#e67e22",
                   "severe":"#e74c3c","critical":"#8e44ad"}.get(s,"#888")
                  for s in rev_by_sev.index]
    axes[0].barh(rev_by_sev.index, rev_by_sev.values / 1000, color=colors_sev)
    axes[0].set_title("Estimated Monthly Revenue Saved by Alert Tier ($K)")
    axes[0].set_xlabel("Revenue ($K/month)")

    # ROI by severity
    roi_by_sev = alerts.groupby("severity")["roi_ratio"].mean().sort_values()
    axes[1].bar(roi_by_sev.index, roi_by_sev.values,
                color=[{"mild":"#f1c40f","moderate":"#e67e22","severe":"#e74c3c",
                        "critical":"#8e44ad"}.get(s,"#888") for s in roi_by_sev.index])
    axes[1].set_title("Average ROI Ratio by Alert Severity")
    axes[1].set_ylabel("ROI (Revenue Saved / Offer Cost)")
    for i, v in enumerate(roi_by_sev.values):
        axes[1].text(i, v + 0.1, f"{v:.1f}×", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig("docs/geo_marketing_value.png", bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## 4. Interactive Folium Map

# %%
map_path = Path("data/processed/anomaly_map.html")
if map_path.exists():
    print(f"Interactive Folium map: {map_path}")
    print("Open in browser or via:\n  streamlit run dashboards/streamlit_app.py")
else:
    print("No Folium map found. Run: python src/models/geo_impact_map.py")

print("\n✅ Geospatial analysis notebook complete.")
