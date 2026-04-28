# %% [markdown]
# # 01 — EDA: Network KPI Timeseries & Anomaly Patterns
# **Project:** Real-Time Network Anomaly Detection & Proactive Marketing
#
# This notebook explores:
# - KPI distributions and inter-correlations
# - Daily/weekly seasonality patterns
# - Anomaly event characteristics (duration, severity by type)
# - Geographic distribution of anomalies
# - KPI behaviour during vs. outside anomaly windows

# %%
import sys
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
sns.set_theme(style="whitegrid")
Path("docs").mkdir(exist_ok=True)

# Load data
df      = pd.read_parquet("data/raw/network_kpis.parquet")
labels  = pd.read_parquet("data/raw/anomaly_labels.parquet")
sites   = pd.read_parquet("data/raw/sites.parquet")

df = df.merge(labels[["site_id", "timestamp", "is_anomaly", "anomaly_type"]],
              on=["site_id", "timestamp"], how="left")

print(f"KPI rows      : {len(df):,}")
print(f"Sites         : {df['site_id'].nunique()}")
print(f"Anomaly rate  : {df['is_anomaly'].mean():.2%}")
print(f"Anomaly types : {labels[labels['is_anomaly']==1]['anomaly_type'].value_counts().to_dict()}")

# %% [markdown]
# ## 1. KPI Distribution by Anomaly Status

# %%
KPI_COLS = ["rsrq_avg", "rsrp_avg", "throughput_mbps",
            "latency_ms", "packet_loss_pct", "prb_utilization"]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, col in zip(axes.flat, KPI_COLS):
    for status, color in [(0, "#2ecc71"), (1, "#e74c3c")]:
        subset = df[df["is_anomaly"] == status][col].dropna()
        subset = subset.clip(subset.quantile(0.01), subset.quantile(0.99))
        ax.hist(subset, bins=50, alpha=0.55, color=color, density=True,
                label="Normal" if status == 0 else "Anomaly")
    ax.set_title(col, fontsize=10)
    ax.legend(fontsize=7)

plt.suptitle("KPI Distributions: Normal vs Anomalous Readings", fontsize=13)
plt.tight_layout()
plt.savefig("docs/eda_kpi_distributions.png", bbox_inches="tight")
plt.show()
print("Saved docs/eda_kpi_distributions.png")

# %% [markdown]
# ## 2. Seasonal Patterns (One Representative Site)

# %%
site_id = df["site_id"].value_counts().index[0]
site_df = df[df["site_id"] == site_id].sort_values("timestamp").copy()
site_df["hour"]    = pd.to_datetime(site_df["timestamp"]).dt.hour
site_df["weekday"] = pd.to_datetime(site_df["timestamp"]).dt.day_name()

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Hourly throughput
hourly = site_df.groupby("hour")[["throughput_mbps", "latency_ms"]].mean()
ax = axes[0]
ax.plot(hourly.index, hourly["throughput_mbps"], "b-o", markersize=4, label="Throughput (Mbps)")
ax2 = ax.twinx()
ax2.plot(hourly.index, hourly["latency_ms"], "r--s", markersize=4, label="Latency (ms)")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Throughput (Mbps)", color="blue")
ax2.set_ylabel("Latency (ms)", color="red")
ax.set_title(f"Hourly Pattern — {site_id}")
ax.set_xticks(range(0, 24, 2))
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

# Weekly heatmap
day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
weekly = site_df.groupby(["weekday","hour"])["throughput_mbps"].mean().unstack()
weekly = weekly.reindex([d for d in day_order if d in weekly.index])
sns.heatmap(weekly, ax=axes[1], cmap="YlOrRd_r", cbar_kws={"label": "Mbps"})
axes[1].set_title("Weekly × Hourly Throughput Heatmap")
axes[1].set_xlabel("Hour")

plt.tight_layout()
plt.savefig("docs/eda_seasonality.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Anomaly Event Analysis

# %%
# Duration analysis
anomaly_events = []
for site_id in labels["site_id"].unique()[:20]:  # sample 20 sites
    s = labels[labels["site_id"] == site_id].sort_values("timestamp")
    s["block"] = (s["is_anomaly"] != s["is_anomaly"].shift()).cumsum()
    for _, grp in s.groupby("block"):
        if grp["is_anomaly"].iloc[0] == 1:
            anomaly_events.append({
                "site_id":     site_id,
                "duration_intervals": len(grp),
                "duration_min":       len(grp) * 5,
                "anomaly_type":       grp["anomaly_type"].mode().iloc[0],
                "start":              grp["timestamp"].min(),
            })

if anomaly_events:
    anom_df = pd.DataFrame(anomaly_events)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Duration distribution by type
    type_order = sorted(anom_df["anomaly_type"].unique())
    for atype in type_order:
        subset = anom_df[anom_df["anomaly_type"] == atype]["duration_min"]
        axes[0].hist(subset, bins=30, alpha=0.6, label=atype, density=True)
    axes[0].set_title("Anomaly Duration by Type (minutes)")
    axes[0].set_xlabel("Duration (minutes)")
    axes[0].legend(fontsize=9)

    # Count by type
    type_counts = anom_df["anomaly_type"].value_counts()
    axes[1].bar(type_counts.index, type_counts.values,
                color=["#e74c3c", "#f39c12", "#9b59b6", "#3498db"])
    axes[1].set_title("Anomaly Events by Type")
    axes[1].set_ylabel("Event Count")

    plt.tight_layout()
    plt.savefig("docs/eda_anomaly_events.png", bbox_inches="tight")
    plt.show()

    print(f"Anomaly events (sample): {len(anom_df)}")
    print(f"Avg duration: {anom_df['duration_min'].mean():.1f} min")
    print(f"By type:\n{anom_df['anomaly_type'].value_counts().to_string()}")

# %% [markdown]
# ## 4. KPI Correlation Matrix

# %%
sample_kpis = df[KPI_COLS + ["is_anomaly"]].dropna().sample(min(10000, len(df)))
corr = sample_kpis.corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, ax=ax, linewidths=0.5, annot_kws={"size": 9})
ax.set_title("KPI + Anomaly Label Correlation Matrix")
plt.tight_layout()
plt.savefig("docs/eda_correlation_matrix.png", bbox_inches="tight")
plt.show()

print("\nTop correlations with is_anomaly:")
print(corr["is_anomaly"].drop("is_anomaly").abs().sort_values(ascending=False).to_string())

# %% [markdown]
# ## 5. Geographic Distribution

# %%
try:
    import folium
    m = folium.Map(
        location=[sites["latitude"].mean(), sites["longitude"].mean()],
        zoom_start=9, tiles="CartoDB positron"
    )
    for _, row in sites.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=4, color="#185FA5", fill=True, fill_opacity=0.6, weight=0.5,
            tooltip=row["site_id"],
        ).add_to(m)
    m.save("docs/eda_site_distribution.html")
    print("Site map → docs/eda_site_distribution.html")
except ImportError:
    print("folium not installed — skipping map")

print("\n✅ EDA complete. Charts saved to docs/")
