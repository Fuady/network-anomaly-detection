# %% [markdown]
# # 02 — Feature Engineering for Anomaly Detection
# **Project:** Real-Time Network Anomaly Detection & Proactive Marketing
#
# This notebook covers:
# - Rolling window feature construction and justification
# - Rate-of-change features (first derivatives)
# - Z-score normalisation for online detection
# - Cross-KPI composite features
# - Feature importance analysis for anomaly classification

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

from src.features.kpi_features import (
    add_rolling_features, add_rate_of_change,
    add_zscore_features, add_cross_kpi_features, add_temporal_features
)

# Load a single site for demonstration
df_full = pd.read_parquet("data/raw/network_kpis.parquet")
labels  = pd.read_parquet("data/raw/anomaly_labels.parquet")
df_full = df_full.merge(labels[["site_id","timestamp","is_anomaly","anomaly_type"]],
                         on=["site_id","timestamp"], how="left")

site_id = df_full["site_id"].value_counts().index[0]
df_site = df_full[df_full["site_id"] == site_id].sort_values("timestamp").copy()
print(f"Working with site: {site_id} ({len(df_site):,} readings)")

# %% [markdown]
# ## 1. Rolling Window Features
#
# **Business rationale:** A single anomalous reading could be noise.
# Rolling statistics over 30-60 minutes detect sustained degradations
# and capture the _rate_ of change — critical for early warning.

# %%
df_rolled = add_rolling_features(df_site, windows_min=[5, 15, 30, 60])

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

# Raw throughput vs rolling means
axes[0].plot(df_site["timestamp"], df_site["throughput_mbps"],
             alpha=0.5, color="#888", linewidth=0.8, label="Raw")
for win, color in [(5, "#2ecc71"), (30, "#3498db"), (60, "#e74c3c")]:
    col = f"throughput_mbps_rmean_{win}"
    if col in df_rolled.columns:
        axes[0].plot(df_rolled["timestamp"], df_rolled[col],
                     linewidth=1.5, label=f"Rolling mean {win}min", color=color)

# Annotate anomaly periods
anom_mask = df_rolled["is_anomaly"] == 1
for _, grp in df_rolled[anom_mask].groupby(
    (anom_mask != anom_mask.shift()).cumsum()
):
    if len(grp) > 0:
        axes[0].axvspan(grp["timestamp"].iloc[0], grp["timestamp"].iloc[-1],
                        alpha=0.15, color="red")
axes[0].set_title(f"{site_id}: Throughput with Rolling Means (red=anomaly)")
axes[0].legend(fontsize=8)

# Rolling std (volatility signal)
for win, color in [(15, "#f39c12"), (30, "#9b59b6")]:
    col = f"rsrq_avg_rstd_{win}"
    if col in df_rolled.columns:
        axes[1].plot(df_rolled["timestamp"], df_rolled[col],
                     linewidth=1.2, label=f"RSRQ std {win}min", color=color)
for _, grp in df_rolled[anom_mask].groupby((anom_mask != anom_mask.shift()).cumsum()):
    if len(grp) > 0:
        axes[1].axvspan(grp["timestamp"].iloc[0], grp["timestamp"].iloc[-1],
                        alpha=0.15, color="red")
axes[1].set_title("RSRQ Rolling Std (volatility increases before outage)")
axes[1].legend(fontsize=8)

# Z-score
df_z = add_zscore_features(df_rolled)
if "rsrq_avg_zscore" in df_z.columns:
    axes[2].plot(df_z["timestamp"], df_z["rsrq_avg_zscore"],
                 color="#185FA5", linewidth=1.2, label="RSRQ z-score")
    axes[2].axhline(3, color="red", linestyle="--", linewidth=1, label="z=3 threshold")
    axes[2].axhline(-3, color="red", linestyle="--", linewidth=1)
    for _, grp in df_z[anom_mask].groupby((anom_mask != anom_mask.shift()).cumsum()):
        if len(grp) > 0:
            axes[2].axvspan(grp["timestamp"].iloc[0], grp["timestamp"].iloc[-1],
                            alpha=0.15, color="red")
    axes[2].set_title("RSRQ Z-Score (standardised deviation from rolling mean)")
    axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig("docs/feat_rolling_window.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2. Rate-of-Change Features
#
# **Business rationale:** Network outages start as sudden drops.
# The first derivative (diff) captures onset speed — a steep drop
# in throughput in a single interval is a strong anomaly signal.

# %%
df_roc = add_rate_of_change(df_site, lag_periods=[1, 3, 6])

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].plot(df_site["timestamp"], df_site["throughput_mbps"],
             color="#3498db", linewidth=0.8, alpha=0.7, label="Throughput")
for _, grp in df_site[df_site["is_anomaly"]==1].groupby(
    (df_site["is_anomaly"] != df_site["is_anomaly"].shift()).cumsum()
):
    if len(grp):
        axes[0].axvspan(grp["timestamp"].iloc[0], grp["timestamp"].iloc[-1],
                        alpha=0.15, color="red")
axes[0].set_title("Raw Throughput (red=anomaly)")

if "throughput_mbps_diff_1" in df_roc.columns:
    axes[1].plot(df_roc["timestamp"], df_roc["throughput_mbps_diff_1"],
                 color="#e74c3c", linewidth=1, alpha=0.7, label="1-interval diff")
    axes[1].axhline(0, color="gray", linewidth=0.5, linestyle="--")
    axes[1].set_title("Throughput Rate-of-Change (1-step diff)")
    axes[1].set_ylabel("Mbps/interval")

plt.tight_layout()
plt.savefig("docs/feat_rate_of_change.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Cross-KPI Composite Features

# %%
df_cross = add_cross_kpi_features(df_site)

composites = [
    ("quality_load_stress",    "Quality-Load Stress (RSRQ × PRB)"),
    ("throughput_per_user",    "Throughput per Connected User"),
    ("user_experience_score",  "User Experience Score (0=bad, 1=good)"),
]
available_composites = [(c, l) for c, l in composites if c in df_cross.columns]

if available_composites:
    fig, axes = plt.subplots(len(available_composites), 1,
                             figsize=(14, 4 * len(available_composites)), sharex=True)
    if len(available_composites) == 1:
        axes = [axes]
    anom_mask = df_site["is_anomaly"] == 1

    for ax, (col, label) in zip(axes, available_composites):
        ax.plot(df_cross["timestamp"], df_cross[col],
                color="#9b59b6", linewidth=1, alpha=0.8, label=col)
        for _, grp in df_site[anom_mask].groupby(
            (anom_mask != anom_mask.shift()).cumsum()
        ):
            if len(grp):
                ax.axvspan(grp["timestamp"].iloc[0], grp["timestamp"].iloc[-1],
                           alpha=0.15, color="red")
        ax.set_title(label)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("docs/feat_cross_kpi.png", bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## 4. Feature Importance (Against Ground Truth Labels)

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

# Load full processed features
feat_path = Path("data/processed/features.parquet")
if feat_path.exists():
    df_feat = pd.read_parquet(feat_path)
    non_feature = {"site_id", "timestamp", "is_anomaly", "anomaly_type",
                   "latitude", "longitude", "h3_r7", "h3_r8"}
    feat_cols = [c for c in df_feat.columns
                 if c not in non_feature and pd.api.types.is_numeric_dtype(df_feat[c])]

    sample = df_feat.sample(min(20000, len(df_feat)), random_state=42)
    X = sample[feat_cols].fillna(0)
    y = sample["is_anomaly"].fillna(0).astype(int)

    print(f"Computing feature importance on {len(sample):,} samples...")
    rf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 7))
    imp.head(25).sort_values().plot.barh(ax=ax, color="#e74c3c", alpha=0.85)
    ax.set_title("Top 25 Features by Random Forest Importance (vs. Anomaly Labels)")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("docs/feat_importance.png", bbox_inches="tight")
    plt.show()

    print("\nTop 10 features:")
    print(imp.head(10).round(4).to_string())
else:
    print("Run feature_pipeline.py first to see feature importance.")

print("\n✅ Feature engineering notebook complete.")
