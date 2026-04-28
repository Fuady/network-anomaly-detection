# %% [markdown]
# # 03 — Anomaly Detection Model Comparison
# **Project:** Real-Time Network Anomaly Detection & Proactive Marketing
#
# Compares all three detection approaches:
# - Prophet: seasonal forecasting + interval violation
# - Isolation Forest: multivariate unsupervised outlier detection
# - LSTM Autoencoder: reconstruction-error-based sequential detector
# - Voting Ensemble: weighted combination
#
# Evaluation: Precision, Recall, F1, ROC-AUC, PR-AUC
# Focus: balancing false positives (alert fatigue) vs false negatives (missed events)

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
Path("docs").mkdir(exist_ok=True)

from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay,
)

# Load all scores
print("Loading scores...")
SCORE_FILES = {
    "Prophet":           "data/processed/prophet_scores.parquet",
    "Isolation Forest":  "data/processed/if_scores.parquet",
    "LSTM Autoencoder":  "data/processed/lstm_scores.parquet",
    "Ensemble":          "data/processed/ensemble_scores.parquet",
}
SCORE_COLS = {
    "Prophet":          "prophet_score",
    "Isolation Forest": "if_score",
    "LSTM Autoencoder": "lstm_score",
    "Ensemble":         "ensemble_score",
}

df_base = pd.read_parquet("data/processed/features.parquet")
df_scores = df_base[["site_id", "timestamp"]].copy()
if "is_anomaly" in df_base.columns:
    df_scores["is_anomaly"] = df_base["is_anomaly"].values

for name, path in SCORE_FILES.items():
    p = Path(path)
    if p.exists():
        sc = pd.read_parquet(p)[["site_id", "timestamp", SCORE_COLS[name]]]
        df_scores = df_scores.merge(sc, on=["site_id", "timestamp"], how="left")
        df_scores[SCORE_COLS[name]] = df_scores[SCORE_COLS[name]].fillna(0)
        print(f"  ✓ {name} scores loaded")
    else:
        print(f"  ✗ {name} scores not found — run the training pipeline")

# %% [markdown]
# ## 1. Precision-Recall Trade-off Across Thresholds

# %%
if "is_anomaly" in df_scores.columns:
    THRESHOLD = 0.65
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"Prophet": "#e74c3c", "Isolation Forest": "#3498db",
              "LSTM Autoencoder": "#9b59b6", "Ensemble": "#2ecc71"}

    for name, col in SCORE_COLS.items():
        if col not in df_scores.columns:
            continue
        # PR curve
        prec, rec, _ = precision_recall_curve(df_scores["is_anomaly"], df_scores[col])
        ap = average_precision_score(df_scores["is_anomaly"], df_scores[col])
        lw = 2.5 if name == "Ensemble" else 1.5
        axes[0].plot(rec, prec, color=colors[name], linewidth=lw,
                     label=f"{name} (AP={ap:.3f})")

        # ROC
        fpr, tpr, _ = roc_curve(df_scores["is_anomaly"], df_scores[col])
        auc = roc_auc_score(df_scores["is_anomaly"], df_scores[col])
        axes[1].plot(fpr, tpr, color=colors[name], linewidth=lw,
                     label=f"{name} (AUC={auc:.3f})")

    axes[0].axhline(df_scores["is_anomaly"].mean(), color="k", ls=":", lw=1.5,
                   label=f"Random ({df_scores['is_anomaly'].mean():.3f})")
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
    axes[0].legend(fontsize=8)

    axes[1].plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    axes[1].set(xlabel="FPR", ylabel="TPR", title="ROC Curve")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("docs/model_pr_roc_curves.png", bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## 2. Metrics Table at Operating Threshold

# %%
if "is_anomaly" in df_scores.columns:
    metrics_rows = []
    for name, col in SCORE_COLS.items():
        if col not in df_scores.columns:
            continue
        pred = (df_scores[col] >= THRESHOLD).astype(int)
        true = df_scores["is_anomaly"].astype(int)
        tp = ((pred==1)&(true==1)).sum()
        fp = ((pred==1)&(true==0)).sum()
        fn = ((pred==0)&(true==1)).sum()
        tn = ((pred==0)&(true==0)).sum()
        metrics_rows.append({
            "Model":     name,
            "Precision": round(precision_score(true, pred, zero_division=0), 3),
            "Recall":    round(recall_score(true, pred, zero_division=0), 3),
            "F1":        round(f1_score(true, pred, zero_division=0), 3),
            "ROC-AUC":   round(roc_auc_score(true, df_scores[col]), 3),
            "PR-AUC":    round(average_precision_score(true, df_scores[col]), 3),
            "FPR":       round(fp / (fp+tn+1e-10), 3),
            "TP": int(tp), "FP": int(fp), "FN": int(fn),
        })

    metrics_df = pd.DataFrame(metrics_rows)
    print("\nModel Comparison at threshold =", THRESHOLD)
    print(metrics_df.to_string(index=False))

    # Visualise
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(metrics_df))
    width = 0.2
    for i, (metric, color) in enumerate([
        ("Precision","#3498db"), ("Recall","#2ecc71"), ("F1","#e74c3c")
    ]):
        ax.bar([xi + i*width for xi in x], metrics_df[metric],
               width, label=metric, color=color, alpha=0.85)
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(metrics_df["Model"], fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title(f"Model Comparison (threshold={THRESHOLD})")
    ax.legend()
    plt.tight_layout()
    plt.savefig("docs/model_comparison_bar.png", bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## 3. Anomaly Score Timeline Comparison (One Site)

# %%
site_id = df_scores["site_id"].value_counts().index[0]
site_scores = df_scores[df_scores["site_id"] == site_id].sort_values("timestamp")

n_cols = sum(1 for c in SCORE_COLS.values() if c in site_scores.columns)
if n_cols > 0:
    fig, axes = plt.subplots(n_cols, 1, figsize=(14, 3.5 * n_cols), sharex=True)
    if n_cols == 1:
        axes = [axes]
    ax_i = 0
    colors = {"prophet_score":"#e74c3c", "if_score":"#3498db",
              "lstm_score":"#9b59b6", "ensemble_score":"#2ecc71"}

    for name, col in SCORE_COLS.items():
        if col not in site_scores.columns:
            continue
        ax = axes[ax_i]
        ax.plot(site_scores["timestamp"], site_scores[col],
                color=colors.get(col, "#888"), linewidth=1, alpha=0.8, label=name)
        ax.axhline(THRESHOLD, color="red", linestyle="--", linewidth=1,
                  label=f"Threshold ({THRESHOLD})")
        ax.fill_between(site_scores["timestamp"], site_scores[col], THRESHOLD,
                       where=site_scores[col] >= THRESHOLD,
                       alpha=0.25, color="red")
        if "is_anomaly" in site_scores.columns:
            anom = site_scores["is_anomaly"] == 1
            ax.fill_between(site_scores["timestamp"], 0, 1,
                           where=anom, alpha=0.12, color="orange",
                           label="True anomaly")
        ax.set_ylabel("Score"); ax.legend(fontsize=8)
        ax.set_title(f"{name} — {site_id}")
        ax_i += 1

    plt.tight_layout()
    plt.savefig("docs/model_score_timeline.png", bbox_inches="tight")
    plt.show()

print("\n✅ Model comparison notebook complete.")
