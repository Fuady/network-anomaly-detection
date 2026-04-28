"""
src/visualization/kpi_plots.py
────────────────────────────────
Reusable chart functions for KPI timeseries and anomaly overlays.
Used by notebooks and the Streamlit dashboard.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.rcParams["figure.dpi"] = 120
sns.set_theme(style="whitegrid")

ANOMALY_COLOR  = "#e74c3c"
NORMAL_COLOR   = "#3498db"
WARNING_COLOR  = "#f39c12"
CRITICAL_COLOR = "#8e44ad"


def plot_kpi_with_anomalies(
    df: pd.DataFrame,
    kpi_col: str,
    score_col: str = "ensemble_score",
    threshold: float = 0.65,
    site_id: Optional[str] = None,
    title: Optional[str] = None,
    figsize: tuple = (14, 5),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot KPI timeseries with anomaly score overlay and shaded anomaly regions."""
    if site_id:
        df = df[df["site_id"] == site_id]
    df = df.sort_values("timestamp")

    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    ax1.plot(df["timestamp"], df[kpi_col], color=NORMAL_COLOR,
             linewidth=1.2, alpha=0.8, label=kpi_col)
    ax1.set_ylabel(kpi_col, color=NORMAL_COLOR)

    if score_col in df.columns:
        ax2.plot(df["timestamp"], df[score_col], color=WARNING_COLOR,
                 linewidth=1, alpha=0.6, linestyle="--", label="Anomaly score")
        ax2.axhline(threshold, color=ANOMALY_COLOR, linestyle=":", linewidth=1.2,
                   label=f"Threshold ({threshold})")
        ax2.fill_between(df["timestamp"], df[score_col], threshold,
                        where=df[score_col] >= threshold,
                        alpha=0.2, color=ANOMALY_COLOR, label="Anomalous")
        ax2.set_ylabel("Anomaly Score", color=WARNING_COLOR)
        ax2.set_ylim(-0.05, 1.1)

    if "is_anomaly" in df.columns:
        anom_mask = df["is_anomaly"] == 1
        for block_id, grp in df[anom_mask].groupby(
            (anom_mask != anom_mask.shift()).cumsum()
        ):
            if len(grp):
                ax1.axvspan(grp["timestamp"].iloc[0], grp["timestamp"].iloc[-1],
                           alpha=0.1, color=ANOMALY_COLOR, zorder=0)

    ax1.set_xlabel("Time")
    ttl = title or f"{kpi_col} — {site_id or 'All Sites'}"
    ax1.set_title(ttl)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig


def plot_kpi_heatmap(
    df: pd.DataFrame,
    kpi_col: str,
    site_sample: int = 20,
    figsize: tuple = (14, 8),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Heatmap of KPI values: sites (rows) × time (columns)."""
    sites = df["site_id"].value_counts().index[:site_sample]
    df_sample = df[df["site_id"].isin(sites)].copy()
    df_sample["hour_bucket"] = pd.to_datetime(df_sample["timestamp"]).dt.floor("1H")

    pivot = (df_sample.groupby(["site_id", "hour_bucket"])[kpi_col]
             .mean()
             .unstack(fill_value=df_sample[kpi_col].median()))
    pivot = pivot.iloc[:, :168]  # first 7 days for display

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(pivot, ax=ax, cmap="RdYlGn", cbar_kws={"label": kpi_col},
                yticklabels=True, xticklabels=False)
    ax.set_title(f"{kpi_col} Heatmap (Sites × Time)")
    ax.set_xlabel("Time (hourly bins)")
    ax.set_ylabel("Site ID")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig


def plot_anomaly_timeline(
    alerts_df: pd.DataFrame,
    figsize: tuple = (14, 4),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Timeline bar chart of alert counts over time, coloured by severity."""
    if len(alerts_df) == 0:
        return None

    df = alerts_df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date

    SEVERITY_COLORS = {
        "mild": "#f1c40f", "moderate": "#e67e22",
        "severe": "#e74c3c", "critical": "#8e44ad",
    }

    fig, ax = plt.subplots(figsize=figsize)
    if "severity" in df.columns:
        for sev, color in SEVERITY_COLORS.items():
            counts = df[df["severity"] == sev].groupby("date").size()
            ax.bar(counts.index, counts.values, label=sev.title(),
                   color=color, alpha=0.85, edgecolor="white")
    else:
        counts = df.groupby("date").size()
        ax.bar(counts.index, counts.values, color=ANOMALY_COLOR, alpha=0.85)

    ax.set_title("Daily Alert Volume by Severity")
    ax.set_xlabel("Date")
    ax.set_ylabel("Alert Count")
    ax.legend(fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig


def plot_score_distribution_comparison(
    df: pd.DataFrame,
    score_cols: dict,
    threshold: float = 0.65,
    figsize: tuple = (12, 5),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Overlapping histograms of anomaly scores for all models."""
    colors = {
        "prophet_score":   "#e74c3c",
        "if_score":        "#3498db",
        "lstm_score":      "#9b59b6",
        "ensemble_score":  "#2ecc71",
    }
    fig, ax = plt.subplots(figsize=figsize)
    for name, col in score_cols.items():
        if col not in df.columns:
            continue
        ax.hist(df[col].dropna(), bins=60, alpha=0.45,
                color=colors.get(col, "#888"), label=name, density=True)

    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.8,
              label=f"Threshold ({threshold})")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.set_title("Anomaly Score Distributions — All Models")
    ax.legend(fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig
