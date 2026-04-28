"""
dashboards/streamlit_app.py
─────────────────────────────
Real-time Network Anomaly Detection & Marketing Dashboard.

Tabs:
  1. 📊 Live Overview    — KPI summary cards, recent anomaly timeline
  2. 🗺️  Geo Map          — H3 zone anomaly severity map (Folium)
  3. 📈 KPI Deep Dive    — Per-site KPI timeseries with anomaly overlay
  4. 🤖 Model Performance — Precision/recall, ROC, model comparison
  5. 📢 Alerts & Offers  — Marketing trigger log

Usage:
    streamlit run dashboards/streamlit_app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import yaml
import joblib
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

st.set_page_config(
    page_title="Network Anomaly Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

SEVERITY_COLORS = {
    "normal":   "#2ecc71",
    "mild":     "#f1c40f",
    "moderate": "#e67e22",
    "severe":   "#e74c3c",
    "critical": "#8e44ad",
}


# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_config():
    try:
        with open("configs/config.yaml") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}


@st.cache_data(ttl=60)
def load_kpis():
    p = Path("data/raw/network_kpis.parquet")
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data(ttl=60)
def load_features():
    p = Path("data/processed/features.parquet")
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data(ttl=60)
def load_ensemble_scores():
    p = Path("data/processed/ensemble_scores.parquet")
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data(ttl=60)
def load_alerts():
    p = Path("data/processed/alerts.parquet")
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=60)
def load_zone_df():
    p = Path("data/processed/anomaly_zones.parquet")
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data(ttl=60)
def load_sites():
    p = Path("data/raw/sites.parquet")
    return pd.read_parquet(p) if p.exists() else None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/network.png", width=55)
    st.title("Anomaly Intelligence")
    st.markdown("---")

    cfg    = load_config()
    df_kpi = load_kpis()
    df_ens = load_ensemble_scores()
    sites  = load_sites()
    alerts = load_alerts()
    zones  = load_zone_df()

    if df_kpi is None:
        st.error("No data found.\n\nRun:\n```\nmake pipeline\n```")
        st.stop()

    n_sites = sites["site_id"].nunique() if sites is not None else df_kpi["site_id"].nunique()
    threshold = cfg.get("anomaly_detection", {}).get("threshold", 0.65)

    st.success(f"✅ {n_sites:,} monitored sites")
    if df_ens is not None and "ensemble_score" in df_ens.columns:
        n_anomalies = (df_ens["ensemble_score"] >= threshold).sum()
        st.error(f"🚨 {n_anomalies:,} anomaly readings") if n_anomalies > 0 else st.success("✅ All systems normal")
    st.markdown("---")
    st.caption("Portfolio Project\nNetwork × Geospatial × ML")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Live Overview", "🗺️ Geo Map", "📈 KPI Deep Dive",
    "🤖 Model Performance", "📢 Alerts & Offers"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Network Anomaly Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Monitored Sites", f"{n_sites:,}")

    if df_ens is not None and "ensemble_score" in df_ens.columns:
        n_anom = (df_ens["ensemble_score"] >= threshold).sum()
        anom_rate = n_anom / len(df_ens) if len(df_ens) > 0 else 0
        col2.metric("Total Anomaly Readings", f"{n_anom:,}", f"{anom_rate:.1%} rate")
    else:
        col2.metric("Anomaly Readings", "—")

    col3.metric("Alert Rules Active", len(cfg.get("marketing", {}).get("offers", {})))
    col4.metric("Alerts Triggered", f"{len(alerts):,}" if len(alerts) > 0 else "0")

    st.markdown("---")

    if df_ens is not None and "ensemble_score" in df_ens.columns:
        # ── Anomaly timeline ──────────────────────────────────────────────────
        col_a, col_b = st.columns(2)
        with col_a:
            if "timestamp" in df_ens.columns:
                ts_anom = (
                    df_ens.groupby("timestamp")["ensemble_score"]
                    .agg(["mean", lambda x: (x >= threshold).mean()])
                    .reset_index()
                )
                ts_anom.columns = ["timestamp", "avg_score", "anomaly_rate"]
                ts_anom = ts_anom.sample(min(2000, len(ts_anom)), random_state=42).sort_values("timestamp")

                fig = px.area(ts_anom, x="timestamp", y="anomaly_rate",
                              title="Anomaly Rate Over Time",
                              labels={"anomaly_rate": "% Sites Anomalous"},
                              color_discrete_sequence=["#e74c3c"])
                fig.add_hline(y=0.15, line_dash="dash", line_color="orange",
                              annotation_text="Alert threshold (15%)")
                st.plotly_chart(fig, use_container_width=True)

        with col_b:
            # Score distribution
            sample = df_ens["ensemble_score"].sample(min(5000, len(df_ens)))
            fig2 = px.histogram(sample, nbins=50,
                                title="Ensemble Anomaly Score Distribution",
                                labels={"value": "Anomaly Score", "count": "Readings"},
                                color_discrete_sequence=["#185FA5"])
            fig2.add_vline(x=threshold, line_dash="dash", line_color="red",
                           annotation_text=f"Threshold ({threshold})")
            st.plotly_chart(fig2, use_container_width=True)

        # ── Severity breakdown ────────────────────────────────────────────────
        def classify_severity(s):
            if s < 0.20:   return "normal"
            elif s < 0.40: return "mild"
            elif s < 0.65: return "moderate"
            elif s < 0.85: return "severe"
            return "critical"

        df_ens["severity"] = df_ens["ensemble_score"].apply(classify_severity)
        sev_counts = df_ens["severity"].value_counts().reset_index()
        sev_counts.columns = ["severity", "count"]
        fig3 = px.pie(sev_counts, values="count", names="severity",
                      title="Readings by Severity",
                      color="severity",
                      color_discrete_map=SEVERITY_COLORS)
        st.plotly_chart(fig3, use_container_width=True)

    # ── KPI summary table ─────────────────────────────────────────────────────
    if df_kpi is not None:
        st.subheader("KPI Summary (Latest Readings)")
        kpi_cols = [c for c in cfg.get("kpi_columns", []) if c in df_kpi.columns]
        if kpi_cols:
            latest = df_kpi.sort_values("timestamp").groupby("site_id")[kpi_cols].last()
            st.dataframe(latest.describe().round(3), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GEO MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Geographic Anomaly Zone Map")

    if zones is not None:
        c1, c2, c3 = st.columns(3)
        n_zones   = len(zones)
        n_alert   = zones.get("is_alert", pd.Series(0)).sum() if "is_alert" in zones.columns else 0
        c1.metric("Total H3 Zones", f"{n_zones:,}")
        c2.metric("Active Alert Zones", f"{n_alert:,}")
        if "avg_anomaly_score" in zones.columns:
            c3.metric("Avg Zone Score", f"{zones['avg_anomaly_score'].mean():.2f}")

        map_html = Path("data/processed/anomaly_map.html")
        if map_html.exists():
            from streamlit.components.v1 import html as st_html
            with open(map_html) as f:
                st_html(f.read(), height=550)
        else:
            st.info("Geo map not generated yet.\n```\npython src/models/geo_impact_map.py\n```")

            if sites is not None and "latitude" in sites.columns and df_ens is not None:
                st.subheader("Fallback: Site Scatter by Anomaly Score")
                latest_scores = df_ens.sort_values("timestamp").groupby("site_id")["ensemble_score"].last().reset_index()
                merged = sites.merge(latest_scores, on="site_id", how="left")
                merged["ensemble_score"] = merged["ensemble_score"].fillna(0)
                fig_map = px.scatter_mapbox(
                    merged, lat="latitude", lon="longitude",
                    color="ensemble_score", size="ensemble_score",
                    color_continuous_scale="RdYlGn_r",
                    zoom=9, height=500, mapbox_style="carto-positron",
                    title="Latest Anomaly Scores per Site",
                )
                st.plotly_chart(fig_map, use_container_width=True)

    else:
        st.warning("Run `python src/models/geo_impact_map.py` to generate zone data.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — KPI DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Per-Site KPI Timeseries")

    if df_kpi is not None:
        site_options = sorted(df_kpi["site_id"].unique())
        selected_site = st.selectbox("Select Site", site_options)

        kpi_options = [c for c in cfg.get("kpi_columns", []) if c in df_kpi.columns]
        selected_kpi = st.selectbox("Select KPI", kpi_options, index=0)

        site_kpi = df_kpi[df_kpi["site_id"] == selected_site].sort_values("timestamp")

        # Merge anomaly scores if available
        has_scores = False
        if df_ens is not None and "ensemble_score" in df_ens.columns:
            site_scores = df_ens[df_ens["site_id"] == selected_site][
                ["timestamp", "ensemble_score"]
            ].sort_values("timestamp")
            if len(site_scores) > 0:
                site_kpi = site_kpi.merge(site_scores, on="timestamp", how="left")
                has_scores = True

        # Downsample for display
        display = site_kpi.sample(min(2000, len(site_kpi)), random_state=42).sort_values("timestamp")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=display["timestamp"], y=display[selected_kpi],
            mode="lines", name=selected_kpi,
            line=dict(color="#3498db", width=1.5),
        ))

        if has_scores and "ensemble_score" in display.columns:
            anomalies = display[display["ensemble_score"] >= threshold]
            fig.add_trace(go.Scatter(
                x=anomalies["timestamp"], y=anomalies[selected_kpi],
                mode="markers", name="Anomaly",
                marker=dict(color="#e74c3c", size=6, symbol="x"),
            ))
            fig.add_trace(go.Scatter(
                x=display["timestamp"], y=display["ensemble_score"] * display[selected_kpi].mean(),
                mode="lines", name="Anomaly Score (scaled)",
                line=dict(color="#f39c12", width=1, dash="dot"),
                yaxis="y2",
            ))
            fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False))

        fig.update_layout(
            title=f"{selected_site} — {selected_kpi}",
            xaxis_title="Time", yaxis_title=selected_kpi,
            height=450, legend=dict(orientation="h"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Stats
        if len(site_kpi) > 0:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(f"Avg {selected_kpi}", f"{site_kpi[selected_kpi].mean():.2f}")
            col2.metric(f"Min {selected_kpi}", f"{site_kpi[selected_kpi].min():.2f}")
            col3.metric(f"Max {selected_kpi}", f"{site_kpi[selected_kpi].max():.2f}")
            if has_scores and "ensemble_score" in site_kpi.columns:
                anom_pct = (site_kpi["ensemble_score"] >= threshold).mean()
                col4.metric("Anomaly %", f"{anom_pct:.1%}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Anomaly Detection Model Performance")

    if df_ens is not None and "is_anomaly" in df_ens.columns:
        from sklearn.metrics import (
            precision_score, recall_score, f1_score,
            roc_auc_score, roc_curve, precision_recall_curve
        )

        models_info = [
            ("Prophet",           "prophet_score"),
            ("Isolation Forest",  "if_score"),
            ("LSTM Autoencoder",  "lstm_score"),
            ("Ensemble",          "ensemble_score"),
        ]

        metrics_rows = []
        for name, score_col in models_info:
            if score_col not in df_ens.columns:
                continue
            pred = (df_ens[score_col] >= threshold).astype(int)
            true = df_ens["is_anomaly"].astype(int)
            try:
                row = {
                    "Model":     name,
                    "Precision": round(precision_score(true, pred, zero_division=0), 3),
                    "Recall":    round(recall_score(true, pred, zero_division=0), 3),
                    "F1":        round(f1_score(true, pred, zero_division=0), 3),
                    "ROC-AUC":   round(roc_auc_score(true, df_ens[score_col]), 3) if true.nunique() > 1 else 0.5,
                }
                metrics_rows.append(row)
            except Exception:
                pass

        if metrics_rows:
            metrics_df = pd.DataFrame(metrics_rows)
            st.dataframe(metrics_df, use_container_width=True)

            # ROC curves
            fig_roc = go.Figure()
            for name, score_col in models_info:
                if score_col not in df_ens.columns:
                    continue
                try:
                    fpr, tpr, _ = roc_curve(df_ens["is_anomaly"], df_ens[score_col])
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=name))
                except Exception:
                    pass
            fig_roc.add_shape(type="line", x0=0, x1=1, y0=0, y1=1,
                              line=dict(dash="dash", color="gray"))
            fig_roc.update_layout(
                title="ROC Curves — All Models",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=400,
            )
            st.plotly_chart(fig_roc, use_container_width=True)
    else:
        st.info("Run the full pipeline to see model performance:\n```\nmake pipeline\n```")

        # Static reference table from docs
        ref_data = {
            "Model":     ["Prophet", "Isolation Forest", "LSTM AE", "Ensemble"],
            "Precision": [0.71, 0.79, 0.82, 0.86],
            "Recall":    [0.84, 0.76, 0.78, 0.81],
            "F1":        [0.77, 0.77, 0.80, 0.83],
            "FPR":       [0.12, 0.08, 0.07, 0.05],
        }
        st.dataframe(pd.DataFrame(ref_data), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ALERTS & OFFERS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Anomaly Alerts & Marketing Offers")

    if len(alerts) > 0:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Alerts", f"{len(alerts):,}")
        if "severity" in alerts.columns:
            col2.metric("Critical Alerts", (alerts["severity"] == "critical").sum())
        if "offer_id" in alerts.columns:
            col3.metric("Unique Offers Sent", alerts["offer_id"].nunique())

        st.markdown("---")

        # Alert severity chart
        if "severity" in alerts.columns:
            sev_counts = alerts["severity"].value_counts().reset_index()
            sev_counts.columns = ["severity", "count"]
            fig_sev = px.bar(sev_counts, x="severity", y="count",
                             title="Alerts by Severity",
                             color="severity",
                             color_discrete_map=SEVERITY_COLORS)
            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(fig_sev, use_container_width=True)
            with col_b:
                if "offer_id" in alerts.columns:
                    offer_counts = alerts["offer_id"].value_counts().reset_index()
                    offer_counts.columns = ["offer", "count"]
                    fig_off = px.pie(offer_counts, values="count", names="offer",
                                     title="Offers Triggered by Type")
                    st.plotly_chart(fig_off, use_container_width=True)

        st.subheader("Recent Alerts (Last 50)")
        display_cols = [c for c in ["site_id", "timestamp", "ensemble_score",
                                     "severity", "offer_id", "channel", "alert_time"]
                        if c in alerts.columns]
        st.dataframe(
            alerts.sort_values("timestamp", ascending=False).head(50)[display_cols],
            use_container_width=True,
        )
    else:
        st.info("No alerts yet. Run the full pipeline:\n```\nmake pipeline\n```")
        st.markdown("**Offer tiers when alerts fire:**")
        offers = cfg.get("marketing", {}).get("offers", {})
        if offers:
            for tier, cfg_tier in offers.items():
                st.markdown(f"- **{tier.title()}** (score ≥ {cfg_tier['threshold']}): "
                            f"{cfg_tier['message']}")
