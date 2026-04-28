"""
src/features/kpi_features.py
──────────────────────────────
Time-series feature engineering for network KPI anomaly detection.

For each KPI, computes per-site rolling window statistics:
  - Rolling mean, std, min, max at 5/15/30/60-minute windows
  - Rate of change (1st derivative)
  - Z-score vs recent baseline
  - Cross-KPI ratio features
  - Hour-of-day and day-of-week features (seasonality context)

These features are inputs to the Isolation Forest and LSTM Autoencoder.
Prophet operates directly on the raw KPI timeseries.
"""

import numpy as np
import pandas as pd
from loguru import logger


KPI_COLS = [
    "rsrq_avg", "rsrp_avg", "throughput_mbps",
    "latency_ms", "packet_loss_pct", "connected_users",
    "prb_utilization", "sinr_avg",
]


def add_rolling_features(
    df: pd.DataFrame,
    windows_min: list[int],
    interval_min: int = 5,
    kpi_cols: list[str] = None,
) -> pd.DataFrame:
    """
    Add rolling window statistics for each KPI.
    Operates per-site (groupby site_id).
    """
    if kpi_cols is None:
        kpi_cols = [c for c in KPI_COLS if c in df.columns]

    df = df.sort_values(["site_id", "timestamp"]).copy()

    for win_min in windows_min:
        win_periods = win_min // interval_min  # convert minutes → periods
        if win_periods < 1:
            continue

        logger.info(f"  Rolling window: {win_min} min ({win_periods} periods)")

        for col in kpi_cols:
            if col not in df.columns:
                continue
            grouped = df.groupby("site_id")[col]

            df[f"{col}_rmean_{win_min}"] = grouped.transform(
                lambda x: x.rolling(win_periods, min_periods=1).mean()
            ).round(4)

            df[f"{col}_rstd_{win_min}"] = grouped.transform(
                lambda x: x.rolling(win_periods, min_periods=2).std().fillna(0)
            ).round(4)

            df[f"{col}_rmin_{win_min}"] = grouped.transform(
                lambda x: x.rolling(win_periods, min_periods=1).min()
            ).round(4)

            df[f"{col}_rmax_{win_min}"] = grouped.transform(
                lambda x: x.rolling(win_periods, min_periods=1).max()
            ).round(4)

    return df


def add_rate_of_change(
    df: pd.DataFrame,
    lag_periods: list[int] = [1, 3, 6],
    kpi_cols: list[str] = None,
) -> pd.DataFrame:
    """Add rate-of-change (diff) features at multiple lag periods."""
    if kpi_cols is None:
        kpi_cols = [c for c in KPI_COLS if c in df.columns]

    df = df.sort_values(["site_id", "timestamp"]).copy()

    for lag in lag_periods:
        for col in kpi_cols:
            if col not in df.columns:
                continue
            df[f"{col}_diff_{lag}"] = df.groupby("site_id")[col].transform(
                lambda x: x.diff(lag).fillna(0)
            ).round(4)

    return df


def add_zscore_features(
    df: pd.DataFrame,
    baseline_window_min: int = 60,
    interval_min: int = 5,
    kpi_cols: list[str] = None,
) -> pd.DataFrame:
    """Add z-score of current value vs. recent rolling baseline."""
    if kpi_cols is None:
        kpi_cols = [c for c in KPI_COLS if c in df.columns]

    win = baseline_window_min // interval_min
    df = df.copy()

    for col in kpi_cols:
        if col not in df.columns:
            continue
        rolling_mean = df.groupby("site_id")[col].transform(
            lambda x: x.rolling(win, min_periods=1).mean()
        )
        rolling_std = df.groupby("site_id")[col].transform(
            lambda x: x.rolling(win, min_periods=2).std().fillna(1)
        ).replace(0, 1)

        df[f"{col}_zscore"] = ((df[col] - rolling_mean) / rolling_std).clip(-6, 6).round(3)

    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour-of-day and day-of-week cyclical encodings."""
    df = df.copy()

    ts = pd.to_datetime(df["timestamp"])
    hour = ts.dt.hour + ts.dt.minute / 60.0
    dow  = ts.dt.dayofweek

    # Cyclical encoding: sin/cos of hour (24h cycle) and weekday (7-day cycle)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24).round(4)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24).round(4)
    df["dow_sin"]  = np.sin(2 * np.pi * dow / 7).round(4)
    df["dow_cos"]  = np.cos(2 * np.pi * dow / 7).round(4)
    df["is_weekend"] = (dow >= 5).astype(int)
    df["is_peak_hour"] = (((hour >= 7) & (hour <= 9)) | ((hour >= 18) & (hour <= 22))).astype(int)

    return df


def add_cross_kpi_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features between KPIs that jointly signal anomalies."""
    df = df.copy()

    # Quality-load interaction: low RSRQ + high PRB = congestion signal
    if "rsrq_avg" in df.columns and "prb_utilization" in df.columns:
        df["quality_load_stress"] = (
            (-df["rsrq_avg"] / 20.0) * df["prb_utilization"]
        ).clip(0, 1).round(4)

    # Throughput per user: low = network stress
    if "throughput_mbps" in df.columns and "connected_users" in df.columns:
        df["throughput_per_user"] = (
            df["throughput_mbps"] / (df["connected_users"].clip(1))
        ).round(4)

    # Latency-packet-loss composite degradation score
    if "latency_ms" in df.columns and "packet_loss_pct" in df.columns:
        lat_norm  = (df["latency_ms"].clip(5, 500) - 5) / 495
        loss_norm = df["packet_loss_pct"].clip(0, 20) / 20
        df["user_experience_score"] = (1 - 0.5 * lat_norm - 0.5 * loss_norm).clip(0, 1).round(4)

    # RSRQ deviation from 30-min mean
    if "rsrq_avg" in df.columns and "rsrq_avg_rmean_30" in df.columns:
        df["rsrq_deviation_30"] = (df["rsrq_avg"] - df["rsrq_avg_rmean_30"]).round(4)

    return df


def build_feature_matrix(
    df: pd.DataFrame,
    windows_min: list[int] = [5, 15, 30, 60],
    lag_periods: list[int] = [1, 3, 6],
    interval_min: int = 5,
) -> pd.DataFrame:
    """Full feature engineering pipeline. Returns feature-enriched DataFrame."""
    logger.info(f"Building features for {len(df):,} rows, {df['site_id'].nunique()} sites...")

    df = add_temporal_features(df)
    logger.info("  ✓ Temporal features")

    df = add_rolling_features(df, windows_min, interval_min)
    logger.info("  ✓ Rolling window features")

    df = add_rate_of_change(df, lag_periods)
    logger.info("  ✓ Rate-of-change features")

    df = add_zscore_features(df, interval_min=interval_min)
    logger.info("  ✓ Z-score features")

    df = add_cross_kpi_features(df)
    logger.info("  ✓ Cross-KPI features")

    new_cols = df.shape[1]
    logger.success(f"Feature matrix: {len(df):,} rows × {new_cols} columns")
    return df
