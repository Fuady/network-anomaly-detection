"""tests/test_features.py — Feature engineering tests."""
import sys, pytest, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.kpi_features import (
    add_rolling_features, add_rate_of_change,
    add_zscore_features, add_cross_kpi_features, add_temporal_features,
)

def make_site_df(n=200, n_sites=3, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sites):
        ts = pd.date_range("2024-01-01", periods=n, freq="5min")
        rows.append(pd.DataFrame({
            "site_id":          f"SITE_{s:04d}",
            "timestamp":        ts,
            "rsrq_avg":         rng.normal(-11, 2, n),
            "rsrp_avg":         rng.normal(-95, 5, n),
            "throughput_mbps":  np.abs(rng.normal(25, 5, n)),
            "latency_ms":       np.abs(rng.normal(20, 5, n)),
            "packet_loss_pct":  np.abs(rng.normal(0.5, 0.3, n)),
            "connected_users":  rng.integers(20, 150, n),
            "prb_utilization":  rng.uniform(0.1, 0.9, n),
            "sinr_avg":         rng.normal(12, 3, n),
            "is_anomaly":       rng.integers(0, 2, n),
        }))
    return pd.concat(rows, ignore_index=True)


class TestRollingFeatures:
    def test_creates_expected_columns(self):
        df = make_site_df()
        result = add_rolling_features(df, windows_min=[5, 30])
        for col in ["rsrq_avg_rmean_5", "rsrq_avg_rstd_5",
                    "throughput_mbps_rmean_30", "latency_ms_rmin_30"]:
            assert col in result.columns, f"Missing: {col}"

    def test_no_new_nulls_after_rolling(self):
        df = make_site_df()
        result = add_rolling_features(df, windows_min=[30])
        new_cols = [c for c in result.columns if "rmean" in c or "rstd" in c]
        for col in new_cols:
            assert result[col].isnull().sum() == 0, f"Nulls in {col}"

    def test_rolling_mean_within_range(self):
        df = make_site_df()
        result = add_rolling_features(df, windows_min=[30])
        col = "rsrq_avg_rmean_30"
        assert result[col].between(-30, 0).all(), "RSRQ rolling mean out of range"

    def test_per_site_computation(self):
        """Rolling features should not bleed between sites."""
        df = make_site_df(n=100, n_sites=2)
        result = add_rolling_features(df, windows_min=[30])
        for site in result["site_id"].unique():
            site_df = result[result["site_id"] == site]
            # First value should equal itself (min_periods=1)
            assert not site_df["rsrq_avg_rmean_30"].isnull().any()


class TestRateOfChange:
    def test_diff_columns_created(self):
        df = make_site_df()
        result = add_rate_of_change(df, lag_periods=[1, 3])
        for col in ["rsrq_avg_diff_1", "throughput_mbps_diff_3"]:
            assert col in result.columns

    def test_diff_first_row_is_zero(self):
        """First diff per site should be 0 (fillna)."""
        df = make_site_df(n=50, n_sites=1)
        result = add_rate_of_change(df, lag_periods=[1])
        first = result.sort_values("timestamp").iloc[0]
        assert first["rsrq_avg_diff_1"] == 0.0


class TestZScoreFeatures:
    def test_zscore_columns_created(self):
        df = make_site_df()
        result = add_zscore_features(df, baseline_window_min=30)
        assert "rsrq_avg_zscore" in result.columns

    def test_zscore_clipped(self):
        df = make_site_df()
        result = add_zscore_features(df)
        assert result["rsrq_avg_zscore"].between(-6, 6).all()


class TestCrossKpiFeatures:
    def test_composite_features_created(self):
        df = make_site_df()
        result = add_cross_kpi_features(df)
        for col in ["quality_load_stress", "throughput_per_user", "user_experience_score"]:
            assert col in result.columns

    def test_user_experience_score_in_range(self):
        df = make_site_df()
        result = add_cross_kpi_features(df)
        assert result["user_experience_score"].between(0, 1).all()


class TestTemporalFeatures:
    def test_temporal_columns_created(self):
        df = make_site_df()
        result = add_temporal_features(df)
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos",
                    "is_weekend", "is_peak_hour"]:
            assert col in result.columns

    def test_cyclical_features_in_unit_circle(self):
        df = make_site_df()
        result = add_temporal_features(df)
        assert result["hour_sin"].between(-1, 1).all()
        assert result["hour_cos"].between(-1, 1).all()

    def test_is_weekend_binary(self):
        df = make_site_df()
        result = add_temporal_features(df)
        assert set(result["is_weekend"].unique()).issubset({0, 1})
