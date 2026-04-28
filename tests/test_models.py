"""tests/test_models.py — Anomaly detection model tests."""
import sys, pytest, numpy as np, pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.isolation_forest import (
    train, score, evaluate, get_feature_cols
)
from src.models.ensemble_detector import (
    compute_ensemble_score, compute_detailed_metrics, generate_alerts
)
from src.models.geo_impact_map import (
    aggregate_to_h3, build_geojson
)


def make_kpi_features(n=1000, n_sites=5, anomaly_rate=0.10, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sites):
        is_anom = rng.random(n) < anomaly_rate
        rows.append(pd.DataFrame({
            "site_id":              f"SITE_{s:04d}",
            "timestamp":            pd.date_range("2024-01-01", periods=n, freq="5min"),
            "rsrq_avg":             rng.normal(-11, 2, n) + np.where(is_anom, -6, 0),
            "rsrp_avg":             rng.normal(-95, 5, n) + np.where(is_anom, -10, 0),
            "throughput_mbps":      np.abs(rng.normal(25, 5, n)) * np.where(is_anom, 0.1, 1.0),
            "latency_ms":           np.abs(rng.normal(20, 3, n)) * np.where(is_anom, 4.0, 1.0),
            "packet_loss_pct":      np.abs(rng.normal(0.5, 0.2, n)) + np.where(is_anom, 5.0, 0),
            "rsrq_avg_rmean_30":    rng.normal(-11, 1, n),
            "rsrq_avg_rstd_30":     np.abs(rng.normal(1.5, 0.5, n)),
            "throughput_mbps_rmean_30": np.abs(rng.normal(25, 3, n)),
            "latency_ms_rmean_30":  np.abs(rng.normal(20, 2, n)),
            "rsrq_rate_of_change":  rng.normal(0, 0.5, n),
            "throughput_rate_of_change": rng.normal(0, 1, n),
            "is_anomaly":           is_anom.astype(int),
        }))
    return pd.concat(rows, ignore_index=True)


class TestIsolationForest:

    def _params(self):
        return {"isolation_forest": {
            "n_estimators":   100,
            "contamination":  0.05,
            "max_samples":    "auto",
            "random_state":   42,
            "n_jobs":         1,
            "feature_set": [
                "rsrq_avg", "throughput_mbps", "latency_ms",
                "rsrq_avg_rmean_30", "rsrq_avg_rstd_30",
            ],
        }}

    def test_train_returns_model_and_scaler(self, tmp_path):
        df = make_kpi_features()
        params = self._params()
        feature_cols = get_feature_cols(df, params)
        model, scaler = train(df, params, feature_cols, tmp_path)
        assert model is not None
        assert scaler is not None

    def test_score_returns_if_score_column(self, tmp_path):
        df = make_kpi_features()
        params = self._params()
        feature_cols = get_feature_cols(df, params)
        model, scaler = train(df, params, feature_cols, tmp_path)
        artifact = {"model": model, "scaler": scaler, "feature_cols": feature_cols}
        df_scored = score(df, artifact)
        assert "if_score" in df_scored.columns

    def test_if_scores_in_range(self, tmp_path):
        df = make_kpi_features()
        params = self._params()
        feature_cols = get_feature_cols(df, params)
        model, scaler = train(df, params, feature_cols, tmp_path)
        artifact = {"model": model, "scaler": scaler, "feature_cols": feature_cols}
        df_scored = score(df, artifact)
        assert df_scored["if_score"].between(0, 1).all()

    def test_evaluate_returns_metrics(self, tmp_path):
        df = make_kpi_features()
        df["if_score"] = 0.0
        df.loc[df["is_anomaly"] == 1, "if_score"] = 0.8
        metrics = evaluate(df, threshold=0.65)
        for k in ["precision", "recall", "f1"]:
            assert k in metrics
            assert 0.0 <= metrics[k] <= 1.0


class TestEnsembleDetector:

    def test_compute_ensemble_score(self):
        n = 100
        df = pd.DataFrame({
            "prophet_score": np.random.uniform(0, 1, n),
            "if_score":      np.random.uniform(0, 1, n),
            "lstm_score":    np.random.uniform(0, 1, n),
        })
        weights = {"prophet": 0.30, "isolation_forest": 0.35, "lstm_autoencoder": 0.35}
        result  = compute_ensemble_score(df, weights)
        assert "ensemble_score" in result.columns
        assert result["ensemble_score"].between(0, 1).all()

    def test_ensemble_score_higher_when_all_models_agree(self):
        df_all_high = pd.DataFrame({
            "prophet_score": [0.9],
            "if_score":      [0.9],
            "lstm_score":    [0.9],
        })
        df_mixed = pd.DataFrame({
            "prophet_score": [0.9],
            "if_score":      [0.1],
            "lstm_score":    [0.5],
        })
        weights = {"prophet": 0.30, "isolation_forest": 0.35, "lstm_autoencoder": 0.35}
        high_score  = compute_ensemble_score(df_all_high, weights)["ensemble_score"].iloc[0]
        mixed_score = compute_ensemble_score(df_mixed,    weights)["ensemble_score"].iloc[0]
        assert high_score > mixed_score

    def test_compute_metrics_keys(self):
        rng = np.random.default_rng(42)
        df  = pd.DataFrame({
            "ensemble_score": rng.uniform(0, 1, 500),
            "is_anomaly":     rng.integers(0, 2, 500),
        })
        metrics = compute_detailed_metrics(df, "ensemble_score", "is_anomaly", 0.5)
        for k in ["precision", "recall", "f1", "roc_auc", "pr_auc", "fpr"]:
            assert k in metrics

    def test_generate_alerts_severity_mapping(self):
        rng = np.random.default_rng(42)
        df  = pd.DataFrame({
            "site_id":           ["SITE_0001"] * 50,
            "timestamp":         pd.date_range("2024-01-01", periods=50, freq="5min"),
            "ensemble_score":    rng.uniform(0.4, 1.0, 50),
            "anomaly_confidence":rng.uniform(0.5, 1.0, 50),
        })
        alert_rules = {
            "severity_tiers": {
                "mild":     {"min_score": 0.20, "max_score": 0.40, "offer_id": "data_1gb"},
                "moderate": {"min_score": 0.40, "max_score": 0.65, "offer_id": "data_5gb"},
                "severe":   {"min_score": 0.65, "max_score": 0.85, "offer_id": "day_pass"},
                "critical": {"min_score": 0.85, "max_score": 1.01, "offer_id": "week_free"},
            }
        }
        alerts = generate_alerts(df, threshold=0.40, alert_rules=alert_rules)
        assert "severity" in alerts.columns
        assert "offer_id" in alerts.columns
        assert alerts["severity"].isin(["mild","moderate","severe","critical"]).all()


class TestGeoImpactMap:

    def test_aggregate_to_h3_structure(self):
        rng = np.random.default_rng(42)
        n   = 200
        df_scores = pd.DataFrame({
            "site_id":        [f"SITE_{i:04d}" for i in range(20)] * (n // 20),
            "timestamp":      pd.date_range("2024-01-01", periods=n, freq="5min"),
            "ensemble_score": rng.uniform(0, 1, n),
            "h3_r8":          rng.choice(["cell_A", "cell_B", "cell_C"], n),
        })
        sites_df = None
        zone_df  = aggregate_to_h3(df_scores, sites_df, h3_col="h3_r8",
                                    threshold=0.65, min_sites=1)
        assert "avg_anomaly_score" in zone_df.columns
        assert "severity" in zone_df.columns
        assert zone_df["avg_anomaly_score"].between(0, 1).all()

    def test_severity_mapping_correct(self):
        df_scores = pd.DataFrame({
            "site_id":        ["S1", "S2"],
            "timestamp":      pd.Timestamp("2024-01-01"),
            "ensemble_score": [0.90, 0.15],
            "h3_r8":          ["cell_X", "cell_Y"],
        })
        zone_df = aggregate_to_h3(df_scores, None, h3_col="h3_r8",
                                   threshold=0.65, min_sites=1)
        # cell_X has score 0.90 → critical
        cell_x = zone_df[zone_df["h3_r8"] == "cell_X"]
        assert cell_x["severity"].iloc[0] == "critical"
        cell_y = zone_df[zone_df["h3_r8"] == "cell_Y"]
        assert cell_y["severity"].iloc[0] == "normal"
