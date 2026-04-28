"""tests/test_api.py — API schema and model loader tests."""
import sys, pytest, numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.schemas import KPIReading, DetectionResult


SAMPLE_EVENT = {
    "site_id":         "SITE_0001",
    "timestamp":       "2024-01-15T08:30:00Z",
    "rsrq_avg":        -18.5,
    "rsrp_avg":        -105.0,
    "throughput_mbps":  4.2,
    "latency_ms":       85.0,
    "packet_loss_pct":  3.2,
    "connected_users":  145,
    "prb_utilization":  0.85,
    "sinr_avg":         3.5,
    "latitude":         -6.2088,
    "longitude":        106.8456,
}


class TestKPIReadingSchema:
    def test_valid_payload_parses(self):
        r = KPIReading(**SAMPLE_EVENT)
        assert r.site_id == "SITE_0001"
        assert r.rsrq_avg == -18.5

    def test_negative_throughput_rejected(self):
        from pydantic import ValidationError
        bad = {**SAMPLE_EVENT, "throughput_mbps": -5.0}
        with pytest.raises(ValidationError):
            KPIReading(**bad)

    def test_packet_loss_over_100_rejected(self):
        from pydantic import ValidationError
        bad = {**SAMPLE_EVENT, "packet_loss_pct": 105.0}
        with pytest.raises(ValidationError):
            KPIReading(**bad)

    def test_prb_out_of_range_rejected(self):
        from pydantic import ValidationError
        bad = {**SAMPLE_EVENT, "prb_utilization": 1.5}
        with pytest.raises(ValidationError):
            KPIReading(**bad)

    def test_optional_lat_lon_can_be_none(self):
        p = {k: v for k, v in SAMPLE_EVENT.items() if k not in ("latitude", "longitude")}
        r = KPIReading(**p)
        assert r.latitude is None

    def test_defaults_applied(self):
        r = KPIReading(site_id="X", timestamp="2024-01-01T00:00:00Z")
        assert r.rsrq_avg == -11.0
        assert r.throughput_mbps == 25.0


class TestModelLoader:
    def test_not_loaded_initially(self):
        from src.api.model_loader import ModelLoader
        loader = ModelLoader()
        assert not loader.is_loaded()

    def test_get_severity_thresholds(self):
        from src.api.model_loader import ModelLoader
        loader = ModelLoader()
        assert loader._get_severity(0.10) == "normal"
        assert loader._get_severity(0.30) == "mild"
        assert loader._get_severity(0.55) == "moderate"
        assert loader._get_severity(0.75) == "severe"
        assert loader._get_severity(0.90) == "critical"

    def test_score_event_returns_required_keys(self):
        from src.api.model_loader import ModelLoader
        import numpy as np

        loader = ModelLoader()
        # Mock the IF artifact
        mock_model = MagicMock()
        mock_model.decision_function.return_value = np.array([-0.3])
        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.zeros((1, 5))
        loader.if_artifact = {
            "model":        mock_model,
            "scaler":       mock_scaler,
            "feature_cols": ["rsrq_avg", "throughput_mbps", "latency_ms",
                             "packet_loss_pct", "rsrp_avg"],
        }

        result = loader.score_event(SAMPLE_EVENT)
        for k in ["site_id", "timestamp", "anomaly_score", "is_anomaly",
                  "severity", "confidence", "scored_at"]:
            assert k in result, f"Missing key: {k}"

    def test_anomaly_score_in_range(self):
        from src.api.model_loader import ModelLoader
        import numpy as np

        loader = ModelLoader()
        mock_model = MagicMock()
        mock_model.decision_function.return_value = np.array([-0.8])
        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.zeros((1, 5))
        loader.if_artifact = {
            "model": mock_model, "scaler": mock_scaler,
            "feature_cols": ["rsrq_avg", "throughput_mbps", "latency_ms",
                             "packet_loss_pct", "rsrp_avg"],
        }
        result = loader.score_event(SAMPLE_EVENT)
        assert 0.0 <= result["anomaly_score"] <= 1.0


class TestAlertPublisher:
    def test_publish_anomaly_returns_alert(self):
        from src.streaming.alert_publisher import AlertPublisher
        import yaml
        try:
            config = yaml.safe_load(open("configs/config.yaml"))
        except FileNotFoundError:
            config = {"kafka": {}, "marketing": {"offers": {}}}
        pub    = AlertPublisher(config, kafka_mode=False)
        alert  = pub.publish_anomaly("SITE_0001", 0.75, "severe", "2024-01-15T08:30:00Z")
        assert alert is not None
        assert alert["site_id"]  == "SITE_0001"
        assert alert["severity"] == "severe"
        assert "offer_id"        in alert

    def test_rate_limiting_prevents_duplicate(self):
        from src.streaming.alert_publisher import AlertPublisher
        config = {"kafka": {}, "marketing": {"offers": {}}}
        pub    = AlertPublisher(config, kafka_mode=False)
        pub._cooldown_hours = 24

        first  = pub.publish_anomaly("SITE_0001", 0.8, "severe", "ts")
        second = pub.publish_anomaly("SITE_0001", 0.9, "critical", "ts2")
        assert first  is not None
        assert second is None   # rate-limited

    def test_zone_alert_published(self):
        from src.streaming.alert_publisher import AlertPublisher
        config = {"kafka": {}, "marketing": {"offers": {}}}
        pub    = AlertPublisher(config, kafka_mode=False)
        zone_alert = pub.publish_zone_alert(
            h3_zone="abc123", affected_sites=["S1","S2","S3"],
            avg_score=0.75, severity="severe", timestamp="2024-01-15T08:30:00Z"
        )
        assert zone_alert["n_affected_sites"] == 3
        assert zone_alert["h3_zone"]          == "abc123"
