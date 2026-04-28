"""
src/models/ensemble_detector.py
──────────────────────────────────
Voting ensemble combining Prophet + Isolation Forest + LSTM Autoencoder.

Ensemble strategy:
  - Weighted average of 3 anomaly scores
  - Final anomaly label by threshold on ensemble score
  - Confidence = max model agreement (all 3 agree = high confidence)

Usage:
    python src/models/ensemble_detector.py --train   # trains all 3 models
    python src/models/ensemble_detector.py --score   # scores and saves results
    python src/models/ensemble_detector.py --evaluate
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import mlflow
import yaml
from loguru import logger
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    classification_report,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_config(path="configs/config.yaml"):
    with open(path) as f: return yaml.safe_load(f)

def load_params(path="configs/anomaly_params.yaml"):
    with open(path) as f: return yaml.safe_load(f)


def merge_scores(
    df_base: pd.DataFrame,
    score_files: dict,
) -> pd.DataFrame:
    """Merge individual model scores into one DataFrame."""
    df = df_base[["site_id", "timestamp"]].copy()
    if "is_anomaly" in df_base.columns:
        df["is_anomaly"] = df_base["is_anomaly"].values

    for score_col, path in score_files.items():
        p = Path(path)
        if p.exists():
            score_df = pd.read_parquet(p)[["site_id", "timestamp", score_col]]
            df = df.merge(score_df, on=["site_id", "timestamp"], how="left")
            df[score_col] = df[score_col].fillna(0.0)
        else:
            logger.warning(f"Score file not found: {path} — using 0")
            df[score_col] = 0.0

    return df


def compute_ensemble_score(
    df: pd.DataFrame,
    weights: dict,
) -> pd.DataFrame:
    """Compute weighted ensemble anomaly score."""
    df = df.copy()
    score_cols = {
        "prophet_score": weights.get("prophet", 0.30),
        "if_score":      weights.get("isolation_forest", 0.35),
        "lstm_score":    weights.get("lstm_autoencoder", 0.35),
    }

    total_w = 0.0
    ensemble = np.zeros(len(df))
    for col, w in score_cols.items():
        if col in df.columns:
            ensemble += w * df[col].fillna(0).values
            total_w  += w

    df["ensemble_score"] = (ensemble / max(total_w, 1e-6)).clip(0, 1).round(4)

    # Confidence: standard deviation of the 3 scores (low std = high agreement)
    avail_cols = [c for c in score_cols if c in df.columns]
    if len(avail_cols) > 1:
        stacked = np.vstack([df[c].fillna(0).values for c in avail_cols])
        df["anomaly_confidence"] = (1 - np.std(stacked, axis=0)).clip(0, 1).round(4)
    else:
        df["anomaly_confidence"] = df["ensemble_score"].clip(0, 1)

    return df


def compute_detailed_metrics(
    df: pd.DataFrame,
    score_col: str,
    label_col: str,
    threshold: float,
) -> dict:
    """Full evaluation metrics for one model/ensemble."""
    if label_col not in df.columns or score_col not in df.columns:
        return {}
    pred = (df[score_col] >= threshold).astype(int)
    true = df[label_col].astype(int)
    tp = ((pred == 1) & (true == 1)).sum()
    fp = ((pred == 1) & (true == 0)).sum()
    fn = ((pred == 0) & (true == 1)).sum()
    tn = ((pred == 0) & (true == 0)).sum()
    return {
        "precision":      round(precision_score(true, pred, zero_division=0), 4),
        "recall":         round(recall_score(true, pred, zero_division=0), 4),
        "f1":             round(f1_score(true, pred, zero_division=0), 4),
        "roc_auc":        round(roc_auc_score(true, df[score_col]), 4) if true.nunique() > 1 else 0.5,
        "pr_auc":         round(average_precision_score(true, df[score_col]), 4) if true.nunique() > 1 else 0.0,
        "fpr":            round(fp / (fp + tn + 1e-10), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def generate_alerts(
    df_scored: pd.DataFrame,
    threshold: float,
    alert_rules: dict,
) -> pd.DataFrame:
    """Convert anomaly detections into structured alert records."""
    anomalies = df_scored[df_scored["ensemble_score"] >= threshold].copy()

    def get_severity(score: float) -> str:
        tiers = alert_rules.get("severity_tiers", {})
        for tier, cfg in tiers.items():
            if cfg["min_score"] <= score < cfg["max_score"]:
                return tier
        return "critical"

    def get_offer(severity: str) -> dict:
        tiers = alert_rules.get("severity_tiers", {})
        tier  = tiers.get(severity, {})
        return {"offer_id": tier.get("offer_id", "data_1gb"),
                "offer_name": tier.get("offer_name", "Data Bonus"),
                "channel": tier.get("channel", "push")}

    anomalies["severity"]   = anomalies["ensemble_score"].apply(get_severity)
    anomalies["offer_id"]   = anomalies["severity"].apply(lambda s: get_offer(s)["offer_id"])
    anomalies["channel"]    = anomalies["severity"].apply(lambda s: get_offer(s)["channel"])
    anomalies["alert_time"] = pd.Timestamp.now()

    return anomalies[["site_id", "timestamp", "ensemble_score",
                       "anomaly_confidence", "severity", "offer_id",
                       "channel", "alert_time"]]


def main():
    parser = argparse.ArgumentParser(description="Ensemble anomaly detector")
    parser.add_argument("--train",    action="store_true", help="Train all 3 sub-models")
    parser.add_argument("--score",    action="store_true", help="Run ensemble scoring")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate against labels")
    parser.add_argument("--config",   default="configs/config.yaml")
    parser.add_argument("--params",   default="configs/anomaly_params.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    params = load_params(args.params)
    weights = config["anomaly_detection"]["ensemble_weights"]
    threshold = config["anomaly_detection"]["threshold"]
    models_dir = Path("data/models")

    data_path = Path("data/processed/features.parquet")
    if not data_path.exists():
        logger.error("Features not found. Run feature_pipeline.py first.")
        sys.exit(1)

    # ── Train all sub-models ──────────────────────────────────────────────────
    if args.train:
        logger.info("Training all sub-models...")
        import subprocess, sys as _sys
        for script, extra_args in [
            ("src/models/isolation_forest.py", ["--train"]),
            ("src/models/lstm_autoencoder.py", ["--train"]),
            ("src/models/prophet_detector.py", ["--train", "--sample_sites", "50"]),
        ]:
            logger.info(f"  Running {script}...")
            result = subprocess.run(
                [_sys.executable, script] + extra_args,
                capture_output=False,
            )
            if result.returncode != 0:
                logger.warning(f"  {script} had non-zero exit — continuing")

    # ── Score ─────────────────────────────────────────────────────────────────
    if args.score:
        df = pd.read_parquet(data_path)

        # Run individual model scoring
        import subprocess, sys as _sys
        for script in [
            "src/models/isolation_forest.py",
            "src/models/lstm_autoencoder.py",
        ]:
            subprocess.run([_sys.executable, script, "--score"], capture_output=False)

        # Merge all model scores
        score_files = {
            "prophet_score": "data/processed/prophet_scores.parquet",
            "if_score":      "data/processed/if_scores.parquet",
            "lstm_score":    "data/processed/lstm_scores.parquet",
        }
        df_merged = merge_scores(df, score_files)
        df_ensemble = compute_ensemble_score(df_merged, weights)

        # Save ensemble scores
        out_path = Path("data/processed") / "ensemble_scores.parquet"
        df_ensemble.to_parquet(out_path, index=False)
        logger.success(f"Ensemble scores → {out_path}")

        # Save alerts
        try:
            import yaml
            with open("configs/alert_rules.yaml") as f:
                alert_rules = yaml.safe_load(f)
        except FileNotFoundError:
            alert_rules = {}
        alerts = generate_alerts(df_ensemble, threshold, alert_rules)
        alerts_path = Path("data/processed") / "alerts.parquet"
        alerts.to_parquet(alerts_path, index=False)
        logger.success(f"Alerts saved ({len(alerts):,} events) → {alerts_path}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    if args.evaluate:
        scores_path = Path("data/processed") / "ensemble_scores.parquet"
        if not scores_path.exists():
            logger.error("No ensemble scores. Run with --score first.")
            sys.exit(1)

        df_scored = pd.read_parquet(scores_path)
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])

        with mlflow.start_run(run_name="ensemble_evaluation"):
            for score_col, name in [
                ("prophet_score", "prophet"),
                ("if_score",      "isolation_forest"),
                ("lstm_score",    "lstm_autoencoder"),
                ("ensemble_score","ensemble"),
            ]:
                m = compute_detailed_metrics(df_scored, score_col, "is_anomaly", threshold)
                if m:
                    logger.info(f"{name}: {m}")
                    mlflow.log_metrics({f"{name}_{k}": v for k, v in m.items()
                                        if isinstance(v, float)})

            # Best model summary
            ens_metrics = compute_detailed_metrics(
                df_scored, "ensemble_score", "is_anomaly", threshold
            )
            print("\n" + "=" * 60)
            print("ENSEMBLE EVALUATION RESULTS")
            print("=" * 60)
            for k, v in ens_metrics.items():
                print(f"  {k:20s}: {v}")
            print("=" * 60)


if __name__ == "__main__":
    main()
