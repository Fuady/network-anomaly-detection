"""
src/models/isolation_forest.py
────────────────────────────────
Isolation Forest multivariate anomaly detector for network KPIs.

Strategy:
  - Trains on rolling-window feature vectors from normal data
  - Isolation Forest isolates anomalies by random feature splits
  - Anomaly score = decision_function output (higher = more anomalous)

Best for: Sudden multivariate anomalies (e.g. ALL KPIs degrade at once = outage).
          Handles high-dimensional feature vectors efficiently.

Usage:
    python src/models/isolation_forest.py --train
    python src/models/isolation_forest.py --score
    python src/models/isolation_forest.py --train --tune
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
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_config(path="configs/config.yaml"):
    with open(path) as f: return yaml.safe_load(f)

def load_params(path="configs/anomaly_params.yaml"):
    with open(path) as f: return yaml.safe_load(f)


def get_feature_cols(df: pd.DataFrame, params: dict) -> list:
    """Get the feature columns to use for Isolation Forest."""
    wanted = params["isolation_forest"].get("feature_set", [])
    return [c for c in wanted if c in df.columns]


def train(
    df: pd.DataFrame,
    params: dict,
    feature_cols: list,
    models_dir: Path,
    use_normal_only: bool = True,
) -> tuple:
    """Train Isolation Forest. Optionally train on normal-only data for better baseline."""
    p = params["isolation_forest"]

    X = df[feature_cols].fillna(0).values

    if use_normal_only and "is_anomaly" in df.columns:
        normal_mask = df["is_anomaly"] == 0
        X_train = df.loc[normal_mask, feature_cols].fillna(0).values
        logger.info(f"Training on {len(X_train):,} normal rows (of {len(df):,} total)")
    else:
        X_train = X
        logger.info(f"Training on {len(X_train):,} rows (mixed)")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = IsolationForest(
        n_estimators=p["n_estimators"],
        contamination=p["contamination"],
        max_samples=p["max_samples"],
        random_state=p["random_state"],
        n_jobs=p["n_jobs"],
    )
    model.fit(X_train_scaled)

    # Save
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "feature_cols": feature_cols},
                models_dir / "isolation_forest.pkl")
    logger.success(f"Isolation Forest saved → {models_dir / 'isolation_forest.pkl'}")
    return model, scaler


def score(
    df: pd.DataFrame,
    artifact: dict,
) -> pd.DataFrame:
    """Score all rows with trained Isolation Forest."""
    model       = artifact["model"]
    scaler      = artifact["scaler"]
    feature_cols = artifact["feature_cols"]

    available = [c for c in feature_cols if c in df.columns]
    X = df[available].fillna(0).values
    X_scaled = scaler.transform(X)

    # decision_function: lower = more anomalous (negative)
    raw_scores = model.decision_function(X_scaled)

    # Convert to [0, 1]: flip sign, scale to positive range
    anomaly_score = (-raw_scores - raw_scores.min()) / (-raw_scores.min() + 1e-6)
    anomaly_score = np.clip(anomaly_score, 0, 1)

    df = df.copy()
    df["if_score"] = anomaly_score.round(4)
    return df


def evaluate(df_scored: pd.DataFrame, threshold: float = 0.65) -> dict:
    """Evaluate against ground truth labels."""
    if "is_anomaly" not in df_scored.columns or "if_score" not in df_scored.columns:
        return {}
    pred = (df_scored["if_score"] >= threshold).astype(int)
    true = df_scored["is_anomaly"].astype(int)
    return {
        "precision": round(precision_score(true, pred, zero_division=0), 4),
        "recall":    round(recall_score(true, pred, zero_division=0), 4),
        "f1":        round(f1_score(true, pred, zero_division=0), 4),
        "threshold": threshold,
    }


def tune_with_optuna(
    df: pd.DataFrame,
    feature_cols: list,
    params: dict,
    n_trials: int = 30,
) -> dict:
    """Auto-tune contamination and n_estimators via Optuna."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if "is_anomaly" not in df.columns:
        logger.warning("No labels for tuning — using default params")
        return params["isolation_forest"]

    X = df[feature_cols].fillna(0).values
    y = df["is_anomaly"].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    def objective(trial):
        contamination = trial.suggest_float("contamination", 0.01, 0.15)
        n_estimators  = trial.suggest_int("n_estimators",   100, 400)
        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=42, n_jobs=-1,
        )
        model.fit(X_scaled[y == 0])
        raw = model.decision_function(X_scaled)
        anom = (-raw - raw.min()) / (-raw.min() + 1e-6)
        pred = (anom >= 0.65).astype(int)
        return f1_score(y, pred, zero_division=0)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=300)
    best = study.best_params
    logger.info(f"Optuna best: {best} → F1={study.best_value:.4f}")
    return {**params["isolation_forest"], **best}


def main():
    parser = argparse.ArgumentParser(description="Isolation Forest anomaly detector")
    parser.add_argument("--train",  action="store_true")
    parser.add_argument("--score",  action="store_true")
    parser.add_argument("--tune",   action="store_true")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--params", default="configs/anomaly_params.yaml")
    args = parser.parse_args()

    config     = load_config(args.config)
    params     = load_params(args.params)
    models_dir = Path("data/models")

    data_path = Path("data/processed/features.parquet")
    if not data_path.exists():
        logger.error("Features not found. Run: python src/features/feature_pipeline.py")
        sys.exit(1)

    df = pd.read_parquet(data_path)
    feature_cols = get_feature_cols(df, params)
    logger.info(f"Feature cols: {len(feature_cols)}")

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    if args.tune:
        tuned_params = tune_with_optuna(df, feature_cols, params)
        params["isolation_forest"].update(tuned_params)

    if args.train:
        with mlflow.start_run(run_name="isolation_forest_training"):
            p = params["isolation_forest"]
            mlflow.log_params({k: v for k, v in p.items() if k != "feature_set"})
            mlflow.log_param("n_features", len(feature_cols))
            model, scaler = train(df, params, feature_cols, models_dir)

    if args.score:
        artifact_path = models_dir / "isolation_forest.pkl"
        if not artifact_path.exists():
            logger.error("No model found — run with --train first")
            sys.exit(1)
        artifact = joblib.load(artifact_path)
        df_scored = score(df, artifact)

        threshold = config["anomaly_detection"]["threshold"]
        metrics   = evaluate(df_scored, threshold)
        logger.info(f"Isolation Forest metrics: {metrics}")

        out_path = Path("data/processed") / "if_scores.parquet"
        df_scored[["site_id", "timestamp", "if_score"]].to_parquet(out_path, index=False)
        logger.success(f"IF scores saved → {out_path}")
        print(f"\nIsolation Forest Evaluation: {metrics}")

        with mlflow.start_run(run_name="isolation_forest_eval"):
            mlflow.log_metrics(metrics)


if __name__ == "__main__":
    main()
