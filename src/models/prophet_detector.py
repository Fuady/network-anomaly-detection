"""
src/models/prophet_detector.py
────────────────────────────────
Facebook Prophet-based seasonal anomaly detector.

Strategy:
  - Train one Prophet model per (site_id, kpi_column) pair
  - Forecasts expected value with confidence interval
  - Points outside the interval are flagged as anomalies
  - Anomaly score = normalised distance from prediction interval

Best for: Gradual degradations that violate expected seasonal patterns.
          Works well when daily/weekly seasonality is strong.

Usage:
    python src/models/prophet_detector.py --train
    python src/models/prophet_detector.py --score
"""

import sys
import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import mlflow
import yaml
from loguru import logger

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_params(path: str = "configs/anomaly_params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train_prophet_for_site_kpi(
    df_site: pd.DataFrame,
    kpi_col: str,
    params: dict,
) -> object:
    """Train a Prophet model for one site×KPI pair."""
    try:
        from prophet import Prophet
    except ImportError:
        logger.error("prophet not installed: pip install prophet")
        return None

    p = params["prophet"]
    model = Prophet(
        daily_seasonality=p["daily_seasonality"],
        weekly_seasonality=p["weekly_seasonality"],
        yearly_seasonality=p["yearly_seasonality"],
        seasonality_mode=p["seasonality_mode"],
        interval_width=p["interval_width"],
        changepoint_prior_scale=p["changepoint_prior_scale"],
        seasonality_prior_scale=p["seasonality_prior_scale"],
    )

    # Prophet expects columns: ds (datetime), y (value)
    train_df = df_site[["timestamp", kpi_col]].rename(
        columns={"timestamp": "ds", kpi_col: "y"}
    ).dropna()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(train_df)

    return model


def score_with_prophet(
    model,
    df_site: pd.DataFrame,
    kpi_col: str,
    interval_width: float = 0.95,
) -> np.ndarray:
    """
    Score a KPI series with a trained Prophet model.
    Returns anomaly scores in [0, 1].
    """
    if model is None:
        return np.zeros(len(df_site))

    future = df_site[["timestamp"]].rename(columns={"timestamp": "ds"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecast = model.predict(future)

    actual   = df_site[kpi_col].values
    yhat     = forecast["yhat"].values
    yhat_lo  = forecast["yhat_lower"].values
    yhat_hi  = forecast["yhat_upper"].values
    interval = (yhat_hi - yhat_lo).clip(1e-6)

    # Score: how far outside the interval is the actual value?
    below = np.maximum(yhat_lo - actual, 0)
    above = np.maximum(actual - yhat_hi, 0)
    deviation = (below + above) / interval

    # Normalise to [0, 1] using sigmoid
    score = 1 / (1 + np.exp(-3 * (deviation - 0.5)))
    return score.clip(0, 1)


def train_all(
    df: pd.DataFrame,
    params: dict,
    kpi_cols: list,
    sample_sites: int = None,
    models_dir: Path = Path("data/models"),
) -> dict:
    """Train Prophet models for all sites × KPIs. Returns {site_id: {kpi: model}}."""
    models_dir.mkdir(parents=True, exist_ok=True)
    p = params["prophet"]
    training_rows = p["training_days"] * (24 * 60 // 5)

    sites = df["site_id"].unique()
    if sample_sites:
        sites = sites[:sample_sites]

    logger.info(f"Training Prophet for {len(sites)} sites × {len(kpi_cols)} KPIs...")
    all_models = {}

    for i, site_id in enumerate(sites):
        site_df = df[df["site_id"] == site_id].sort_values("timestamp")
        train_df = site_df.head(training_rows)
        site_models = {}

        for kpi in kpi_cols:
            if kpi not in df.columns:
                continue
            model = train_prophet_for_site_kpi(train_df, kpi, params)
            site_models[kpi] = model

        all_models[site_id] = site_models
        if (i + 1) % 20 == 0:
            logger.info(f"  Trained {i+1}/{len(sites)} sites")

    # Save
    model_path = models_dir / "prophet_models.pkl"
    joblib.dump(all_models, model_path)
    logger.success(f"Prophet models saved → {model_path}")
    return all_models


def score_all(
    df: pd.DataFrame,
    models: dict,
    kpi_cols: list,
    params: dict,
) -> pd.DataFrame:
    """Score all sites with trained Prophet models."""
    logger.info("Scoring with Prophet models...")
    interval_width = params["prophet"]["interval_width"]
    score_cols = []

    all_scores = []
    for site_id, site_models in models.items():
        site_df = df[df["site_id"] == site_id].sort_values("timestamp").copy()
        kpi_scores = []
        for kpi, model in site_models.items():
            if kpi not in df.columns or model is None:
                continue
            s = score_with_prophet(model, site_df, kpi, interval_width)
            site_df[f"prophet_score_{kpi}"] = s
            kpi_scores.append(s)
            if f"prophet_score_{kpi}" not in score_cols:
                score_cols.append(f"prophet_score_{kpi}")

        # Aggregate score across KPIs (max = most severe anomaly wins)
        if kpi_scores:
            site_df["prophet_score"] = np.max(np.vstack(kpi_scores), axis=0)
        else:
            site_df["prophet_score"] = 0.0

        all_scores.append(site_df)

    return pd.concat(all_scores, ignore_index=True)


def evaluate(df_scored: pd.DataFrame) -> dict:
    """Compute precision/recall against ground truth labels."""
    if "is_anomaly" not in df_scored.columns or "prophet_score" not in df_scored.columns:
        return {}

    threshold = 0.5
    pred = (df_scored["prophet_score"] >= threshold).astype(int)
    true = df_scored["is_anomaly"].astype(int)

    tp = ((pred == 1) & (true == 1)).sum()
    fp = ((pred == 1) & (true == 0)).sum()
    fn = ((pred == 0) & (true == 1)).sum()
    tn = ((pred == 0) & (true == 0)).sum()

    prec = tp / (tp + fp + 1e-10)
    rec  = tp / (tp + fn + 1e-10)
    f1   = 2 * prec * rec / (prec + rec + 1e-10)
    fpr  = fp / (fp + tn + 1e-10)

    return {
        "precision": round(prec, 4),
        "recall":    round(rec, 4),
        "f1":        round(f1, 4),
        "fpr":       round(fpr, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Prophet anomaly detector")
    parser.add_argument("--train",        action="store_true")
    parser.add_argument("--score",        action="store_true")
    parser.add_argument("--sample_sites", type=int, default=None,
                        help="Train on first N sites only (for quick testing)")
    parser.add_argument("--config",  default="configs/config.yaml")
    parser.add_argument("--params",  default="configs/anomaly_params.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    params = load_params(args.params)
    kpi_cols = config["kpi_columns"]
    models_dir = Path("data/models")

    data_path = Path("data/processed/features.parquet")
    if not data_path.exists():
        logger.error(f"Features not found: {data_path}")
        logger.error("Run: python src/features/feature_pipeline.py")
        sys.exit(1)

    df = pd.read_parquet(data_path)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    if args.train:
        with mlflow.start_run(run_name="prophet_training"):
            mlflow.log_params({
                "n_sites": df["site_id"].nunique(),
                "kpi_cols": len(kpi_cols),
                "interval_width": params["prophet"]["interval_width"],
            })
            models = train_all(df, params, kpi_cols, args.sample_sites, models_dir)
            logger.success(f"Trained {len(models)} site models")

    if args.score:
        model_path = models_dir / "prophet_models.pkl"
        if not model_path.exists():
            logger.error("No trained models found — run with --train first")
            sys.exit(1)
        models = joblib.load(model_path)
        df_scored = score_all(df, models, kpi_cols, params)

        # Evaluate
        metrics = evaluate(df_scored)
        if metrics:
            logger.info(f"Prophet metrics: {metrics}")

        out_path = Path("data/processed") / "prophet_scores.parquet"
        df_scored[["site_id", "timestamp", "prophet_score"]].to_parquet(out_path, index=False)
        logger.success(f"Prophet scores saved → {out_path}")
        print(f"\nProphet Evaluation: {metrics}")


if __name__ == "__main__":
    main()
