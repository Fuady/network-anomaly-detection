"""
src/features/feature_pipeline.py
──────────────────────────────────
Orchestrates all feature engineering steps for anomaly detection.

Usage:
    python src/features/feature_pipeline.py
    python src/features/feature_pipeline.py --input data/raw/ --output data/processed/
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml
import joblib
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.kpi_features import build_feature_matrix, KPI_COLS
from src.features.geo_features import add_h3_neighbour_features, add_site_metadata_features


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_pipeline(config: dict, input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    interval_min = config["data_generation"]["interval_min"]
    windows_min  = config["features"]["windows_min"]
    lag_periods  = config["features"]["lag_periods"]

    # ── Load raw KPI data ─────────────────────────────────────────────────────
    logger.info("Loading raw KPI data...")
    kpi_path = input_dir / "network_kpis.parquet"
    if not kpi_path.exists():
        logger.error(f"KPI data not found: {kpi_path}")
        logger.error("Run: python src/data_engineering/generate_data.py")
        sys.exit(1)

    df = pd.read_parquet(kpi_path)
    logger.info(f"  {len(df):,} rows, {df['site_id'].nunique()} sites")

    # Load site metadata
    sites_path = input_dir / "sites.parquet"
    sites_df   = pd.read_parquet(sites_path) if sites_path.exists() else None

    # Load ground truth labels (for evaluation)
    labels_path = input_dir / "anomaly_labels.parquet"
    labels_df   = pd.read_parquet(labels_path) if labels_path.exists() else None

    # ── Add site metadata ─────────────────────────────────────────────────────
    if sites_df is not None:
        logger.info("Adding site metadata features...")
        df = add_site_metadata_features(df, sites_df)
        # Merge H3 columns from sites
        h3_cols = [c for c in sites_df.columns if c.startswith("h3_")]
        if h3_cols:
            df = df.merge(sites_df[["site_id"] + h3_cols], on="site_id", how="left")

    # ── KPI features ──────────────────────────────────────────────────────────
    logger.info("Building KPI feature matrix...")
    df = build_feature_matrix(df, windows_min, lag_periods, interval_min)

    # ── H3 neighbour features ─────────────────────────────────────────────────
    if "h3_r8" in df.columns:
        logger.info("Adding H3 neighbour features...")
        df = add_h3_neighbour_features(df, h3_col="h3_r8")

    # ── Merge labels ──────────────────────────────────────────────────────────
    if labels_df is not None:
        df = df.merge(
            labels_df[["site_id", "timestamp", "is_anomaly", "anomaly_type"]],
            on=["site_id", "timestamp"], how="left"
        )
        df["is_anomaly"] = df["is_anomaly"].fillna(0).astype(int)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = output_dir / "features.parquet"
    df.to_parquet(out_path, index=False)
    logger.success(f"Features saved → {out_path}")

    # Save feature column list for model training
    non_feature_cols = {
        "site_id", "timestamp", "is_anomaly", "anomaly_type",
        "latitude", "longitude", "h3_r7", "h3_r8", "cluster_id",
        "radio_type", "urban_density",
    }
    feature_cols = [c for c in df.columns if c not in non_feature_cols
                    and pd.api.types.is_numeric_dtype(df[c])]
    feat_list_path = output_dir / "feature_columns.txt"
    feat_list_path.write_text("\n".join(feature_cols))
    joblib.dump(feature_cols, Path("data/models") / "feature_columns.pkl")

    print("\n" + "=" * 55)
    print("FEATURE PIPELINE COMPLETE")
    print("=" * 55)
    print(f"  Rows           : {len(df):,}")
    print(f"  Feature cols   : {len(feature_cols)}")
    if "is_anomaly" in df.columns:
        print(f"  Anomaly rate   : {df['is_anomaly'].mean():.2%}")
    print(f"  Output         : {out_path.resolve()}")
    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(description="Run feature engineering pipeline")
    parser.add_argument("--input",  default="data/raw")
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    Path("data/models").mkdir(parents=True, exist_ok=True)
    run_pipeline(config, Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
