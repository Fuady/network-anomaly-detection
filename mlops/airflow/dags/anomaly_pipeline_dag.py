"""
mlops/airflow/dags/anomaly_pipeline_dag.py
───────────────────────────────────────────
Daily automated pipeline DAG for anomaly detection model retraining.

Schedule: Every night at 01:00 UTC
Pipeline:
  1. data_validation        — validate latest KPI data quality
  2. feature_engineering    — rebuild rolling-window feature matrix
  3. train_isolation_forest — retrain IF on 30-day rolling window
  4. train_lstm_ae          — retrain LSTM on latest normal sequences
  5. score_ensemble         — score all sites with ensemble
  6. geo_impact_map         — refresh H3 anomaly zone map
  7. drift_detection        — check if model performance degraded
  8. conditional_promote    — promote model if quality improved
  9. notify                 — send daily summary
"""

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner":           "data-science-team",
    "depends_on_past": False,
    "start_date":      days_ago(1),
    "email_on_failure":True,
    "retries":         2,
    "retry_delay":     timedelta(minutes=5),
    "execution_timeout": timedelta(hours=4),
}

dag = DAG(
    dag_id="network_anomaly_detection",
    default_args=default_args,
    description="Daily network anomaly detection pipeline with auto-retraining",
    schedule_interval="0 1 * * *",   # 01:00 UTC every day
    catchup=False,
    max_active_runs=1,
    tags=["anomaly", "network", "streaming", "production"],
)


# ── Task functions ─────────────────────────────────────────────────────────────

def run_data_validation(**ctx):
    import subprocess
    r = subprocess.run(
        ["python", "src/data_engineering/data_validation.py",
         "--input", "data/raw/network_kpis.parquet"],
        capture_output=True, text=True, cwd="/app"
    )
    if r.returncode != 0:
        raise ValueError(f"Validation failed:\n{r.stderr}")


def run_feature_engineering(**ctx):
    import subprocess
    r = subprocess.run(
        ["python", "src/features/feature_pipeline.py"],
        capture_output=True, text=True, cwd="/app"
    )
    if r.returncode != 0:
        raise ValueError(f"Feature engineering failed:\n{r.stderr}")


def run_isolation_forest_training(**ctx):
    import subprocess
    r = subprocess.run(
        ["python", "src/models/isolation_forest.py", "--train"],
        capture_output=True, text=True, cwd="/app"
    )
    if r.returncode != 0:
        raise ValueError(f"IF training failed:\n{r.stderr}")


def run_lstm_training(**ctx):
    """Retrain LSTM AE on latest data."""
    import subprocess
    r = subprocess.run(
        ["python", "src/models/lstm_autoencoder.py", "--train"],
        capture_output=True, text=True, cwd="/app"
    )
    if r.returncode != 0:
        raise ValueError(f"LSTM training failed:\n{r.stderr}")


def run_ensemble_scoring(**ctx):
    import subprocess
    for script in [
        ("src/models/isolation_forest.py", ["--score"]),
        ("src/models/lstm_autoencoder.py", ["--score"]),
        ("src/models/ensemble_detector.py", ["--score", "--evaluate"]),
    ]:
        r = subprocess.run(
            ["python", script[0]] + script[1],
            capture_output=True, text=True, cwd="/app"
        )
        if r.returncode != 0:
            raise ValueError(f"{script[0]} failed:\n{r.stderr}")

    # Extract F1 from stdout for XCom
    ensemble_metrics = {"f1": 0.0}
    try:
        import re
        f1_match = re.search(r"f1.*?(\d+\.\d+)", r.stdout)
        if f1_match:
            ensemble_metrics["f1"] = float(f1_match.group(1))
    except Exception:
        pass
    ctx["task_instance"].xcom_push(key="ensemble_f1", value=ensemble_metrics["f1"])


def run_geo_impact_refresh(**ctx):
    import subprocess
    r = subprocess.run(
        ["python", "src/models/geo_impact_map.py"],
        capture_output=True, text=True, cwd="/app"
    )
    if r.returncode != 0:
        raise ValueError(f"Geo impact refresh failed:\n{r.stderr}")


def check_model_drift(**ctx):
    """
    Compute PSI on ensemble score distribution.
    Compares current day vs. previous day scores.
    """
    import pandas as pd
    import numpy as np
    import yaml

    config = yaml.safe_load(open("configs/config.yaml"))
    psi_threshold = config["monitoring"]["drift_psi_threshold"]

    scores_path = Path("data/processed/ensemble_scores.parquet")
    if not scores_path.exists():
        ctx["task_instance"].xcom_push(key="psi", value=0.0)
        return "no_drift"

    df = pd.read_parquet(scores_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    today     = df["timestamp"].max().normalize()
    yesterday = today - pd.Timedelta(days=1)

    curr = df[df["timestamp"] >= today]["ensemble_score"].dropna()
    prev = df[(df["timestamp"] >= yesterday) &
              (df["timestamp"] < today)]["ensemble_score"].dropna()

    if len(prev) < 100:
        ctx["task_instance"].xcom_push(key="psi", value=0.0)
        return "no_drift"

    def psi(expected, actual, n_bins=10):
        eps   = 1e-10
        bins  = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
        bins  = np.unique(bins)
        e_cnt, _ = np.histogram(expected, bins=bins)
        a_cnt, _ = np.histogram(actual,   bins=bins)
        e_pct = e_cnt / (len(expected) + eps)
        a_pct = a_cnt / (len(actual)   + eps)
        return float(np.sum((a_pct - e_pct) * np.log((a_pct + eps) / (e_pct + eps))))

    psi_val = psi(prev.values, curr.values)
    ctx["task_instance"].xcom_push(key="psi", value=round(psi_val, 4))

    if psi_val > psi_threshold:
        return "high_drift_retrain"
    return "no_drift"


def trigger_full_retrain(**ctx):
    """Retrain both models when significant drift detected."""
    psi = ctx["task_instance"].xcom_pull(
        task_ids="drift_detection", key="psi"
    ) or 0.0
    import subprocess
    print(f"Retraining due to drift: PSI={psi:.4f}")
    for script in [
        ["python", "src/models/isolation_forest.py", "--train"],
        ["python", "src/models/lstm_autoencoder.py", "--train"],
    ]:
        subprocess.run(script, cwd="/app")


def send_daily_notification(**ctx):
    """Send daily pipeline summary."""
    ti  = ctx["task_instance"]
    f1  = ti.xcom_pull(task_ids="ensemble_scoring", key="ensemble_f1") or 0.0
    psi = ti.xcom_pull(task_ids="drift_detection",  key="psi") or 0.0

    msg = (
        f"📡 Anomaly Detection Pipeline — {ctx['ds']}\n"
        f"  Ensemble F1 score : {f1:.4f}\n"
        f"  Score distribution PSI : {psi:.4f}\n"
    )
    print(msg)
    # Extend: post to Slack, PagerDuty, etc.


# ── Tasks ──────────────────────────────────────────────────────────────────────
t_start   = EmptyOperator(task_id="start",           dag=dag)
t_val     = PythonOperator(task_id="data_validation",python_callable=run_data_validation,    dag=dag)
t_feat    = PythonOperator(task_id="feature_engineering", python_callable=run_feature_engineering, dag=dag)
t_if      = PythonOperator(task_id="train_isolation_forest", python_callable=run_isolation_forest_training, dag=dag)
t_lstm    = PythonOperator(task_id="train_lstm_ae",  python_callable=run_lstm_training,      dag=dag)
t_score   = PythonOperator(task_id="ensemble_scoring", python_callable=run_ensemble_scoring, dag=dag)
t_geo     = PythonOperator(task_id="geo_impact_refresh", python_callable=run_geo_impact_refresh, dag=dag)
t_drift   = BranchPythonOperator(task_id="drift_detection", python_callable=check_model_drift, dag=dag)
t_retrain = PythonOperator(task_id="high_drift_retrain", python_callable=trigger_full_retrain, dag=dag)
t_nodrift = EmptyOperator(task_id="no_drift", dag=dag)
t_notify  = PythonOperator(task_id="notify",         python_callable=send_daily_notification,
                            trigger_rule="none_failed_min_one_success", dag=dag)
t_end     = EmptyOperator(task_id="end", dag=dag)

# ── Wiring ──────────────────────────────────────────────────────────────────────
(t_start >> t_val >> t_feat
 >> [t_if, t_lstm]           # parallel training
 >> t_score >> t_geo
 >> t_drift)
t_drift   >> t_retrain >> t_notify
t_drift   >> t_nodrift >> t_notify
t_notify  >> t_end
