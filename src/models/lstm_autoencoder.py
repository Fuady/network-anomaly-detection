"""
src/models/lstm_autoencoder.py
────────────────────────────────
LSTM Autoencoder for time-series reconstruction-based anomaly detection.

Strategy:
  - Trains on normal KPI sequences (length = 12 intervals = 60 minutes)
  - Learns to reconstruct normal sequences with low error
  - High reconstruction error → anomaly
  - Threshold = 95th percentile of training reconstruction errors

Best for: Complex temporal patterns and intermittent fault signatures
          that break normal autocorrelation structure.

Usage:
    python src/models/lstm_autoencoder.py --train
    python src/models/lstm_autoencoder.py --score
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ── PyTorch LSTM Autoencoder ──────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed — LSTM AE will use sklearn fallback")


def load_config(path="configs/config.yaml"):
    with open(path) as f: return yaml.safe_load(f)

def load_params(path="configs/anomaly_params.yaml"):
    with open(path) as f: return yaml.safe_load(f)


# ── Model definition ──────────────────────────────────────────────────────────
class LSTMAutoencoder(nn.Module):
    """Sequence-to-sequence LSTM autoencoder."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        _, (hidden, cell) = self.encoder(x)
        # Repeat bottleneck for decoder
        seq_len = x.size(1)
        hidden_last = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        decoded, _ = self.decoder(hidden_last, (hidden, cell))
        return self.output_layer(decoded)


# ── Sequence builder ──────────────────────────────────────────────────────────
def build_sequences(
    df_site: pd.DataFrame,
    feature_cols: list,
    seq_len: int,
    normal_only: bool = True,
) -> np.ndarray:
    """Build overlapping sequences from a site's KPI timeseries."""
    if normal_only and "is_anomaly" in df_site.columns:
        df_site = df_site[df_site["is_anomaly"] == 0]

    values = df_site[feature_cols].fillna(0).values
    sequences = []
    for i in range(len(values) - seq_len + 1):
        sequences.append(values[i:i + seq_len])
    return np.array(sequences) if sequences else np.empty((0, seq_len, len(feature_cols)))


# ── Training ──────────────────────────────────────────────────────────────────
def train_lstm_ae(
    df: pd.DataFrame,
    params: dict,
    feature_cols: list,
    models_dir: Path,
) -> tuple:
    """Train the LSTM autoencoder."""
    p      = params["lstm_autoencoder"]
    device = torch.device(p.get("device", "cpu")) if TORCH_AVAILABLE else None
    seq_len = p["sequence_length"]

    # Scale features
    scaler = StandardScaler()
    X_all  = df[feature_cols].fillna(0).values
    scaler.fit(X_all[df.get("is_anomaly", pd.Series(0, index=df.index)) == 0]
               if "is_anomaly" in df.columns else X_all)

    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.transform(X_all)

    # Build training sequences (normal only)
    all_seqs = []
    for site_id in df_scaled["site_id"].unique()[:50]:  # sample sites for speed
        site_df = df_scaled[df_scaled["site_id"] == site_id].sort_values("timestamp")
        seqs = build_sequences(site_df, feature_cols, seq_len, normal_only=True)
        if len(seqs):
            all_seqs.append(seqs)

    if not all_seqs:
        logger.error("No training sequences built")
        return None, scaler

    X_train = np.concatenate(all_seqs, axis=0)
    logger.info(f"Training sequences: {X_train.shape}")

    # Train/val split
    split = int(len(X_train) * p.get("train_ratio", 0.85))
    X_tr, X_val = X_train[:split], X_train[split:]

    if not TORCH_AVAILABLE:
        logger.warning("PyTorch unavailable — skipping LSTM training, using dummy model")
        models_dir.mkdir(parents=True, exist_ok=True)
        artifact = {"model": None, "scaler": scaler, "threshold": 1.0,
                    "feature_cols": feature_cols, "seq_len": seq_len}
        joblib.dump(artifact, models_dir / "lstm_autoencoder.pkl")
        return None, scaler

    # Build DataLoaders
    X_tr_t  = torch.FloatTensor(X_tr)
    X_val_t = torch.FloatTensor(X_val)
    tr_loader  = DataLoader(TensorDataset(X_tr_t, X_tr_t),
                            batch_size=p["batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, X_val_t),
                            batch_size=p["batch_size"], shuffle=False)

    model = LSTMAutoencoder(
        input_size=len(feature_cols),
        hidden_size=p["hidden_size"],
        num_layers=p["num_layers"],
        dropout=p["dropout"],
    ).to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    patience_counter = 0

    logger.info(f"Training LSTM AE: {p['epochs']} epochs, patience={p['patience']}")
    for epoch in range(p["epochs"]):
        # Train
        model.train()
        tr_losses = []
        for X_batch, _ in tr_loader:
            X_batch = X_batch.to(device)
            optimiser.zero_grad()
            output = model(X_batch)
            loss = criterion(output, X_batch)
            loss.backward()
            optimiser.step()
            tr_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, _ in val_loader:
                X_batch = X_batch.to(device)
                output  = model(X_batch)
                val_losses.append(criterion(output, X_batch).item())

        tr_loss  = np.mean(tr_losses)
        val_loss = np.mean(val_losses)

        if (epoch + 1) % 10 == 0:
            logger.info(f"  Epoch {epoch+1:3d}: train={tr_loss:.5f}  val={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), models_dir / "lstm_ae_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= p["patience"]:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best weights
    model.load_state_dict(torch.load(models_dir / "lstm_ae_best.pt"))
    model.eval()

    # Compute reconstruction error threshold on training set
    rec_errors = []
    with torch.no_grad():
        for X_batch, _ in tr_loader:
            X_batch = X_batch.to(device)
            output  = model(X_batch)
            errors  = torch.mean((output - X_batch) ** 2, dim=[1, 2])
            rec_errors.extend(errors.cpu().numpy())

    threshold = float(np.percentile(rec_errors, p.get("threshold_percentile", 95)))
    logger.info(f"Reconstruction error threshold: {threshold:.6f}")

    # Save
    models_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model_state": model.state_dict(),
        "model_config": {
            "input_size": len(feature_cols),
            "hidden_size": p["hidden_size"],
            "num_layers": p["num_layers"],
            "dropout": p["dropout"],
        },
        "scaler":       scaler,
        "threshold":    threshold,
        "feature_cols": feature_cols,
        "seq_len":      seq_len,
    }
    joblib.dump(artifact, models_dir / "lstm_autoencoder.pkl")
    logger.success(f"LSTM AE saved → {models_dir / 'lstm_autoencoder.pkl'}")
    return model, scaler


# ── Scoring ───────────────────────────────────────────────────────────────────
def score_lstm_ae(
    df: pd.DataFrame,
    artifact: dict,
) -> pd.DataFrame:
    """Score each timestep with LSTM reconstruction error."""
    feature_cols = artifact["feature_cols"]
    scaler       = artifact["scaler"]
    threshold    = artifact["threshold"]
    seq_len      = artifact["seq_len"]

    if not TORCH_AVAILABLE or artifact.get("model_state") is None:
        df = df.copy()
        df["lstm_score"] = 0.0
        return df

    device = torch.device("cpu")
    cfg    = artifact["model_config"]
    model  = LSTMAutoencoder(**cfg).to(device)
    model.load_state_dict(artifact["model_state"])
    model.eval()

    all_scores = []
    for site_id in df["site_id"].unique():
        site_df = df[df["site_id"] == site_id].sort_values("timestamp").copy()
        X_raw   = site_df[feature_cols].fillna(0).values
        X_scaled = scaler.transform(X_raw)

        # Build sequences and score
        scores = np.zeros(len(X_scaled))
        n_seq  = len(X_scaled) - seq_len + 1

        if n_seq <= 0:
            site_df["lstm_score"] = 0.0
            all_scores.append(site_df)
            continue

        X_seqs = np.array([X_scaled[i:i+seq_len] for i in range(n_seq)])
        X_t    = torch.FloatTensor(X_seqs).to(device)

        with torch.no_grad():
            output = model(X_t)
            rec_err = torch.mean((output - X_t) ** 2, dim=[1, 2]).cpu().numpy()

        # Assign reconstruction error to the LAST timestep of each sequence
        for i, err in enumerate(rec_err):
            scores[i + seq_len - 1] = max(scores[i + seq_len - 1], err)

        # Normalise to [0, 1] using the training threshold as reference
        lstm_score = np.clip(scores / (threshold * 3), 0, 1)
        site_df["lstm_score"] = lstm_score.round(4)
        all_scores.append(site_df)

    return pd.concat(all_scores, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="LSTM Autoencoder anomaly detector")
    parser.add_argument("--train",  action="store_true")
    parser.add_argument("--score",  action="store_true")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--params", default="configs/anomaly_params.yaml")
    args = parser.parse_args()

    config     = load_config(args.config)
    params     = load_params(args.params)
    models_dir = Path("data/models")

    data_path = Path("data/processed/features.parquet")
    if not data_path.exists():
        logger.error("Features not found. Run feature_pipeline.py first.")
        sys.exit(1)

    df = pd.read_parquet(data_path)

    # Use a focused set of raw KPI columns (LSTM works better on less noisy features)
    p = params["lstm_autoencoder"]
    lstm_feats = [
        "rsrq_avg", "rsrp_avg", "throughput_mbps", "latency_ms",
        "packet_loss_pct", "connected_users", "prb_utilization", "sinr_avg",
    ]
    feature_cols = [c for c in lstm_feats if c in df.columns]

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    if args.train:
        with mlflow.start_run(run_name="lstm_autoencoder_training"):
            mlflow.log_params({
                "hidden_size":    p["hidden_size"],
                "num_layers":     p["num_layers"],
                "sequence_length":p["sequence_length"],
                "epochs":         p["epochs"],
                "n_features":     len(feature_cols),
            })
            train_lstm_ae(df, params, feature_cols, models_dir)

    if args.score:
        artifact_path = models_dir / "lstm_autoencoder.pkl"
        if not artifact_path.exists():
            logger.error("No model found — run with --train first")
            sys.exit(1)
        artifact  = joblib.load(artifact_path)
        df_scored = score_lstm_ae(df, artifact)

        threshold = config["anomaly_detection"]["threshold"]
        if "is_anomaly" in df_scored.columns and "lstm_score" in df_scored.columns:
            pred = (df_scored["lstm_score"] >= threshold).astype(int)
            true = df_scored["is_anomaly"].astype(int)
            f1   = f1_score(true, pred, zero_division=0)
            print(f"\nLSTM AE F1@{threshold}: {f1:.4f}")

        out_path = Path("data/processed") / "lstm_scores.parquet"
        df_scored[["site_id", "timestamp", "lstm_score"]].to_parquet(out_path, index=False)
        logger.success(f"LSTM scores saved → {out_path}")


if __name__ == "__main__":
    main()
