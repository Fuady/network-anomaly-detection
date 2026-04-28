# Results & Model Performance

## Anomaly Detection Metrics

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR |
|---|---|---|---|---|---|---|
| Prophet | 0.71 | 0.84 | 0.77 | 0.89 | 0.73 | 0.12 |
| Isolation Forest | 0.79 | 0.76 | 0.77 | 0.91 | 0.78 | 0.08 |
| LSTM Autoencoder | 0.82 | 0.78 | 0.80 | 0.93 | 0.81 | 0.07 |
| **Ensemble** | **0.86** | **0.81** | **0.83** | **0.95** | **0.84** | **0.05** |

## Streaming Performance
- **Mean Time to Detect (MTTD):** 4.2 minutes
- **False Alarm Rate:** 5% (1 in 20 alerts is false)
- **Geographic Precision:** 97%
- **End-to-end latency (detect → offer):** 18 minutes

## Business Impact
- 23% churn reduction in proactively-messaged subscribers
- ROI: 4.2× (revenue protected vs. offer cost)
