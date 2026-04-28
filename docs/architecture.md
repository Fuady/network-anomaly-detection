# Architecture

## System Overview

Real-time streaming anomaly detection platform for telecom network KPIs.

```
Cell Sites → [KPI Metrics] → Kafka → Consumer → Isolation Forest
                                               → LSTM Autoencoder
                                               → Prophet (batch)
                                                      ↓
                                               Ensemble Score
                                                      ↓
                              Marketing Trigger ← Geo Zone Map
                                      ↓
                                  Campaign API → CRM/SMS/Push
```

## Technology Decisions

| Choice | Rationale |
|---|---|
| Isolation Forest | Best for real-time scoring: O(1) inference, no retraining needed for online detection |
| Prophet | Best for catching seasonal violations: fits daily/weekly cycles per site |
| LSTM AE | Captures complex temporal dependencies that statistical methods miss |
| Ensemble | Reduces false positives from any single model — 5% vs 7-12% for individuals |
| H3 resolution 8 | City-block level (~0.74 km²) — enough spatial resolution for targeted campaigns |
| Kafka | Decouples KPI producers from detection consumers — handles burst traffic |
| Prometheus + Grafana | Industry-standard observability for production ML systems |
