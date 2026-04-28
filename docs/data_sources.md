# Data Sources Guide

## 1. Synthetic Network KPI Data (Auto-generated)

```bash
python src/data_engineering/generate_data.py --n_sites 200 --days 90
```

Generates `data/raw/network_kpis.parquet` — 90 days × 200 sites × 5-min intervals.

| Column | Type | Description |
|---|---|---|
| `site_id` | str | Cell site identifier |
| `timestamp` | datetime | 5-minute interval timestamp |
| `rsrq_avg` | float | Reference Signal Received Quality (dB). Good: > -10, Poor: < -15 |
| `rsrp_avg` | float | Reference Signal Received Power (dBm). Good: > -80, Poor: < -110 |
| `throughput_mbps` | float | Average downlink throughput |
| `latency_ms` | float | Round-trip latency |
| `packet_loss_pct` | float | Packet loss percentage |
| `connected_users` | int | Number of active UEs |
| `prb_utilization` | float | Physical Resource Block utilisation (0–1) |
| `sinr_avg` | float | Signal-to-Interference-plus-Noise Ratio (dB) |

Also generates:
- `data/raw/sites.parquet` — site metadata (lat/lon, urban density, radio type, H3 cells)
- `data/raw/anomaly_labels.parquet` — ground truth anomaly labels with types

---

## 2. OpenCelliD Tower Data (Free — requires free account)

```bash
python src/data_engineering/ingest_opencellid.py --country ID
```

1. Register at https://opencellid.org/register
2. Get token from profile page
3. Add `OPENCELLID_TOKEN=your_token` to `.env`

Used to enrich H3 cells with real tower density — helps distinguish "low signal because no towers nearby" from genuine network degradation.

---

## 3. OSM Road Network (Free — no account)

```bash
# Download via overpass-api.de
python -c "
import requests, json
from pathlib import Path
q = '''[out:json];way['highway']['highway'!='footway'](bbox:-6.5,106.6,-5.9,107.1);(._;>;);out body;'''
r = requests.post('https://overpass-api.de/api/interpreter', data={'data': q}, timeout=120)
Path('data/external/roads_jakarta.json').write_text(r.text)
print('Done')
"
```

Road network data helps contextualise anomalies — sites near major roads may experience predictable congestion spikes vs. true hardware failures.

---

## 4. GADM Administrative Boundaries (Optional)

```bash
wget https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_IDN_2.json.zip
unzip gadm41_IDN_2.json.zip -d data/external/
```

Used to label H3 anomaly zones with district/province names in alerts.
