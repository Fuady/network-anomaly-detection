# ─────────────────────────────────────────────────────────────
# Makefile — Network Anomaly Detection Project
# ─────────────────────────────────────────────────────────────

.PHONY: help install setup data validate features train score evaluate \
        geo-map pipeline stream-demo stream-live api dashboard mlflow \
        test test-cov lint format docker-up docker-down clean

PYTHON := python

help:
	@echo ""
	@echo "  network-anomaly-detection"
	@echo "  ──────────────────────────────────────────────────"
	@echo "  Setup"
	@echo "    make install       Install Python dependencies"
	@echo "    make setup         First-time project setup"
	@echo ""
	@echo "  Batch Pipeline (run in order)"
	@echo "    make data          Generate 90d × 200-site KPI data"
	@echo "    make validate      Validate raw data quality"
	@echo "    make features      Build rolling-window feature matrix"
	@echo "    make train         Train all 3 anomaly detectors"
	@echo "    make score         Score ensemble + evaluate"
	@echo "    make geo-map       Generate H3 anomaly zone map"
	@echo "    make pipeline      Run ALL steps end-to-end"
	@echo ""
	@echo "  Streaming"
	@echo "    make stream-demo   Run streaming demo (no Kafka needed)"
	@echo "    make stream-live   Start Kafka consumer (requires Docker)"
	@echo "    make produce       Start KPI event producer"
	@echo ""
	@echo "  Serving"
	@echo "    make api           Start FastAPI at :8000"
	@echo "    make dashboard     Start Streamlit at :8501"
	@echo "    make mlflow        Start MLflow UI at :5000"
	@echo ""
	@echo "  Notebooks"
	@echo "    make notebook      Launch Jupyter Lab"
	@echo "    make run-notebooks Run all notebooks as scripts"
	@echo ""
	@echo "  Quality"
	@echo "    make test          Run pytest suite"
	@echo "    make test-cov      Tests with coverage"
	@echo "    make lint          flake8 + black check"
	@echo ""
	@echo "  Docker (includes Kafka + Prometheus + Grafana)"
	@echo "    make docker-up     Start full stack"
	@echo "    make docker-down   Stop all containers"
	@echo ""
	@echo "    make clean         Remove Python cache"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────
install:
	pip install -r requirements.txt && pip install -e .

setup: install
	cp -n .env.example .env || true
	mkdir -p data/raw data/processed data/external data/models logs docs
	@echo "✅ Setup complete. Run: make pipeline"

# ── Pipeline ──────────────────────────────────────────────────
data:
	$(PYTHON) src/data_engineering/generate_data.py \
		--n_sites 200 --days 90 --output data/raw/

validate:
	$(PYTHON) src/data_engineering/data_validation.py \
		--input data/raw/network_kpis.parquet

features:
	$(PYTHON) src/features/feature_pipeline.py \
		--input data/raw --output data/processed

train:
	$(PYTHON) src/models/ensemble_detector.py --train

score:
	$(PYTHON) src/models/ensemble_detector.py --score --evaluate

geo-map:
	$(PYTHON) src/models/geo_impact_map.py \
		--output data/processed/anomaly_zones.geojson

pipeline: data validate features train score geo-map
	@echo ""
	@echo "✅ Full pipeline complete!"
	@echo "   make dashboard  → Streamlit at :8501"
	@echo "   make api        → FastAPI at :8000"
	@echo "   make mlflow     → Experiments at :5000"

# ── Train individual models ───────────────────────────────────
train-prophet:
	$(PYTHON) src/models/prophet_detector.py --train --sample_sites 50

train-isolation-forest:
	$(PYTHON) src/models/isolation_forest.py --train

train-lstm:
	$(PYTHON) src/models/lstm_autoencoder.py --train

tune-isolation-forest:
	$(PYTHON) src/models/isolation_forest.py --train --tune

# ── Streaming ─────────────────────────────────────────────────
stream-demo:
	$(PYTHON) src/streaming/consumer.py --demo

stream-live:
	$(PYTHON) src/streaming/consumer.py --live

produce:
	$(PYTHON) src/streaming/producer.py --live --speed 60

# ── Serving ───────────────────────────────────────────────────
api:
	uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

dashboard:
	streamlit run dashboards/streamlit_app.py \
		--server.port 8501 --server.address 0.0.0.0

mlflow:
	mlflow ui --host 0.0.0.0 --port 5000

# ── Notebooks ─────────────────────────────────────────────────
notebook:
	jupyter lab notebooks/

run-notebooks:
	$(PYTHON) notebooks/01_eda_kpi_timeseries.py
	$(PYTHON) notebooks/02_feature_engineering.py
	$(PYTHON) notebooks/03_anomaly_detection_models.py
	$(PYTHON) notebooks/04_geo_impact_analysis.py
	@echo "✅ All notebooks complete. Charts in docs/"

# ── Testing ───────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# ── Code quality ──────────────────────────────────────────────
lint:
	flake8 src/ --max-line-length=100 --ignore=E501,W503 || true
	black src/ --check --line-length=100 || true

format:
	black src/ --line-length=100 && isort src/ --profile=black

# ── Docker ────────────────────────────────────────────────────
docker-up:
	cd mlops/docker && docker-compose up --build -d
	@echo "API       : http://localhost:8000"
	@echo "Dashboard : http://localhost:8501"
	@echo "MLflow    : http://localhost:5000"
	@echo "Kafka     : localhost:9092"
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana   : http://localhost:3000  (admin/admin)"

docker-down:
	cd mlops/docker && docker-compose down

docker-logs:
	cd mlops/docker && docker-compose logs -f

# ── Cleanup ───────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov/ .pytest_cache/

clean-data:
	rm -f data/raw/*.parquet data/processed/*.parquet
	rm -f data/processed/*.geojson data/processed/*.html data/processed/*.txt
	rm -f data/models/*.pkl data/models/*.pt data/models/*.png
	@echo "✅ Data removed. Run: make pipeline"
