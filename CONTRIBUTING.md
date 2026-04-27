# Contributing

## Setup
```bash
git clone https://github.com/YOUR/network-anomaly-detection.git
cd network-anomaly-detection
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

## Tests
```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing
```

## Code Style
```bash
make format   # black + isort
make lint     # check
```

## Pull Requests
1. Feature branch off main
2. Write tests for new functionality
3. Run `make test && make lint`
4. Submit PR with clear description
