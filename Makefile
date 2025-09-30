PY=python
PIP=pip

.PHONY: install test cov run docker-build docker-run docker-test clean

install:
	$(PIP) install -r requirements.txt

test:
	pytest -q

cov:
	pytest --cov=src --cov-report=term-missing

run:
	$(PY) -m src.stock_analysis_cli --ticker AAPL --start 2020-01-01 --end 2025-01-01

docker-build:
	docker build -t ids706-stocks .

docker-run:
	docker run --rm ids706-stocks

docker-test:
	docker run --rm ids706-stocks pytest -q

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} +
	rm -rf .pytest_cache

format:
	docker run --rm -v "$$PWD":/app -w /app python:3.11-slim \
	  bash -lc "pip install -q black==24.4.2 && black src tests stock_analysis.py"

lint:
	flake8

check: lint
	pytest -q

