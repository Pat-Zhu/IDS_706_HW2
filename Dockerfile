FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY tests/ tests/
COPY pytest.ini .

# 默认跑 CLI；用 `docker run ... pytest -q` 会覆盖这个 CMD
CMD ["python", "-m", "src.stock_analysis_cli", "--ticker", "AAPL", "--start", "2020-01-01", "--end", "2025-01-01"]
