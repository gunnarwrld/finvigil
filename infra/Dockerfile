FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY ml/ ./ml/
COPY data/ ./data/
COPY evals/ ./evals/

RUN pip install --no-cache-dir -e ".[prod]"

EXPOSE 8000

CMD ["uvicorn", "finvigil.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
