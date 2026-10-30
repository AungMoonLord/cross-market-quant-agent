FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md* ./
COPY app ./app
COPY agent ./agent
COPY corpus ./corpus
COPY eval ./eval
COPY web ./web
COPY tests ./tests

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[dev]"

# Build the deterministic SQLite database, 56-1/10-K filings, and default vector index
RUN python -m corpus.build_corpus && \
    python -m app.ingest --strategy markdown_section

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]