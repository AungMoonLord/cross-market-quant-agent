"""
Environment Configuration Loader for Cross-Market Quant & Filing Analyst Agent.
Strictly reads OPENAI_API_KEY, OPENAI_BASE_URL, and model/runtime parameters from environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration populated from environment variables."""

    openai_api_key: str
    openai_base_url: str | None
    llm_model: str
    embedding_model: str
    min_score: float
    top_k: int
    max_agent_steps: int
    chunking_strategy: str
    corpus_dir: Path
    raw_filings_dir: Path
    sqlite_db_path: Path
    vector_index_path: Path
    host: str
    port: int


def load_config() -> AppConfig:
    """Load configuration from environment variables with safe institutional defaults."""
    api_key = os.getenv("OPENAI_API_KEY", "sk-local-offline-fallback-key")
    base_url = os.getenv("OPENAI_BASE_URL") or None
    llm_model = os.getenv("LLM_MODEL", "gemma-4-E4B-it")
    embedding_model = os.getenv("EMBEDDING_MODEL", "bge-m3")
    min_score = float(os.getenv("MIN_SCORE", "0.30"))
    top_k = int(os.getenv("TOP_K", "5"))
    max_agent_steps = int(os.getenv("MAX_AGENT_STEPS", "6"))
    chunking_strategy = os.getenv("CHUNKING_STRATEGY", "markdown_section")

    corpus_dir = PROJECT_ROOT / os.getenv("CORPUS_DIR", "corpus")
    raw_filings_dir = PROJECT_ROOT / os.getenv("RAW_FILINGS_DIR", "corpus/raw_filings")
    sqlite_db_path = PROJECT_ROOT / os.getenv("SQLITE_DB_PATH", "corpus/financials.db")
    vector_index_path = PROJECT_ROOT / os.getenv("VECTOR_INDEX_PATH", "corpus/vector_index.json")

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))

    return AppConfig(
        openai_api_key=api_key,
        openai_base_url=base_url,
        llm_model=llm_model,
        embedding_model=embedding_model,
        min_score=min_score,
        top_k=top_k,
        max_agent_steps=max_agent_steps,
        chunking_strategy=chunking_strategy,
        corpus_dir=corpus_dir,
        raw_filings_dir=raw_filings_dir,
        sqlite_db_path=sqlite_db_path,
        vector_index_path=vector_index_path,
        host=host,
        port=port,
    )


settings = load_config()