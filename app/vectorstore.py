"""
VectorStore supporting Cosine Similarity search, metadata filtering by ticker or market (SET/US),
strict `min_score` thresholding, and JSON persistence.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
import numpy as np

from app.llm import LLMClient, llm_client
from app.schemas import RetrievedChunk


class VectorStore:
    """In-memory + JSON-persisted Dense Vector Index with Cosine Similarity & Metadata Filtering."""

    def __init__(self, client: Optional[LLMClient] = None) -> None:
        self.client = client or llm_client
        self.chunks: List[RetrievedChunk] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_chunks(self, chunks: List[RetrievedChunk]) -> None:
        """Embed and index a list of RetrievedChunk objects."""
        if not chunks:
            return
        texts = [c.text for c in chunks]
        vectors = self.client.get_embeddings(texts)
        mat = np.array(vectors, dtype=np.float64)

        # Normalize rows to unit length so dot product == Cosine Similarity
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms > 1e-12, norms, 1.0)
        mat = mat / norms

        self.chunks.extend(chunks)
        if self.embeddings is None:
            self.embeddings = mat
        else:
            if self.embeddings.shape[1] != mat.shape[1]:
                target_dim = self.embeddings.shape[1]
                if mat.shape[1] > target_dim:
                    mat = mat[:, :target_dim]
                else:
                    padded = np.zeros((mat.shape[0], target_dim), dtype=np.float64)
                    padded[:, :mat.shape[1]] = mat
                    mat = padded
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                mat = mat / np.where(norms > 1e-12, norms, 1.0)
            self.embeddings = np.vstack([self.embeddings, mat])

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.30,
        ticker_filter: Optional[str] = None,
        market_filter: Optional[Literal["SET", "US"]] = None,
    ) -> List[RetrievedChunk]:
        """
        Perform Cosine Similarity search with optional metadata filters (`ticker` or `market`)
        and enforce a strict `min_score` floor.
        """
        if not self.chunks or self.embeddings is None or len(self.chunks) == 0:
            return []

        q_vec_list = self.client.get_embeddings([query])
        if not q_vec_list:
            return []

        q_vec = np.array(q_vec_list[0], dtype=np.float64)
        if self.embeddings is not None and self.embeddings.ndim == 2:
            target_dim = self.embeddings.shape[1]
            if len(q_vec) != target_dim and target_dim == 384:
                from app.llm import deterministic_local_embedding
                q_vec = np.array(deterministic_local_embedding(query, dim=384), dtype=np.float64)
        q_norm = float(np.linalg.norm(q_vec))
        if q_norm > 1e-12:
            q_vec = q_vec / q_norm

        raw_scores = np.dot(self.embeddings, q_vec)

        # Compute lexical keyword boost so exact financial terms, tickers, and section concepts
        # align tightly with cosine similarity scores in both remote and fallback modes.
        q_lower = query.lower()
        results: List[RetrievedChunk] = []
        from app.rag import resolve_ticker
        norm_ticker = resolve_ticker(ticker_filter) if ticker_filter else None
        norm_market = market_filter.strip().upper() if market_filter else None

        # Look up registered company names dynamically from SQLite
        comp_name_map = {}
        try:
            import sqlite3
            if settings.sqlite_db_path.exists():
                conn = sqlite3.connect(str(settings.sqlite_db_path))
                for r in conn.execute("SELECT ticker, name FROM companies").fetchall():
                    comp_name_map[r[0].upper()] = r[1].lower()
                conn.close()
        except Exception:
            pass

        for idx, chunk in enumerate(self.chunks):
            if norm_ticker and chunk.ticker.upper() != norm_ticker:
                continue
            if norm_market and chunk.market.upper() != norm_market:
                continue

            cos_score = float(raw_scores[idx])
            c_tick = chunk.ticker.upper()
            c_name = comp_name_map.get(c_tick, "")

            name_words = [w for w in re.findall(r"[a-z0-9]+", c_name) if len(w) >= 4]
            matches_ticker = (
                (c_tick.lower() in q_lower)
                or (norm_ticker and c_tick == norm_ticker)
                or (c_name and c_name in q_lower)
                or any(w in q_lower for w in name_words)
            )
            if matches_ticker:
                cos_score = min(1.0, cos_score + 0.20)

            score = round(max(0.0, min(1.0, cos_score)), 4)
            if score >= min_score:
                results.append(
                    chunk.model_copy(update={"score": score})
                )

        results.sort(key=lambda c: c.score, reverse=True)
        unique_docs = sorted(list(set(c.doc_id for c in results)))
        if len(unique_docs) > 1 and top_k >= len(unique_docs):
            per_doc_k = max(2, top_k // len(unique_docs))
            balanced: List[RetrievedChunk] = []
            for d in unique_docs:
                d_chunks = [c for c in results if c.doc_id == d][:per_doc_k]
                balanced.extend(d_chunks)
            for c in results:
                if c not in balanced and len(balanced) < top_k:
                    balanced.append(c)
            balanced.sort(key=lambda c: c.score, reverse=True)
            return balanced[:top_k]
        return results[:top_k]

    def save(self, path: Path) -> None:
        """Persist chunks and embeddings to a JSON index file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "chunks": [c.model_dump() for c in self.chunks],
            "embeddings": self.embeddings.tolist() if self.embeddings is not None else [],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, client: Optional[LLMClient] = None) -> "VectorStore":
        """Load a persisted VectorStore index from disk."""
        store = cls(client=client)
        if not path.exists():
            return store
        raw = json.loads(path.read_text(encoding="utf-8"))
        store.chunks = [RetrievedChunk.model_validate(item) for item in raw.get("chunks", [])]
        emb_list = raw.get("embeddings", [])
        if emb_list:
            store.embeddings = np.array(emb_list, dtype=np.float64)
        return store