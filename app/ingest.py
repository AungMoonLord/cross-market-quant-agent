"""
Filing Ingestion & Chunking Pipeline supporting two selectable strategies for Ablation Study:
1. strategy="naive_fixed": Fixed 500-character sliding window (overlap=50) without header awareness.
2. strategy="markdown_section": Splits cleanly by Markdown `## ` headers and prepends
   `[MARKET | TICKER | FILING_TYPE | Section]` context headers inside each chunk.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Literal, Optional

from app.config import settings
from app.llm import LLMClient, llm_client
from app.schemas import RetrievedChunk
from app.vectorstore import VectorStore
from corpus.build_corpus import COMPANIES_DATA, build_all

TICKER_META = {
    row[0]: {"name": row[1], "market": row[2], "filing_type": row[5]}
    for row in COMPANIES_DATA
}


def parse_filing_filename(file_path: Path) -> tuple[str, str, str, str]:
    """
    Extract (doc_id, ticker, market, filing_type) from a filename or SQLite.
    """
    stem = file_path.stem  # e.g., ADVANC_56-1_2025 or SPCX_10-K_2026_Q2
    parts = stem.split("_")
    ticker = parts[0].upper()
    
    # Check SQLite first for accurate dynamic metadata
    market = "SET"
    filing_type = "56-1"
    try:
        import sqlite3
        from app.config import settings
        if settings.sqlite_db_path.exists():
            conn = sqlite3.connect(str(settings.sqlite_db_path))
            row = conn.execute("SELECT market, filing_type FROM companies WHERE ticker = ?", (ticker,)).fetchone()
            conn.close()
            if row:
                market = "SET" if row[0] in ("SET", "TH") else "US"
                filing_type = row[1]
                return stem, ticker, market, filing_type
    except Exception:
        pass

    # Fallback to parts if doc_id was formed as TICKER_FILINGTYPE_YEAR_QUARTER
    if len(parts) >= 2:
        candidate_ft = parts[1]
        if candidate_ft in ("56-1", "10-K", "10-Q", "20-F", "8-K", "MDA", "QUARTERLY_FINANCIALS"):
            filing_type = candidate_ft
            market = "US" if candidate_ft in ("10-K", "10-Q", "20-F", "8-K") else "SET"
            return stem, ticker, market, filing_type

    meta = TICKER_META.get(ticker, {"market": "SET", "filing_type": "56-1"})
    market = "SET" if meta["market"] in ("SET", "TH") else "US"
    filing_type = meta["filing_type"]
    return stem, ticker, market, filing_type


def chunk_filing_naive_fixed(
    file_path: Path,
    window_chars: int = 500,
    overlap_chars: int = 50,
) -> List[RetrievedChunk]:
    """
    Strategy 1 (`naive_fixed`):
    Fixed 500-character sliding window across the raw file without Markdown header awareness
    or prepended metadata headers.
    """
    raw_text = file_path.read_text(encoding="utf-8").strip()
    doc_id, ticker, market, filing_type = parse_filing_filename(file_path)

    chunks: List[RetrievedChunk] = []
    step = max(1, window_chars - overlap_chars)
    idx = 0
    chunk_num = 0
    while idx < len(raw_text):
        window_text = raw_text[idx : idx + window_chars].strip()
        if window_text:
            chunks.append(
                RetrievedChunk(
                    doc_id=doc_id,
                    chunk_id=f"c{chunk_num}",
                    ticker=ticker,
                    market=market,
                    filing_type=filing_type,
                    section="Naive Fixed Window",
                    text=window_text,
                    score=0.0,
                )
            )
            chunk_num += 1
        idx += step
    return chunks


def _split_text_into_subchunks(text: str, max_chars: int = 2500, overlap: int = 100) -> list[str]:
    """Split long section text into manageable subchunks cleanly at paragraph/newline boundaries."""
    if len(text) <= max_chars:
        return [text]
    
    paragraphs = text.split("\n\n")
    subchunks: list[str] = []
    current_chunk: list[str] = []
    current_len = 0

    for p in paragraphs:
        p_len = len(p)
        if current_len + p_len + 2 > max_chars and current_chunk:
            subchunks.append("\n\n".join(current_chunk).strip())
            current_chunk = [p]
            current_len = p_len
        else:
            current_chunk.append(p)
            current_len += p_len + 2

    if current_chunk:
        subchunks.append("\n\n".join(current_chunk).strip())

    final_chunks: list[str] = []
    for sc in subchunks:
        if len(sc) > max_chars * 1.5:
            step = max_chars - overlap
            for i in range(0, len(sc), step):
                window = sc[i : i + max_chars].strip()
                if window:
                    final_chunks.append(window)
        else:
            if sc.strip():
                final_chunks.append(sc.strip())

    return final_chunks if final_chunks else [text[:max_chars]]


def chunk_filing_markdown_section(file_path: Path) -> List[RetrievedChunk]:
    """
    Strategy 2 (`markdown_section`):
    Splits cleanly by Markdown `## ` headers, bounds chunk size to ~1200 chars
    to ensure bge-m3 embeddings never timeout or exceed token limits,
    and prepends `[MARKET | TICKER | FILING_TYPE | Section]` metadata headers.
    """
    raw_text = file_path.read_text(encoding="utf-8").strip()
    doc_id, ticker, market, filing_type = parse_filing_filename(file_path)

    section_blocks = re.split(r"(?m)^##\s+", raw_text)
    chunks: List[RetrievedChunk] = []
    chunk_num = 0

    # If document has no ## headers, chunk preamble directly
    if len(section_blocks) <= 1:
        subchunks = _split_text_into_subchunks(raw_text, max_chars=2500)
        for sc in subchunks:
            context_header = f"[{market} | {ticker} | {filing_type} | Overview]"
            chunks.append(
                RetrievedChunk(
                    doc_id=doc_id,
                    chunk_id=f"c{chunk_num}",
                    ticker=ticker,
                    market=market,
                    filing_type=filing_type,
                    section="Overview",
                    text=f"{context_header}\n{sc}",
                    score=0.0,
                )
            )
            chunk_num += 1
        return chunks

    for block in section_blocks[1:]:
        lines = block.strip().splitlines()
        if not lines:
            continue
        section_title = lines[0].strip()
        section_body = "\n".join(lines[1:]).strip()
        subchunks = _split_text_into_subchunks(section_body, max_chars=2500)

        for sc in subchunks:
            context_header = f"[{market} | {ticker} | {filing_type} | {section_title}]"
            enriched_text = f"{context_header}\n## {section_title}\n{sc}"
            chunks.append(
                RetrievedChunk(
                    doc_id=doc_id,
                    chunk_id=f"c{chunk_num}",
                    ticker=ticker,
                    market=market,
                    filing_type=filing_type,
                    section=section_title,
                    text=enriched_text,
                    score=0.0,
                )
            )
            chunk_num += 1

    return chunks


def ingest_corpus(
    strategy: Literal["naive_fixed", "markdown_section"] = "markdown_section",
    raw_filings_dir: Optional[Path] = None,
    output_index_path: Optional[Path] = None,
    client: Optional[LLMClient] = None,
) -> VectorStore:
    """
    Ingest all 8 Markdown filings using the requested chunking strategy and return a populated VectorStore.
    Automatically generates the raw corpus first if missing.
    """
    filings_dir = raw_filings_dir or settings.raw_filings_dir
    if not filings_dir.exists() or not list(filings_dir.glob("*.md")):
        build_all(db_path=settings.sqlite_db_path, raw_dir=filings_dir)

    md_files = sorted(filings_dir.glob("*.md"))
    all_chunks: List[RetrievedChunk] = []

    for md_file in md_files:
        if strategy == "naive_fixed":
            file_chunks = chunk_filing_naive_fixed(md_file)
        elif strategy == "markdown_section":
            file_chunks = chunk_filing_markdown_section(md_file)
        else:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")
        all_chunks.extend(file_chunks)

    store = VectorStore(client=client or llm_client)
    store.add_chunks(all_chunks)

    if output_index_path is not None:
        store.save(output_index_path)
    elif strategy == settings.chunking_strategy:
        store.save(settings.vector_index_path)

    return store


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest 56-1 and 10-K filings into VectorStore.")
    parser.add_argument(
        "--strategy",
        choices=["naive_fixed", "markdown_section"],
        default=settings.chunking_strategy,
        help="Chunking strategy to build.",
    )
    args = parser.parse_args()
    vs = ingest_corpus(strategy=args.strategy)
    print(f"[ingest] Indexed {len(vs.chunks)} chunks using strategy='{args.strategy}'.")