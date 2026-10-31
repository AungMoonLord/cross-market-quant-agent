"""
Dynamic Filing & PDF Ingestion Service (`app/upload.py`).
Allows uploading financial filings (PDF, Markdown, Text) directly from the Web UI,
automatically extracting text, generating structured sections, upserting company
metadata into SQLite, embedding chunks with bge-m3, and updating the live VectorStore.
"""

from __future__ import annotations

import io
import json
import logging
import re
import sqlite3
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from typing import Any, Dict, List, Literal, Optional, Tuple

import pypdf

from app.config import settings
from app.ingest import chunk_filing_markdown_section
from app.llm import llm_client
from app.schemas import TH_FILING_TYPES, US_FILING_TYPES, RetrievedChunk
from app.vectorstore import VectorStore

logger = logging.getLogger("quant_agent.upload")


def extract_text_from_pdf_ocr(file_bytes: bytes, filename: str) -> str:
    """
    Automated OCR Fallback for Scanned / Image-only PDFs (using pypdfium2 + RapidOCR).
    Allows ingestion of scanned filings, printed-as-image PDFs, and image-only reports.
    """
    try:
        import numpy as np
        import pypdfium2 as pdfium
        from rapidocr_onnxruntime import RapidOCR

        logger.info(f"Initiating RapidOCR fallback for scanned/image-only PDF: '{filename}'")
        pdf = pdfium.PdfDocument(file_bytes)
        ocr = RapidOCR()
        pages_text: List[str] = []
        for i, page in enumerate(pdf):
            img = page.render(scale=1.2).to_pil()
            result, _ = ocr(np.array(img))
            if result:
                text = "\n".join([line[1] for line in result])
                pages_text.append(f"<!-- Page {i+1} -->\n{text}")
        return "\n\n".join(pages_text)
    except Exception as e:
        logger.warning(f"RapidOCR fallback error for '{filename}': {e}")
        return ""


def extract_text_from_bytes(file_bytes: bytes, filename: str) -> str:
    """Extract clean text from PDF, Markdown, or text file bytes with automated OCR fallback."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        try:
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            pages_text: List[str] = []
            for i, page in enumerate(reader.pages):
                txt = page.extract_text() or ""
                txt = txt.strip()
                if txt:
                    pages_text.append(f"<!-- Page {i+1} -->\n{txt}")
            full_text = "\n\n".join(pages_text)
            if not full_text.strip():
                # Scanned / Image-only PDF: trigger automated RapidOCR fallback
                logger.info(f"PDF '{filename}' contains no digital text layer. Running automated OCR fallback...")
                full_text = extract_text_from_pdf_ocr(file_bytes, filename)

            if not full_text.strip():
                raise ValueError(
                    f"Could not extract readable text from PDF '{filename}' even after OCR analysis. "
                    f"Please check that the file is not blank or corrupted."
                )
            return full_text
        except ValueError:
            raise
        except Exception as e:
            try:
                ocr_text = extract_text_from_pdf_ocr(file_bytes, filename)
                if ocr_text.strip():
                    return ocr_text
            except Exception:
                pass
            raise ValueError(f"Invalid or unreadable PDF file '{filename}': {str(e)}")
    else:
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return file_bytes.decode("latin-1", errors="replace")


def format_into_markdown_filing(
    raw_text: str,
    ticker: str,
    name: str,
    market: str,
    filing_type: str,
    year: int,
    quarter: str = "FY",
) -> str:
    """
    Format extracted text into structured Markdown with `## ` sections
    so that `chunk_filing_markdown_section` creates clean, grounded chunks.
    """
    norm_ticker = ticker.strip().upper()
    period_str = f"{year} {quarter}".strip()

    # Check if text already has markdown `## ` sections
    existing_sections = re.findall(r"(?m)^##\s+(.+)$", raw_text)
    if len(existing_sections) >= 2:
        # Prepend main title header if not present
        if not raw_text.strip().startswith("# "):
            raw_text = f"# {norm_ticker} ({name}) - {filing_type} ({period_str})\n\n" + raw_text
        return raw_text

    # Otherwise, split raw text into balanced sections with meaningful headers
    clean_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw_text) if p.strip()]

    section_titles = [
        "1. Business & Segment Overview",
        "2. Management Discussion & Analysis (MD&A)",
        "3. Financial Performance & Statements",
        "4. Footnote Disclosures: Contingencies & Policies",
        "5. Guidance, Outlook & Risk Factors",
    ]

    # Distribute paragraphs into ~5 sections
    total_paras = len(clean_paragraphs)
    if total_paras == 0:
        clean_paragraphs = [raw_text.strip()]
        total_paras = 1

    chunk_size = max(1, (total_paras + len(section_titles) - 1) // len(section_titles))
    sections: List[str] = [f"# {norm_ticker} ({name}) - {filing_type} ({period_str})\n"]

    for i, title in enumerate(section_titles):
        start = i * chunk_size
        end = min(total_paras, (i + 1) * chunk_size)
        if start < total_paras:
            sec_body = "\n\n".join(clean_paragraphs[start:end])
            sections.append(f"## {title}\n{sec_body}\n")

    # If extra paragraphs remain
    remainder_start = len(section_titles) * chunk_size
    if remainder_start < total_paras:
        remainder_body = "\n\n".join(clean_paragraphs[remainder_start:])
        sections.append(f"## 6. Additional Disclosures & Notes\n{remainder_body}\n")

    return "\n".join(sections)


def upsert_company_and_financials(
    ticker: str,
    name: str,
    market: Literal["SET", "US"],
    sector: str,
    currency: Literal["THB", "USD"],
    filing_type: Literal["56-1", "10-K"],
    accounting_standard: Literal["TFRS", "US_GAAP"],
    year: int,
    metrics: Optional[Dict[str, float]] = None,
    db_path: Path = settings.sqlite_db_path,
) -> None:
    """Insert or update company metadata and financial statement metrics in SQLite."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Ensure tables exist
    cursor.executescript(
        """
        CREATE TABLE IF NOT EXISTS companies (
            ticker TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            market TEXT NOT NULL,
            sector TEXT NOT NULL,
            currency TEXT NOT NULL,
            filing_type TEXT NOT NULL,
            accounting_standard TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS fx_rates (
            year INTEGER NOT NULL,
            pair TEXT NOT NULL,
            avg_rate REAL NOT NULL,
            year_end_rate REAL NOT NULL,
            PRIMARY KEY (year, pair)
        );

        CREATE TABLE IF NOT EXISTS financial_statements (
            ticker TEXT NOT NULL,
            year INTEGER NOT NULL,
            currency TEXT NOT NULL,
            revenue REAL NOT NULL,
            gross_profit REAL NOT NULL,
            ebit REAL NOT NULL,
            ebitda REAL NOT NULL,
            net_profit REAL NOT NULL,
            total_assets REAL NOT NULL,
            current_assets REAL NOT NULL,
            total_liabilities REAL NOT NULL,
            current_liabilities REAL NOT NULL,
            retained_earnings REAL NOT NULL,
            total_equity REAL NOT NULL,
            interest_bearing_debt REAL NOT NULL,
            cash REAL NOT NULL,
            operating_cash_flow REAL NOT NULL,
            capex REAL NOT NULL,
            market_cap REAL NOT NULL,
            shares_outstanding REAL NOT NULL,
            PRIMARY KEY (ticker, year),
            FOREIGN KEY (ticker) REFERENCES companies(ticker)
        );
        """
    )

    # Ensure fx_rates has baseline rates
    cursor.execute("SELECT COUNT(*) FROM fx_rates")
    if cursor.fetchone()[0] == 0:
        cursor.executemany(
            "INSERT OR IGNORE INTO fx_rates VALUES (?, ?, ?, ?)",
            [
                (2023, "USD_THB", 34.80, 34.35),
                (2024, "USD_THB", 35.20, 34.10),
                (2025, "USD_THB", 34.00, 33.85),
                (2026, "USD_THB", 34.00, 34.00),
            ],
        )

    # Upsert company
    cursor.execute(
        """
        INSERT INTO companies (ticker, name, market, sector, currency, filing_type, accounting_standard)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
            name=excluded.name,
            market=excluded.market,
            sector=excluded.sector,
            currency=excluded.currency,
            filing_type=excluded.filing_type,
            accounting_standard=excluded.accounting_standard
        """,
        (ticker.upper(), name, market, sector, currency, filing_type, accounting_standard),
    )

    # Upsert financial statements row if metrics provided, or basic default row
    m = metrics or {}
    rev = float(m.get("revenue", 0.0))
    gp = float(m.get("gross_profit", rev * 0.35 if rev > 0 else 0.0))
    np = float(m.get("net_profit", rev * 0.12 if rev > 0 else 0.0))
    ebit = float(m.get("ebit", np * 1.25 if np > 0 else 0.0))
    ebitda = float(m.get("ebitda", ebit * 1.35 if ebit > 0 else 0.0))
    assets = float(m.get("total_assets", rev * 1.5 if rev > 0 else 1000.0))
    cur_assets = float(m.get("current_assets", assets * 0.3))
    liab = float(m.get("total_liabilities", assets * 0.5))
    cur_liab = float(m.get("current_liabilities", liab * 0.4))
    equity = float(m.get("total_equity", max(1.0, assets - liab)))
    ret_earn = float(m.get("retained_earnings", equity * 0.6))
    debt = float(m.get("interest_bearing_debt", liab * 0.4))
    cash = float(m.get("cash", cur_assets * 0.4))
    ocf = float(m.get("operating_cash_flow", np * 1.1 if np > 0 else 0.0))
    capex = float(m.get("capex", ocf * 0.4 if ocf > 0 else 0.0))
    mcap = float(m.get("market_cap", np * 15.0 if np > 0 else rev * 2.0 if rev > 0 else 10000.0))
    shares = float(m.get("shares_outstanding", 1000.0))

    cursor.execute(
        """
        INSERT INTO financial_statements (
            ticker, year, currency, revenue, gross_profit, ebit, ebitda,
            net_profit, total_assets, current_assets, total_liabilities,
            current_liabilities, retained_earnings, total_equity,
            interest_bearing_debt, cash, operating_cash_flow, capex,
            market_cap, shares_outstanding
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, year) DO UPDATE SET
            currency=excluded.currency,
            revenue=excluded.revenue,
            gross_profit=excluded.gross_profit,
            ebit=excluded.ebit,
            ebitda=excluded.ebitda,
            net_profit=excluded.net_profit,
            total_assets=excluded.total_assets,
            current_assets=excluded.current_assets,
            total_liabilities=excluded.total_liabilities,
            current_liabilities=excluded.current_liabilities,
            retained_earnings=excluded.retained_earnings,
            total_equity=excluded.total_equity,
            interest_bearing_debt=excluded.interest_bearing_debt,
            cash=excluded.cash,
            operating_cash_flow=excluded.operating_cash_flow,
            capex=excluded.capex,
            market_cap=excluded.market_cap,
            shares_outstanding=excluded.shares_outstanding
        """,
        (
            ticker.upper(),
            year,
            currency,
            rev,
            gp,
            ebit,
            ebitda,
            np,
            assets,
            cur_assets,
            liab,
            cur_liab,
            ret_earn,
            equity,
            debt,
            cash,
            ocf,
            capex,
            mcap,
            shares,
        ),
    )

    conn.commit()
    conn.close()


def ingest_filing_to_corpus(
    file_bytes: bytes,
    filename: str,
    ticker: str,
    name: str,
    market: Literal["SET", "US"] = "SET",
    sector: str = "General",
    currency: Literal["THB", "USD"] = "THB",
    year: int = 2025,
    quarter: str = "FY",
    filing_type: Literal["56-1", "10-K"] = "56-1",
    accounting_standard: Literal["TFRS", "US_GAAP"] = "TFRS",
    metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Complete end-to-end user upload & ingestion pipeline:
    1. Extracts text from PDF / MD / TXT.
    2. Formats into structured Markdown sections.
    3. Saves Markdown filing into `corpus/raw_filings/{doc_id}.md`.
    4. Upserts company & financial statement in SQLite `corpus/financials.db`.
    5. Chunks document via `markdown_section` strategy.
    6. Generates bge-m3 embeddings and appends to `corpus/vector_index.json`.
    7. Refreshes live in-memory VectorStore and dynamic known tickers.
    """
    norm_ticker = ticker.strip().upper()
    norm_market = "SET" if market.upper() in ("SET", "TH") else "US"
    norm_ft = filing_type.strip()

    # Validate cascading market & filing_type compatibility
    if market.upper() in ("SET", "TH"):
        if norm_ft not in TH_FILING_TYPES:
            raise ValueError(
                f"Filing type '{norm_ft}' is not valid for Thai SET market (TH). "
                f"Allowed types: Form 56-1 One Report (56-1 / 56-1_ONE_REPORT), "
                f"งบการเงินรายไตรมาส (QUARTERLY_FINANCIALS), or MD&A (MDA)."
            )
    elif market.upper() == "US":
        if norm_ft not in US_FILING_TYPES:
            raise ValueError(
                f"Filing type '{norm_ft}' is not valid for US SEC market (US). "
                f"Allowed types: SEC Form 10-K, SEC Form 10-Q, SEC Form 20-F, or SEC Form 8-K."
            )
    settings.raw_filings_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract text
    raw_text = extract_text_from_bytes(file_bytes, filename)

    # 2. Structure into Markdown
    quarter_slug = quarter.replace(" ", "_")
    doc_id = f"{norm_ticker}_{filing_type}_{year}_{quarter_slug}"
    markdown_content = format_into_markdown_filing(
        raw_text=raw_text,
        ticker=norm_ticker,
        name=name,
        market=market,
        filing_type=filing_type,
        year=year,
        quarter=quarter,
    )

    # 3. Save to raw_filings
    target_path = settings.raw_filings_dir / f"{doc_id}.md"
    target_path.write_text(markdown_content, encoding="utf-8")

    # 4. Upsert SQLite DB
    upsert_company_and_financials(
        ticker=norm_ticker,
        name=name,
        market=market,
        sector=sector,
        currency=currency,
        filing_type=filing_type,
        accounting_standard=accounting_standard,
        year=year,
        metrics=metrics,
        db_path=settings.sqlite_db_path,
    )

    # 5. Chunking via markdown_section
    new_chunks = chunk_filing_markdown_section(target_path)
    if not new_chunks:
        # Fallback if markdown headers were sparse: single chunk
        new_chunks = [
            RetrievedChunk(
                doc_id=doc_id,
                chunk_id="c0",
                ticker=norm_ticker,
                market=market,
                filing_type=filing_type,
                section="Overview & Financial Disclosures",
                text=f"[{market} | {norm_ticker} | {filing_type} | Overview]\n{markdown_content[:3000]}",
                score=0.0,
            )
        ]

    # 6. Update VectorStore
    index_path = settings.vector_index_path
    if index_path.exists():
        try:
            store = VectorStore.load(index_path, client=llm_client)
        except Exception:
            store = VectorStore(client=llm_client)
    else:
        store = VectorStore(client=llm_client)

    # 6. Update VectorStore incrementally without re-embedding baseline chunks
    import numpy as np

    remaining_indices = [i for i, c in enumerate(store.chunks) if c.doc_id != doc_id]
    remaining_chunks = [store.chunks[i] for i in remaining_indices]

    # Only embed new_chunks (takes 1-2s, never times out)
    new_vectors = llm_client.get_embeddings([c.text for c in new_chunks])
    new_mat = np.array(new_vectors, dtype=np.float64)
    new_norms = np.linalg.norm(new_mat, axis=1, keepdims=True)
    new_mat = new_mat / np.where(new_norms > 1e-12, new_norms, 1.0)

    if store.embeddings is not None and len(remaining_indices) > 0 and store.embeddings.ndim == 2:
        remaining_embeddings = store.embeddings[remaining_indices]
        if remaining_embeddings.shape[1] != new_mat.shape[1]:
            target_dim = remaining_embeddings.shape[1]
            if new_mat.shape[1] > target_dim:
                new_mat = new_mat[:, :target_dim]
            else:
                padded = np.zeros((new_mat.shape[0], target_dim), dtype=np.float64)
                padded[:, :new_mat.shape[1]] = new_mat
                new_mat = padded
            n = np.linalg.norm(new_mat, axis=1, keepdims=True)
            new_mat = new_mat / np.where(n > 1e-12, n, 1.0)
        combined_embeddings = np.vstack([remaining_embeddings, new_mat])
    else:
        combined_embeddings = new_mat

    fresh_store = VectorStore(client=llm_client)
    fresh_store.chunks = remaining_chunks + new_chunks
    fresh_store.embeddings = combined_embeddings
    fresh_store.save(index_path)

    # 7. Refresh in-memory agent tools RAG pipeline
    import agent.tools as tools_mod
    from app.rag import GroundedRAGPipeline, refresh_known_tickers

    tools_mod._rag_pipeline_instance = GroundedRAGPipeline(vectorstore=fresh_store, client=llm_client)
    refresh_known_tickers()

    return {
        "status": "success",
        "ticker": norm_ticker,
        "name": name,
        "doc_id": doc_id,
        "filing_path": str(target_path),
        "chunks_added": len(new_chunks),
        "total_corpus_chunks": len(fresh_store.chunks),
        "characters_extracted": len(raw_text),
        "preview_text": raw_text[:400].replace("\n", " "),
    }


def clear_corpus() -> Dict[str, Any]:
    """
    Clear all pre-seeded and uploaded corpus data:
    1. Removes all rows from companies and financial_statements in SQLite.
    2. Deletes all Markdown files in corpus/raw_filings/.
    3. Resets corpus/vector_index.json to 0 chunks.
    4. Refreshes in-memory VectorStore and KNOWN_TICKERS.
    """
    # 1. Clean DB tables
    if settings.sqlite_db_path.exists():
        conn = sqlite3.connect(str(settings.sqlite_db_path))
        cursor = conn.cursor()
        cursor.execute("DELETE FROM financial_statements")
        cursor.execute("DELETE FROM companies")
        conn.commit()
        conn.close()

    # 2. Delete raw filings
    if settings.raw_filings_dir.exists():
        for f in settings.raw_filings_dir.glob("*.md"):
            try:
                f.unlink()
            except Exception:
                pass

    # 3. Reset VectorStore file
    empty_index = {"chunks": [], "strategy": "markdown_section"}
    settings.vector_index_path.parent.mkdir(parents=True, exist_ok=True)
    settings.vector_index_path.write_text(json.dumps(empty_index), encoding="utf-8")

    # 4. Refresh in-memory RAG pipeline and dynamic tickers
    import agent.tools as tools_mod
    from app.rag import GroundedRAGPipeline, refresh_known_tickers

    empty_store = VectorStore(client=llm_client)
    empty_store.save(settings.vector_index_path)
    tools_mod._rag_pipeline_instance = GroundedRAGPipeline(vectorstore=empty_store, client=llm_client)
    refresh_known_tickers()

    return {
        "status": "cleared",
        "message": "All corpus filings, database companies, and vector index chunks cleared successfully.",
        "active_tickers": 0,
        "total_chunks": 0,
    }
