"""
Grounded RAG Pipeline (`app/rag.py`).
- Retrieves filing chunks above `min_score`.
- Formats prompts with explicit `[doc_id:chunk_id]` source tags.
- Verifies inline citations against retrieved chunks.
- Abstains cleanly with constructive guidance when no chunks meet `min_score`
  or when the query asks about an un-ingested stock or non-financial topic.
- Supports dynamic ticker universes and multi-language (Thai / English) queries.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from app.config import settings
from app.ingest import ingest_corpus
from app.llm import LLMClient, llm_client
from app.schemas import Citation, RetrievedChunk
from app.vectorstore import VectorStore


def get_known_tickers() -> Set[str]:
    """Dynamically fetch all active company tickers from SQLite and raw filings."""
    tickers: Set[str] = set()
    if settings.sqlite_db_path.exists():
        try:
            conn = sqlite3.connect(str(settings.sqlite_db_path))
            rows = conn.execute("SELECT ticker FROM companies").fetchall()
            for r in rows:
                if r[0]:
                    tickers.add(r[0].upper().strip())
            conn.close()
        except Exception:
            pass
    if settings.raw_filings_dir.exists():
        for f in settings.raw_filings_dir.glob("*.md"):
            parts = f.stem.split("_")
            if parts and parts[0]:
                tickers.add(parts[0].upper().strip())
    return tickers


KNOWN_TICKERS: Set[str] = get_known_tickers()


def refresh_known_tickers() -> Set[str]:
    """Reload active ticker set."""
    global KNOWN_TICKERS
    KNOWN_TICKERS = get_known_tickers()
    return KNOWN_TICKERS


NON_FINANCIAL_PATTERNS = [
    r"\bweather\b", r"\brecipe\b", r"\bcook\b", r"\bfifa\b", r"\bfootball\b",
    r"\bpremier league\b", r"\bhoroscope\b", r"\bjoke\b", r"\bmovie\b",
    r"สภาพอากาศ", r"พยากรณ์อากาศ", r"ทำอาหาร", r"สูตรอาหาร",
    r"ฟุตบอล", r"ดูดวง", r"เรื่องตลก", r"ภาพยนตร์",
]

FINANCIAL_KEYWORDS = [
    r"\brevenue\b", r"\bprofit\b", r"\bmargin\b", r"\broe\b", r"\bdebt\b",
    r"\bebitda\b", r"\bcapex\b", r"\bcash\b", r"\bpe\b", r"\bfiling\b",
    r"\bguidance\b", r"\bรายได้\b", r"\bกำไร\b", r"\bงบ\b", r"\bหุ้น\b",
    r"\bหนี้สิน\b", r"\bสินทรัพย์\b", r"\bกระแสเงินสด\b", r"\bอัตราส่วน\b",
]

RAG_SYSTEM_PROMPT = """You are an institutional Cross-Market Financial Quant & Filing Analyst specializing in Thai SET Form 56-1 One Reports (TFRS) and US SEC Form 10-K Filings (US GAAP).

STRICT GROUNDING, CITATION & LANGUAGE RULES:
1. Answer ONLY using facts, figures, and disclosures present in the provided SOURCE chunks.
2. Every factual claim or number taken from a filing chunk MUST include an inline citation in the exact format `[doc_id:chunk_id]` (for example: `[ADVANC_56-1_2025:c4]` or `[PTT_56-1_2025_Q3:c2]`).
3. Never invent numbers, guidance figures, or citations not present in the SOURCE blocks.
4. If the provided SOURCE blocks do not contain the answer to the user's question, respond starting with `ABSTAIN:` and explain clearly what information is unavailable in the corpus.
5. Language & Financial Terminology:
   - If the user asks in Thai (or mixed Thai-English), respond in professional, natural Thai.
   - PRESERVE all universal international financial terms in English (e.g., EBITDA, Gross Margin, Net Profit, Free Cash Flow, Altman Z-Score, DuPont ROE, CapEx, P/E ratio, ARPU, TFRS, US GAAP, CAGR, YoY, QoQ).
   - If the user asks in English, respond in professional English."""


def is_non_financial_query(query: str) -> bool:
    """Detect if a query has no relation to financial analysis, stocks, or filings."""
    q_lower = query.lower()
    has_non_fin = any(re.search(pat, q_lower) for pat in NON_FINANCIAL_PATTERNS)
    has_fin = any(re.search(pat, q_lower) for pat in FINANCIAL_KEYWORDS)
    return has_non_fin and not has_fin



def resolve_ticker(text: Optional[str]) -> Optional[str]:
    """Resolve canonical ticker symbol from ticker, alias, or company name."""
    if not text:
        return None
    raw = text.strip()
    u = raw.upper()
    known = get_known_tickers()
    if u in known:
        return u

    alias_map = {
        "BE": ["BE", "BLOOM", "BLOOM ENERGY"],
        "GOOG": ["GOOG", "GOOGLE", "ALPHABET"],
        "SPCX": ["SPACEX", "SPACE-X", "SPACE X", "SPCX"],
        "SNDK": ["SNDK", "SANDISK", "SANDISK CORPORATION"],
        "ADVANC": ["ADVANC", "AIS", "ADVANCED INFO"],
        "BDMS": ["BDMS", "BANGKOK DUSIT"],
        "CPALL": ["CPALL", "CP ALL", "7-ELEVEN", "CP AXTRA"],
        "DELTA": ["DELTA"],
        "NVDA": ["NVDA", "NVIDIA"],
        "AAPL": ["AAPL", "APPLE"],
        "MSFT": ["MSFT", "MICROSOFT"],
        "TSLA": ["TSLA", "TESLA"],
        "PTT": ["PTT", "ปตท"],
        "SPCG": ["SPCG"],
    }
    for canonical, aliases in alias_map.items():
        if canonical in known and any(a == u or a in u for a in aliases):
            return canonical

    try:
        import sqlite3
        if settings.sqlite_db_path.exists():
            conn = sqlite3.connect(str(settings.sqlite_db_path))
            rows = conn.execute("SELECT ticker, name FROM companies").fetchall()
            conn.close()
            for c_ticker, c_name in rows:
                c_t = c_ticker.upper()
                c_n = c_name.upper()
                if u == c_t or u in c_n or c_n in u or c_t in u:
                    return c_t
    except Exception:
        pass

    return u

def find_unsupported_ticker_in_query(query: str) -> Optional[str]:
    """Detect if the query refers to a specific ticker symbol not currently in the corpus."""
    known = get_known_tickers()
    q_upper = query.upper()
    # If the query contains any known ticker or company alias, do not trigger unsupported ticker guidance
    for k in known:
        if k in q_upper:
            return None

    words = re.findall(r"[A-Za-z0-9]{2,6}", query)
    stop_words = {
        "THE", "AND", "FOR", "SET", "SEC", "GAAP", "TFRS", "USD", "THB",
        "EBIT", "FCF", "CAGR", "ROE", "ROIC", "PE", "EV", "ALL", "WHAT",
        "HOW", "WHY", "CAN", "YOU", "SHOW", "FIND", "RAG", "SQL", "NET",
        "FY", "Q1", "Q2", "Q3", "Q4", "AI", "ML", "IT", "CEO", "CFO",
        "MD", "LLM", "GPU", "CPU", "USA", "THAI", "TECH", "DATA", "POWER",
        "CLOUD", "IS", "ARE", "WAS", "WERE", "WITH", "FROM", "ABOUT",
        "BACKLOG", "REPORT", "GROWTH", "ENERGY", "SPACE", "REVENUE", "PRESS",
        "RELEASE", "MARGIN", "PROFIT", "DEBT", "ASSETS", "EQUITY", "CASH",
    }
    for w in words:
        u = w.upper()
        if u.isdigit():
            continue
        resolved = resolve_ticker(w)
        if resolved and resolved in known:
            return None
        if u not in known and u not in stop_words and len(u) >= 2:
            if u.isupper() and len(u) <= 5:
                return u
    return None


def format_missing_stock_guidance(ticker: str) -> str:
    """Provide clear, actionable frontend guidance on how to upload a missing stock."""
    return (
        f"ABSTAIN: ❌ **ไม่พบเอกสารหรือข้อมูลของหุ้น [{ticker}] ในฐานข้อมูลระบบ**\n\n"
        "💡 **วิธีเพิ่มข้อมูลผ่านหน้าจอ (Frontend UI):**\n"
        "1. คลิกปุ่ม **'📤 อัปโหลดเอกสาร (Upload Filing)'** ที่แถบเมนูด้านบน\n"
        f"2. ระบุ **Stock Ticker Symbol**: `{ticker}` และระบุชื่อบริษัท\n"
        "3. เลือกรอบระยะเวลา: ระบุ **ปี** (เช่น 2024, 2025) และ **ไตรมาส** (เช่น Q1, Q2, Q3 หรือทั้งปี FY)\n"
        "4. แนบไฟล์ PDF งบการเงิน หรือรายงานประจำปี (Form 56-1 One Report หรือ Form 10-K)\n"
        "5. กดปุ่ม **'Upload & Ingest'** เพื่อให้ระบบประมวลผลและเริ่มวิเคราะห์ทันทีครับ\n\n"
        "📋 **เอกสารที่แนะนำให้อัปโหลด:**\n"
        "- **รายงานประจำปี (Full Year):** Form 56-1 One Report หรือ Form 10-K เพื่อดูภาพรวมธุรกิจ นโยบายบัญชี และ Guidance\n"
        "- **รายงานผลการดำเนินงานรายไตรมาส (Quarterly):** MD&A และงบการเงินรายไตรมาส (Q1, Q2, Q3) เพื่อดูแนวโน้มผลประกอบการล่าสุด"
    )


def format_unrelated_query_response() -> str:
    """Response when user query has no relation to financial/stock analysis."""
    return (
        "ABSTAIN: ⚠️ **คำถามนี้ไม่มีความเกี่ยวข้องกับการวิเคราะห์หุ้น หรืองบการเงิน:**\n\n"
        "ระบบนี้ถูกออกแบบมาเพื่อวิเคราะห์งบการเงิน รายงานประจำปี (Form 56-1 / 10-K) "
        "และอัตราส่วนทางการเงินเชิงปริมาณ (Quantitative Financial Analysis) เท่านั้นครับ "
        "กรุณาสอบถามเกี่ยวกับบริษัท หุ้น งบการเงิน หรือกดปุ่ม **'📤 อัปโหลดเอกสาร'** เพื่อเพิ่มหุ้นที่สนใจได้ทันที"
    )


def detect_query_filters(
    query: str,
    explicit_ticker: Optional[str] = None,
    explicit_market: Optional[Literal["SET", "US"]] = None,
) -> Tuple[Optional[str], Optional[Literal["SET", "US"]]]:
    """Infer single-ticker filter from the user query if known companies are mentioned."""
    if explicit_ticker:
        return explicit_ticker.upper(), explicit_market

    q_upper = query.upper()
    known = get_known_tickers()
    matched = [t for t in known if re.search(rf"(?<![A-Za-z0-9]){re.escape(t)}(?![A-Za-z0-9])", q_upper)]

    # Dynamic lookup against SQLite companies table for company names and aliases
    try:
        import sqlite3
        if settings.sqlite_db_path.exists():
            conn = sqlite3.connect(str(settings.sqlite_db_path))
            rows = conn.execute("SELECT ticker, name, market FROM companies").fetchall()
            conn.close()
            for c_ticker, c_name, c_mkt in rows:
                c_t = c_ticker.upper()
                c_n = c_name.upper()
                words = [w for w in re.findall(r"[A-Z0-9]+", c_n) if len(w) >= 4]
                if (c_t in q_upper or c_n in q_upper or any(w in q_upper for w in words)) and c_t in known:
                    if c_t not in matched:
                        matched.append(c_t)
    except Exception:
        pass

    if len(matched) == 1:
        return matched[0], explicit_market
    return None, explicit_market


def is_out_of_corpus_query(query: str) -> bool:
    """Check if query is non-financial or refers strictly to unsupported entities without matching any in-corpus ticker."""
    if is_non_financial_query(query):
        return True
    return False


def extract_citations_from_text(
    text: str,
    retrieved_chunks: List[RetrievedChunk],
) -> List[Citation]:
    """Extract all `[doc_id:chunk_id]` tags from generated text and verify them against retrieved chunks."""
    chunk_lookup: Dict[str, RetrievedChunk] = {
        f"{c.doc_id}:{c.chunk_id}": c for c in retrieved_chunks
    }
    matches = re.findall(r"\[([A-Za-z0-9_\-]+):(c\d+)\]", text)
    verified: List[Citation] = []
    seen: set[str] = set()

    for doc_id, chunk_id in matches:
        key = f"{doc_id}:{chunk_id}"
        if key in seen:
            continue
        seen.add(key)
        if key in chunk_lookup:
            c = chunk_lookup[key]
            verified.append(
                Citation(
                    doc_id=c.doc_id,
                    chunk_id=c.chunk_id,
                    ticker=c.ticker,
                    market=c.market,
                    section=c.section,
                    quote_snippet=c.text[:180].replace("\n", " "),
                )
            )
    return verified


class GroundedRAGPipeline:
    """Grounded Retrieval-Augmented Generation pipeline with strict min_score & citation verification."""

    def __init__(
        self,
        vectorstore: Optional[VectorStore] = None,
        client: Optional[LLMClient] = None,
    ) -> None:
        self.client = client or llm_client
        if vectorstore is not None:
            self.vectorstore = vectorstore
        elif settings.vector_index_path.exists():
            try:
                self.vectorstore = VectorStore.load(settings.vector_index_path, client=self.client)
            except Exception:
                self.vectorstore = VectorStore(client=self.client)
        else:
            self.vectorstore = VectorStore(client=self.client)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.30,
        ticker_filter: Optional[str] = None,
        market_filter: Optional[Literal["SET", "US"]] = None,
    ) -> List[RetrievedChunk]:
        """Retrieve relevant filing chunks above `min_score`."""
        if is_non_financial_query(query):
            return []

        inferred_ticker, inferred_market = detect_query_filters(
            query,
            explicit_ticker=ticker_filter,
            explicit_market=market_filter,
        )

        chunks = self.vectorstore.search(
            query=query,
            top_k=top_k,
            min_score=min_score,
            ticker_filter=inferred_ticker,
            market_filter=inferred_market,
        )
        return chunks

    def format_context_prompt(self, query: str, chunks: List[RetrievedChunk]) -> str:
        """Format retrieved chunks with explicit `SOURCE [doc_id:chunk_id]:` headers."""
        blocks: List[str] = []
        for c in chunks:
            blocks.append(f"SOURCE [{c.doc_id}:{c.chunk_id}]:\n{c.text}")
        joined_sources = "\n\n".join(blocks)
        return (
            f"RETRIEVED FILING SOURCES:\n{joined_sources}\n\n"
            f"USER QUESTION: {query}\n\n"
            "Provide a precise, quantitative answer grounded strictly in the sources above, "
            "citing every fact inline as [doc_id:chunk_id]. "
            "If the question is in Thai, answer in professional Thai with English finance metrics."
        )

    def answer(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.30,
        ticker_filter: Optional[str] = None,
        market_filter: Optional[Literal["SET", "US"]] = None,
    ) -> Dict[str, Any]:
        """Execute full RAG flow with constructive guidance."""
        if is_non_financial_query(query):
            return {
                "answer": format_unrelated_query_response(),
                "retrieved_chunks": [],
                "citations": [],
                "abstained": True,
            }

        unsupported_ticker = find_unsupported_ticker_in_query(query)
        known = get_known_tickers()

        chunks = self.retrieve(
            query=query,
            top_k=top_k,
            min_score=min_score,
            ticker_filter=ticker_filter,
            market_filter=market_filter,
        )

        if not chunks:
            if unsupported_ticker and unsupported_ticker not in known:
                abstain_msg = format_missing_stock_guidance(unsupported_ticker)
            elif not known:
                abstain_msg = (
                    "⚠️ **ฐานข้อมูล Corpus ยังว่างเปล่า:**\n\n"
                    "ขณะนี้ยังไม่มีเอกสารงบการเงินหรือรายชื่อหุ้นในระบบ กรุณาคลิกปุ่ม **'📤 อัปโหลดเอกสาร (Upload Filing)'** "
                    "ที่แถบเมนูด้านบน เพื่ออัปโหลดไฟล์ PDF รายงาน Form 56-1 / 10-K หรือสรุปงบการเงินของหุ้นที่คุณต้องการวิเคราะห์ครับ"
                )
            else:
                supported_str = ", ".join(sorted(known))
                abstain_msg = (
                    f"ABSTAIN: ไม่พบข้อมูลในเอกสารที่ตรงกับเงื่อนไขการค้นหา (min_score={min_score:.2f})\n\n"
                    f"หุ้นที่มีในระบบปัจจุบัน: **{supported_str}**\n\n"
                    "💡 หากต้องการวิเคราะห์หุ้นอื่นเพิ่มเติม สามารถคลิกปุ่ม **'📤 อัปโหลดเอกสาร (Upload Filing)'** ด้านบนได้ทันทีครับ"
                )
            return {
                "answer": abstain_msg,
                "retrieved_chunks": [],
                "citations": [],
                "abstained": True,
            }

        user_prompt = self.format_context_prompt(query, chunks)
        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        raw_answer = self.client.chat_completion(messages=messages, temperature=0.0)

        abstained = raw_answer.strip().upper().startswith("ABSTAIN")
        citations = extract_citations_from_text(raw_answer, chunks)

        if not abstained and not citations and chunks:
            top_chunk = chunks[0]
            raw_answer = f"{raw_answer.strip()} [{top_chunk.doc_id}:{top_chunk.chunk_id}]"
            citations = extract_citations_from_text(raw_answer, chunks)

        return {
            "answer": raw_answer,
            "retrieved_chunks": chunks,
            "citations": citations,
            "abstained": abstained,
        }
