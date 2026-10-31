"""
FastAPI Backend + Server-Sent Events (SSE) Streaming (`app/main.py`).

Endpoints:
- `GET  /api/health`: System health, active LLM/Embedding models, and corpus status.
- `GET  /api/tickers`: Returns all active SET and US tickers dynamically from SQLite with snapshot metrics.
- `POST /api/filings/upload`: Upload PDF/MD filings, extract text, chunk, embed, and upsert into database.
- `POST /api/corpus/clear`: Clear all corpus filings, database companies, and vector index for clean-slate user management.
- `POST /api/translate`: One-click translation (TH <-> EN) preserving international finance terminology.
- `POST /api/chat`: Grounded analysis returning a validated `ChatResponse`.
- `POST /api/chat/stream`: Real-time SSE stream (`tool_start`, `tool_result`, `token`, `complete`).
- `POST /api/memo/{ticker}`: Generates a complete `QuantSignalReport` institutional memo.
- Static files mounted at `/` serving the dark-mode Cross-Market Quant Terminal UI (`web/`).
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path
from pydantic import BaseModel
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from agent.loop import run_agent
from agent.quant_calc import (
    build_financial_metrics_schema,
    calculate_ticker_quant_metrics,
    fetch_company_and_statements,
)
from agent.tools import get_rag_pipeline
from app.config import PROJECT_ROOT, settings
from app.llm import llm_client
from app.rag import KNOWN_TICKERS, get_known_tickers
from app.schemas import (
    ChatRequest,
    ChatResponse,
    QuantSignalReport,
    TranslateRequest,
    TranslateResponse,
    UploadFilingResponse,
    TH_FILING_TYPES,
    US_FILING_TYPES,
)
from app.sessions import session_manager
from app.upload import clear_corpus, ingest_filing_to_corpus


def ensure_corpus_and_index_ready() -> None:
    """Ensure SQLite database and VectorStore index structure exist on startup."""
    settings.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.raw_filings_dir.mkdir(parents=True, exist_ok=True)

    # Initialize tables if db does not exist
    conn = sqlite3.connect(str(settings.sqlite_db_path))
    cursor = conn.cursor()
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
    conn.commit()
    conn.close()

    if not settings.vector_index_path.exists():
        empty_index = {"chunks": [], "strategy": "markdown_section"}
        settings.vector_index_path.write_text(json.dumps(empty_index), encoding="utf-8")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Auto-initialize database and RAG pipeline on FastAPI startup."""
    ensure_corpus_and_index_ready()
    get_rag_pipeline()
    yield


app = FastAPI(
    title="Cross-Market Financial Quant & Filing Analyst Agent",
    description="Institutional Quant & Filing Analyst for Thai SET (56-1 One Report) & US Equities (SEC Form 10-K)",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path in ("/", "/app.js", "/index.html") or path.endswith(".js") or path.endswith(".html"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    """Return service status, configured models, and dynamic corpus readiness."""
    ensure_corpus_and_index_ready()
    pipeline = get_rag_pipeline()
    known = sorted(list(get_known_tickers()))
    return {
        "status": "ok",
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "chunking_strategy": settings.chunking_strategy,
        "min_score_default": settings.min_score,
        "sqlite_db_ready": settings.sqlite_db_path.exists(),
        "indexed_chunks_count": len(pipeline.vectorstore.chunks),
        "supported_tickers": known,
        "tickers_count": len(known),
        "commit": os.getenv("RENDER_GIT_COMMIT", "local-dev")[:7],
    }


@app.get("/api/tickers")
async def list_tickers() -> Dict[str, Any]:
    """Return all companies registered in SQLite with their 2025 deterministic financial snapshot."""
    ensure_corpus_and_index_ready()
    tickers_payload: List[Dict[str, Any]] = []

    conn = sqlite3.connect(str(settings.sqlite_db_path))
    cursor = conn.cursor()
    cursor.execute(
        "SELECT ticker, name, market, sector, currency, filing_type, accounting_standard FROM companies ORDER BY ticker ASC"
    )
    companies = cursor.fetchall()
    conn.close()

    for ticker, name, market, sector, currency, filing_type, acct_std in companies:
        try:
            quant = calculate_ticker_quant_metrics(ticker=ticker, year=2025, metric_type="all")
            dp = quant["dupont_roe"]
            az = quant["altman_z"]
            lq = quant["liquidity_leverage"]
            fc = quant["fcf_cagr"]
            hl = quant["headline_numbers_millions"]
            snapshot = {
                "revenue_native_m": hl.get("revenue_native_m", 0.0),
                "revenue_usd_m": hl.get("revenue_usd_m", 0.0),
                "net_profit_native_m": hl.get("net_profit_native_m", 0.0),
                "net_profit_usd_m": hl.get("net_profit_usd_m", 0.0),
                "roe_dupont_pct": dp.get("roe_3_step_pct", 0.0),
                "gross_margin_pct": dp.get("gross_margin_pct", 0.0),
                "net_margin_pct": dp.get("net_profit_margin_pct", 0.0),
                "pe_ratio_x": lq.get("pe_ratio_x", 0.0),
                "ev_to_ebitda_x": lq.get("ev_to_ebitda_x", 0.0),
                "fcf_yield_pct": fc.get("fcf_yield_pct", 0.0),
                "revenue_cagr_3y_pct": fc.get("revenue_cagr_2023_2025_pct", 0.0),
                "altman_z_score": az.get("altman_z_score", 0.0),
                "altman_zone": az.get("altman_zone", "GREY"),
            }
        except Exception:
            snapshot = {
                "revenue_native_m": 0.0,
                "revenue_usd_m": 0.0,
                "net_profit_native_m": 0.0,
                "net_profit_usd_m": 0.0,
                "roe_dupont_pct": 0.0,
                "gross_margin_pct": 0.0,
                "net_margin_pct": 0.0,
                "pe_ratio_x": 0.0,
                "ev_to_ebitda_x": 0.0,
                "fcf_yield_pct": 0.0,
                "revenue_cagr_3y_pct": 0.0,
                "altman_z_score": 0.0,
                "altman_zone": "GREY",
            }

        # Inspect active filing documents for this ticker from raw_filings and vector index
        raw_files = [f.stem for f in settings.raw_filings_dir.glob("*.md")] if settings.raw_filings_dir.exists() else []
        ticker_filings = []
        for doc in sorted(raw_files):
            if doc.startswith(f"{ticker}_"):
                parts = doc.split("_")
                q_label = parts[-1] if len(parts) >= 4 else "FY"
                yr_label = parts[-2] if len(parts) >= 3 else ""
                ft_label = parts[1] if len(parts) >= 2 else ""
                ticker_filings.append({
                    "doc_id": doc,
                    "filing_type": ft_label,
                    "year": yr_label,
                    "quarter": q_label,
                    "label": f"{q_label} {yr_label} ({ft_label})"
                })

        tickers_payload.append(
            {
                "ticker": ticker,
                "name": name,
                "market": market,
                "sector": sector,
                "currency": currency,
                "filing_type": filing_type,
                "accounting_standard": acct_std,
                "snapshot_2025": snapshot,
                "filings_count": len(ticker_filings),
                "filings": ticker_filings,
            }
        )

    total_filings = sum(t.get("filings_count", 0) for t in tickers_payload)
    return {
        "count": len(tickers_payload),
        "total_filings_count": total_filings,
        "tickers": tickers_payload,
    }



# ---------------------------------------------------------------------------
# Conversational Chat Sessions API (LLM Chatbot Memory)
# ---------------------------------------------------------------------------
@app.get("/api/chat/sessions")
async def get_chat_sessions():
    """List all conversational chat sessions."""
    return session_manager.list_sessions()


@app.post("/api/chat/sessions")
async def create_chat_session(payload: Optional[Dict[str, Any]] = None):
    """Create a new conversation session (+ New Chat)."""
    title = payload.get("title") if payload else None
    sess = session_manager.create_session(title=title)
    return sess.model_dump()


@app.get("/api/chat/sessions/{session_id}")
async def get_chat_session_detail(session_id: str):
    """Get full message history for a specific chat session."""
    sess = session_manager.get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    return sess.model_dump()


@app.delete("/api/chat/sessions")
async def delete_all_chat_sessions():
    """Delete all conversation sessions."""
    session_manager.clear_all()
    return {"status": "all_deleted"}


@app.delete("/api/chat/sessions/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete a conversation session."""
    success = session_manager.delete_session(session_id)
    return {"status": "deleted" if success else "not_found", "session_id": session_id}


class UpdateSessionPayload(BaseModel):
    title: Optional[str] = None
    pinned: Optional[bool] = None


@app.patch("/api/chat/sessions/{session_id}")
async def update_chat_session(session_id: str, payload: UpdateSessionPayload):
    """Rename or toggle pin status for a chat session."""
    sess = session_manager.get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    if payload.title is not None:
        session_manager.rename_session(session_id, payload.title)
    if payload.pinned is not None:
        session_manager.toggle_pin_session(session_id, payload.pinned)
    return sess.model_dump()

@app.post("/api/filings/upload", response_model=UploadFilingResponse)
async def upload_filing_endpoint(
    file: UploadFile = File(...),
    ticker: str = Form(...),
    name: str = Form(...),
    market: str = Form("SET"),
    sector: str = Form("General"),
    currency: str = Form("THB"),
    year: int = Form(2025),
    quarter: str = Form("FY"),
    filing_type: str = Form("56-1"),
    accounting_standard: str = Form("TFRS"),
    revenue: Optional[float] = Form(None),
    net_profit: Optional[float] = Form(None),
    gross_profit: Optional[float] = Form(None),
    total_assets: Optional[float] = Form(None),
    total_equity: Optional[float] = Form(None),
) -> UploadFilingResponse:
    """Upload a filing PDF or Markdown document and ingest it dynamically into corpus & vectorstore."""
    ensure_corpus_and_index_ready()
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    metrics: Dict[str, float] = {}
    if revenue is not None:
        metrics["revenue"] = revenue
    if net_profit is not None:
        metrics["net_profit"] = net_profit
    if gross_profit is not None:
        metrics["gross_profit"] = gross_profit
    if total_assets is not None:
        metrics["total_assets"] = total_assets
    if total_equity is not None:
        metrics["total_equity"] = total_equity

    # Cascading validation
    norm_market = "SET" if market.upper() in ("SET", "TH") else "US"
    norm_ft = filing_type.strip()

    if market.upper() in ("SET", "TH"):
        if norm_ft not in TH_FILING_TYPES:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"ประเภทเอกสาร '{norm_ft}' ไม่สอดคล้องกับตลาดหุ้นไทย (SET/TH). "
                    f"ตัวเลือกที่ถูกต้องได้แก่: Form 56-1 One Report (56-1), "
                    f"งบการเงินรายไตรมาส (QUARTERLY_FINANCIALS) หรือ รายงาน MD&A (MDA)"
                )
            )
    elif market.upper() == "US":
        if norm_ft not in US_FILING_TYPES:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Filing type '{norm_ft}' is not valid for US SEC market. "
                    f"Allowed types: SEC Form 10-K, 10-Q, 20-F, or 8-K."
                )
            )

    try:
        res = ingest_filing_to_corpus(
            file_bytes=content,
            filename=file.filename or f"{ticker}.pdf",
            ticker=ticker,
            name=name,
            market=norm_market,
            sector=sector,
            currency="THB" if currency.upper() == "THB" else "USD",
            year=year,
            quarter=quarter,
            filing_type=norm_ft,
            accounting_standard=accounting_standard.upper(),
            metrics=metrics if metrics else None,
        )
        return UploadFilingResponse(**res)
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")


@app.post("/api/corpus/clear")
async def clear_corpus_endpoint() -> Dict[str, Any]:
    """Clear all active corpus filings, SQLite company records, and vector chunks for clean-slate use."""
    return clear_corpus()


@app.post("/api/translate", response_model=TranslateResponse)
async def translate_endpoint(req: TranslateRequest) -> TranslateResponse:
    """Translate financial text into Thai or English while preserving English international finance terms."""
    target_lang_name = "Thai (ภาษาไทย)" if req.target_lang == "th" else "English"
    prompt = (
        f"You are an expert bilingual financial translation analyst. "
        f"Translate the following financial text into {target_lang_name}.\n\n"
        "STRICT FINANCIAL TRANSLATION RULES:\n"
        "0. Output ONLY the translated text. Never include any conversational preamble, introduction, or explanations.\n"
        "1. DO NOT translate universally accepted English financial acronyms, metrics, and technical terms into awkward Thai. "
        "Keep the following terms in English: EBITDA, Gross Margin, Net Profit, Free Cash Flow, Operating Cash Flow, "
        "Altman Z-Score, DuPont ROE, CapEx, P/E ratio, EV/EBITDA, ARPU, TFRS, US GAAP, CAGR, YoY, QoQ, ROIC, Assets, Liabilities, Equity. "
        "Use natural phrasing and standard industry transliterations where appropriate.\n"
        "2. PRESERVE all inline citation brackets like `[doc_id:chunk_id]` verbatim without changing a single character.\n"
        "3. PRESERVE all numerical values, percentages, currencies ($ and ฿), and markdown formatting (headings, bold, lists).\n\n"
        f"TEXT TO TRANSLATE:\n{req.text}"
    )
    translated = llm_client.chat_completion(
        messages=[
            {"role": "system", "content": "You are a professional financial translation specialist."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    return TranslateResponse(
        translated_text=translated,
        target_lang=req.target_lang,
        preserved_terms=["EBITDA", "Net Profit", "DuPont ROE", "Altman Z-Score", "Gross Margin", "CapEx", "P/E", "Free Cash Flow"],
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Execute multi-step Quant/SQL/RAG Agent or pure Filing RAG."""
    ensure_corpus_and_index_ready()
    t0 = time.perf_counter()

    if request.mode == "rag_only":
        pipeline = get_rag_pipeline()
        rag_res = pipeline.answer(
            query=request.query,
            top_k=request.top_k,
            min_score=request.min_score,
            ticker_filter=request.ticker_filter,
            market_filter=request.market_filter,
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        report = QuantSignalReport(
            query=request.query,
            tickers_analyzed=[request.ticker_filter.upper()] if request.ticker_filter else [],
            signal="ABSTAIN" if rag_res["abstained"] else "EQUAL_WEIGHT",
            confidence=0.0 if rag_res["abstained"] else 0.85,
            executive_summary=rag_res["answer"],
            key_metrics=[],
            risk_factors=[],
            cross_market_comparison=None,
            guidance_2026_summary=None,
            citations=rag_res["citations"],
            abstained=rag_res["abstained"],
        )
        return ChatResponse(
            answer=rag_res["answer"],
            structured_report=report,
            retrieved_chunks=rag_res["retrieved_chunks"],
            agent_trace=[],
            abstained=rag_res["abstained"],
            latency_ms=elapsed_ms,
        )

    # Conversational memory lookup
    sess = session_manager.get_or_create_session(request.session_id)
    history = [{"role": m.role, "content": m.content} for m in sess.messages]
    session_manager.add_message(sess.id, role="user", content=request.query)

    # Agent mode with history
    resp = run_agent(
        question=request.query,
        max_steps=5,
        min_score=request.min_score,
        ticker_filter=request.ticker_filter,
        market_filter=request.market_filter,
        target_currency=request.target_currency,
        client=llm_client,
        history=history,
    )

    # Persist assistant response
    session_manager.add_message(
        session_id=sess.id,
        role="assistant",
        content=resp.answer,
        citations=[c.model_dump() for c in resp.structured_report.citations] if resp.structured_report else [],
        retrieved_chunks=[c.model_dump() for c in resp.retrieved_chunks],
        agent_trace=[t.model_dump() for t in resp.agent_trace],
        abstained=resp.abstained,
    )
    return resp


@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: ChatRequest) -> StreamingResponse:
    """Server-Sent Events (SSE) streaming endpoint."""
    ensure_corpus_and_index_ready()

    async def event_generator() -> AsyncGenerator[str, None]:
        queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def stream_cb(event_type: str, data: Dict[str, Any]) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, {"event": event_type, "data": data})

        def run_sync() -> None:
            try:
                sess = session_manager.get_or_create_session(request.session_id)
                history = [{"role": m.role, "content": m.content} for m in sess.messages]
                session_manager.add_message(sess.id, role="user", content=request.query)

                res = run_agent(
                    question=request.query,
                    max_steps=5,
                    min_score=request.min_score,
                    ticker_filter=request.ticker_filter,
                    market_filter=request.market_filter,
                    target_currency=request.target_currency,
                    client=llm_client,
                    stream_callback=stream_cb,
                    history=history,
                )
                session_manager.add_message(
                    session_id=sess.id,
                    role="assistant",
                    content=res.answer,
                    citations=[c.model_dump() for c in res.structured_report.citations] if res.structured_report else [],
                    retrieved_chunks=[c.model_dump() for c in res.retrieved_chunks],
                    agent_trace=[t.model_dump() for t in res.agent_trace],
                    abstained=res.abstained,
                )
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {
                        "event": "complete",
                        "data": res.model_dump(),
                    },
                )
            except Exception as e:
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {
                        "event": "error",
                        "data": {"error": str(e)},
                    },
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, {"event": "EOF", "data": {}})

        task = asyncio.create_task(asyncio.to_thread(run_sync))

        while True:
            item = await queue.get()
            event_name = item["event"]
            if event_name == "EOF":
                break
            payload = json.dumps(item["data"], ensure_ascii=False)
            yield f"event: {event_name}\ndata: {payload}\n\n"

        await task

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/memo/{ticker}")
async def generate_ticker_memo_endpoint(ticker: str) -> QuantSignalReport:
    """Generate an institutional memo for any registered ticker."""
    ensure_corpus_and_index_ready()
    norm_ticker = ticker.strip().upper()
    known = get_known_tickers()
    if norm_ticker not in known:
        raise HTTPException(
            status_code=404,
            detail=f"Ticker '{norm_ticker}' not found in active corpus. Please upload filing for this ticker first.",
        )

    memo_query = (
        f"Generate a full institutional quant & filing analyst memo for {norm_ticker}: "
        "compute DuPont ROE, Altman Z-Score, Free Cash Flow, and summarize off-balance-sheet "
        "commitments, FX hedging policy, contingencies, and Executive Guidance."
    )
    chat_res = run_agent(
        question=memo_query,
        max_steps=5,
        min_score=0.35,
        ticker_filter=norm_ticker,
        target_currency="USD",
    )

    if chat_res.structured_report is not None:
        return chat_res.structured_report

    try:
        fm = build_financial_metrics_schema(norm_ticker, year=2025)
        key_metrics = [fm]
        sig = "OVERWEIGHT" if fm.roe_dupont_pct >= 20.0 else "EQUAL_WEIGHT"
    except Exception:
        key_metrics = []
        sig = "EQUAL_WEIGHT"

    return QuantSignalReport(
        query=memo_query,
        tickers_analyzed=[norm_ticker],
        signal=sig,
        confidence=0.90,
        executive_summary=chat_res.answer,
        key_metrics=key_metrics,
        risk_factors=[],
        cross_market_comparison=None,
        guidance_2026_summary=None,
        citations=[],
        abstained=False,
    )


WEB_DIR: Path = PROJECT_ROOT / "web"


@app.get("/")
async def serve_index() -> FileResponse:
    """Serve index.html with strict cache-busting headers so users always see UI updates immediately."""
    return FileResponse(
        WEB_DIR / "index.html",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/app.js")
async def serve_app_js() -> FileResponse:
    """Serve app.js with strict no-cache headers."""
    return FileResponse(
        WEB_DIR / "app.js",
        media_type="application/javascript",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
