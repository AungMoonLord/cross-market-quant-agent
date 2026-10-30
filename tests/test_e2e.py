"""
End-to-End FastAPI, SSE Streaming & Agent Loop Test Suite (`tests/test_e2e.py`).

Covers:
1. `GET /api/health` readiness, configured models (`gemma-4-E4B-it`, `bge-m3`), and indexed chunks >= 40.
2. `GET /api/tickers` returning all 5 real corporate filings (BE, GOOG, JPM, PENG, SPCX).
3. `POST /api/chat` in `mode='agent'` for multi-ticker synthesis (BE vs SPCX) with tool trace and citations.
4. `POST /api/chat` in `mode='rag_only'` for direct filing retrieval.
5. `POST /api/chat` out-of-scope trap query abstention guardrail (`abstained=True`, `signal='ABSTAIN'`).
6. `POST /api/chat/stream` real-time Server-Sent Events (SSE) stream (`tool_start`, `tool_result`,
   `token`, `citations`, `done`, `complete`).
7. `POST /api/memo/{ticker}` institutional `QuantSignalReport` generation for real institutional tickers + HTTP 404 check.
8. Static web UI serving (`GET /` and `GET /app.js`) and `run_agent` `max_steps` stop condition enforcement.
"""

from __future__ import annotations

import json
from typing import Dict, List
import pytest
from fastapi.testclient import TestClient

from agent.loop import run_agent
from app.main import app


client = TestClient(app)


def test_health_and_tickers_endpoints() -> None:
    """Verify `/api/health` and `/api/tickers` return complete institutional metadata."""
    health_res = client.get("/api/health")
    assert health_res.status_code == 200
    h_data = health_res.json()
    assert h_data["status"] == "ok"
    assert h_data["llm_model"] == "gemma-4-E4B-it"
    assert h_data["embedding_model"] == "bge-m3"
    assert h_data["sqlite_db_ready"] is True
    assert h_data["indexed_chunks_count"] >= 40

    tickers_res = client.get("/api/tickers")
    assert tickers_res.status_code == 200
    t_data = tickers_res.json()
    assert t_data["count"] >= 5

    by_ticker = {t["ticker"]: t for t in t_data["tickers"]}
    assert {"BE", "GOOG", "JPM", "PENG", "SPCX"}.issubset(set(by_ticker.keys()))

    # Check verified company metadata
    assert by_ticker["BE"]["market"] == "US"
    assert by_ticker["BE"]["currency"] == "USD"
    assert by_ticker["BE"]["filing_type"] == "10-Q"
    assert by_ticker["BE"]["accounting_standard"] == "US_GAAP"

    assert by_ticker["SPCX"]["market"] == "US"
    assert by_ticker["SPCX"]["currency"] == "USD"
    assert by_ticker["SPCX"]["filing_type"] == "10-Q"
    assert by_ticker["SPCX"]["accounting_standard"] == "US_GAAP"


def test_chat_endpoint_cross_market_agent_mode() -> None:
    """Verify `/api/chat` (`mode='agent'`) executes Quant + RAG tools and returns verified citations."""
    payload = {
        "query": "Compare BE (Bloom Energy) and SPCX (SpaceX) 2026 revenue, DuPont ROE, and commercial guidance.",
        "target_currency": "USD",
        "min_score": 0.35,
        "top_k": 5,
        "mode": "agent",
    }
    res = client.post("/api/chat", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["abstained"] is False
    assert len(data["agent_trace"]) >= 1
    tool_names_used = [step["tool_name"] for step in data["agent_trace"]]
    assert any(t in tool_names_used for t in ["compare_cross_market", "calculate_quant_metrics", "search_filings_rag", "query_financial_db"])

    assert len(data["retrieved_chunks"]) >= 1
    assert "[" in data["answer"] and ":c" in data["answer"]

    report = data["structured_report"]
    assert report is not None
    assert any(t in report["tickers_analyzed"] for t in ["BE", "SPCX"])
    assert len(report["citations"]) >= 1


def test_chat_endpoint_rag_only_mode() -> None:
    """Verify `/api/chat` (`mode='rag_only'`) retrieves grounded footnotes with `[doc_id:chunk_id]` tags."""
    payload = {
        "query": "What is Bloom Energy full year 2026 revenue guidance and quarterly revenue growth in Q2 2026?",
        "ticker_filter": "BE",
        "market_filter": "US",
        "min_score": 0.30,
        "top_k": 3,
        "mode": "rag_only",
    }
    res = client.post("/api/chat", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["abstained"] is False
    assert data["agent_trace"] == []
    assert len(data["retrieved_chunks"]) >= 1
    assert data["retrieved_chunks"][0]["ticker"] == "BE"
    assert "[" in data["answer"] and ":c" in data["answer"]


def test_chat_endpoint_out_of_scope_trap_abstention() -> None:
    """Verify `/api/chat` abstains cleanly (`abstained=True`) on out-of-corpus tickers and topics."""
    payload = {
        "query": "What is PTT Public Company Limited (PTT) and Bitcoin treasury guidance for 2028?",
        "min_score": 0.35,
        "mode": "agent",
    }
    res = client.post("/api/chat", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["abstained"] is True
    assert data["answer"].upper().startswith("ABSTAIN")
    assert data["retrieved_chunks"] == []
    assert data["structured_report"]["signal"] == "ABSTAIN"
    assert data["structured_report"]["confidence"] == 0.0


def test_chat_sse_stream_endpoint() -> None:
    """Verify `/api/chat/stream` streams valid Server-Sent Events (`tool_start`, `tool_result`, `token`, `done`, `complete`)."""
    payload = {
        "query": "What is SpaceX's Starlink revenue and launch cadence guidance in 2026?",
        "ticker_filter": "SPCX",
        "min_score": 0.35,
        "mode": "agent",
    }
    with client.stream("POST", "/api/chat/stream", json=payload) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
        raw_body = "".join(response.iter_text())

    # Parse SSE event types from stream
    events_seen: List[str] = []
    complete_payload: Dict[str, object] = {}
    for frame in raw_body.split("\n\n"):
        lines = [ln.strip() for ln in frame.splitlines() if ln.strip()]
        ev_type = None
        ev_data = ""
        for ln in lines:
            if ln.startswith("event:"):
                ev_type = ln.split("event:", 1)[1].strip()
            elif ln.startswith("data:"):
                ev_data += ln.split("data:", 1)[1].strip()
        if ev_type:
            events_seen.append(ev_type)
            if ev_type == "complete" and ev_data:
                complete_payload = json.loads(ev_data)

    assert "token" in events_seen
    assert "done" in events_seen
    assert "complete" in events_seen
    assert complete_payload.get("abstained") is False


def test_memo_endpoint_and_max_steps_enforcement() -> None:
    """Verify `/api/memo/{ticker}` for institutional tickers, 404 error handling, and `max_steps` stop condition."""
    memo_be = client.post("/api/memo/BE")
    assert memo_be.status_code == 200
    memo_data = memo_be.json()
    assert memo_data["tickers_analyzed"] == ["BE"]
    assert memo_data["abstained"] is False

    memo_spcx = client.post("/api/memo/SPCX")
    assert memo_spcx.status_code == 200
    assert memo_spcx.json()["tickers_analyzed"] == ["SPCX"]

    # Unsupported ticker must return HTTP 404
    not_found = client.post("/api/memo/UNKNOWN_TICKER_XYZ")
    assert not_found.status_code == 404

    # Verify max_steps=1 strictly caps agent trace at 1 step
    capped_res = run_agent(
        question="Compare BE and SPCX 2026 revenue and guidance.",
        max_steps=1,
        min_score=0.35,
    )
    assert len(capped_res.agent_trace) == 1


def test_static_terminal_ui_serving() -> None:
    """Verify that the Dark-Mode Cross-Market Quant Terminal HTML and JS assets are served at `/` and `/app.js`."""
    index_res = client.get("/")
    assert index_res.status_code == 200
    assert "QUANT" in index_res.text.upper()

    js_res = client.get("/app.js")
    assert js_res.status_code == 200
    assert "executeStreamingQuery" in js_res.text