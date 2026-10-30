"""
Allow-Listed Tool Registry & Boundary Validation (`agent/tools.py`).

Defines Pydantic v2 input validation schemas for the 4 allow-listed tools:
1. `search_filings_rag(query, ticker, market, min_score)`
2. `query_financial_sql(sql_query)`
3. `compute_quant_metrics(ticker, year, metric_type)`
4. `compare_cross_market_peers(tickers, target_currency)` (aliased with `compare_cross_market`)

Security & Boundary Guardrails:
- Enforces strict allow-list checking (`ALLOWED_TOOLS`). Any unapproved tool name is rejected.
- Validates all tool arguments through Pydantic v2 before execution.
- Sanitizes and truncates tool outputs as untrusted data before returning to the LLM context.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Literal, Optional, Type
from pydantic import BaseModel, Field, ValidationError, field_validator

from agent.quant_calc import calculate_ticker_quant_metrics, compare_cross_market
from agent.sql_tool import execute_readonly_sql, nl_to_sql
from app.config import settings
from app.rag import GroundedRAGPipeline

MAX_TOOL_OUTPUT_CHARS = 4500


class SearchFilingsRAGInput(BaseModel):
    """Pydantic v2 input schema for `search_filings_rag`."""

    query: str = Field(..., min_length=2, description="Filing search query (e.g., 'SpaceX Q3 2026 revenue' or 'Bloom Energy 2026 guidance')")
    ticker: Optional[str] = Field(default=None, description="Optional ticker filter: BE, GOOG, JPM, PENG, SPCX")
    market: Optional[Literal["SET", "US"]] = Field(default=None, description="Optional market filter: 'SET' or 'US'")
    min_score: float = Field(default=0.35, ge=0.0, le=1.0, description="Minimum cosine similarity threshold")

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: Optional[str]) -> Optional[str]:
        if v is None or not v.strip():
            return None
        return v.strip().upper()


class QueryFinancialSQLInput(BaseModel):
    """Pydantic v2 input schema for `query_financial_sql`."""

    sql_query: str = Field(
        ...,
        min_length=6,
        description="Read-only SQLite SELECT query against `companies`, `financial_statements`, and `fx_rates` tables (no semicolons)",
    )


class ComputeQuantMetricsInput(BaseModel):
    """Pydantic v2 input schema for `compute_quant_metrics`."""

    ticker: str = Field(..., min_length=2, description="Company ticker symbol (e.g., 'SPCX', 'GOOG', 'BE', 'JPM', 'PENG')")
    year: Optional[int] = Field(default=None, ge=2020, le=2050, description="Fiscal year (e.g. 2024, 2025, 2026)")
    metric_type: Literal["dupont_roe", "altman_z", "liquidity_leverage", "fcf_cagr", "all"] = Field(
        default="all",
        description="Subset of deterministic quant metrics to compute",
    )

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        return v.strip().upper()


class CompareCrossMarketPeersInput(BaseModel):
    """Pydantic v2 input schema for `compare_cross_market_peers`."""

    tickers: List[str] = Field(
        ...,
        min_length=1,
        max_length=8,
        description="List of SET and/or US tickers to compare side-by-side (e.g., ['DELTA', 'NVDA'])",
    )
    target_currency: Literal["USD", "THB"] = Field(
        default="USD",
        description="Common reporting currency ('USD' or 'THB') for normalizing absolute financial figures via fx_rates",
    )

    @field_validator("tickers")
    @classmethod
    def normalize_tickers(cls, v: List[str]) -> List[str]:
        cleaned = [t.strip().upper() for t in v if t and t.strip()]
        if not cleaned:
            raise ValueError("tickers list cannot be empty.")
        return cleaned


def sanitize_untrusted_tool_output(raw_data: Any, max_chars: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    """
    Sanitize and truncate tool output before returning it to the LLM context window:
    - Strips prompt-injection control patterns (`IGNORE PREVIOUS INSTRUCTIONS`, `<|im_start|>`, etc.)
    - Enforces `max_chars` ceiling so oversized payloads never overflow context.
    """
    if isinstance(raw_data, str):
        text = raw_data
    else:
        text = json.dumps(raw_data, ensure_ascii=False, indent=2)

    # Neutralize potential prompt injection directives embedded inside untrusted text
    injection_patterns = [
        r"(?i)ignore\s+(?:all\s+)?previous\s+instructions",
        r"(?i)system\s+override",
        r"<\|im_start\|>",
        r"<\|im_end\|>",
    ]
    for pat in injection_patterns:
        text = re.sub(pat, "[REDACTED_UNTRUSTED_DIRECTIVE]", text)

    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[TRUNCATED_TOOL_OUTPUT]"
    return text


# Shared RAG pipeline singleton for tool invocations
_rag_pipeline_instance: Optional[GroundedRAGPipeline] = None


def get_rag_pipeline(force_reload: bool = False) -> GroundedRAGPipeline:
    """Lazy-initialize or reload the shared GroundedRAGPipeline instance from disk."""
    global _rag_pipeline_instance
    if _rag_pipeline_instance is None or force_reload or len(_rag_pipeline_instance.vectorstore.chunks) == 0:
        if settings.vector_index_path.exists():
            try:
                store = VectorStore.load(settings.vector_index_path, client=llm_client)
                _rag_pipeline_instance = GroundedRAGPipeline(vectorstore=store, client=llm_client)
            except Exception:
                _rag_pipeline_instance = GroundedRAGPipeline()
        else:
            _rag_pipeline_instance = GroundedRAGPipeline()
    return _rag_pipeline_instance


def _exec_search_filings_rag(args: SearchFilingsRAGInput) -> Dict[str, Any]:
    pipeline = get_rag_pipeline()
    from app.rag import resolve_ticker
    canonical_ticker = resolve_ticker(args.ticker) if args.ticker else None
    chunks = pipeline.retrieve(
        query=args.query,
        top_k=settings.top_k,
        min_score=args.min_score,
        ticker_filter=canonical_ticker,
        market_filter=args.market,
    )
    return {
        "tool": "search_filings_rag",
        "query": args.query,
        "ticker_filter": args.ticker,
        "market_filter": args.market,
        "min_score": args.min_score,
        "chunks_found": len(chunks),
        "chunks": [
            {
                "citation_tag": c.citation_tag,
                "doc_id": c.doc_id,
                "chunk_id": c.chunk_id,
                "ticker": c.ticker,
                "market": c.market,
                "section": c.section,
                "score": c.score,
                "text": c.text,
            }
            for c in chunks
        ],
    }


def _exec_query_financial_sql(args: QueryFinancialSQLInput) -> Dict[str, Any]:
    sql_text = args.sql_query.strip()
    # If the caller passed a natural language question instead of SELECT, route via nl_to_sql
    if not sql_text.upper().startswith(("SELECT", "WITH")) and not any(
        kw in sql_text.upper() for kw in ("DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "ATTACH", "PRAGMA", ";")
    ):
        return nl_to_sql(sql_text)
    return execute_readonly_sql(sql_text)


def _exec_compute_quant_metrics(args: ComputeQuantMetricsInput) -> Dict[str, Any]:
    return calculate_ticker_quant_metrics(
        ticker=args.ticker,
        year=args.year if args.year is not None else 2025,
        metric_type=args.metric_type,
    )


def _exec_compare_cross_market_peers(args: CompareCrossMarketPeersInput) -> Dict[str, Any]:
    return compare_cross_market(
        tickers=args.tickers,
        target_currency=args.target_currency,
    )


ALLOWED_TOOLS: Dict[str, Dict[str, Any]] = {
    "search_filings_rag": {
        "schema": SearchFilingsRAGInput,
        "handler": _exec_search_filings_rag,
        "canonical_trace_name": "search_filings_rag",
        "description": (
            "Search Thai SET Form 56-1 One Reports and US SEC Form 10-K filings for qualitative disclosures, "
            "MD&A, footnote commitments (leases, FX hedging, litigation, concentration, TFRS vs US GAAP R&D rules), "
            "and 2026 Executive Guidance numbers. Returns chunks with [doc_id:chunk_id] citations."
        ),
    },
    "query_financial_sql": {
        "schema": QueryFinancialSQLInput,
        "handler": _exec_query_financial_sql,
        "canonical_trace_name": "query_financial_sql",
        "description": (
            "Execute a read-only SQLite SELECT query against `companies`, `financial_statements` (2023-2025), "
            "and `fx_rates` tables. Never include semicolons (;) or destructive DDL/DML keywords."
        ),
    },
    "compute_quant_metrics": {
        "schema": ComputeQuantMetricsInput,
        "handler": _exec_compute_quant_metrics,
        "canonical_trace_name": "compute_quant_metrics",
        "description": (
            "Run deterministic Python calculations for a SINGLE SET or US ticker only (e.g. DELTA or NVDA). "
            "If 2 or more tickers or a comparison is requested, you MUST call compare_cross_market_peers instead. "
            "Computes 3-Step & 5-Step DuPont ROE, Altman Z-Score & distress zone, Current Ratio, IBD/E, Net Debt/EBITDA, "
            "Free Cash Flow, FCF Yield, and 3-Year Revenue & Net Profit CAGR (2023-2025)."
        ),
    },
    "compare_cross_market_peers": {
        "schema": CompareCrossMarketPeersInput,
        "handler": _exec_compare_cross_market_peers,
        "canonical_trace_name": "compare_cross_market",
        "description": (
            "MANDATORY when comparing 2 or more companies or cross-market peers side-by-side (e.g., DELTA and NVDA). "
            "Normalizes absolute figures (Revenue, Net Profit, EBITDA, FCF, Market Cap) into a common target currency "
            "(USD or THB) using fx_rates while comparing currency-neutral ratios (ROE %, Margins, FCF Yield %, Altman Z, CAGR %) directly."
        ),
    },
    "compare_cross_market": {
        "schema": CompareCrossMarketPeersInput,
        "handler": _exec_compare_cross_market_peers,
        "canonical_trace_name": "compare_cross_market",
        "description": "Alias for compare_cross_market_peers.",
    },
}


def get_openai_tools_spec() -> List[Dict[str, Any]]:
    """Return the OpenAI function-calling `tools` specification for the 4 primary allow-listed tools."""
    primary_names = [
        "compare_cross_market_peers",
        "compute_quant_metrics",
        "search_filings_rag",
        "query_financial_sql",
    ]
    specs: List[Dict[str, Any]] = []
    for name in primary_names:
        entry = ALLOWED_TOOLS[name]
        schema_cls: Type[BaseModel] = entry["schema"]
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": entry["description"],
                    "parameters": schema_cls.model_json_schema(),
                },
            }
        )
    return specs


def execute_allowlisted_tool(
    tool_name: str,
    raw_arguments: Dict[str, Any] | str,
) -> Dict[str, Any]:
    """
    Validate tool name against `ALLOWED_TOOLS`, validate arguments via Pydantic v2,
    execute the tool safely, and return both raw structured data and sanitized string output.
    """
    clean_name = (tool_name or "").strip()
    if clean_name not in ALLOWED_TOOLS:
        return {
            "ok": False,
            "tool_name": clean_name,
            "canonical_trace_name": "final_answer",
            "error": (
                f"Security Guardrail Violation: Tool '{clean_name}' is not in ALLOWED_TOOLS. "
                f"Permitted tools: {list(ALLOWED_TOOLS.keys())}"
            ),
            "sanitized_output": f"ERROR: Tool '{clean_name}' is not permitted.",
            "raw_result": None,
        }

    entry = ALLOWED_TOOLS[clean_name]
    schema_cls: Type[BaseModel] = entry["schema"]
    handler: Callable[[Any], Dict[str, Any]] = entry["handler"]
    canonical_name: str = entry["canonical_trace_name"]

    # Parse raw_arguments if provided as JSON string
    if isinstance(raw_arguments, str):
        try:
            parsed_args = json.loads(raw_arguments) if raw_arguments.strip() else {}
        except json.JSONDecodeError as exc:
            return {
                "ok": False,
                "tool_name": clean_name,
                "canonical_trace_name": canonical_name,
                "error": f"Invalid JSON arguments for tool '{clean_name}': {str(exc)}",
                "sanitized_output": f"ERROR: Invalid JSON arguments for '{clean_name}'.",
                "raw_result": None,
            }
    elif isinstance(raw_arguments, dict):
        parsed_args = raw_arguments
    else:
        parsed_args = {}

    try:
        validated_input = schema_cls.model_validate(parsed_args)
    except ValidationError as val_err:
        return {
            "ok": False,
            "tool_name": clean_name,
            "canonical_trace_name": canonical_name,
            "error": f"Pydantic Argument Validation Error for '{clean_name}': {str(val_err)}",
            "sanitized_output": f"ERROR: Argument validation failed for '{clean_name}': {str(val_err)}",
            "raw_result": None,
        }

    try:
        raw_result = handler(validated_input)
        sanitized = sanitize_untrusted_tool_output(raw_result)
        return {
            "ok": True,
            "tool_name": clean_name,
            "canonical_trace_name": canonical_name,
            "validated_args": validated_input.model_dump(),
            "raw_result": raw_result,
            "sanitized_output": sanitized,
        }
    except Exception as exc:
        err_msg = f"Tool Execution Error in '{clean_name}': {str(exc)}"
        return {
            "ok": False,
            "tool_name": clean_name,
            "canonical_trace_name": canonical_name,
            "validated_args": validated_input.model_dump(),
            "error": err_msg,
            "sanitized_output": err_msg,
            "raw_result": None,
        }