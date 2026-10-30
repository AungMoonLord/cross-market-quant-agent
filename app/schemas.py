"""
Comprehensive Pydantic v2 Schemas for Structured Extraction, RAG Citations,
Quant Signals, Agent Traces, Chat API, and Evaluation Suite.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class Citation(BaseModel):
    """Inline citation pointing to a specific filing document and chunk."""

    doc_id: str = Field(..., description="Document identifier, e.g., 'ADVANC_56-1_2025' or 'NVDA_10-K_2025'")
    chunk_id: str = Field(..., description="Chunk identifier within the document, e.g., 'c0', 'c4'")
    ticker: Optional[str] = Field(default=None, description="Company ticker symbol")
    market: Optional[str] = Field(default=None, description="Market jurisdiction: SET, TH, or US")
    section: Optional[str] = Field(default=None, description="Section header title")
    quote_snippet: Optional[str] = Field(default=None, description="Supporting verbatim snippet from the chunk")

    @property
    def tag(self) -> str:
        return f"[{self.doc_id}:{self.chunk_id}]"


MarketType = Literal["SET", "TH", "US"]

TH_FILING_TYPES = {"56-1", "56-1_ONE_REPORT", "QUARTERLY_FINANCIALS", "MDA"}
US_FILING_TYPES = {"10-K", "10-Q", "20-F", "8-K"}
ALL_FILING_TYPES = TH_FILING_TYPES | US_FILING_TYPES

class RetrievedChunk(BaseModel):
    """A single chunk stored in or retrieved from the VectorStore."""

    doc_id: str = Field(..., description="Document identifier, e.g., 'ADVANC_56-1_2025'")
    chunk_id: str = Field(..., description="Chunk identifier, e.g., 'c0'")
    ticker: str = Field(..., description="Company ticker symbol")
    market: str = Field(default="SET", description="Market: SET, TH, or US")
    filing_type: str = Field(default="56-1", description="Filing type: 56-1, 10-K, 10-Q, 20-F, 8-K, QUARTERLY_FINANCIALS, MDA")
    section: str = Field(..., description="Markdown section header")
    text: str = Field(..., description="Full chunk text (including context header when applicable)")
    score: float = Field(default=0.0, description="Cosine similarity score in [0.0, 1.0]")

    @property
    def citation_tag(self) -> str:
        return f"[{self.doc_id}:{self.chunk_id}]"


class FinancialMetrics(BaseModel):
    """Deterministic DuPont, Valuation, Cash Flow, and Solvency Quant Metrics."""

    ticker: str = Field(..., description="Company ticker symbol")
    market: Literal["SET", "US"] = Field(..., description="Market: SET or US")
    year: int = Field(..., description="Fiscal year")
    native_currency: Literal["THB", "USD"] = Field(..., description="Original reporting currency")
    reporting_currency: Literal["THB", "USD"] = Field(..., description="Currency in which monetary figures are expressed")
    fx_rate_applied: float = Field(..., description="FX conversion rate applied (1.0 if unadjusted)")
    revenue: float = Field(..., description="Revenue in Millions of reporting_currency")
    net_profit: float = Field(..., description="Net profit in Millions of reporting_currency")
    ebitda: float = Field(..., description="EBITDA in Millions of reporting_currency")
    free_cash_flow: float = Field(..., description="Operating Cash Flow minus CapEx in Millions")
    net_profit_margin_pct: float = Field(..., description="Net Profit / Revenue (%)")
    ebitda_margin_pct: float = Field(..., description="EBITDA / Revenue (%)")
    asset_turnover: float = Field(..., description="Revenue / Total Assets (x)")
    equity_multiplier: float = Field(..., description="Total Assets / Total Equity (x)")
    roe_dupont_pct: float = Field(..., description="3-Stage DuPont ROE (%)")
    roic_pct: float = Field(..., description="Return on Invested Capital (%)")
    debt_to_equity: float = Field(..., description="Interest-Bearing Debt / Total Equity (x)")
    net_debt_to_ebitda: float = Field(..., description="(Interest-Bearing Debt - Cash) / EBITDA (x)")
    current_ratio: float = Field(..., description="Current Assets / Current Liabilities (x)")
    pe_ratio: float = Field(..., description="Market Cap / Net Profit (x)")
    ev_to_ebitda: float = Field(..., description="Enterprise Value / EBITDA (x)")
    altman_z_score: float = Field(..., description="Manufacturing/Non-Manufacturing Altman Z-Score")
    altman_zone: Literal["SAFE", "GREY", "DISTRESS"] = Field(..., description="Credit risk zone classification")


class RiskFactor(BaseModel):
    """Structured risk factor extracted from a 56-1 One Report or Form 10-K filing."""

    category: Literal[
        "FX_AND_DERIVATIVES",
        "OFF_BALANCE_SHEET_LEASES",
        "LITIGATION_AND_REGULATORY",
        "CONCENTRATION_RISK",
        "ACCOUNTING_STANDARD_DIFF",
        "OPERATIONAL_MACRO",
    ] = Field(..., description="Risk classification category")
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(..., description="Institutional risk severity")
    summary: str = Field(..., description="Concise quantitative summary of the risk factor")
    quantified_exposure: Optional[str] = Field(
        default=None,
        description="Explicit monetary amount or percentage concentration mentioned in filing",
    )
    citation: str = Field(..., description="Inline citation in [doc_id:chunk_id] format")

    @field_validator("citation")
    @classmethod
    def validate_citation_format(cls, v: str) -> str:
        v = v.strip()
        if not (v.startswith("[") and v.endswith("]") and ":" in v):
            return f"[{v}]" if ":" in v else "[UNKNOWN:c0]"
        return v


class CrossMarketComparison(BaseModel):
    """Structured cross-market comparison between a Thai SET company and a US peer."""

    primary_ticker: str = Field(..., description="First company ticker (e.g., DELTA or ADVANC)")
    peer_ticker: str = Field(..., description="Comparison company ticker (e.g., NVDA or AAPL)")
    normalized_currency: Literal["USD", "THB"] = Field(default="USD", description="Common currency used for normalization")
    fx_rate_used: float = Field(..., description="USD/THB exchange rate used for normalization")
    valuation_spread_summary: str = Field(..., description="Comparison of P/E, EV/EBITDA, and FCF yield")
    dupont_roe_comparison: str = Field(..., description="Comparison of Net Margin, Asset Turnover, and Leverage drivers")
    accounting_standard_adjustments: str = Field(
        ...,
        description="Key differences between TFRS (TAS 38 R&D capitalization, leases) and US GAAP (ASC 730 immediate R&D expensing)",
    )
    citations: List[str] = Field(default_factory=list, description="List of [doc_id:chunk_id] citations")


class QuantSignalReport(BaseModel):
    """Institutional Quant & Filing Analyst Signal Report (Pydantic v2 structured output)."""

    query: str = Field(..., description="Original user question")
    tickers_analyzed: List[str] = Field(default_factory=list, description="Tickers covered in the analysis")
    signal: Literal["OVERWEIGHT", "EQUAL_WEIGHT", "UNDERWEIGHT", "ABSTAIN"] = Field(
        ...,
        description="Institutional quant recommendation or ABSTAIN if insufficient evidence",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    executive_summary: str = Field(
        ...,
        description="Grounded synthesis with mandatory inline [doc_id:chunk_id] citations for filing facts",
    )
    key_metrics: List[FinancialMetrics] = Field(
        default_factory=list,
        description="Deterministic financial metrics computed from SQLite/Quant calculator",
    )
    risk_factors: List[RiskFactor] = Field(
        default_factory=list,
        description="Structured risk factors extracted from 56-1 / 10-K footnotes",
    )
    cross_market_comparison: Optional[CrossMarketComparison] = Field(
        default=None,
        description="Populated when comparing SET and US equities across currencies/standards",
    )
    guidance_2026_summary: Optional[str] = Field(
        default=None,
        description="2026 executive guidance retrieved from filing text with [doc_id:chunk_id] citation",
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="Verified list of citations used in the report",
    )
    abstained: bool = Field(
        default=False,
        description="True if the agent abstained due to low retrieval scores or out-of-scope query",
    )


class AgentStepTrace(BaseModel):
    """Single step in the multi-step Agent execution loop."""

    step_number: int = Field(..., description="1-indexed step number")
    thought: str = Field(..., description="Agent reasoning for selecting this action")
    tool_name: Literal[
        "search_filings_rag",
        "query_financial_sql",
        "compute_quant_metrics",
        "compare_cross_market",
        "final_answer",
    ] = Field(..., description="Allow-listed tool invoked at this step")
    tool_input: Dict[str, Any] = Field(default_factory=dict, description="Arguments passed to the tool")
    tool_output_summary: str = Field(..., description="Concise summary or JSON snippet returned by the tool")
    latency_ms: float = Field(default=0.0, description="Execution time of the step in milliseconds")


class ChatRequest(BaseModel):
    """REST / SSE API request payload for the Quant Terminal."""

    query: str = Field(..., min_length=2, description="Financial or filing question to analyze")
    session_id: Optional[str] = Field(default=None, description="Chat session identifier for conversational memory")
    history: Optional[List[Dict[str, str]]] = Field(default_factory=list, description="Conversation history for context memory")
    ticker_filter: Optional[str] = Field(default=None, description="Optional ticker filter (e.g., ADVANC, NVDA)")
    market_filter: Optional[Literal["SET", "US"]] = Field(default=None, description="Optional market filter (SET or US)")
    target_currency: Literal["NATIVE", "USD", "THB"] = Field(
        default="USD",
        description="Target currency for cross-market normalization",
    )
    min_score: float = Field(default=0.35, ge=0.0, le=1.0, description="Minimum cosine similarity threshold")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    mode: Literal["agent", "rag_only"] = Field(
        default="agent",
        description="'agent' runs full tool loop (SQL + Quant + RAG); 'rag_only' runs pure filing RAG",
    )
    language: Optional[Literal["auto", "th", "en"]] = Field(
        default="auto",
        description="Response language preference: 'auto' (inferred from query), 'th' (Thai), or 'en' (English)",
    )


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Financial text to translate")
    target_lang: Literal["th", "en"] = Field(default="th", description="Target language: 'th' or 'en'")


class TranslateResponse(BaseModel):
    translated_text: str = Field(..., description="Translated financial text preserving finance terms and citations")
    target_lang: str = Field(..., description="Target language used")
    preserved_terms: List[str] = Field(default_factory=list, description="Preserved international financial terminology")


class UploadFilingResponse(BaseModel):
    status: str = Field(..., description="Upload status")
    ticker: str = Field(..., description="Stock ticker symbol")
    name: str = Field(..., description="Company name")
    doc_id: str = Field(..., description="Document identifier")
    filing_path: str = Field(..., description="Path to saved Markdown filing")
    chunks_added: int = Field(..., description="Number of vector chunks generated and embedded")
    total_corpus_chunks: int = Field(..., description="Total active chunks in corpus")
    characters_extracted: int = Field(..., description="Total characters extracted from file")
    preview_text: str = Field(..., description="Preview snippet of extracted text")


class ChatResponse(BaseModel):
    """Complete response returned by /api/chat."""

    answer: str = Field(..., description="Grounded markdown response with inline [doc_id:chunk_id] citations")
    structured_report: Optional[QuantSignalReport] = Field(
        default=None,
        description="Validated Pydantic v2 QuantSignalReport",
    )
    retrieved_chunks: List[RetrievedChunk] = Field(
        default_factory=list,
        description="Filing chunks retrieved above min_score",
    )
    agent_trace: List[AgentStepTrace] = Field(
        default_factory=list,
        description="Step-by-step tool invocation trace",
    )
    abstained: bool = Field(default=False, description="Whether the system abstained")
    latency_ms: float = Field(default=0.0, description="Total end-to-end latency in milliseconds")


class EvalResult(BaseModel):
    """Per-example evaluation metrics for Retrieval + End-to-End Judge."""

    qid: str = Field(..., description="Evaluation question ID, e.g., 'Q01'")
    category: str = Field(..., description="Question category (SET_56_1, US_10_K, CROSS_MARKET, SQL_QUANT, ABSTAIN)")
    query: str = Field(..., description="Evaluation question text")
    expected_doc_ids: List[str] = Field(default_factory=list, description="Gold relevant doc_id:chunk_id list")
    retrieved_doc_ids: List[str] = Field(default_factory=list, description="Retrieved doc_id:chunk_id list")
    recall_at_k: float = Field(..., description="Recall@K score")
    mrr: float = Field(..., description="Mean Reciprocal Rank score")
    ndcg_at_k: float = Field(..., description="nDCG@K score")
    citation_precision: float = Field(..., description="Fraction of cited chunks that match gold or retrieved context")
    numeric_exact_match: float = Field(..., description="Fraction of required gold numbers/keywords present in answer")
    faithfulness_score: float = Field(..., description="Groundedness/faithfulness score in [0.0, 1.0]")
    abstain_correct: bool = Field(..., description="Whether abstention behavior matched gold expectation")
    answer_preview: str = Field(default="", description="First 250 characters of generated answer")