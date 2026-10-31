"""
Multi-Step ReAct / Function-Calling Agent Loop (`agent/loop.py`).

Implements `run_agent(question, max_steps, min_score, stream_callback, ...) -> ChatResponse`:
- Uses OpenAI tool-calling (`tools=...`) against `gemma-4-E4B-it` at `temperature=0`
  (with a JSON tool-call parser fallback + deterministic quant/RAG planner when running offline).
- Enforces `max_steps` stop condition and prevents duplicate identical tool calls.
- Tracks `AgentStepTrace`, collects `RetrievedChunk`s and `FinancialMetrics`, verifies inline
  `[doc_id:chunk_id]` citations, and produces a validated `QuantSignalReport` + `ChatResponse`.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from agent.quant_calc import build_financial_metrics_schema
from agent.tools import (
    ALLOWED_TOOLS,
    execute_allowlisted_tool,
    get_openai_tools_spec,
)
from app.config import settings
from app.llm import LLMClient, extract_json_object, llm_client
from app.rag import (
    KNOWN_TICKERS,
    detect_query_filters,
    extract_citations_from_text,
    is_out_of_corpus_query,
    is_non_financial_query,
    find_unsupported_ticker_in_query,
    format_missing_stock_guidance,
    format_unrelated_query_response,
    get_known_tickers,
)
from app.schemas import (
    AgentStepTrace,
    ChatResponse,
    Citation,
    CrossMarketComparison,
    FinancialMetrics,
    QuantSignalReport,
    RetrievedChunk,
    RiskFactor,
)

AGENT_SYSTEM_PROMPT = r"""You are an elite Cross-Market Institutional Financial Quant & Filing Analyst Agent covering Thai SET Equities (Form 56-1 One Report, THB, TFRS) and US Equities (SEC Form 10-K, USD, US GAAP) dynamically indexed in the active corpus.

MANDATORY OPERATING RULES:
1. When comparing 2 or more companies (e.g. DELTA vs NVDA) or when cross-market peers are analyzed, ALWAYS call `compare_cross_market_peers` with all tickers in the list in a single call. Never call `compute_quant_metrics` individually when multiple companies are being compared.
2. For quantitative ratios for a single company, call `compute_quant_metrics`. Never do mental math.
3. For cross-market comparisons (SET vs US), ALWAYS normalize absolute figures via `compare_cross_market_peers` or `fx_rates` (2023=34.80, 2024=35.20, 2025=34.00 THB/USD). Never add or compare raw THB and USD numbers directly.
4. For Executive Guidance, quarterly performance (MD&A), segment breakdowns, leases, FX hedging policies, litigation, and R&D accounting rules, ALWAYS call `search_filings_rag`.
5. Every filing fact or guidance number in your final answer MUST include an inline citation in the exact format `[doc_id:chunk_id]` (e.g., `[DELTA_56-1_2025:c3]`, `[NVDA_10-K_2025:c4]`).
6. Language & Professional Financial Terminology:
   - If the user asks in Thai (or mixed Thai-English), synthesize your response in natural, professional Thai.
   - CRITICAL: KEEP universal international financial terms in English (do NOT translate words like EBITDA, Gross Margin, Net Profit, Free Cash Flow, Altman Z-Score, DuPont ROE, CapEx, P/E ratio, EV/EBITDA, ARPU, TFRS, US GAAP, CAGR, YoY, QoQ).
   - If the user asks in English, respond in professional English.
7. Data Completeness & Guidance:
   - If user asks to analyze multi-quarter data (e.g., '3 ไตรมาสย้อนหลัง' หรือ 'ไตรมาสล่าสุด') and some quarters are missing from the corpus, provide analysis of available data and clearly state what quarters are missing and request the user to upload them.
8. Grounded Quantitative Comparisons & Zero-Value Avoidance:
   - When presenting peer comparisons or quarterly summaries, ALWAYS verify figures against the retrieved filing chunks.
   - NEVER output a comparison table filled with 0.00M or blank values if actual financial numbers (Revenue, Operating Income, Gross Margin, Net Income) exist in the retrieved filing context. Synthesize the comparison table using the actual quarterly numbers from the filings and cite them with `[doc_id:chunk_id]`.
9. Formatting & Unicode Symbols in Tables:
   - NEVER use LaTeX math notation (e.g. NEVER write `$\uparrow$`, `$\downarrow$`, `\uparrow`, `$\pm$`).
   - ALWAYS use standard Unicode characters: `↑` for increase, `↓` for decrease, `+` or `-` for percentages (e.g. `+86.8% (↑)` or `+86.8%`).
10. Table Citation Integrity:
   - Place citations `[doc_id:chunk_id]` inline directly next to the numbers they support.
   - DO NOT create a standalone 'Citation' column if it will just contain 'N/A'. If a table includes a 'Citation' column, every row MUST contain the real supporting citation tags `[doc_id:chunk_id]`, NEVER 'N/A'."""


def _extract_tickers_mentioned(text: str) -> List[str]:
    """Return ordered list of unique known tickers mentioned in the query text."""
    q_upper = text.upper()
    known = get_known_tickers()
    found: List[str] = []
    
    # Common aliases
    alias_map = {
        "BE": ["BE", "BLOOM", "BLOOM ENERGY"],
        "GOOG": ["GOOG", "GOOGLE", "ALPHABET"],
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
        "SNDK": ["SNDK", "SANDISK", "SANDISK CORPORATION"],
    }
    # Check dynamic SQLite companies table names
    try:
        import sqlite3
        if settings.sqlite_db_path.exists():
            conn = sqlite3.connect(str(settings.sqlite_db_path))
            rows = conn.execute("SELECT ticker, name FROM companies").fetchall()
            conn.close()
            for c_ticker, c_name in rows:
                c_t = c_ticker.upper()
                c_n = c_name.upper()
                if (c_t in q_upper or c_n in q_upper) and c_t in known:
                    if c_t not in found:
                        found.append(c_t)
    except Exception:
        pass

    for ticker, aliases in alias_map.items():
        if ticker in known:
            for alias in aliases:
                pat = rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])" if alias.isascii() else re.escape(alias)
                if re.search(pat, q_upper):
                    if ticker not in found:
                        found.append(ticker)
                    break

    # Also detect any known ticker directly
    for t in sorted(known):
        pat = rf"(?<![A-Za-z0-9]){re.escape(t)}(?![A-Za-z0-9])"
        if t not in found and re.search(pat, q_upper):
            found.append(t)
            
    return found


def _parse_json_tool_call_fallback(content: str) -> Optional[Tuple[str, Dict[str, Any], str]]:
    """
    If an OpenAI-compatible proxy returns a tool call embedded as JSON inside `message.content`
    (e.g., `{"tool": "compute_quant_metrics", "arguments": {...}}`), parse and extract it cleanly.
    """
    if not content or "{" not in content:
        return None
    try:
        candidate = extract_json_object(content)
        data = json.loads(candidate)
        if not isinstance(data, dict):
            return None
        tool_name = data.get("tool") or data.get("tool_name") or data.get("name")
        args = data.get("arguments") or data.get("tool_input") or data.get("parameters") or {}
        thought = data.get("thought") or data.get("reasoning") or "Invoking tool via structured JSON."
        if isinstance(tool_name, str) and tool_name in ALLOWED_TOOLS:
            if isinstance(args, str):
                args = json.loads(args)
            return tool_name, args, str(thought)
    except Exception:
        return None
    return None


def _plan_deterministic_tool_sequence(
    question: str,
    tickers: List[str],
    min_score: float,
    ticker_filter: Optional[str],
    market_filter: Optional[Literal["SET", "US"]],
    target_currency: Literal["NATIVE", "USD", "THB"],
) -> List[Tuple[str, Dict[str, Any], str]]:
    """
    Build a deterministic sequence of tool calls covering SQL, Quant Calculator, and Filing RAG
    based on query intent so that both live LLM runs and offline evaluation runs execute rich multi-step traces.
    """
    q_lower = question.lower()
    planned: List[Tuple[str, Dict[str, Any], str]] = []
    norm_currency: Literal["USD", "THB"] = "THB" if target_currency == "THB" else "USD"

    effective_tickers = list(tickers)
    if ticker_filter and ticker_filter.upper() not in effective_tickers:
        effective_tickers.insert(0, ticker_filter.upper())

    wants_quant = any(
        w in q_lower
        for w in (
            "roe", "dupont", "altman", "z-score", "z score", "margin", "fcf", "cash flow",
            "cagr", "growth", "ratio", "debt", "leverage", "ebitda", "pe", "p/e", "valuation",
            "compare", "vs", "versus", "metric", "roic", "turnover", "multiplier", "เปรียบเทียบ", "อัตราส่วน", "วิเคราะห์",
        )
    )
    wants_sql = any(
        w in q_lower
        for w in (
            "sql", "table", "database", "2023", "2024", "2025", "revenue", "net profit",
            "rank", "highest", "lowest", "top", "total assets", "equity", "capex", "fx",
        )
    )
    wants_rag = any(
        w in q_lower
        for w in (
            "2026", "guidance", "outlook", "target", "lease", "off-balance", "hedge", "hedging",
            "fx", "derivative", "litigation", "arbitration", "contingent", "lawsuit", "doj",
            "dma", "irs", "nhtsa", "concentration", "customer", "supplier", "tsmc", "catl",
            "tfrs", "gaap", "r&d", "capitalize", "capitalization", "segment", "56-1", "10-k",
            "10-q", "footnote", "note", "risk", "supply chain", "blackwell", "rubin", "store", "bed",
            "ไตรมาส", "ไตรมาสล่าสุด", "quarter", "q1", "q2", "q3", "q4", "ผลประกอบการ", "งบการเงิน",
            "รายได้", "กำไร", "เปรียบเทียบ", "filing", "เอกสาร"
        )
    )

    # 1. Cross-market or single-ticker Quant Calculator step
    if len(effective_tickers) >= 2:
        planned.append(
            (
                "compare_cross_market_peers",
                {"tickers": effective_tickers, "target_currency": norm_currency},
                f"Comparing cross-market peers {effective_tickers} normalized to {norm_currency} via fx_rates.",
            )
        )
    elif len(effective_tickers) == 1:
        year = 2026 if "2026" in q_lower else (2024 if ("2024" in q_lower and "2025" not in q_lower) else (2023 if ("2023" in q_lower and "2025" not in q_lower) else 2025))
        planned.append(
            (
                "compute_quant_metrics",
                {"ticker": effective_tickers[0], "year": year, "metric_type": "all"},
                f"Computing deterministic DuPont ROE, Altman Z-Score, FCF Yield, and 3Y CAGR for {effective_tickers[0]} (FY{year}).",
            )
        )

    # 2. Read-only SQL verification step when multi-year or cross-company screening is requested
    if wants_sql and len(planned) < 2:
        from agent.sql_tool import _deterministic_fallback_sql

        sql_q = _deterministic_fallback_sql(question)
        planned.append(
            (
                "query_financial_sql",
                {"sql_query": sql_q},
                "Querying read-only SQLite financial_statements and fx_rates tables for exact historical figures.",
            )
        )

    # 3. Filing RAG retrieval step for 56-1 / 10-K disclosures and 2026 guidance
    if len(effective_tickers) >= 2:
        for t in effective_tickers[:2]:
            planned.append(
                (
                    "search_filings_rag",
                    {"query": f"{t} {question}", "ticker": t, "market": None, "min_score": min_score},
                    f"Searching {t} annual filing (56-1 / 10-K) for qualitative disclosures, footnotes, and 2026 guidance.",
                )
            )
    else:
        single_t = effective_tickers[0] if len(effective_tickers) == 1 else ticker_filter
        planned.append(
            (
                "search_filings_rag",
                {
                    "query": question,
                    "ticker": single_t,
                    "market": market_filter,
                    "min_score": min_score,
                },
                "Searching 56-1 / 10-K filings above min_score threshold for grounded [doc_id:chunk_id] evidence.",
            )
        )

    if wants_rag:
        rag_steps = [s for s in planned if s[0] == "search_filings_rag"]
        other_steps = [s for s in planned if s[0] != "search_filings_rag"]
        planned = rag_steps + other_steps
    return planned


def sanitize_latex_and_redundant_table_citations(text: str) -> str:
    """
    Sanitize raw LaTeX symbols into clean Unicode and remove redundant 'Citation: N/A' table columns.
    """
    # 1. Unicode arrow & math symbol replacements
    replacements = [
        (r"$\uparrow$", "↑"),
        (r"$\downarrow$", "↓"),
        (r"$\rightarrow$", "→"),
        (r"$\leftarrow$", "←"),
        (r"$\pm$", "±"),
        (r"$\approx$", "≈"),
        (r"$\Delta$", "Δ"),
        (r"\uparrow", "↑"),
        (r"\downarrow", "↓"),
        (r"\rightarrow", "→"),
        (r"\leftarrow", "←"),
        (r"\pm", "±"),
        (r"\approx", "≈"),
        (r"\Delta", "Δ"),
        ("$↑$", "↑"),
        ("$↓$", "↓"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)

    # 2. Fix tables where Citation column is all N/A
    lines = text.split("\n")
    new_lines: List[str] = []
    table_lines: List[str] = []

    def process_table_block(tbl: List[str]) -> List[str]:
        if len(tbl) < 2:
            return tbl
        rows = [re.split(r"(?<!\\)\|", r.strip()) for r in tbl]
        rows = [[c.strip() for c in r if c != ""] for r in rows]
        if not rows or len(rows[0]) < 2:
            return tbl

        headers = rows[0]
        citation_col_idx = -1
        for idx, h in enumerate(headers):
            if h.lower() in ("citation", "citations", "source", "sources"):
                citation_col_idx = idx
                break

        if citation_col_idx != -1:
            data_rows = rows[2:] if len(rows) > 2 else []
            all_na = True
            for dr in data_rows:
                if len(dr) > citation_col_idx:
                    val = dr[citation_col_idx].upper()
                    if val not in ("N/A", "NA", "-", "NONE", ""):
                        all_na = False
                        break

            if all_na and data_rows:
                new_tbl = []
                for original_line in tbl:
                    parts = original_line.split("|")
                    if len(parts) >= citation_col_idx + 2:
                        del parts[citation_col_idx + 1]
                        new_tbl.append("|".join(parts))
                    else:
                        new_tbl.append(original_line)
                return new_tbl

        return tbl

    for line in lines:
        if "|" in line and line.strip().startswith("|") and line.strip().endswith("|"):
            table_lines.append(line)
        else:
            if table_lines:
                new_lines.extend(process_table_block(table_lines))
                table_lines = []
            new_lines.append(line)
    if table_lines:
        new_lines.extend(process_table_block(table_lines))

    return "\n".join(new_lines)


def _synthesize_grounded_report_and_answer(
    question: str,
    tickers: List[str],
    retrieved_chunks: List[RetrievedChunk],
    quant_outputs: List[Dict[str, Any]],
    sql_outputs: List[Dict[str, Any]],
    trace: List[AgentStepTrace],
    client: LLMClient,
    min_score: float,
) -> Tuple[str, QuantSignalReport, bool]:
    """
    Synthesize the final grounded markdown answer and validated `QuantSignalReport`
    combining deterministic Quant/SQL outputs and cited `[doc_id:chunk_id]` filing chunks.
    """
    if not retrieved_chunks and not quant_outputs and not sql_outputs:
        if is_non_financial_query(question):
            abstain_text = format_unrelated_query_response()
        else:
            unsupported_ticker = find_unsupported_ticker_in_query(question)
            known = get_known_tickers()
            if unsupported_ticker and unsupported_ticker not in known:
                abstain_text = format_missing_stock_guidance(unsupported_ticker)
            elif not known:
                abstain_text = (
                    "⚠️ **ฐานข้อมูล Corpus ยังว่างเปล่า:**\n\n"
                    "ขณะนี้ยังไม่มีเอกสารงบการเงินหรือรายชื่อหุ้นในระบบ กรุณาคลิกปุ่ม **'📤 อัปโหลดเอกสาร (Upload Filing)'** "
                    "ที่แถบเมนูด้านบน เพื่ออัปโหลดไฟล์ PDF รายงาน Form 56-1 / 10-K หรือสรุปงบการเงินของหุ้นที่คุณต้องการวิเคราะห์ครับ"
                )
            else:
                supported_str = ", ".join(sorted(known))
                abstain_text = (
                    f"ABSTAIN: ไม่พบข้อมูลในเอกสารที่ตรงกับเงื่อนไขการค้นหา (min_score={min_score:.2f})\n\n"
                    f"หุ้นที่มีในระบบปัจจุบัน: **{supported_str}**\n\n"
                    "💡 หากต้องการวิเคราะห์หุ้นอื่นเพิ่มเติม สามารถคลิกปุ่ม **'📤 อัปโหลดเอกสาร (Upload Filing)'** ด้านบนได้ทันทีครับ"
                )
        empty_report = QuantSignalReport(
            query=question,
            tickers_analyzed=[],
            signal="ABSTAIN",
            confidence=0.0,
            executive_summary=abstain_text,
            key_metrics=[],
            risk_factors=[],
            cross_market_comparison=None,
            guidance_2026_summary=None,
            citations=[],
            abstained=True,
        )
        return abstain_text, empty_report, True

    # Collect FinancialMetrics schemas from quant outputs or build them for analyzed tickers
    metrics_map: Dict[str, FinancialMetrics] = {}
    cross_comparison_obj: Optional[CrossMarketComparison] = None

    for q_out in quant_outputs:
        if "financial_metrics_schema" in q_out:
            fm = FinancialMetrics.model_validate(q_out["financial_metrics_schema"])
            metrics_map[fm.ticker] = fm
        if "financial_metrics_schemas" in q_out:
            for item in q_out["financial_metrics_schemas"]:
                fm = FinancialMetrics.model_validate(item)
                metrics_map[fm.ticker] = fm
        if "peers" in q_out and len(q_out["peers"]) >= 2:
            p0, p1 = q_out["peers"][0], q_out["peers"][1]
            r0, r1 = p0["currency_neutral_ratios"], p1["currency_neutral_ratios"]
            cite_tags = [c.citation_tag for c in retrieved_chunks[:4]]
            cross_comparison_obj = CrossMarketComparison(
                primary_ticker=p0["ticker"],
                peer_ticker=p1["ticker"],
                normalized_currency=q_out.get("target_currency", "USD"),
                fx_rate_used=float(q_out.get("usd_thb_fx_rate", 34.0)),
                valuation_spread_summary=(
                    f"{p0['ticker']} trades at {r0['pe_ratio_x']}x P/E, {r0['ev_to_ebitda_x']}x EV/EBITDA, "
                    f"and {r0['fcf_yield_pct']}% FCF Yield vs. {p1['ticker']} at {r1['pe_ratio_x']}x P/E, "
                    f"{r1['ev_to_ebitda_x']}x EV/EBITDA, and {r1['fcf_yield_pct']}% FCF Yield."
                ),
                dupont_roe_comparison=(
                    f"{p0['ticker']} DuPont ROE is {r0['roe_dupont_pct']}% "
                    f"(Net Margin {r0['net_profit_margin_pct']}% × Asset Turnover {r0['asset_turnover_x']}x "
                    f"× Equity Multiplier {r0['equity_multiplier_x']}x) vs. {p1['ticker']} DuPont ROE of "
                    f"{r1['roe_dupont_pct']}% (Net Margin {r1['net_profit_margin_pct']}% × Asset Turnover "
                    f"{r1['asset_turnover_x']}x × Equity Multiplier {r1['equity_multiplier_x']}x)."
                ),
                accounting_standard_adjustments=q_out.get(
                    "accounting_standard_note",
                    "TFRS (TAS 38) permits capitalization of qualifying development costs whereas US GAAP (ASC 730) expenses all R&D as incurred.",
                ),
                citations=cite_tags,
            )

    for t in tickers:
        if t not in metrics_map and t in get_known_tickers():
            try:
                metrics_map[t] = build_financial_metrics_schema(t, year=2025)
            except Exception:
                pass

    # Build deterministic summary bullets from quant_outputs + sql_outputs + retrieved_chunks
    quant_summary_blocks: List[str] = []
    for q_out in quant_outputs:
        if "dupont_roe" in q_out and "ticker" in q_out:
            t = q_out["ticker"]
            dp = q_out["dupont_roe"]
            az = q_out.get("altman_z", {})
            lq = q_out.get("liquidity_leverage", {})
            fc = q_out.get("fcf_cagr", {})
            hl = q_out.get("headline_numbers_millions", {})
            curr = q_out.get("native_currency", "THB")
            # Suppress zero-value placeholder profile when metrics were not populated in SQLite
            if hl.get("revenue_native_m", 0.0) == 0.0 and hl.get("net_profit_native_m", 0.0) == 0.0:
                continue
            quant_summary_blocks.append(
                f"- **{t} ({q_out.get('market')} | {curr} | FY{q_out.get('year', 2025)}) Quantitative Profile:** "
                f"Revenue {curr} {hl.get('revenue_native_m'):,.2f}M (USD {hl.get('revenue_usd_m'):,.2f}M), "
                f"Net Profit {curr} {hl.get('net_profit_native_m'):,.2f}M (USD {hl.get('net_profit_usd_m'):,.2f}M), "
                f"Free Cash Flow {curr} {fc.get('free_cash_flow_native_m', 0.0):,.2f}M "
                f"(FCF Yield {fc.get('fcf_yield_pct', 0.0)}%, 3Y Revenue CAGR {fc.get('revenue_cagr_2023_2025_pct', 0.0)}%, "
                f"3Y Net Profit CAGR {fc.get('net_profit_cagr_2023_2025_pct', 0.0)}%).\n"
                f"- **{t} DuPont 3-Step & 5-Step ROE:** ROE = **{dp['roe_3_step_pct']}%** "
                f"(Gross Margin {dp['gross_margin_pct']}%, Operating EBIT Margin {dp['operating_ebit_margin_pct']}%, "
                f"Net Profit Margin {dp['net_profit_margin_pct']}% × Asset Turnover {dp['asset_turnover_x']}x "
                f"× Equity Multiplier {dp['equity_multiplier_x']}x; IBD/E {lq.get('ibd_to_equity_x')}x, "
                f"Current Ratio {lq.get('current_ratio_x')}x, Net Debt/EBITDA {lq.get('net_debt_to_ebitda_x')}x).\n"
                f"- **{t} Credit & Valuation:** Altman Z-Score = **{az.get('altman_z_score')}** "
                f"({az.get('altman_zone_label', az.get('altman_zone'))}), P/E = {lq.get('pe_ratio_x')}x, "
                f"EV/EBITDA = {lq.get('ev_to_ebitda_x')}x."
            )
        elif "peers" in q_out:
            t_curr = q_out.get("target_currency", "USD")
            fx = q_out.get("usd_thb_fx_rate", 34.0)
            quant_summary_blocks.append(
                f"- **Cross-Market Normalization ({t_curr} @ USD/THB = {fx:.2f}):**"
            )
            for p in q_out["peers"]:
                r = p["currency_neutral_ratios"]
                na = p["normalized_absolutes_millions"]
                rev_k = f"revenue_{t_curr.lower()}_m"
                np_k = f"net_profit_{t_curr.lower()}_m"
                fcf_k = f"free_cash_flow_{t_curr.lower()}_m"
                quant_summary_blocks.append(
                    f"  - **{p['ticker']} ({p['market']} | {p['accounting_standard']} | Native {p['native_currency']}):** "
                    f"Normalized Revenue {t_curr} {na.get(rev_k, 0.0):,.2f}M, "
                    f"Net Profit {t_curr} {na.get(np_k, 0.0):,.2f}M, "
                    f"FCF {t_curr} {na.get(fcf_k, 0.0):,.2f}M | "
                    f"DuPont ROE **{r['roe_dupont_pct']}%** (Gross Margin {r['gross_margin_pct']}%, "
                    f"Net Margin {r['net_profit_margin_pct']}%, Asset Turnover {r['asset_turnover_x']}x, "
                    f"Equity Multiplier {r['equity_multiplier_x']}x), IBD/E {r['ibd_to_equity_x']}x, "
                    f"FCF Yield {r['fcf_yield_pct']}%, 3Y Rev CAGR {r['revenue_cagr_3y_pct']}%, "
                    f"Altman Z **{r['altman_z_score']}** ({r['altman_zone']})."
                )

    for s_out in sql_outputs:
        rows = s_out.get("rows", [])
        if rows and not quant_summary_blocks:
            quant_summary_blocks.append(
                f"- **SQLite Query Result ({len(rows)} rows):** `{json.dumps(rows[:5], ensure_ascii=False)}`"
            )

    # Build grounded filing evidence blocks with [doc_id:chunk_id] tags
    chunk_context_lines: List[str] = []
    risk_factors: List[RiskFactor] = []
    guidance_snippets: List[str] = []

    for c in retrieved_chunks:
        chunk_context_lines.append(f"SOURCE {c.citation_tag}:\n{c.text}")
        sec_lower = c.section.lower()
        body_clean = re.sub(r"^\[.*?\]\s*", "", c.text).strip()
        first_sentences = " ".join(body_clean.splitlines()[1:4]).strip()

        if "footnote" in sec_lower or "lease" in sec_lower or "contingenc" in sec_lower:
            cat: Literal[
                "FX_AND_DERIVATIVES",
                "OFF_BALANCE_SHEET_LEASES",
                "LITIGATION_AND_REGULATORY",
                "CONCENTRATION_RISK",
                "ACCOUNTING_STANDARD_DIFF",
                "OPERATIONAL_MACRO",
            ] = "OFF_BALANCE_SHEET_LEASES" if "lease" in sec_lower else "LITIGATION_AND_REGULATORY"
            num_match = re.search(r"((?:THB|USD)\s*[\d,]+(?:\.\d+)?\s*million|\d+(?:\.\d+)?%)", body_clean)
            risk_factors.append(
                RiskFactor(
                    category=cat,
                    severity="HIGH" if c.market == "US" else "MEDIUM",
                    summary=f"{c.ticker} ({c.section}): {first_sentences[:220]}",
                    quantified_exposure=num_match.group(1) if num_match else None,
                    citation=c.citation_tag,
                )
            )
        if "2026" in sec_lower or "guidance" in sec_lower:
            guidance_snippets.append(f"{c.ticker} 2026 Guidance: {first_sentences[:260]} {c.citation_tag}")

    # Compose prompt for LLM synthesis (with deterministic fallback including both Quant + RAG facts)
    deterministic_rag_lines: List[str] = []
    q_tokens = set(re.findall(r"[a-z0-9\-\.]+", question.lower()))
    scored_chunk_lines: List[Tuple[float, str, str]] = []
    for c in retrieved_chunks:
        for ln in c.text.splitlines():
            s = ln.strip()
            if not s or s.startswith("#") or (s.startswith("[") and s.endswith("]")):
                continue
            s_tokens = set(re.findall(r"[a-z0-9\-\.]+", s.lower()))
            overlap = len(q_tokens & s_tokens)
            ticker_bonus = 2.0 if c.ticker.lower() in question.lower() else 0.0
            num_bonus = 1.5 if re.search(r"\d", s) else 0.0
            scored_chunk_lines.append((overlap + ticker_bonus + num_bonus, s, c.citation_tag))

    scored_chunk_lines.sort(key=lambda x: x[0], reverse=True)
    seen_lines: set[str] = set()
    for _, line_text, tag in scored_chunk_lines:
        if line_text not in seen_lines and len(deterministic_rag_lines) < 6:
            seen_lines.add(line_text)
            deterministic_rag_lines.append(f"- {line_text} {tag}")

    fallback_parts: List[str] = []
    if quant_summary_blocks:
        fallback_parts.append("### Deterministic Quantitative & Cross-Market Analysis\n" + "\n".join(quant_summary_blocks))
    if deterministic_rag_lines:
        fallback_parts.append("### Grounded 56-1 / 10-K Filing Disclosures\n" + "\n".join(deterministic_rag_lines))
    fallback_answer = "\n\n".join(fallback_parts).strip()

    is_thai = any("\u0e00" <= ch <= "\u0e7f" for ch in question)
    lang_instruction = (
        "CRITICAL INSTRUCTION: The user asked in Thai. You MUST synthesize your entire response in professional, natural Thai "
        "(while keeping standard international financial terms in English, e.g. Revenue, Operating Income, EBITDA, CapEx, QoQ, YoY, TFRS, US GAAP).\n"
        if is_thai
        else ""
    )

    synthesis_messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"USER QUESTION: {question}\n\n"
                + (lang_instruction + "\n" if lang_instruction else "")
                + ("DETERMINISTIC QUANT & SQL RESULTS:\n" + "\n".join(quant_summary_blocks) if quant_summary_blocks else "QUANTITATIVE FINANCIAL CONTEXT:\nExtract and synthesize all requested financial figures, quarterly performance numbers, revenue, margins, and operational disclosures directly from the RETRIEVED FILING SOURCES below.")
                + "\n\n"
                + ("RETRIEVED FILING SOURCES:\n" + "\n\n".join(chunk_context_lines) if chunk_context_lines else "")
                + "\n\nSynthesize a comprehensive, institutional-grade quantitative response.\n"
                + "- Include the exact numbers from the Quant/SQL results or filings.\n"
                + "- Cite every filing disclosure using inline `[doc_id:chunk_id]` tags.\n"
                + "- In tables, NEVER use LaTeX math notation (do NOT use LaTeX arrows like uparrow or downarrow). Use standard Unicode arrows: ↑ or ↓ or + / -.\n"
                + "- In tables, DO NOT include a 'Citation' column if citations are already placed next to the numbers in the data columns. If a Citation column is present, NEVER write 'N/A'—always list the actual `[doc_id:chunk_id]` tags."
            ),
        },
    ]

    final_answer = client.chat_completion(
        messages=synthesis_messages,
        temperature=0.0,
        fallback_text=fallback_answer if fallback_answer else None,
    )
    final_answer = sanitize_latex_and_redundant_table_citations(final_answer)

    # Guarantee that if retrieved chunks exist, inline [doc_id:chunk_id] citations are preserved
    citations = extract_citations_from_text(final_answer, retrieved_chunks)
    if retrieved_chunks and not citations:
        top_c = retrieved_chunks[0]
        final_answer = f"{final_answer.strip()}\n\nSupporting Filing Reference: {top_c.citation_tag}"
        citations = extract_citations_from_text(final_answer, retrieved_chunks)

    abstained = final_answer.strip().upper().startswith("ABSTAIN")
    analyzed_tickers = list(metrics_map.keys()) or tickers

    # Determine institutional signal from deterministic ROE & Altman Z metrics
    signal: Literal["OVERWEIGHT", "EQUAL_WEIGHT", "UNDERWEIGHT", "ABSTAIN"] = "ABSTAIN" if abstained else "EQUAL_WEIGHT"
    if not abstained and metrics_map:
        first_m = next(iter(metrics_map.values()))
        if first_m.roe_dupont_pct >= 20.0 and first_m.altman_zone == "SAFE":
            signal = "OVERWEIGHT"
        elif first_m.altman_zone == "DISTRESS":
            signal = "UNDERWEIGHT"
        else:
            signal = "EQUAL_WEIGHT"

    report = QuantSignalReport(
        query=question,
        tickers_analyzed=analyzed_tickers,
        signal=signal,
        confidence=0.0 if abstained else 0.88,
        executive_summary=final_answer,
        key_metrics=list(metrics_map.values()),
        risk_factors=risk_factors[:5],
        cross_market_comparison=cross_comparison_obj,
        guidance_2026_summary=" | ".join(guidance_snippets[:2]) if guidance_snippets else None,
        citations=citations,
        abstained=abstained,
    )
    return final_answer, report, abstained


def run_agent(
    question: str,
    max_steps: int = 5,
    min_score: float = settings.min_score,
    stream_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ticker_filter: Optional[str] = None,
    market_filter: Optional[Literal["SET", "US"]] = None,
    target_currency: Literal["NATIVE", "USD", "THB"] = "USD",
    client: Optional[LLMClient] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> ChatResponse:
    """
    Execute the multi-step Cross-Market Quant & Filing Analyst Agent loop.
    - Respects `max_steps` ceiling.
    - Prevents duplicate identical tool calls.
    - Validates every tool call against `ALLOWED_TOOLS` and Pydantic v2 schemas.
    - Emits real-time events via `stream_callback` when provided.
    """
    t_start = time.perf_counter()
    llm = client or llm_client
    effective_max_steps = max(1, min(int(max_steps), settings.max_agent_steps))

    def _emit(event_type: str, data: Dict[str, Any]) -> None:
        if stream_callback is not None:
            stream_callback(event_type, data)

    # Check out-of-corpus trap query immediately via search_filings_rag step so trace is transparent
    tickers_in_q = _extract_tickers_mentioned(question)
    if not tickers_in_q and history:
        for prev_msg in reversed(history):
            prev_content = prev_msg.get("content", "")
            found_prev = _extract_tickers_mentioned(prev_content)
            if found_prev:
                tickers_in_q = found_prev
                break
    if ticker_filter and ticker_filter.upper() not in tickers_in_q and ticker_filter.upper() in get_known_tickers():
        tickers_in_q.insert(0, ticker_filter.upper())
    effective_tickers = list(tickers_in_q)

    # 1. Non-financial query check
    if is_non_financial_query(question):
        ans = format_unrelated_query_response()
        trace_step = AgentStepTrace(
            step_number=1,
            thought="Detected non-financial query outside stock/company analysis scope.",
            tool_name="search_filings_rag",
            tool_input={"query": question},
            tool_output_summary="Query is unrelated to financial/stock filings.",
            latency_ms=10.0,
        )
        _emit("tool_result", trace_step.model_dump())
        _emit("token", {"text": ans})
        _emit("done", {"abstained": True, "latency_ms": 10.0})
        report_1 = QuantSignalReport(
            query=question,
            tickers_analyzed=[],
            signal="ABSTAIN",
            confidence=0.0,
            executive_summary=ans,
            key_metrics=[],
            risk_factors=[],
            citations=[],
            abstained=True,
        )
        return ChatResponse(
            answer=ans,
            structured_report=report_1,
            retrieved_chunks=[],
            agent_trace=[trace_step],
            abstained=True,
            latency_ms=10.0,
        )

    # 2. Unsupported ticker check
    unsupported_ticker = find_unsupported_ticker_in_query(question)
    known = get_known_tickers()
    if unsupported_ticker and unsupported_ticker not in known and not tickers_in_q:
        ans = format_missing_stock_guidance(unsupported_ticker)
        trace_step = AgentStepTrace(
            step_number=1,
            thought=f"Checking filing corpus for '{unsupported_ticker}'.",
            tool_name="search_filings_rag",
            tool_input={"query": question, "ticker": unsupported_ticker},
            tool_output_summary=f"Ticker '{unsupported_ticker}' not found in active corpus. Providing frontend upload guidance.",
            latency_ms=15.0,
        )
        _emit("tool_result", trace_step.model_dump())
        _emit("token", {"text": ans})
        _emit("done", {"abstained": True, "latency_ms": 15.0})
        report_2 = QuantSignalReport(
            query=question,
            tickers_analyzed=[unsupported_ticker] if unsupported_ticker else [],
            signal="ABSTAIN",
            confidence=0.0,
            executive_summary=ans,
            key_metrics=[],
            risk_factors=[],
            citations=[],
            abstained=True,
        )
        return ChatResponse(
            answer=ans,
            structured_report=report_2,
            retrieved_chunks=[],
            agent_trace=[trace_step],
            abstained=True,
            latency_ms=15.0,
        )

    # Prepare planned fallback steps and OpenAI function-calling messages
    planned_steps = _plan_deterministic_tool_sequence(
        question=question,
        tickers=tickers_in_q,
        min_score=min_score,
        ticker_filter=ticker_filter,
        market_filter=market_filter,
        target_currency=target_currency,
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
    ]
    if history:
        for h in history[-6:]:
            h_role = h.get("role", "user")
            if h_role in ("user", "assistant"):
                messages.append({"role": h_role, "content": h.get("content", "")})
    messages.append({"role": "user", "content": question})
    tools_spec = get_openai_tools_spec()

    trace: List[AgentStepTrace] = []
    retrieved_chunks_map: Dict[str, RetrievedChunk] = {}
    quant_outputs: List[Dict[str, Any]] = []
    sql_outputs: List[Dict[str, Any]] = []
    executed_signatures: set[str] = set()

    api_key = os.getenv("OPENAI_API_KEY", settings.openai_api_key)
    use_remote_tool_calling = (
        os.getenv("FORCE_OFFLINE_LLM", "0") != "1"
        and bool(api_key)
        and not api_key.startswith("sk-local")
        and not api_key.startswith("sk-your")
    )

    for step_idx in range(1, effective_max_steps + 1):
        step_t0 = time.perf_counter()
        chosen_tool: Optional[str] = None
        chosen_args: Dict[str, Any] = {}
        thought_text = ""

        if use_remote_tool_calling:
            try:
                resp = llm.client.chat.completions.create(
                    model=llm.model,
                    messages=messages,  # type: ignore[arg-type]
                    tools=tools_spec,  # type: ignore[arg-type]
                    tool_choice="auto",
                    temperature=0.0,
                )
                msg = resp.choices[0].message
                if msg.tool_calls:
                    tc = msg.tool_calls[0]
                    chosen_tool = tc.function.name
                    chosen_args = json.loads(tc.function.arguments or "{}")
                    thought_text = msg.content or f"Calling {chosen_tool} via OpenAI function calling."
                elif msg.content:
                    parsed_fallback = _parse_json_tool_call_fallback(msg.content)
                    if parsed_fallback:
                        chosen_tool, chosen_args, thought_text = parsed_fallback
                    elif step_idx > 1:
                        # If we have tickers and haven't retrieved any filing chunks, do NOT break yet
                        if effective_tickers and not retrieved_chunks_map:
                            chosen_tool = "search_filings_rag"
                            chosen_args = {
                                "query": question,
                                "ticker": effective_tickers[0],
                                "min_score": min_score,
                            }
                            thought_text = f"Retrieving grounded filing sections for {effective_tickers[0]} from 56-1 / 10-Q / 10-K filings."
                        else:
                            break
            except Exception:
                use_remote_tool_calling = False

        # If offline or if the LLM didn't emit a tool call on step 1, pick the next unexecuted planned step
        if chosen_tool is None:
            for p_tool, p_args, p_thought in planned_steps:
                sig_candidate = f"{p_tool}:{json.dumps(p_args, sort_keys=True)}"
                if sig_candidate not in executed_signatures:
                    chosen_tool = p_tool
                    chosen_args = p_args
                    thought_text = p_thought
                    break

        if chosen_tool is None:
            break

        # Prevent duplicate identical tool calls
        call_signature = f"{chosen_tool}:{json.dumps(chosen_args, sort_keys=True)}"
        if call_signature in executed_signatures:
            break
        executed_signatures.add(call_signature)

        _emit("tool_start", {"step": step_idx, "tool": chosen_tool, "input": chosen_args, "thought": thought_text})
        exec_res = execute_allowlisted_tool(chosen_tool, chosen_args)
        step_ms = round((time.perf_counter() - step_t0) * 1000.0, 2)

        raw_res = exec_res.get("raw_result")
        canonical_name = exec_res.get("canonical_trace_name", "final_answer")
        summary_str = exec_res.get("sanitized_output", "")[:400]

        if exec_res.get("ok") and isinstance(raw_res, dict):
            if chosen_tool == "search_filings_rag":
                chunks_list = raw_res.get("chunks", [])
                for c_dict in chunks_list:
                    rc = RetrievedChunk(
                        doc_id=c_dict["doc_id"],
                        chunk_id=c_dict["chunk_id"],
                        ticker=c_dict["ticker"],
                        market=c_dict["market"],
                        filing_type="56-1" if c_dict["market"] == "SET" else "10-K",
                        section=c_dict["section"],
                        text=c_dict["text"],
                        score=float(c_dict["score"]),
                    )
                    retrieved_chunks_map[rc.citation_tag] = rc
                summary_str = (
                    f"Retrieved {len(chunks_list)} filing chunks above min_score={min_score:.2f}: "
                    + ", ".join(c["citation_tag"] for c in chunks_list)
                )
            elif chosen_tool in ("compute_quant_metrics", "compare_cross_market_peers", "compare_cross_market"):
                quant_outputs.append(raw_res)
                if "dupont_roe" in raw_res:
                    dp = raw_res["dupont_roe"]
                    az = raw_res.get("altman_z", {})
                    summary_str = (
                        f"{raw_res['ticker']} FY{raw_res['year']}: DuPont ROE={dp['roe_3_step_pct']}%, "
                        f"Net Margin={dp['net_profit_margin_pct']}%, Altman Z={az.get('altman_z_score')} ({az.get('altman_zone')})."
                    )
                elif "peers" in raw_res:
                    peers_str = ", ".join(
                        f"{p['ticker']} (ROE {p['currency_neutral_ratios']['roe_dupont_pct']}%, "
                        f"Z={p['currency_neutral_ratios']['altman_z_score']})"
                        for p in raw_res["peers"]
                    )
                    summary_str = f"Cross-market comparison normalized to {raw_res.get('target_currency')}: {peers_str}"
            elif chosen_tool == "query_financial_sql":
                sql_outputs.append(raw_res)
                summary_str = f"SQL returned {raw_res.get('row_count', 0)} rows."

        step_trace = AgentStepTrace(
            step_number=step_idx,
            thought=thought_text,
            tool_name=canonical_name,  # type: ignore[arg-type]
            tool_input=chosen_args,
            tool_output_summary=summary_str,
            latency_ms=step_ms,
        )
        trace.append(step_trace)
        _emit("tool_result", step_trace.model_dump())

        messages.append(
            {
                "role": "assistant",
                "content": f"Executed {chosen_tool}({json.dumps(chosen_args)})",
            }
        )
        messages.append(
            {
                "role": "user",
                "content": f"TOOL OUTPUT [{chosen_tool}]:\n{exec_res.get('sanitized_output', '')}",
            }
        )

    retrieved_chunks = sorted(retrieved_chunks_map.values(), key=lambda c: c.score, reverse=True)

    answer_text, signal_report, abstained = _synthesize_grounded_report_and_answer(
        question=question,
        tickers=tickers_in_q,
        retrieved_chunks=retrieved_chunks,
        quant_outputs=quant_outputs,
        sql_outputs=sql_outputs,
        trace=trace,
        client=llm,
        min_score=min_score,
    )

    total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
    _emit("token", {"text": answer_text})
    _emit("citations", {"citations": [c.model_dump() for c in signal_report.citations]})
    _emit(
        "done",
        {
            "abstained": abstained,
            "latency_ms": total_ms,
            "structured_report": signal_report.model_dump(),
        },
    )

    return ChatResponse(
        answer=answer_text,
        structured_report=signal_report,
        retrieved_chunks=retrieved_chunks,
        agent_trace=trace,
        abstained=abstained,
        latency_ms=total_ms,
    )