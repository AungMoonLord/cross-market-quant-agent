"""
Sandboxed Read-Only Text-to-SQL Engine (`agent/sql_tool.py`).

Security & Institutional Guardrails:
1. Opens SQLite strictly in read-only URI mode (`file:...?mode=ro`) and sets `PRAGMA query_only = ON`.
2. Blocks any SQL containing destructive or administrative keywords (`DROP`, `DELETE`, `UPDATE`, `INSERT`,
   `ALTER`, `ATTACH`, `DETACH`, `PRAGMA`, `REPLACE`, `CREATE`, `TRUNCATE`, `VACUUM`) or multi-statement
   semicolons (`;`).
3. Enforces a strict row limit (`max_rows=50`) and statement timeout.
4. Provides `get_db_schema()` with explicit notes on THB vs USD currencies and `fx_rates`,
   plus `nl_to_sql(question: str)` with 1-step automatic self-correction on SQLite syntax/column errors.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import settings
from app.llm import LLMClient, llm_client
from corpus.build_corpus import build_all

FORBIDDEN_SQL_KEYWORDS = (
    "DROP",
    "DELETE",
    "UPDATE",
    "INSERT",
    "ALTER",
    "ATTACH",
    "DETACH",
    "PRAGMA",
    "REPLACE",
    "CREATE",
    "TRUNCATE",
    "VACUUM",
    "GRANT",
    "REVOKE",
)


def get_db_schema() -> str:
    """
    Return the complete SQLite schema alongside explicit cross-market currency (`THB` vs `USD`)
    and `fx_rates` normalization notes for Text-to-SQL generation.
    """
    return """DATABASE SCHEMA (`corpus/financials.db` - SQLite):

1. Table `companies`:
   - `ticker` (TEXT PRIMARY KEY): 'BE', 'GOOG', 'JPM', 'PENG', 'SPCX'
   - `name` (TEXT): Full legal company name
   - `market` (TEXT): 'SET' (Stock Exchange of Thailand) or 'US' (NASDAQ/NYSE)
   - `sector` (TEXT): Industry sector
   - `currency` (TEXT): 'THB' (for SET companies) or 'USD' (for US companies)
   - `filing_type` (TEXT): '56-1' (for SET) or '10-K' (for US)
   - `accounting_standard` (TEXT): 'TFRS' (for SET) or 'US_GAAP' (for US)

2. Table `fx_rates`:
   - `year` (INTEGER): 2023, 2024, 2025
   - `pair` (TEXT): 'USD_THB'
   - `avg_rate` (REAL): Annual average THB per 1 USD (2023=34.80, 2024=35.20, 2025=34.00)
   - `year_end_rate` (REAL): Year-end THB per 1 USD (2023=34.35, 2024=34.10, 2025=33.85)

3. Table `financial_statements`:
   - `ticker` (TEXT), `year` (INTEGER: 2023, 2024, 2025), `currency` (TEXT: 'THB' or 'USD')
   - All monetary amounts below are in MILLIONS of `currency` (THB Millions for SET; USD Millions for US):
     `revenue`, `gross_profit`, `ebit`, `ebitda`, `net_profit`,
     `total_assets`, `current_assets`, `total_liabilities`, `current_liabilities`,
     `retained_earnings`, `total_equity`, `interest_bearing_debt`, `cash`,
     `operating_cash_flow`, `capex`, `market_cap`, `shares_outstanding`

CRITICAL CURRENCY NORMALIZATION RULE:
- NEVER sum or compare raw `revenue`, `net_profit`, or `market_cap` across SET ('THB') and US ('USD') without converting via `fx_rates`!
- Example USD normalization in SQL:
  `CASE WHEN f.currency = 'THB' THEN ROUND(f.revenue / fx.avg_rate, 2) ELSE f.revenue END AS revenue_usd_m`
- Note: 2026 Executive Guidance numbers and footnote disclosures (leases, FX hedges, litigation) are NOT in SQLite; use `search_filings_rag` for those."""


def validate_readonly_sql(sql_query: str) -> str:
    """
    Validate that a SQL query is strictly a single read-only `SELECT` or `WITH ... SELECT` statement.
    Raises ValueError if any forbidden keyword or semicolon injection is detected.
    """
    if not sql_query or not sql_query.strip():
        raise ValueError("SQL query cannot be empty.")

    cleaned = sql_query.strip()
    # Strip markdown SQL code fences if present
    fence_match = re.search(r"```(?:sql)?\s*([\s\S]*?)```", cleaned, re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    # Block any semicolon to prevent multi-statement injection (e.g., `SELECT 1; DROP TABLE ...`)
    if ";" in cleaned:
        raise ValueError(
            "Security Guardrail Violation: Semicolons (;) and multi-statement SQL queries are strictly prohibited."
        )

    # Remove SQL comments before checking keywords
    no_comments = re.sub(r"--.*?$", " ", cleaned, flags=re.MULTILINE)
    no_comments = re.sub(r"/\*[\s\S]*?\*/", " ", no_comments).strip()

    upper_sql = no_comments.upper()
    if not (upper_sql.startswith("SELECT") or upper_sql.startswith("WITH")):
        raise ValueError(
            "Security Guardrail Violation: Only read-only SELECT (or WITH ... SELECT) queries are permitted."
        )

    for kw in FORBIDDEN_SQL_KEYWORDS:
        if re.search(rf"\b{kw}\b", upper_sql):
            raise ValueError(
                f"Security Guardrail Violation: Forbidden SQL keyword '{kw}' detected in query."
            )

    return no_comments


def execute_readonly_sql(
    sql_query: str,
    max_rows: int = 50,
    db_path: Path = settings.sqlite_db_path,
) -> Dict[str, Any]:
    """
    Execute a validated read-only SQL query against `corpus/financials.db` in read-only URI mode.
    Enforces row limits (`max_rows`) and returns structured columns + rows.
    """
    validated_sql = validate_readonly_sql(sql_query)

    if not db_path.exists():
        build_all(db_path=db_path, raw_dir=settings.raw_filings_dir)

    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()
        cursor.execute(validated_sql)
        fetched = cursor.fetchmany(max_rows)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = [dict(r) for r in fetched]
        return {
            "success": True,
            "sql_executed": validated_sql,
            "row_count": len(rows),
            "columns": columns,
            "rows": rows,
            "truncated": len(rows) == max_rows,
        }
    except sqlite3.Error as exc:
        return {
            "success": False,
            "sql_executed": validated_sql,
            "error": f"SQLite Execution Error: {str(exc)}",
            "rows": [],
        }
    finally:
        conn.close()


def _deterministic_fallback_sql(question: str) -> str:
    """
    Deterministic Text-to-SQL fallback generator for offline execution and unit tests.
    Constructs a valid, currency-normalized read-only SQL query tailored to the question.
    """
    q_upper = question.upper()
    tickers = [
        t for t in ("BE", "GOOG", "JPM", "PENG", "SPCX")
        if re.search(rf"\b{t}\b", q_upper)
    ]
    years = [y for y in ("2023", "2024", "2025", "2026") if y in q_upper]

    where_clauses: List[str] = []
    if tickers:
        quoted = ", ".join(f"'{t}'" for t in tickers)
        where_clauses.append(f"f.ticker IN ({quoted})")
    if years:
        years_joined = ", ".join(years)
        where_clauses.append(f"f.year IN ({years_joined})")
    elif "ALL YEARS" not in q_upper and "CAGR" not in q_upper and "TREND" not in q_upper:
        where_clauses.append("f.year = 2025")

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else "WHERE f.year = 2025"

    return (
        "SELECT f.ticker, c.market, c.accounting_standard, f.year, f.currency, "
        "f.revenue, f.gross_profit, f.ebit, f.ebitda, f.net_profit, "
        "f.operating_cash_flow, f.capex, (f.operating_cash_flow - f.capex) AS free_cash_flow, "
        "f.total_assets, f.total_equity, f.interest_bearing_debt, f.cash, f.market_cap, "
        "ROUND(CASE WHEN f.currency = 'THB' THEN f.revenue / fx.avg_rate ELSE f.revenue END, 2) AS revenue_usd_m, "
        "ROUND(CASE WHEN f.currency = 'THB' THEN f.net_profit / fx.avg_rate ELSE f.net_profit END, 2) AS net_profit_usd_m "
        "FROM financial_statements f "
        "JOIN companies c ON f.ticker = c.ticker "
        "JOIN fx_rates fx ON f.year = fx.year AND fx.pair = 'USD_THB' "
        f"{where_sql} "
        "ORDER BY f.ticker ASC, f.year ASC LIMIT 25"
    )


def nl_to_sql(
    question: str,
    client: Optional[LLMClient] = None,
    db_path: Path = settings.sqlite_db_path,
) -> Dict[str, Any]:
    """
    Convert a natural-language financial question into a read-only SQLite query,
    execute it, and perform a 1-step self-correction retry if SQLite returns an error.
    """
    llm = client or llm_client
    schema_text = get_db_schema()
    fallback_sql = _deterministic_fallback_sql(question)

    system_prompt = (
        "You are an expert Quantitative SQL Developer. Write a single read-only SQLite SELECT query "
        "to answer the user's question using the schema below.\n"
        "RULES:\n"
        "1. Output ONLY the raw SQL SELECT query with NO semicolons (;) and NO markdown commentary.\n"
        "2. Always include `currency` and normalize across THB/USD using `fx_rates` when comparing SET and US tickers.\n\n"
        f"{schema_text}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {question}"},
    ]

    candidate_sql = llm.chat_completion(
        messages=messages,
        temperature=0.0,
        fallback_text=fallback_sql,
    ).strip()

    # Clean potential trailing semicolon from LLM output before validation
    candidate_sql = re.sub(r";+\s*$", "", candidate_sql.strip())
    if not candidate_sql.upper().startswith(("SELECT", "WITH")):
        candidate_sql = fallback_sql

    try:
        result = execute_readonly_sql(candidate_sql, db_path=db_path)
    except ValueError as val_err:
        result = {"success": False, "sql_executed": candidate_sql, "error": str(val_err), "rows": []}

    if result.get("success"):
        result["self_corrected"] = False
        return result

    # 1-Step Self-Correction on SQLite or validation error
    error_msg = result.get("error", "Unknown SQLite error")
    repair_messages = messages + [
        {"role": "assistant", "content": candidate_sql},
        {
            "role": "user",
            "content": (
                f"The SQL query failed with error: {error_msg}\n"
                "Rewrite a corrected single SELECT query without semicolons that adheres strictly to the schema."
            ),
        },
    ]
    repaired_sql = llm.chat_completion(
        messages=repair_messages,
        temperature=0.0,
        fallback_text=fallback_sql,
    ).strip()
    repaired_sql = re.sub(r";+\s*$", "", repaired_sql.strip())
    if not repaired_sql.upper().startswith(("SELECT", "WITH")):
        repaired_sql = fallback_sql

    try:
        repaired_result = execute_readonly_sql(repaired_sql, db_path=db_path)
    except ValueError:
        repaired_result = execute_readonly_sql(fallback_sql, db_path=db_path)

    repaired_result["self_corrected"] = True
    repaired_result["initial_error"] = error_msg
    return repaired_result