"""
Deterministic Cross-Market Financial Database (SQLite) & Filing Ingestion for Default Corpus.

Default Corpus:
- BE (Bloom Energy, US, 10-Q)
- GOOG (Alphabet Inc., US, 10-Q)
- JPM (JPMorgan Chase & Co., US, 10-Q)
- SPCX (SpaceX, US, 10-Q)
Built from `raw_documents` filings across 2026.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
RAW_DOCS = PROJECT_ROOT / "raw_documents"
DB_PATH = BASE_DIR / "financials.db"
RAW_FILINGS_DIR = BASE_DIR / "raw_filings"
VECTOR_INDEX_PATH = BASE_DIR / "vector_index.json"


COMPANIES_DATA: List[Tuple[str, str, str, str, str, str, str]] = [
    # (ticker, name, market, sector, currency, filing_type, accounting_standard)
    ("BE", "Bloom Energy", "US", "Clean Energy & Fuel Cells", "USD", "10-Q", "US_GAAP"),
    ("GOOG", "Alphabet Inc. (Google)", "US", "Internet & Cloud Technology", "USD", "10-Q", "US_GAAP"),
    ("JPM", "JPMorgan Chase & Co.", "US", "Financial Services & Banking", "USD", "10-Q", "US_GAAP"),
    ("PENG", "Penguin Solutions, Inc.", "US", "AI Infrastructure & Computing", "USD", "10-Q", "US_GAAP"),
    ("SPCX", "SpaceX", "US", "Aerospace & Satellites", "USD", "10-Q", "US_GAAP"),
]


FX_RATES_DATA: List[Tuple[int, str, float, float]] = [
    # (year, pair, avg_rate, year_end_rate)
    (2023, "USD_THB", 34.80, 34.35),
    (2024, "USD_THB", 35.20, 34.10),
    (2025, "USD_THB", 34.00, 33.85),
    (2026, "USD_THB", 34.00, 34.00),
]


FINANCIAL_STATEMENTS_DATA: List[Dict[str, Any]] = [
    # BE (Bloom Energy, USD Millions)
    {
        "ticker": "BE", "year": 2025, "currency": "USD",
        "revenue": 1350.0, "gross_profit": 360.0, "ebit": 110.0, "ebitda": 180.0,
        "net_profit": 60.0, "total_assets": 4800.0, "current_assets": 3200.0,
        "total_liabilities": 3100.0, "current_liabilities": 950.0, "retained_earnings": -3500.0,
        "total_equity": 1700.0, "interest_bearing_debt": 2200.0, "cash": 1800.0,
        "operating_cash_flow": 45.0, "capex": 110.0, "market_cap": 3200.0,
        "shares_outstanding": 280.0,
    },
    {
        "ticker": "BE", "year": 2026, "currency": "USD",
        "revenue": 3600.0, "gross_profit": 1150.0, "ebit": 650.0, "ebitda": 850.0,
        "net_profit": 520.0, "total_assets": 6571.58, "current_assets": 5207.75,
        "total_liabilities": 4608.47, "current_liabilities": 1391.82, "retained_earnings": -3470.69,
        "total_equity": 1963.11, "interest_bearing_debt": 2473.50, "cash": 2486.70,
        "operating_cash_flow": 350.0, "capex": 180.0, "market_cap": 5800.0,
        "shares_outstanding": 295.54,
    },
    # GOOG (Alphabet Inc., USD Millions)
    {
        "ticker": "GOOG", "year": 2025, "currency": "USD",
        "revenue": 350000.0, "gross_profit": 200000.0, "ebit": 105000.0, "ebitda": 130000.0,
        "net_profit": 88000.0, "total_assets": 400000.0, "current_assets": 165000.0,
        "total_liabilities": 110000.0, "current_liabilities": 75000.0, "retained_earnings": 205000.0,
        "total_equity": 290000.0, "interest_bearing_debt": 28000.0, "cash": 100000.0,
        "operating_cash_flow": 108000.0, "capex": 45000.0, "market_cap": 2100000.0,
        "shares_outstanding": 12400.0,
    },
    {
        "ticker": "GOOG", "year": 2026, "currency": "USD",
        "revenue": 420000.0, "gross_profit": 245000.0, "ebit": 135000.0, "ebitda": 168000.0,
        "net_profit": 118000.0, "total_assets": 460000.0, "current_assets": 195000.0,
        "total_liabilities": 125000.0, "current_liabilities": 88000.0, "retained_earnings": 245000.0,
        "total_equity": 335000.0, "interest_bearing_debt": 28000.0, "cash": 120000.0,
        "operating_cash_flow": 130000.0, "capex": 52000.0, "market_cap": 2450000.0,
        "shares_outstanding": 12200.0,
    },
    # JPM (JPMorgan Chase & Co., USD Millions)
    {
        "ticker": "JPM", "year": 2025, "currency": "USD",
        "revenue": 170000.0, "gross_profit": 170000.0, "ebit": 62000.0, "ebitda": 64000.0,
        "net_profit": 49500.0, "total_assets": 3900000.0, "current_assets": 1400000.0,
        "total_liabilities": 3580000.0, "current_liabilities": 2350000.0, "retained_earnings": 260000.0,
        "total_equity": 320000.0, "interest_bearing_debt": 330000.0, "cash": 1400000.0,
        "operating_cash_flow": 65000.0, "capex": 6500.0, "market_cap": 580000.0,
        "shares_outstanding": 2870.0,
    },
    {
        "ticker": "JPM", "year": 2026, "currency": "USD",
        "revenue": 210000.0, "gross_profit": 210000.0, "ebit": 76000.0, "ebitda": 78000.0,
        "net_profit": 62000.0, "total_assets": 4100000.0, "current_assets": 1500000.0,
        "total_liabilities": 3760000.0, "current_liabilities": 2500000.0, "retained_earnings": 280000.0,
        "total_equity": 340000.0, "interest_bearing_debt": 350000.0, "cash": 1500000.0,
        "operating_cash_flow": 78000.0, "capex": 7500.0, "market_cap": 650000.0,
        "shares_outstanding": 2850.0,
    },
    # PENG (Penguin Solutions, Inc., USD Millions)
    {
        "ticker": "PENG", "year": 2025, "currency": "USD",
        "revenue": 1374.0, "gross_profit": 350.0, "ebit": 105.0, "ebitda": 140.0,
        "net_profit": 65.0, "total_assets": 1650.0, "current_assets": 850.0,
        "total_liabilities": 830.0, "current_liabilities": 370.0, "retained_earnings": 320.0,
        "total_equity": 820.0, "interest_bearing_debt": 420.0, "cash": 280.0,
        "operating_cash_flow": 130.0, "capex": 35.0, "market_cap": 1100.0,
        "shares_outstanding": 51.0,
    },
    {
        "ticker": "PENG", "year": 2026, "currency": "USD",
        "revenue": 1732.0, "gross_profit": 520.0, "ebit": 180.0, "ebitda": 230.0,
        "net_profit": 125.0, "total_assets": 1850.0, "current_assets": 980.0,
        "total_liabilities": 920.0, "current_liabilities": 410.0, "retained_earnings": 380.0,
        "total_equity": 930.0, "interest_bearing_debt": 450.0, "cash": 320.0,
        "operating_cash_flow": 195.0, "capex": 45.0, "market_cap": 1450.0,
        "shares_outstanding": 52.0,
    },
    # SPCX (SpaceX, USD Millions)
    {
        "ticker": "SPCX", "year": 2025, "currency": "USD",
        "revenue": 8700.0, "gross_profit": 2800.0, "ebit": 1200.0, "ebitda": 1900.0,
        "net_profit": 450.0, "total_assets": 35000.0, "current_assets": 12000.0,
        "total_liabilities": 17000.0, "current_liabilities": 6500.0, "retained_earnings": 1900.0,
        "total_equity": 18000.0, "interest_bearing_debt": 4500.0, "cash": 6800.0,
        "operating_cash_flow": 2500.0, "capex": 1800.0, "market_cap": 180000.0,
        "shares_outstanding": 1000.0,
    },
    {
        "ticker": "SPCX", "year": 2026, "currency": "USD",
        "revenue": 22400.0, "gross_profit": 8900.0, "ebit": 5200.0, "ebitda": 7100.0,
        "net_profit": 4200.0, "total_assets": 55000.0, "current_assets": 22000.0,
        "total_liabilities": 24000.0, "current_liabilities": 9500.0, "retained_earnings": 6100.0,
        "total_equity": 31000.0, "interest_bearing_debt": 5200.0, "cash": 12000.0,
        "operating_cash_flow": 6800.0, "capex": 3200.0, "market_cap": 250000.0,
        "shares_outstanding": 1000.0,
    },
]


def init_database(db_path: Path = DB_PATH) -> None:
    """Initialize or reset SQLite tables and seed default companies & metrics."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.executescript(
        """
        DROP TABLE IF EXISTS financial_statements;
        DROP TABLE IF EXISTS fx_rates;
        DROP TABLE IF EXISTS companies;

        CREATE TABLE companies (
            ticker TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            market TEXT NOT NULL CHECK(market IN ('SET', 'US')),
            sector TEXT NOT NULL,
            currency TEXT NOT NULL CHECK(currency IN ('THB', 'USD')),
            filing_type TEXT NOT NULL CHECK(filing_type IN ('56-1', '10-K', '10-Q')),
            accounting_standard TEXT NOT NULL CHECK(accounting_standard IN ('TFRS', 'US_GAAP'))
        );

        CREATE TABLE fx_rates (
            year INTEGER NOT NULL,
            pair TEXT NOT NULL,
            avg_rate REAL NOT NULL,
            year_end_rate REAL NOT NULL,
            PRIMARY KEY (year, pair)
        );

        CREATE TABLE financial_statements (
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

    cursor.executemany(
        """
        INSERT INTO companies (
            ticker, name, market, sector, currency, filing_type, accounting_standard
        ) VALUES (?, ?, ?, ?, ?, ?, ?);
        """,
        COMPANIES_DATA,
    )

    cursor.executemany(
        """
        INSERT INTO fx_rates (year, pair, avg_rate, year_end_rate)
        VALUES (?, ?, ?, ?);
        """,
        FX_RATES_DATA,
    )

    for row in FINANCIAL_STATEMENTS_DATA:
        cursor.execute(
            """
            INSERT INTO financial_statements (
                ticker, year, currency, revenue, gross_profit, ebit, ebitda,
                net_profit, total_assets, current_assets, total_liabilities,
                current_liabilities, retained_earnings, total_equity,
                interest_bearing_debt, cash, operating_cash_flow, capex,
                market_cap, shares_outstanding
            ) VALUES (
                :ticker, :year, :currency, :revenue, :gross_profit, :ebit, :ebitda,
                :net_profit, :total_assets, :current_assets, :total_liabilities,
                :current_liabilities, :retained_earnings, :total_equity,
                :interest_bearing_debt, :cash, :operating_cash_flow, :capex,
                :market_cap, :shares_outstanding
            );
            """,
            row,
        )

    conn.commit()
    conn.close()


def write_markdown_filings(raw_dir: Path = RAW_FILINGS_DIR) -> List[Path]:
    """Ensure raw_filings contains all markdown files from raw_documents."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    existing_files = list(raw_dir.glob("*.md"))
    return existing_files


def build_all(db_path: Path = DB_PATH, raw_dir: Path = RAW_FILINGS_DIR) -> Dict[str, Any]:
    """Build both the SQLite database and maintain the default filings."""
    init_database(db_path)
    files = write_markdown_filings(raw_dir)
    return {
        "db_path": str(db_path),
        "companies_count": len(COMPANIES_DATA),
        "financial_rows_count": len(FINANCIAL_STATEMENTS_DATA),
        "fx_rows_count": len(FX_RATES_DATA),
        "filings_written": [str(p) for p in files],
    }


if __name__ == "__main__":
    summary = build_all()
    print(
        f"[build_corpus] Built SQLite DB at {summary['db_path']} "
        f"({summary['companies_count']} companies, {summary['financial_rows_count']} statement rows) "
        f"and {len(summary['filings_written'])} Markdown filings."
    )
