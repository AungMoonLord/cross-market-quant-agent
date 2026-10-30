"""
Deterministic Cross-Market Institutional Quant Calculator (`agent/quant_calc.py`).

Connects to `corpus/financials.db` and performs exact, hallucination-free Python calculations
for any Thai SET (THB, TFRS) or US (USD, US_GAAP) ticker:
1. DuPont 3-Step & 5-Step ROE Decomposition:
   - 3-Step: Net Profit Margin × Asset Turnover × Equity Multiplier
   - 5-Step: Tax/Interest Burden (Net Profit / EBIT) × Operating Margin (EBIT / Revenue)
             × Asset Turnover (Revenue / Total Assets) × Equity Multiplier (Total Assets / Total Equity)
             plus Gross Margin % and Debt-to-Equity.
2. Altman Z-Score:
   - Computes exact Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
   - Zone classification: Safe Zone (Z > 2.99), Grey Zone (1.81 <= Z <= 2.99), Distress Zone (Z < 1.81).
3. Liquidity, Cash Flow & Growth:
   - Current Ratio, Interest-Bearing Debt to Equity (IBD/E), Net Debt to EBITDA,
     Free Cash Flow (OCF - CapEx), FCF Yield (FCF / Market Cap), and 3-Year Revenue & Net Profit CAGR (2023-2025).
4. Cross-Market Peer Comparison & FX Normalization (`compare_cross_market`):
   - Compares any subset of SET (THB) and US (USD) tickers side-by-side.
   - Currency-neutral ratios (ROE %, Gross Margin %, Net Margin %, FCF Yield %, IBD/E, CAGR %) are compared directly.
   - Absolute figures (Revenue, Net Profit, EBITDA, Market Cap, FCF) are automatically converted to a common
     target currency (`USD` or `THB`) using the `fx_rates` table so raw THB and USD figures are never mixed.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from app.config import settings
from app.schemas import FinancialMetrics
from corpus.build_corpus import build_all


def _ensure_db_exists(db_path: Path = settings.sqlite_db_path) -> Path:
    """Ensure the SQLite database exists before querying."""
    if not db_path.exists():
        build_all(db_path=db_path, raw_dir=settings.raw_filings_dir)
    return db_path


def _get_readonly_connection(db_path: Path = settings.sqlite_db_path) -> sqlite3.Connection:
    """Open a read-only SQLite connection with row factory enabled."""
    resolved = _ensure_db_exists(db_path)
    uri = f"file:{resolved.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def get_fx_rate(
    year: int = 2025,
    rate_type: Literal["avg_rate", "year_end_rate"] = "avg_rate",
    db_path: Path = settings.sqlite_db_path,
) -> float:
    """Fetch USD_THB exchange rate for a given year from `fx_rates` table."""
    conn = _get_readonly_connection(db_path)
    try:
        row = conn.execute(
            "SELECT avg_rate, year_end_rate FROM fx_rates WHERE year = ? AND pair = 'USD_THB'",
            (year,),
        ).fetchone()
        if not row:
            row = conn.execute(
                "SELECT avg_rate, year_end_rate FROM fx_rates WHERE pair = 'USD_THB' ORDER BY year DESC LIMIT 1"
            ).fetchone()
        if not row:
            return 34.00
        return float(row[rate_type])
    finally:
        conn.close()


def fetch_company_and_statements(
    ticker: str,
    db_path: Path = settings.sqlite_db_path,
) -> TupleCompanyRows:
    """Fetch company metadata and all historical financial statement rows (2023-2025) for a ticker."""
    norm_ticker = ticker.strip().upper()
    conn = _get_readonly_connection(db_path)
    try:
        comp = conn.execute(
            "SELECT * FROM companies WHERE UPPER(ticker) = ?",
            (norm_ticker,),
        ).fetchone()
        if not comp:
            raise ValueError(
                f"Ticker '{norm_ticker}' not found in database. "
                "Supported tickers: BE, GOOG, JPM, PENG, SPCX (US)."
            )
        rows = conn.execute(
            "SELECT * FROM financial_statements WHERE UPPER(ticker) = ? ORDER BY year ASC",
            (norm_ticker,),
        ).fetchall()
        if not rows:
            raise ValueError(f"No financial statement rows found for ticker '{norm_ticker}'.")
        return dict(comp), [dict(r) for r in rows]
    finally:
        conn.close()


TupleCompanyRows = tuple[Dict[str, Any], List[Dict[str, Any]]]


def compute_dupont_roe(statement_row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute 3-Step and 5-Step DuPont ROE decomposition for a single fiscal year.
    """
    rev = float(statement_row["revenue"])
    gp = float(statement_row["gross_profit"])
    ebit = float(statement_row["ebit"])
    net_profit = float(statement_row["net_profit"])
    assets = float(statement_row["total_assets"])
    equity = float(statement_row["total_equity"])
    ibd = float(statement_row["interest_bearing_debt"])

    gross_margin = (gp / rev) if rev else 0.0
    operating_margin = (ebit / rev) if rev else 0.0
    net_profit_margin = (net_profit / rev) if rev else 0.0
    tax_and_interest_burden = (net_profit / ebit) if ebit else 0.0
    asset_turnover = (rev / assets) if assets else 0.0
    equity_multiplier = (assets / equity) if equity else 0.0
    debt_to_equity = (ibd / equity) if equity else 0.0

    # 3-Step DuPont ROE = Net Profit Margin * Asset Turnover * Equity Multiplier
    roe_3_step = net_profit_margin * asset_turnover * equity_multiplier
    # 5-Step DuPont ROE = (Net Profit / EBIT) * (EBIT / Revenue) * (Revenue / Assets) * (Assets / Equity)
    roe_5_step = tax_and_interest_burden * operating_margin * asset_turnover * equity_multiplier

    return {
        "gross_margin_pct": round(gross_margin * 100.0, 2),
        "operating_ebit_margin_pct": round(operating_margin * 100.0, 2),
        "net_profit_margin_pct": round(net_profit_margin * 100.0, 2),
        "tax_and_interest_burden_ratio": round(tax_and_interest_burden, 4),
        "asset_turnover_x": round(asset_turnover, 4),
        "equity_multiplier_x": round(equity_multiplier, 4),
        "debt_to_equity_x": round(debt_to_equity, 4),
        "roe_3_step_pct": round(roe_3_step * 100.0, 2),
        "roe_5_step_pct": round(roe_5_step * 100.0, 2),
    }


def compute_altman_z(statement_row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute Altman Z-Score:
      X1 = Working Capital / Total Assets = (Current Assets - Current Liabilities) / Total Assets
      X2 = Retained Earnings / Total Assets
      X3 = EBIT / Total Assets
      X4 = Market Value of Equity (Market Cap) / Total Liabilities
      X5 = Revenue / Total Assets
      Z  = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
    Zones:
      Z > 2.99          -> Safe Zone ("SAFE")
      1.81 <= Z <= 2.99 -> Grey Zone ("GREY")
      Z < 1.81          -> Distress Zone ("DISTRESS")
    """
    assets = float(statement_row["total_assets"])
    curr_assets = float(statement_row["current_assets"])
    curr_liab = float(statement_row["current_liabilities"])
    total_liab = float(statement_row["total_liabilities"])
    retained = float(statement_row["retained_earnings"])
    ebit = float(statement_row["ebit"])
    mcap = float(statement_row["market_cap"])
    rev = float(statement_row["revenue"])

    working_capital = curr_assets - curr_liab
    x1 = (working_capital / assets) if assets else 0.0
    x2 = (retained / assets) if assets else 0.0
    x3 = (ebit / assets) if assets else 0.0
    x4 = (mcap / total_liab) if total_liab else 0.0
    x5 = (rev / assets) if assets else 0.0

    z_score = (1.2 * x1) + (1.4 * x2) + (3.3 * x3) + (0.6 * x4) + (1.0 * x5)
    if z_score > 2.99:
        zone_code: Literal["SAFE", "GREY", "DISTRESS"] = "SAFE"
        zone_label = "Safe Zone"
    elif z_score >= 1.81:
        zone_code = "GREY"
        zone_label = "Grey Zone"
    else:
        zone_code = "DISTRESS"
        zone_label = "Distress Zone"

    return {
        "working_capital_to_assets_x1": round(x1, 4),
        "retained_earnings_to_assets_x2": round(x2, 4),
        "ebit_to_assets_x3": round(x3, 4),
        "market_cap_to_total_liabilities_x4": round(x4, 4),
        "asset_turnover_x5": round(x5, 4),
        "altman_z_score": round(z_score, 2),
        "altman_zone": zone_code,
        "altman_zone_label": zone_label,
    }


def compute_liquidity_leverage(statement_row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute Current Ratio, IBD/E, Net Debt to EBITDA, ROIC %, P/E, and EV/EBITDA.
    """
    curr_assets = float(statement_row["current_assets"])
    curr_liab = float(statement_row["current_liabilities"])
    ibd = float(statement_row["interest_bearing_debt"])
    cash = float(statement_row["cash"])
    equity = float(statement_row["total_equity"])
    ebit = float(statement_row["ebit"])
    ebitda = float(statement_row["ebitda"])
    net_profit = float(statement_row["net_profit"])
    mcap = float(statement_row["market_cap"])
    rev = float(statement_row["revenue"])

    current_ratio = (curr_assets / curr_liab) if curr_liab else 0.0
    ibd_to_equity = (ibd / equity) if equity else 0.0
    net_debt = ibd - cash
    net_debt_to_ebitda = (net_debt / ebitda) if ebitda else 0.0

    # Invested Capital = Total Equity + Interest-Bearing Debt - Cash (floor at 10% of Equity if net cash > equity)
    invested_capital = max(equity * 0.1, equity + ibd - cash)
    nopat = ebit * 0.80  # Standardized 20% effective statutory corporate tax rate proxy for NOPAT
    roic_pct = (nopat / invested_capital) * 100.0 if invested_capital else 0.0

    pe_ratio = (mcap / net_profit) if net_profit > 0 else 0.0
    enterprise_value = mcap + ibd - cash
    ev_to_ebitda = (enterprise_value / ebitda) if ebitda > 0 else 0.0
    ebitda_margin_pct = (ebitda / rev) * 100.0 if rev else 0.0

    return {
        "current_ratio_x": round(current_ratio, 2),
        "ibd_to_equity_x": round(ibd_to_equity, 2),
        "net_debt_native_m": round(net_debt, 2),
        "net_debt_to_ebitda_x": round(net_debt_to_ebitda, 2),
        "roic_pct": round(roic_pct, 2),
        "pe_ratio_x": round(pe_ratio, 2),
        "enterprise_value_native_m": round(enterprise_value, 2),
        "ev_to_ebitda_x": round(ev_to_ebitda, 2),
        "ebitda_margin_pct": round(ebitda_margin_pct, 2),
    }


def compute_fcf_and_cagr(
    statement_rows: List[Dict[str, Any]],
    target_year: int = 2025,
) -> Dict[str, Any]:
    """
    Compute Free Cash Flow (OCF - CapEx), FCF Yield (FCF / Market Cap),
    and 3-Year (2023 to 2025, n=2 compounding intervals) Revenue & Net Profit CAGR.
    """
    by_year = {int(r["year"]): r for r in statement_rows}
    row_target = by_year.get(target_year) or statement_rows[-1]

    ocf = float(row_target["operating_cash_flow"])
    capex = float(row_target["capex"])
    mcap = float(row_target["market_cap"])
    rev_target = float(row_target["revenue"])

    fcf = ocf - capex
    fcf_yield_pct = (fcf / mcap) * 100.0 if mcap > 0 else 0.0
    fcf_margin_pct = (fcf / rev_target) * 100.0 if rev_target > 0 else 0.0

    # 3-Year series 2023 -> 2025 (2 annual compounding periods)
    row_2023 = by_year.get(2023)
    row_2025 = by_year.get(2025)
    if row_2023 and row_2025:
        rev_2023 = float(row_2023["revenue"])
        rev_2025 = float(row_2025["revenue"])
        np_2023 = float(row_2023["net_profit"])
        np_2025 = float(row_2025["net_profit"])

        rev_cagr_pct = ((math.pow(rev_2025 / rev_2023, 1.0 / 2.0) - 1.0) * 100.0) if rev_2023 > 0 else 0.0
        np_cagr_pct = (
            ((math.pow(np_2025 / np_2023, 1.0 / 2.0) - 1.0) * 100.0)
            if (np_2023 > 0 and np_2025 > 0)
            else 0.0
        )
    else:
        rev_cagr_pct = 0.0
        np_cagr_pct = 0.0

    return {
        "operating_cash_flow_native_m": round(ocf, 2),
        "capex_native_m": round(capex, 2),
        "free_cash_flow_native_m": round(fcf, 2),
        "fcf_margin_pct": round(fcf_margin_pct, 2),
        "fcf_yield_pct": round(fcf_yield_pct, 2),
        "revenue_cagr_2023_2025_pct": round(rev_cagr_pct, 2),
        "net_profit_cagr_2023_2025_pct": round(np_cagr_pct, 2),
    }


def _convert_amount(
    amount: float,
    native_currency: str,
    target_currency: Literal["NATIVE", "USD", "THB"],
    usd_thb_rate: float,
) -> tuple[float, Literal["THB", "USD"], float]:
    """
    Convert an amount from `native_currency` ('THB' or 'USD') into `target_currency`.
    Returns (converted_amount, reporting_currency, applied_fx_rate).
    """
    nat: Literal["THB", "USD"] = "THB" if native_currency.upper() == "THB" else "USD"
    if target_currency == "NATIVE" or target_currency == nat:
        return round(amount, 2), nat, 1.0

    if nat == "THB" and target_currency == "USD":
        return round(amount / usd_thb_rate, 2), "USD", usd_thb_rate

    if nat == "USD" and target_currency == "THB":
        return round(amount * usd_thb_rate, 2), "THB", usd_thb_rate

    return round(amount, 2), nat, 1.0


def build_financial_metrics_schema(
    ticker: str,
    year: int = 2025,
    target_currency: Literal["NATIVE", "USD", "THB"] = "NATIVE",
    db_path: Path = settings.sqlite_db_path,
) -> FinancialMetrics:
    """
    Build a validated Pydantic v2 `FinancialMetrics` object for a ticker and year.
    """
    comp, rows = fetch_company_and_statements(ticker, db_path=db_path)
    by_year = {int(r["year"]): r for r in rows}
    if not by_year:
        raise ValueError(f"No financial statements found for {ticker}")
    if year not in by_year:
        # Fall back gracefully to latest available year instead of crashing
        year = max(by_year.keys())

    row = by_year[year]
    fx_rate = get_fx_rate(year=year, db_path=db_path)
    native_curr: Literal["THB", "USD"] = "THB" if comp["currency"] == "THB" else "USD"

    rev_conv, rep_curr, fx_applied = _convert_amount(float(row["revenue"]), native_curr, target_currency, fx_rate)
    np_conv, _, _ = _convert_amount(float(row["net_profit"]), native_curr, target_currency, fx_rate)
    ebitda_conv, _, _ = _convert_amount(float(row["ebitda"]), native_curr, target_currency, fx_rate)
    fcf_native = float(row["operating_cash_flow"]) - float(row["capex"])
    fcf_conv, _, _ = _convert_amount(fcf_native, native_curr, target_currency, fx_rate)

    dupont = compute_dupont_roe(row)
    altman = compute_altman_z(row)
    liq = compute_liquidity_leverage(row)

    return FinancialMetrics(
        ticker=comp["ticker"],
        market=comp["market"],
        year=year,
        native_currency=native_curr,
        reporting_currency=rep_curr,
        fx_rate_applied=fx_applied,
        revenue=rev_conv,
        net_profit=np_conv,
        ebitda=ebitda_conv,
        free_cash_flow=fcf_conv,
        net_profit_margin_pct=dupont["net_profit_margin_pct"],
        ebitda_margin_pct=liq["ebitda_margin_pct"],
        asset_turnover=round(dupont["asset_turnover_x"], 2),
        equity_multiplier=round(dupont["equity_multiplier_x"], 2),
        roe_dupont_pct=dupont["roe_3_step_pct"],
        roic_pct=liq["roic_pct"],
        debt_to_equity=round(dupont["debt_to_equity_x"], 2),
        net_debt_to_ebitda=liq["net_debt_to_ebitda_x"],
        current_ratio=liq["current_ratio_x"],
        pe_ratio=liq["pe_ratio_x"],
        ev_to_ebitda=liq["ev_to_ebitda_x"],
        altman_z_score=altman["altman_z_score"],
        altman_zone=altman["altman_zone"],
    )


def calculate_ticker_quant_metrics(
    ticker: str,
    year: int = 2025,
    metric_type: Literal["dupont_roe", "altman_z", "liquidity_leverage", "fcf_cagr", "all"] = "all",
    target_currency: Literal["NATIVE", "USD", "THB"] = "NATIVE",
    db_path: Path = settings.sqlite_db_path,
) -> Dict[str, Any]:
    """
    Primary deterministic quant calculation entry point for `compute_quant_metrics` tool.
    """
    comp, rows = fetch_company_and_statements(ticker, db_path=db_path)
    by_year = {int(r["year"]): r for r in rows}
    if not by_year:
        raise ValueError(f"No financial statements found for {ticker}")
    if year not in by_year:
        # Fall back gracefully to latest available year instead of crashing
        year = max(by_year.keys())

    row = by_year[year]
    usd_thb = get_fx_rate(year=year, db_path=db_path)
    native_curr: Literal["THB", "USD"] = "THB" if comp["currency"] == "THB" else "USD"

    dupont = compute_dupont_roe(row)
    altman = compute_altman_z(row)
    liq = compute_liquidity_leverage(row)
    fcf_cagr = compute_fcf_and_cagr(rows, target_year=year)
    schema_obj = build_financial_metrics_schema(
        ticker=ticker,
        year=year,
        target_currency=target_currency,
        db_path=db_path,
    )

    # Also provide USD and THB normalized headline numbers so cross-currency questions are always exact
    rev_usd, _, _ = _convert_amount(float(row["revenue"]), native_curr, "USD", usd_thb)
    rev_thb, _, _ = _convert_amount(float(row["revenue"]), native_curr, "THB", usd_thb)
    np_usd, _, _ = _convert_amount(float(row["net_profit"]), native_curr, "USD", usd_thb)
    np_thb, _, _ = _convert_amount(float(row["net_profit"]), native_curr, "THB", usd_thb)
    fcf_usd, _, _ = _convert_amount(fcf_cagr["free_cash_flow_native_m"], native_curr, "USD", usd_thb)
    fcf_thb, _, _ = _convert_amount(fcf_cagr["free_cash_flow_native_m"], native_curr, "THB", usd_thb)
    mcap_usd, _, _ = _convert_amount(float(row["market_cap"]), native_curr, "USD", usd_thb)

    payload: Dict[str, Any] = {
        "ticker": comp["ticker"],
        "company_name": comp["name"],
        "market": comp["market"],
        "sector": comp["sector"],
        "filing_type": comp["filing_type"],
        "accounting_standard": comp["accounting_standard"],
        "year": year,
        "native_currency": native_curr,
        "usd_thb_fx_rate": usd_thb,
        "headline_numbers_millions": {
            "revenue_native_m": float(row["revenue"]),
            "revenue_usd_m": rev_usd,
            "revenue_thb_m": rev_thb,
            "net_profit_native_m": float(row["net_profit"]),
            "net_profit_usd_m": np_usd,
            "net_profit_thb_m": np_thb,
            "ebitda_native_m": float(row["ebitda"]),
            "free_cash_flow_native_m": fcf_cagr["free_cash_flow_native_m"],
            "free_cash_flow_usd_m": fcf_usd,
            "free_cash_flow_thb_m": fcf_thb,
            "market_cap_native_m": float(row["market_cap"]),
            "market_cap_usd_m": mcap_usd,
        },
        "financial_metrics_schema": schema_obj.model_dump(),
    }

    if metric_type in ("dupont_roe", "all"):
        payload["dupont_roe"] = dupont
    if metric_type in ("altman_z", "all"):
        payload["altman_z"] = altman
    if metric_type in ("liquidity_leverage", "all"):
        payload["liquidity_leverage"] = liq
    if metric_type in ("fcf_cagr", "all"):
        payload["fcf_cagr"] = fcf_cagr

    return payload


def compare_cross_market(
    tickers: List[str],
    target_currency: Literal["USD", "THB"] = "USD",
    year: int = 2025,
    db_path: Path = settings.sqlite_db_path,
) -> Dict[str, Any]:
    """
    Deterministic Cross-Market Peer Comparison & FX Normalization.
    Compares any subset of SET (THB) and US (USD) tickers side-by-side:
    - Currency-neutral ratios (ROE %, Gross Margin %, Net Margin %, FCF Yield %, IBD/E, 3Y CAGR %)
      are compared directly without distortion.
    - Absolute monetary figures (Revenue, Net Profit, EBITDA, Free Cash Flow, Market Cap)
      are converted to `target_currency` ('USD' or 'THB') using the `fx_rates` table.
    """
    if not tickers or len(tickers) < 1:
        raise ValueError("At least one ticker must be provided for cross-market comparison.")

    usd_thb_rate = get_fx_rate(year=year, db_path=db_path)
    peer_table: List[Dict[str, Any]] = []
    metrics_schemas: List[Dict[str, Any]] = []

    for raw_t in tickers:
        t = raw_t.strip().upper()
        comp, rows = fetch_company_and_statements(t, db_path=db_path)
        by_year = {int(r["year"]): r for r in rows}
        row = by_year.get(year) or rows[-1]
        native_curr: Literal["THB", "USD"] = "THB" if comp["currency"] == "THB" else "USD"

        dupont = compute_dupont_roe(row)
        altman = compute_altman_z(row)
        liq = compute_liquidity_leverage(row)
        fcf_cagr = compute_fcf_and_cagr(rows, target_year=year)
        fm_schema = build_financial_metrics_schema(
            ticker=t,
            year=year,
            target_currency=target_currency,
            db_path=db_path,
        )
        metrics_schemas.append(fm_schema.model_dump())

        rev_norm, _, fx_used = _convert_amount(float(row["revenue"]), native_curr, target_currency, usd_thb_rate)
        np_norm, _, _ = _convert_amount(float(row["net_profit"]), native_curr, target_currency, usd_thb_rate)
        ebitda_norm, _, _ = _convert_amount(float(row["ebitda"]), native_curr, target_currency, usd_thb_rate)
        fcf_norm, _, _ = _convert_amount(fcf_cagr["free_cash_flow_native_m"], native_curr, target_currency, usd_thb_rate)
        mcap_norm, _, _ = _convert_amount(float(row["market_cap"]), native_curr, target_currency, usd_thb_rate)

        peer_table.append(
            {
                "ticker": comp["ticker"],
                "name": comp["name"],
                "market": comp["market"],
                "filing_type": comp["filing_type"],
                "accounting_standard": comp["accounting_standard"],
                "native_currency": native_curr,
                "normalized_currency": target_currency,
                "fx_rate_applied": fx_used,
                "normalized_absolutes_millions": {
                    f"revenue_{target_currency.lower()}_m": rev_norm,
                    f"ebitda_{target_currency.lower()}_m": ebitda_norm,
                    f"net_profit_{target_currency.lower()}_m": np_norm,
                    f"free_cash_flow_{target_currency.lower()}_m": fcf_norm,
                    f"market_cap_{target_currency.lower()}_m": mcap_norm,
                },
                "native_absolutes_millions": {
                    "revenue_native_m": float(row["revenue"]),
                    "net_profit_native_m": float(row["net_profit"]),
                    "free_cash_flow_native_m": fcf_cagr["free_cash_flow_native_m"],
                    "market_cap_native_m": float(row["market_cap"]),
                },
                "currency_neutral_ratios": {
                    "roe_dupont_pct": dupont["roe_3_step_pct"],
                    "gross_margin_pct": dupont["gross_margin_pct"],
                    "operating_margin_pct": dupont["operating_ebit_margin_pct"],
                    "net_profit_margin_pct": dupont["net_profit_margin_pct"],
                    "asset_turnover_x": dupont["asset_turnover_x"],
                    "equity_multiplier_x": dupont["equity_multiplier_x"],
                    "ibd_to_equity_x": liq["ibd_to_equity_x"],
                    "net_debt_to_ebitda_x": liq["net_debt_to_ebitda_x"],
                    "current_ratio_x": liq["current_ratio_x"],
                    "fcf_yield_pct": fcf_cagr["fcf_yield_pct"],
                    "pe_ratio_x": liq["pe_ratio_x"],
                    "ev_to_ebitda_x": liq["ev_to_ebitda_x"],
                    "altman_z_score": altman["altman_z_score"],
                    "altman_zone": altman["altman_zone_label"],
                    "revenue_cagr_3y_pct": fcf_cagr["revenue_cagr_2023_2025_pct"],
                    "net_profit_cagr_3y_pct": fcf_cagr["net_profit_cagr_2023_2025_pct"],
                },
            }
        )

    # Generate deterministic comparative ranking insights
    highest_roe = max(peer_table, key=lambda x: x["currency_neutral_ratios"]["roe_dupont_pct"])
    highest_margin = max(peer_table, key=lambda x: x["currency_neutral_ratios"]["net_profit_margin_pct"])
    highest_fcf_yield = max(peer_table, key=lambda x: x["currency_neutral_ratios"]["fcf_yield_pct"])
    highest_rev_cagr = max(peer_table, key=lambda x: x["currency_neutral_ratios"]["revenue_cagr_3y_pct"])

    return {
        "year": year,
        "target_currency": target_currency,
        "usd_thb_fx_rate": usd_thb_rate,
        "accounting_standard_note": (
            "SET companies report under TFRS (TAS 38 allows capitalization of qualifying development costs), "
            "whereas US companies report under US GAAP (ASC 730 requires immediate expensing of R&D)."
        ),
        "peers": peer_table,
        "financial_metrics_schemas": metrics_schemas,
        "summary_leaders": {
            "highest_roe_ticker": f"{highest_roe['ticker']} ({highest_roe['currency_neutral_ratios']['roe_dupont_pct']}%)",
            "highest_net_margin_ticker": (
                f"{highest_margin['ticker']} ({highest_margin['currency_neutral_ratios']['net_profit_margin_pct']}%)"
            ),
            "highest_fcf_yield_ticker": (
                f"{highest_fcf_yield['ticker']} ({highest_fcf_yield['currency_neutral_ratios']['fcf_yield_pct']}%)"
            ),
            "fastest_3y_revenue_cagr_ticker": (
                f"{highest_rev_cagr['ticker']} ({highest_rev_cagr['currency_neutral_ratios']['revenue_cagr_3y_pct']}%)"
            ),
        },
    }