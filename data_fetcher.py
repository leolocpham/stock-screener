# =============================================================================
# data_fetcher.py – yfinance data retrieval with robust error handling
#
# Design principles:
#   • Never crash the pipeline on a single bad ticker – return None instead.
#   • Fill every missing metric with None (displayed as "N/A" in the UI).
#   • Use Streamlit's @st.cache_data to avoid redundant API calls within a session.
#   • Batch-fetch using a thread pool to stay inside API rate limits.
# =============================================================================

from __future__ import annotations
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

from config import FX_TICKERS, TREASURY_TICKER, CACHE_TTL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(info: dict, *keys, default=None):
    """Return the first non-None value for any of the given keys."""
    for k in keys:
        v = info.get(k)
        if v is not None and v != "":
            try:
                f = float(v)
                if np.isfinite(f):
                    return f
            except (TypeError, ValueError):
                return v   # return as-is for non-numeric fields
    return default


def _get_free_cash_flow(ticker_obj: yf.Ticker) -> Optional[float]:
    """
    Derive Free Cash Flow = Operating Cash Flow − Capital Expenditures.
    Handles multiple yfinance naming conventions across library versions.
    """
    try:
        cf = ticker_obj.cashflow
        if cf is None or cf.empty:
            return None

        ocf_keys  = ["Operating Cash Flow",
                     "Total Cash From Operating Activities",
                     "Cash Flow From Continuing Operating Activities"]
        capex_keys = ["Capital Expenditure",
                      "Capital Expenditures",
                      "Purchase Of Property Plant And Equipment",
                      "Purchases Of Property Plant And Equipment"]

        ocf = capex = None
        for k in ocf_keys:
            if k in cf.index:
                ocf = float(cf.loc[k].iloc[0])
                break
        for k in capex_keys:
            if k in cf.index:
                capex = float(cf.loc[k].iloc[0])
                break

        if ocf is not None:
            # capex is stored as a negative number in yfinance
            return ocf + (capex if capex is not None else 0.0)
    except Exception as exc:
        logger.debug(f"FCF extraction failed: {exc}")
    return None


# ---------------------------------------------------------------------------
# Single-ticker fetch
# ---------------------------------------------------------------------------

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_ticker_data(ticker: str) -> Optional[Dict]:
    """
    Fetch all financial metrics for a single ticker via yfinance.

    Returns a flat dictionary of metrics, or None if the ticker is invalid /
    data is unavailable.  All numeric fields that cannot be resolved are None.
    """
    try:
        t = yf.Ticker(ticker)
        info: dict = t.info or {}

        # Verify the ticker is tradeable
        price = _safe(info, "currentPrice", "regularMarketPrice",
                      "navPrice", "previousClose")
        if price is None:
            return None

        fcf      = _get_free_cash_flow(t)
        currency = info.get("currency", "USD") or "USD"

        # Some LSE stocks quote in GBX (pence); normalise to GBP
        if currency == "GBp":
            currency = "GBP"
            if price:
                price /= 100

        return {
            "ticker":             ticker,
            "name":               info.get("longName") or info.get("shortName", ticker),
            "exchange":           info.get("exchange", ""),
            "sector":             info.get("sector", ""),
            "industry":           info.get("industry", ""),
            "currency":           currency,
            # Price & market data
            "price":              price,
            "market_cap":         _safe(info, "marketCap"),
            "fifty_two_wk_high":  _safe(info, "fiftyTwoWeekHigh"),
            "fifty_two_wk_low":   _safe(info, "fiftyTwoWeekLow"),
            # Valuation ratios
            "pe_ratio":           _safe(info, "trailingPE"),
            "forward_pe":         _safe(info, "forwardPE"),
            "pb_ratio":           _safe(info, "priceToBook"),
            "ev_ebitda":          _safe(info, "enterpriseToEbitda"),
            "ps_ratio":           _safe(info, "priceToSalesTrailing12Months"),
            "peg_ratio":          _safe(info, "pegRatio"),
            # Income / EPS
            "eps":                _safe(info, "trailingEps"),
            "forward_eps":        _safe(info, "forwardEps"),
            "earnings_growth":    _safe(info, "earningsGrowth"),  # decimal
            "revenue_growth":     _safe(info, "revenueGrowth"),   # decimal
            "profit_margin":      _safe(info, "profitMargins"),
            # Balance sheet
            "book_value":         _safe(info, "bookValue"),
            "debt_to_equity":     _safe(info, "debtToEquity"),
            "current_ratio":      _safe(info, "currentRatio"),
            "quick_ratio":        _safe(info, "quickRatio"),
            # Returns
            "roe":                _safe(info, "returnOnEquity"),   # decimal
            "roa":                _safe(info, "returnOnAssets"),
            # Cash flow
            "fcf":                fcf,
            "operating_cashflow": _safe(info, "operatingCashflow"),
            "free_cashflow":      _safe(info, "freeCashflow") or fcf,
            # Shares
            "shares_outstanding": _safe(info, "sharesOutstanding"),
            "float_shares":       _safe(info, "floatShares"),
            # Dividend
            "dividend_yield":     _safe(info, "dividendYield"),
        }

    except Exception as exc:
        logger.warning(f"[{ticker}] fetch failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Batch fetch with threading & progress reporting
# ---------------------------------------------------------------------------

def fetch_batch_data(
    tickers: List[str],
    progress_callback=None,
    max_workers: int = 8,
    delay: float = 0.1,
) -> pd.DataFrame:
    """
    Fetch financial data for a list of tickers concurrently.

    Args:
        tickers:           Ticker symbols to fetch.
        progress_callback: Optional callable(current, total, ticker) for UI updates.
        max_workers:       Thread-pool size (keep ≤10 to respect yfinance rate limits).
        delay:             Per-request sleep (seconds) to avoid rate limiting.

    Returns:
        DataFrame with one row per successful ticker.
    """
    results = []
    completed = 0
    total = len(tickers)

    def _fetch_one(ticker: str) -> Optional[Dict]:
        time.sleep(delay)
        return fetch_ticker_data(ticker)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one, t): t for t in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            completed += 1
            try:
                data = future.result()
                if data:
                    results.append(data)
            except Exception as exc:
                logger.warning(f"[{ticker}] unexpected error: {exc}")
            if progress_callback:
                progress_callback(completed, total, ticker)

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Price history (for charts)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_price_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Fetch OHLCV price history for a single ticker.

    Returns DataFrame with DatetimeIndex, or empty DataFrame on failure.
    """
    try:
        hist = yf.Ticker(ticker).history(period=period)
        if hist.empty:
            return pd.DataFrame()
        hist.index = hist.index.tz_localize(None)   # strip timezone for Plotly
        return hist[["Open", "High", "Low", "Close", "Volume"]]
    except Exception as exc:
        logger.warning(f"[{ticker}] price history failed: {exc}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Currency conversion rates (via yfinance FX tickers)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_fx_rates(base_currency: str = "USD") -> Dict[str, float]:
    """
    Return a dict mapping currency code → 1 unit in base_currency.
    Falls back to 1.0 on failure (no conversion).
    """
    rates: Dict[str, float] = {"USD": 1.0}

    for ccy, fx_sym in FX_TICKERS.items():
        if fx_sym is None:
            rates[ccy] = 1.0
            continue
        try:
            info = yf.Ticker(fx_sym).info or {}
            rate = info.get("regularMarketPrice") or info.get("previousClose")
            if rate:
                # rates[ccy] is "1 ccy in USD"
                rates[ccy] = float(rate)
        except Exception:
            rates[ccy] = 1.0   # fallback: no conversion

    # If base is not USD, re-base everything
    if base_currency != "USD" and base_currency in rates:
        usd_per_base = rates[base_currency]
        if usd_per_base and usd_per_base > 0:
            rates = {k: v / usd_per_base for k, v in rates.items()}

    return rates


# ---------------------------------------------------------------------------
# US Treasury yield (used in Graham formula)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_treasury_yield() -> float:
    """
    Return the current US 10-year Treasury yield as a percentage.
    Falls back to 4.4 (Graham's original constant) on failure.
    """
    try:
        info = yf.Ticker(TREASURY_TICKER).info or {}
        rate = info.get("regularMarketPrice") or info.get("previousClose")
        if rate:
            return float(rate)
    except Exception:
        pass
    return 4.4   # Graham's original bond yield constant
