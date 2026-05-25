# =============================================================================
# screening_engine.py – Apply all filters and build the results DataFrame
# =============================================================================

from __future__ import annotations
import logging
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from valuation_models import compute_intrinsic_value
from data_fetcher import fetch_fx_rates

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Currency normalisation
# ---------------------------------------------------------------------------

def normalise_currency(
    df: pd.DataFrame,
    base_currency: str,
    fx_rates: Dict[str, float],
) -> pd.DataFrame:
    """
    Convert all monetary columns to `base_currency` using pre-fetched fx_rates.
    Only converts columns that are per-share prices / values.
    """
    price_cols = ["price", "dcf_value", "graham_value", "intrinsic_value",
                  "eps", "book_value", "forward_eps"]

    def _convert(row):
        ccy = row.get("currency", "USD") or "USD"
        if ccy == base_currency:
            return row
        rate = fx_rates.get(ccy, 1.0)   # rate = 1 unit of ccy in base_currency
        for col in price_cols:
            if col in row and pd.notna(row[col]) and row[col] is not None:
                row = row.copy()
                row[col] = row[col] * rate
        row["currency"] = base_currency
        return row

    return df.apply(_convert, axis=1)


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

def _within(series: pd.Series, max_val: Optional[float]) -> pd.Series:
    """Return a boolean mask: value <= max_val (NaN → excluded)."""
    if max_val is None:
        return pd.Series([True] * len(series), index=series.index)
    return series.notna() & (series <= max_val)


def _above(series: pd.Series, min_val: Optional[float]) -> pd.Series:
    """Return a boolean mask: value >= min_val (NaN → excluded)."""
    if min_val is None:
        return pd.Series([True] * len(series), index=series.index)
    return series.notna() & (series >= min_val)


# ---------------------------------------------------------------------------
# Main screening function
# ---------------------------------------------------------------------------

def run_screen(
    raw_df: pd.DataFrame,
    params: Dict,
    bond_yield: float,
    base_currency: Optional[str] = None,
    normalise: bool = False,
) -> pd.DataFrame:
    """
    Full screening pipeline:
        1. Compute intrinsic values for every stock.
        2. Optional: normalise monetary values to base_currency.
        3. Apply valuation ratio filters.
        4. Apply financial health / quality filters.
        5. Apply DCF margin-of-safety filter.
        6. Return clean, sorted results DataFrame.

    Args:
        raw_df:       Output of data_fetcher.fetch_batch_data().
        params:       Screening parameters from the sidebar.
        bond_yield:   Current treasury yield (%) for Graham.
        base_currency:Target currency for normalisation.
        normalise:    If True, convert all prices to base_currency.

    Returns:
        Filtered and enriched DataFrame ready for display.
    """
    if raw_df.empty:
        return pd.DataFrame()

    df = raw_df.copy()

    # --- Step 1: Compute intrinsic value for each row -------------------------
    iv_records = df.apply(
        lambda row: compute_intrinsic_value(row.to_dict(), params, bond_yield),
        axis=1,
        result_type="expand",
    )
    df = pd.concat([df, iv_records], axis=1)

    # --- Step 2: Currency normalisation ---------------------------------------
    if normalise and base_currency:
        fx = fetch_fx_rates(base_currency)
        df = normalise_currency(df, base_currency, fx)

    # --- Step 3: Valuation ratio filters --------------------------------------
    mask = pd.Series([True] * len(df), index=df.index)

    if params.get("pe_enabled", True):
        mask &= _within(df["pe_ratio"], params["pe_max"])

    if params.get("pb_enabled", True):
        mask &= _within(df["pb_ratio"], params["pb_max"])

    if params.get("ev_ebitda_enabled", True):
        mask &= _within(df["ev_ebitda"], params["ev_ebitda_max"])

    if params.get("ps_enabled", True):
        mask &= _within(df["ps_ratio"], params["ps_max"])

    # --- Step 4: Quality / health filters -------------------------------------
    if params.get("de_enabled", True):
        mask &= _within(df["debt_to_equity"], params["de_max"])

    if params.get("current_enabled", True):
        mask &= _above(df["current_ratio"], params["current_min"])

    if params.get("roe_enabled", True):
        # ROE from yfinance is a decimal (0.15 = 15 %)
        roe_threshold = params["roe_min"] / 100
        mask &= _above(df["roe"], roe_threshold)

    # --- Step 5: DCF margin-of-safety filter ----------------------------------
    if params.get("dcf_enabled") and params.get("dcf_mos_filter", False):
        mos = params["dcf_margin_safety"] / 100
        dcf_mask = (
            df["dcf_value"].isna() |   # keep if no DCF (model may be partly off)
            (df["price"] <= df["dcf_value"] * (1 - mos))
        )
        mask &= dcf_mask

    df = df[mask].copy()

    # --- Step 6: Require at least one intrinsic value ------------------------
    df = df[df["intrinsic_value"].notna()]

    # --- Step 7: Build display columns ----------------------------------------
    df = _build_display_df(df)

    return df.sort_values("Upside (%)", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Display DataFrame builder
# ---------------------------------------------------------------------------

def _fmt(val, decimals: int = 2, suffix: str = "") -> str:
    """Format numeric value or return 'N/A'."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.{decimals}f}{suffix}"


def _build_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """Select and rename columns for the results table."""
    display = pd.DataFrame()

    display["Ticker"]              = df["ticker"]
    display["Company"]             = df["name"].fillna(df["ticker"])
    display["Exchange"]            = df["exchange"].fillna("—")
    display["Currency"]            = df["currency"].fillna("USD")
    display["Price"]               = df["price"].round(2)
    display["Intrinsic Value"]     = df["intrinsic_value"].round(2)
    display["Upside (%)"]          = df["upside_pct"].round(1)
    display["DCF Value"]           = df["dcf_value"].apply(lambda x: round(x, 2) if pd.notna(x) else None)
    display["Graham Value"]        = df["graham_value"].apply(lambda x: round(x, 2) if pd.notna(x) else None)
    display["Model"]               = df["model_used"]
    display["P/E"]                 = df["pe_ratio"].apply(lambda x: round(x, 1) if pd.notna(x) else None)
    display["P/B"]                 = df["pb_ratio"].apply(lambda x: round(x, 2) if pd.notna(x) else None)
    display["EV/EBITDA"]           = df["ev_ebitda"].apply(lambda x: round(x, 1) if pd.notna(x) else None)
    display["P/S"]                 = df["ps_ratio"].apply(lambda x: round(x, 2) if pd.notna(x) else None)
    display["Debt/Equity"]         = df["debt_to_equity"].apply(lambda x: round(x, 2) if pd.notna(x) else None)
    display["Current Ratio"]       = df["current_ratio"].apply(lambda x: round(x, 2) if pd.notna(x) else None)
    display["ROE (%)"]             = df["roe"].apply(
        lambda x: round(x * 100, 1) if pd.notna(x) else None
    )
    display["Market Cap (M)"]      = df["market_cap"].apply(
        lambda x: round(x / 1e6, 0) if pd.notna(x) else None
    )
    display["Sector"]              = df["sector"].fillna("—")

    return display
