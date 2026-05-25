# =============================================================================
# valuation_models.py – DCF and Benjamin Graham intrinsic value calculators
# =============================================================================

from __future__ import annotations
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Discounted Cash Flow (DCF) model
# ---------------------------------------------------------------------------

def calculate_dcf(
    fcf: float,
    shares_outstanding: float,
    growth_rate: float,        # decimal, e.g. 0.08 for 8 %
    discount_rate: float,      # decimal, e.g. 0.09 for 9 %
    terminal_growth: float,    # decimal, e.g. 0.03 for 3 %
    years: int = 10,
) -> Optional[float]:
    """
    Two-stage DCF model:
        Stage 1: Project FCF for `years` years at `growth_rate`.
        Stage 2: Terminal value via Gordon Growth Model.
        Discount everything back at `discount_rate`.

    Returns intrinsic value per share, or None if inputs are invalid.
    """
    if not (fcf and shares_outstanding and shares_outstanding > 0):
        return None
    if fcf <= 0:
        # Negative FCF → can't compute meaningful DCF
        return None
    if discount_rate <= terminal_growth:
        logger.debug("DCF: discount_rate must exceed terminal_growth")
        return None

    pv_stage1 = 0.0
    for yr in range(1, years + 1):
        projected_fcf = fcf * (1 + growth_rate) ** yr
        pv_stage1 += projected_fcf / (1 + discount_rate) ** yr

    # Terminal value at end of projection horizon
    terminal_fcf = fcf * (1 + growth_rate) ** years * (1 + terminal_growth)
    terminal_value = terminal_fcf / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / (1 + discount_rate) ** years

    total_equity_value = pv_stage1 + pv_terminal
    intrinsic_per_share = total_equity_value / shares_outstanding
    return max(intrinsic_per_share, 0.0)   # floor at 0


# ---------------------------------------------------------------------------
# Benjamin Graham Formula
# ---------------------------------------------------------------------------

def calculate_graham_value(
    eps: float,
    growth_rate_pct: float,    # percent, e.g. 7.0 for 7 %
    bond_yield_pct: float,     # percent, e.g. 4.4 for 4.4 %
    no_growth_pe: float = 8.5,
) -> Optional[float]:
    """
    Benjamin Graham intrinsic value formula:
        V = EPS × (no_growth_pe + 2g) × 4.4 / Y

    Where:
        g = expected annual EPS growth (%)
        Y = current AAA corporate bond yield (%)
        4.4 = Graham's original average bond yield constant

    Returns intrinsic value, or None if inputs are invalid.
    """
    if not (eps and eps > 0):
        return None
    if bond_yield_pct <= 0:
        return None

    value = eps * (no_growth_pe + 2 * growth_rate_pct) * 4.4 / bond_yield_pct
    return max(value, 0.0)


# ---------------------------------------------------------------------------
# Composite intrinsic value + upside
# ---------------------------------------------------------------------------

def compute_intrinsic_value(
    row: Dict,
    params: Dict,
    bond_yield: float,
) -> Dict:
    """
    Compute intrinsic value for a single stock row using enabled models.

    Args:
        row:        Dict of stock metrics from data_fetcher.
        params:     User-configured screening parameters.
        bond_yield: Current treasury yield (%) for Graham formula.

    Returns:
        Dict with keys:
            dcf_value      – DCF intrinsic value per share (or None)
            graham_value   – Graham intrinsic value (or None)
            intrinsic_value– Weighted average of enabled models (or None)
            upside_pct     – (intrinsic - price) / price × 100 (or None)
            model_used     – String label of model(s) used
    """
    price = row.get("price")
    results = {
        "dcf_value":       None,
        "graham_value":    None,
        "intrinsic_value": None,
        "upside_pct":      None,
        "model_used":      "—",
    }

    if not price or price <= 0:
        return results

    values, labels = [], []

    # — DCF —
    if params.get("dcf_enabled"):
        fcf    = row.get("free_cashflow") or row.get("fcf")
        shares = row.get("shares_outstanding")
        gr     = params["dcf_growth_rate"]   / 100
        dr     = params["dcf_discount_rate"] / 100
        tg     = params["dcf_terminal_growth"] / 100

        dcf_val = calculate_dcf(fcf, shares, gr, dr, tg, params["dcf_years"])
        results["dcf_value"] = dcf_val
        if dcf_val:
            # Apply margin-of-safety threshold before including
            mos = params["dcf_margin_safety"] / 100
            threshold = dcf_val * (1 - mos)
            if price <= threshold:
                values.append(dcf_val)
                labels.append("DCF")
            elif params.get("use_dcf_for_upside", True):
                # Still contribute to intrinsic value even if below threshold
                values.append(dcf_val)
                labels.append("DCF")

    # — Graham —
    if params.get("graham_enabled"):
        eps = row.get("eps")
        g   = params["graham_growth"]
        pe0 = params["graham_no_growth_pe"]

        graham_val = calculate_graham_value(eps, g, bond_yield, pe0)
        results["graham_value"] = graham_val
        if graham_val:
            values.append(graham_val)
            labels.append("Graham")

    # Composite: simple average of available models
    if values:
        intrinsic = sum(values) / len(values)
        results["intrinsic_value"] = intrinsic
        results["upside_pct"] = (intrinsic - price) / price * 100
        results["model_used"] = " + ".join(labels)

    return results
