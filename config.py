# =============================================================================
# config.py – Global constants, defaults, and exchange definitions
# =============================================================================

# ---------------------------------------------------------------------------
# Exchange definitions
# Each entry: display_name → {suffix, currency, label}
# ---------------------------------------------------------------------------
EXCHANGES = {
    "NASDAQ":   {"suffix": "",    "currency": "USD", "label": "NASDAQ (US)"},
    "NYSE":     {"suffix": "",    "currency": "USD", "label": "NYSE (US)"},
    "LSE":      {"suffix": ".L",  "currency": "GBP", "label": "London Stock Exchange"},
    "TSX":      {"suffix": ".TO", "currency": "CAD", "label": "Toronto Stock Exchange"},
    "ASX":      {"suffix": ".AX", "currency": "AUD", "label": "Australian Securities Exchange"},
    "Tokyo":    {"suffix": ".T",  "currency": "JPY", "label": "Tokyo Stock Exchange"},
    "EuroNext": {"suffix": ".PA", "currency": "EUR", "label": "EuroNext (Paris)"},
}

# Currencies that can serve as the normalization base
BASE_CURRENCIES = ["USD", "EUR", "GBP", "CAD", "AUD", "JPY"]

# yfinance ticker symbols for FX rates (relative to USD)
FX_TICKERS = {
    "GBP": "GBPUSD=X",
    "EUR": "EURUSD=X",
    "CAD": "CADUSD=X",
    "AUD": "AUDUSD=X",
    "JPY": "JPYUSD=X",
    "USD": None,          # no conversion needed
}

# US 10-year Treasury yield (used in Graham formula denominator)
TREASURY_TICKER = "^TNX"

# ---------------------------------------------------------------------------
# Screening default values  (users adjust these in the sidebar)
# ---------------------------------------------------------------------------
DEFAULTS = {
    # Valuation
    "pe_max":        15.0,
    "pb_max":        1.5,
    "ev_ebitda_max": 10.0,
    "ps_max":        2.0,

    # DCF parameters
    "dcf_enabled":        True,
    "dcf_growth_rate":    8.0,    # % per year
    "dcf_discount_rate":  9.0,    # % (WACC / required return)
    "dcf_terminal_growth":3.0,    # % perpetual growth rate
    "dcf_years":         10,      # projection horizon
    "dcf_margin_safety":  30.0,   # show only if price ≤ (1-margin) × DCF value

    # Graham formula
    "graham_enabled":     True,
    "graham_growth":      7.0,    # % expected EPS growth used in formula
    "graham_no_growth_pe":8.5,    # base P/E for zero-growth company

    # Quality / health filters
    "de_max":         1.5,   # Debt-to-Equity upper bound
    "current_min":    1.0,   # Current Ratio lower bound
    "roe_min":       10.0,   # Return on Equity lower bound (%)
}

# Number of projection years for DCF (fixed choices in UI)
DCF_YEAR_OPTIONS = [5, 7, 10, 15]

# Maximum tickers to scan in one run (prevents UI timeout)
MAX_TICKERS_DEFAULT = 100

# Streamlit cache TTL (seconds) – 1 hour
CACHE_TTL = 3600
