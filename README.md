# Stock Screener

Production-ready stock screening app for NYSE, NASDAQ, LSE, TSX, ASX, Tokyo, and EuroNext.

## Quick Start

```bash
cd stock-screener
pip install -r requirements.txt
streamlit run app.py
```

Opens at http://localhost:8501

## File Structure

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI – entry point |
| `config.py` | Constants, defaults, exchange definitions |
| `exchange_data.py` | Curated ticker lists + live NASDAQ/NYSE download |
| `data_fetcher.py` | yfinance data fetching with caching & error handling |
| `valuation_models.py` | DCF and Benjamin Graham intrinsic value models |
| `screening_engine.py` | Filter logic and results DataFrame builder |

## Features

- **Multi-exchange:** NASDAQ, NYSE, LSE, TSX, ASX, Tokyo, EuroNext
- **Custom tickers:** Paste a list or upload a CSV/TXT file
- **DCF model:** Adjustable growth rate, discount rate, terminal growth, projection years, margin of safety
- **Graham formula:** `V = EPS × (8.5 + 2g) × 4.4 / Y`
- **Ratio filters:** P/E, P/B, EV/EBITDA, P/S with per-filter toggles
- **Quality filters:** Debt/Equity, Current Ratio, ROE
- **Currency normalisation:** Convert all prices to USD, EUR, GBP, etc.
- **Interactive table:** Colour-coded upside, sortable columns, row selection
- **Stock detail:** 12-month price chart with intrinsic value line
- **Export:** CSV download of screened results

## Notes

- yfinance data may be delayed 15–20 min for some exchanges
- International tickers require exchange suffixes: `.L` (LSE), `.TO` (TSX), `.AX` (ASX), `.T` (Tokyo), `.PA` (EuroNext Paris)
- Scanning 100+ tickers may take 1–3 minutes
