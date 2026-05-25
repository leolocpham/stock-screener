# =============================================================================
# app.py – Stock Screening Web Application
# Run with:  streamlit run app.py
# =============================================================================

from __future__ import annotations
import io
import json
import logging
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    EXCHANGES, BASE_CURRENCIES, DEFAULTS, DCF_YEAR_OPTIONS, MAX_TICKERS_DEFAULT
)
from exchange_data import get_exchange_tickers, parse_custom_tickers
from data_fetcher import fetch_batch_data, fetch_price_history, fetch_treasury_yield
from screening_engine import run_screen

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS_FILE = "presets.json"

BUILTIN_PRESETS = {
    "Conservative Value": {
        "pe_enabled": True,  "pe_max": 15.0,
        "pb_enabled": True,  "pb_max": 1.5,
        "ev_ebitda_enabled": True, "ev_ebitda_max": 12.0,
        "ps_enabled": True,  "ps_max": 2.0,
        "de_enabled": True,  "de_max": 0.5,
        "current_enabled": True, "current_min": 1.5,
        "roe_enabled": True, "roe_min": 12.0,
        "dcf_enabled": True, "dcf_growth_rate": 6.0,
        "dcf_discount_rate": 10.0, "dcf_terminal_growth": 2.5,
        "dcf_years": 10,     "dcf_margin_safety": 30.0, "dcf_mos_filter": True,
        "graham_enabled": True, "graham_growth": 7.0, "graham_no_growth_pe": 8.5,
    },
    "GARP": {
        "pe_enabled": True,  "pe_max": 25.0,
        "pb_enabled": True,  "pb_max": 3.0,
        "ev_ebitda_enabled": True, "ev_ebitda_max": 18.0,
        "ps_enabled": True,  "ps_max": 3.0,
        "de_enabled": True,  "de_max": 1.0,
        "current_enabled": True, "current_min": 1.2,
        "roe_enabled": True, "roe_min": 15.0,
        "dcf_enabled": True, "dcf_growth_rate": 12.0,
        "dcf_discount_rate": 9.0, "dcf_terminal_growth": 3.0,
        "dcf_years": 10,     "dcf_margin_safety": 20.0, "dcf_mos_filter": False,
        "graham_enabled": False, "graham_growth": 10.0, "graham_no_growth_pe": 8.5,
    },
    "Deep Value": {
        "pe_enabled": True,  "pe_max": 10.0,
        "pb_enabled": True,  "pb_max": 1.0,
        "ev_ebitda_enabled": True, "ev_ebitda_max": 8.0,
        "ps_enabled": True,  "ps_max": 1.0,
        "de_enabled": True,  "de_max": 0.3,
        "current_enabled": True, "current_min": 2.0,
        "roe_enabled": True, "roe_min": 10.0,
        "dcf_enabled": True, "dcf_growth_rate": 4.0,
        "dcf_discount_rate": 10.0, "dcf_terminal_growth": 2.0,
        "dcf_years": 10,     "dcf_margin_safety": 40.0, "dcf_mos_filter": True,
        "graham_enabled": True, "graham_growth": 5.0, "graham_no_growth_pe": 8.5,
    },
}

# Keys that are saved/restored by presets (widget keys only — not exchange/ticker config)
PRESET_KEYS = [
    "pe_enabled", "pe_max", "pb_enabled", "pb_max",
    "ev_ebitda_enabled", "ev_ebitda_max", "ps_enabled", "ps_max",
    "de_enabled", "de_max", "current_enabled", "current_min",
    "roe_enabled", "roe_min",
    "dcf_enabled", "dcf_growth_rate", "dcf_discount_rate",
    "dcf_terminal_growth", "dcf_years", "dcf_margin_safety", "dcf_mos_filter",
    "graham_enabled", "graham_growth", "graham_no_growth_pe",
]


def _load_presets_file() -> dict:
    if os.path.exists(PRESETS_FILE):
        try:
            with open(PRESETS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_presets_file(presets: dict) -> None:
    with open(PRESETS_FILE, "w") as f:
        json.dump(presets, f, indent=2)


def _get_all_presets() -> dict:
    return {**BUILTIN_PRESETS, **_load_presets_file()}


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Stock Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Minimal custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #1e2130; border-radius: 8px;
        padding: 12px 16px; text-align: center;
    }
    .upside-high  { color: #00c853; font-weight: 700; }
    .upside-mid   { color: #69f0ae; }
    .upside-low   { color: #ff6d00; }
    .upside-neg   { color: #f44336; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Preset panel (rendered at top of sidebar)
# ---------------------------------------------------------------------------

def _render_preset_panel() -> None:
    all_presets = _get_all_presets()
    user_presets = _load_presets_file()
    preset_names = list(all_presets.keys())

    st.sidebar.header("💾 Presets")

    # ── Load ─────────────────────────────────────────────────────────────────
    col_sel, col_load = st.sidebar.columns([3, 1])
    selected = col_sel.selectbox(
        "Load preset",
        ["— select —"] + preset_names,
        key="_preset_select",
        label_visibility="collapsed",
    )
    if col_load.button("Load", key="_preset_load", use_container_width=True):
        if selected != "— select —":
            for k, v in all_presets[selected].items():
                st.session_state[k] = v
            st.toast(f"Loaded: {selected}", icon="✅")
            st.rerun()

    # ── Save ──────────────────────────────────────────────────────────────────
    col_name, col_save = st.sidebar.columns([3, 1])
    new_name = col_name.text_input(
        "Preset name",
        placeholder="My preset name",
        key="_preset_name",
        label_visibility="collapsed",
    )
    if col_save.button("Save", key="_preset_save", use_container_width=True):
        name = (new_name or "").strip()
        if not name:
            st.sidebar.warning("Enter a name first.")
        elif name in BUILTIN_PRESETS:
            st.sidebar.error("Can't overwrite built-in presets.")
        else:
            vals = {k: st.session_state[k] for k in PRESET_KEYS if k in st.session_state}
            user_presets[name] = vals
            _save_presets_file(user_presets)
            st.toast(f'Saved preset "{name}"', icon="💾")
            st.rerun()

    # ── Delete ────────────────────────────────────────────────────────────────
    if user_presets:
        col_del, col_del_btn = st.sidebar.columns([3, 1])
        del_name = col_del.selectbox(
            "Delete preset",
            ["— select —"] + list(user_presets.keys()),
            key="_preset_del_select",
            label_visibility="collapsed",
        )
        if col_del_btn.button("Del", key="_preset_del", use_container_width=True):
            if del_name != "— select —":
                user_presets.pop(del_name, None)
                _save_presets_file(user_presets)
                st.toast(f'Deleted "{del_name}"', icon="🗑️")
                st.rerun()

    st.sidebar.divider()


# ---------------------------------------------------------------------------
# Sidebar – configuration panel
# ---------------------------------------------------------------------------

def render_sidebar() -> dict:
    """Render all sidebar controls and return the full params dict."""
    st.sidebar.title("⚙️ Screener Config")

    _render_preset_panel()

    # ── Exchange & Ticker Selection ──────────────────────────────────────────
    st.sidebar.header("🌎 Universe")

    exchange_choices = list(EXCHANGES.keys())
    selected_exchanges: List[str] = st.sidebar.multiselect(
        "Target Exchanges",
        options=exchange_choices,
        default=["NASDAQ", "NYSE"],
        help="Select one or more exchanges to scan.",
    )

    ticker_mode = st.sidebar.radio(
        "Ticker source",
        ["Exchange directory", "Custom list", "Upload CSV/TXT"],
        horizontal=True,
    )

    custom_tickers: List[str] = []
    if ticker_mode == "Custom list":
        raw = st.sidebar.text_area(
            "Enter tickers (comma or newline separated)",
            placeholder="AAPL, MSFT\nTSLA\nNVDA",
            height=120,
        )
        if raw:
            custom_tickers = parse_custom_tickers(raw)

    elif ticker_mode == "Upload CSV/TXT":
        uploaded = st.sidebar.file_uploader(
            "Upload file (one ticker per line or comma-separated)",
            type=["csv", "txt"],
        )
        if uploaded:
            content = uploaded.read().decode("utf-8")
            custom_tickers = parse_custom_tickers(content)

    use_live_download = False
    max_tickers = MAX_TICKERS_DEFAULT
    if ticker_mode == "Exchange directory":
        use_live_download = st.sidebar.toggle(
            "Live download (NASDAQ/NYSE)",
            value=False,
            help="Attempt to download the full current ticker list from NASDAQ's API.",
        )
        max_tickers = st.sidebar.slider(
            "Max tickers to scan",
            min_value=10, max_value=500,
            value=MAX_TICKERS_DEFAULT, step=10,
            help="Cap the number of stocks fetched (higher = slower).",
        )

    # ── Currency Normalisation ───────────────────────────────────────────────
    st.sidebar.header("💱 Currency")
    normalise_currency = st.sidebar.toggle(
        "Normalise to single currency", value=False
    )
    base_currency = "USD"
    if normalise_currency:
        base_currency = st.sidebar.selectbox(
            "Base currency", BASE_CURRENCIES, index=0
        )

    # ── Valuation Ratio Filters ──────────────────────────────────────────────
    st.sidebar.header("📊 Valuation Ratios")

    pe_enabled = st.sidebar.toggle("Filter by P/E Ratio", value=True, key="pe_enabled")
    pe_max = st.sidebar.slider("Max P/E", 0.0, 150.0, DEFAULTS["pe_max"], 0.5,
                                disabled=not pe_enabled, key="pe_max")

    pb_enabled = st.sidebar.toggle("Filter by P/B Ratio", value=True, key="pb_enabled")
    pb_max = st.sidebar.slider("Max P/B", 0.0, 20.0, DEFAULTS["pb_max"], 0.1,
                                disabled=not pb_enabled, key="pb_max")

    ev_enabled = st.sidebar.toggle("Filter by EV/EBITDA", value=True, key="ev_ebitda_enabled")
    ev_max = st.sidebar.slider("Max EV/EBITDA", 0.0, 60.0, DEFAULTS["ev_ebitda_max"], 0.5,
                                disabled=not ev_enabled, key="ev_ebitda_max")

    ps_enabled = st.sidebar.toggle("Filter by P/S Ratio", value=True, key="ps_enabled")
    ps_max = st.sidebar.slider("Max P/S", 0.0, 30.0, DEFAULTS["ps_max"], 0.1,
                                disabled=not ps_enabled, key="ps_max")

    # ── Intrinsic Value Models ───────────────────────────────────────────────
    st.sidebar.header("🧮 Intrinsic Value Models")

    dcf_enabled = st.sidebar.toggle("DCF Model", value=DEFAULTS["dcf_enabled"], key="dcf_enabled")
    if dcf_enabled:
        dcf_growth   = st.sidebar.slider("Expected Growth Rate (%)",
                                          0.0, 40.0, DEFAULTS["dcf_growth_rate"], 0.5,
                                          key="dcf_growth_rate")
        dcf_discount = st.sidebar.slider("Discount Rate / WACC (%)",
                                          4.0, 25.0, DEFAULTS["dcf_discount_rate"], 0.5,
                                          key="dcf_discount_rate")
        dcf_terminal = st.sidebar.slider("Terminal Growth Rate (%)",
                                          0.0, 6.0, DEFAULTS["dcf_terminal_growth"], 0.5,
                                          key="dcf_terminal_growth")
        dcf_years    = st.sidebar.select_slider("Projection Years",
                                                 options=DCF_YEAR_OPTIONS,
                                                 value=DEFAULTS["dcf_years"],
                                                 key="dcf_years")
        dcf_mos      = st.sidebar.slider("Margin of Safety (%)",
                                          0.0, 60.0, DEFAULTS["dcf_margin_safety"], 5.0,
                                          help="Show only stocks trading ≥ this % below DCF value.",
                                          key="dcf_margin_safety")
        dcf_mos_filter = st.sidebar.toggle("Enforce MoS as hard filter", value=False,
                                            key="dcf_mos_filter")
    else:
        dcf_growth = dcf_discount = dcf_terminal = dcf_mos = 0.0
        dcf_years = 10
        dcf_mos_filter = False

    graham_enabled = st.sidebar.toggle("Graham Formula", value=DEFAULTS["graham_enabled"],
                                        key="graham_enabled")
    if graham_enabled:
        graham_growth = st.sidebar.slider("Graham Expected Growth (%)",
                                           0.0, 25.0, DEFAULTS["graham_growth"], 0.5,
                                           key="graham_growth")
        graham_no_gpe = st.sidebar.slider("No-Growth Base P/E",
                                           5.0, 15.0, DEFAULTS["graham_no_growth_pe"], 0.5,
                                           key="graham_no_growth_pe")
    else:
        graham_growth = graham_no_gpe = 0.0

    # ── Financial Health Filters ─────────────────────────────────────────────
    st.sidebar.header("🛡️ Financial Health")

    de_enabled = st.sidebar.toggle("Filter by Debt/Equity", value=True, key="de_enabled")
    de_max = st.sidebar.slider("Max Debt/Equity", 0.0, 10.0, DEFAULTS["de_max"], 0.1,
                                disabled=not de_enabled, key="de_max")

    cr_enabled = st.sidebar.toggle("Filter by Current Ratio", value=True, key="current_enabled")
    cr_min = st.sidebar.slider("Min Current Ratio", 0.0, 5.0, DEFAULTS["current_min"], 0.1,
                                disabled=not cr_enabled, key="current_min")

    roe_enabled = st.sidebar.toggle("Filter by ROE", value=True, key="roe_enabled")
    roe_min = st.sidebar.slider("Min ROE (%)", 0.0, 50.0, DEFAULTS["roe_min"], 1.0,
                                 disabled=not roe_enabled, key="roe_min")

    return {
        # Universe
        "selected_exchanges":    selected_exchanges,
        "ticker_mode":           ticker_mode,
        "custom_tickers":        custom_tickers,
        "use_live_download":     use_live_download,
        "max_tickers":           max_tickers,
        "normalise_currency":    normalise_currency,
        "base_currency":         base_currency,
        # Valuation ratios
        "pe_enabled":            pe_enabled,
        "pe_max":                pe_max,
        "pb_enabled":            pb_enabled,
        "pb_max":                pb_max,
        "ev_ebitda_enabled":     ev_enabled,
        "ev_ebitda_max":         ev_max,
        "ps_enabled":            ps_enabled,
        "ps_max":                ps_max,
        # DCF
        "dcf_enabled":           dcf_enabled,
        "dcf_growth_rate":       dcf_growth,
        "dcf_discount_rate":     dcf_discount,
        "dcf_terminal_growth":   dcf_terminal,
        "dcf_years":             dcf_years,
        "dcf_margin_safety":     dcf_mos,
        "dcf_mos_filter":        dcf_mos_filter,
        "use_dcf_for_upside":    True,
        # Graham
        "graham_enabled":        graham_enabled,
        "graham_growth":         graham_growth,
        "graham_no_growth_pe":   graham_no_gpe,
        # Quality
        "de_enabled":            de_enabled,
        "de_max":                de_max,
        "current_enabled":       cr_enabled,
        "current_min":           cr_min,
        "roe_enabled":           roe_enabled,
        "roe_min":               roe_min,
    }


# ---------------------------------------------------------------------------
# Results table with colour-coded upside
# ---------------------------------------------------------------------------

def style_results(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def upside_color(val):
        if not isinstance(val, (int, float)) or np.isnan(val):
            return ""
        if val >= 40:
            return "background-color:#1b5e20; color:#e8f5e9; font-weight:bold"
        elif val >= 15:
            return "background-color:#2e7d32; color:#e8f5e9"
        elif val >= 0:
            return "background-color:#388e3c; color:#f1f8e9"
        elif val >= -15:
            return "background-color:#bf360c; color:#fbe9e7"
        else:
            return "background-color:#b71c1c; color:#ffebee; font-weight:bold"

    return (
        df.style
        .map(upside_color, subset=["Upside (%)"])
        .format({
            "Price":             "{:.2f}",
            "Intrinsic Value":   "{:.2f}",
            "Upside (%)":        "{:.1f}%",
            "DCF Value":         lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
            "Graham Value":      lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
            "P/E":               lambda x: f"{x:.1f}" if pd.notna(x) else "N/A",
            "P/B":               lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
            "EV/EBITDA":         lambda x: f"{x:.1f}" if pd.notna(x) else "N/A",
            "P/S":               lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
            "Debt/Equity":       lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
            "Current Ratio":     lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
            "ROE (%)":           lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A",
            "Market Cap (M)":    lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A",
        }, na_rep="N/A")
    )


# ---------------------------------------------------------------------------
# Stock detail chart
# ---------------------------------------------------------------------------

def render_stock_chart(ticker: str, intrinsic_value: Optional[float],
                        currency: str = "USD") -> None:
    with st.spinner(f"Loading chart for {ticker}…"):
        hist = fetch_price_history(ticker, period="1y")

    if hist.empty:
        st.warning(f"No price history available for {ticker}.")
        return

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["Close"],
        mode="lines", name="Close Price",
        line=dict(color="#4fc3f7", width=2),
        fill="tozeroy", fillcolor="rgba(79,195,247,0.08)",
    ))

    if intrinsic_value and intrinsic_value > 0:
        fig.add_hline(
            y=intrinsic_value,
            line=dict(color="#00e676", width=2, dash="dash"),
            annotation_text=f"Intrinsic Value: {intrinsic_value:.2f} {currency}",
            annotation_position="top left",
            annotation_font=dict(color="#00e676"),
        )

    fig.add_hline(y=hist["Close"].max(),
                  line=dict(color="#ffca28", width=1, dash="dot"),
                  annotation_text="52W High", annotation_position="top right",
                  annotation_font=dict(color="#ffca28", size=10))
    fig.add_hline(y=hist["Close"].min(),
                  line=dict(color="#ef9a9a", width=1, dash="dot"),
                  annotation_text="52W Low", annotation_position="bottom right",
                  annotation_font=dict(color="#ef9a9a", size=10))

    fig.update_layout(
        title=f"{ticker} – 12-Month Price History",
        xaxis_title="Date",
        yaxis_title=f"Price ({currency})",
        height=380,
        template="plotly_dark",
        margin=dict(l=40, r=20, t=50, b=40),
        showlegend=False,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.title("📈 Global Stock Screener")
    st.caption(
        "Screens NYSE, NASDAQ, LSE, TSX, ASX, Tokyo & EuroNext for undervalued stocks "
        "using DCF, Benjamin Graham, and ratio-based filters. Data via yfinance."
    )

    params = render_sidebar()

    # ── Ticker resolution ────────────────────────────────────────────────────
    if params["ticker_mode"] in ("Custom list", "Upload CSV/TXT"):
        tickers = params["custom_tickers"]
        if not tickers:
            st.info("⬅️  Enter or upload tickers in the sidebar to begin.")
            st.stop()
    else:
        if not params["selected_exchanges"]:
            st.warning("Select at least one exchange in the sidebar.")
            st.stop()
        tickers = get_exchange_tickers(
            params["selected_exchanges"],
            use_live_download=params["use_live_download"],
            limit=params["max_tickers"],
        )

    st.markdown(f"**Universe:** {len(tickers)} ticker(s) queued for analysis.")

    # ── Run button ────────────────────────────────────────────────────────────
    col_run, col_clear = st.columns([1, 6])
    run_clicked = col_run.button("🔍 Run Screener", type="primary", use_container_width=True)
    if col_clear.button("🗑️ Clear Results", use_container_width=False):
        for k in ("raw_df", "results_df"):
            st.session_state.pop(k, None)
        st.rerun()

    # ── Fetch & screen ────────────────────────────────────────────────────────
    if run_clicked:
        progress_bar = st.progress(0, text="Initialising…")
        status_text  = st.empty()

        def _progress(current, total, ticker):
            frac = current / total
            progress_bar.progress(frac, text=f"Fetching {ticker} ({current}/{total})")
            status_text.caption(f"Last fetched: **{ticker}**")

        with st.spinner("Fetching financial data…"):
            raw_df = fetch_batch_data(
                tickers,
                progress_callback=_progress,
                max_workers=8,
            )

        progress_bar.empty()
        status_text.empty()

        if raw_df.empty:
            st.error("No data returned. Check your ticker list or network connection.")
            st.stop()

        st.session_state["raw_df"] = raw_df

        bond_yield = fetch_treasury_yield()

        with st.spinner("Applying filters…"):
            results_df = run_screen(
                raw_df,
                params,
                bond_yield=bond_yield,
                base_currency=params["base_currency"],
                normalise=params["normalise_currency"],
            )

        st.session_state["results_df"] = results_df
        st.session_state["bond_yield"]  = bond_yield

    # ── Display results ───────────────────────────────────────────────────────
    results_df: Optional[pd.DataFrame] = st.session_state.get("results_df")
    raw_df:     Optional[pd.DataFrame] = st.session_state.get("raw_df")

    if results_df is None:
        st.info("Configure your filters in the sidebar and click **Run Screener** to begin.")
        return

    bond_yield = st.session_state.get("bond_yield", 4.4)
    total_scanned = len(raw_df) if raw_df is not None else 0
    total_passed  = len(results_df)
    avg_upside    = results_df["Upside (%)"].mean() if total_passed else 0
    top_ticker    = results_df["Ticker"].iloc[0] if total_passed else "—"

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Scanned",        total_scanned)
    k2.metric("Passed Filters", total_passed)
    k3.metric("Avg Upside",     f"{avg_upside:.1f}%")
    k4.metric("Top Pick",       top_ticker)
    k5.metric("10Y Treasury",   f"{bond_yield:.2f}%")

    st.divider()

    if results_df.empty:
        st.warning("No stocks passed the current filters. Try relaxing the criteria.")
        return

    # ── Interactive results table ─────────────────────────────────────────────
    st.subheader(f"📋 Screened Results ({total_passed} stocks)")

    display_cols = [
        "Ticker", "Company", "Exchange", "Currency", "Price",
        "Intrinsic Value", "Upside (%)", "P/E", "P/B", "EV/EBITDA",
        "Debt/Equity", "ROE (%)", "Model", "Market Cap (M)", "Sector",
    ]
    display_df = results_df[[c for c in display_cols if c in results_df.columns]]

    try:
        event = st.dataframe(
            style_results(display_df),
            use_container_width=True,
            height=460,
            on_select="rerun",
            selection_mode="single-row",
            key="results_table",
        )
        selected_rows = event.selection.rows if hasattr(event, "selection") else []
    except TypeError:
        st.dataframe(style_results(display_df), use_container_width=True, height=460)
        selected_rows = []

    # ── Export to CSV ─────────────────────────────────────────────────────────
    csv_buf = io.StringIO()
    results_df.to_csv(csv_buf, index=False)
    st.download_button(
        label="⬇️  Export to CSV",
        data=csv_buf.getvalue(),
        file_name="screened_stocks.csv",
        mime="text/csv",
    )

    # ── Stock detail panel ────────────────────────────────────────────────────
    selected_ticker = None
    if selected_rows:
        selected_ticker = display_df.iloc[selected_rows[0]]["Ticker"]
    else:
        ticker_options = ["(none)"] + display_df["Ticker"].tolist()
        chosen = st.selectbox("🔎 Inspect a stock", ticker_options,
                               index=0, key="manual_select")
        if chosen != "(none)":
            selected_ticker = chosen

    if selected_ticker:
        sel_row = results_df[results_df["Ticker"] == selected_ticker].iloc[0]

        with st.expander(f"📊 Detail: {selected_ticker} — {sel_row['Company']}", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current Price",   f"{sel_row['Price']:.2f} {sel_row['Currency']}")
            c2.metric("Intrinsic Value", f"{sel_row['Intrinsic Value']:.2f} {sel_row['Currency']}")
            c3.metric("Upside Potential", f"{sel_row['Upside (%)']:.1f}%")
            c4.metric("Model Used",      sel_row["Model"])

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Valuation Ratios**")
                st.markdown(
                    f"P/E: `{sel_row['P/E']}`  ·  P/B: `{sel_row['P/B']}`  ·  "
                    f"EV/EBITDA: `{sel_row['EV/EBITDA']}`  ·  P/S: "
                    f"`{sel_row.get('P/S', 'N/A')}`"
                )
            with col_b:
                st.markdown("**Financial Health**")
                st.markdown(
                    f"Debt/Equity: `{sel_row['Debt/Equity']}`  ·  "
                    f"Current Ratio: `{sel_row.get('Current Ratio', 'N/A')}`  ·  "
                    f"ROE: `{sel_row['ROE (%)']}`"
                )

            render_stock_chart(
                selected_ticker,
                sel_row["Intrinsic Value"],
                sel_row["Currency"],
            )

    # ── Sector distribution chart ─────────────────────────────────────────────
    if total_passed > 0 and "Sector" in results_df.columns:
        with st.expander("🗂️ Sector Distribution", expanded=False):
            sector_counts = (
                results_df["Sector"]
                .replace("—", "Unknown")
                .value_counts()
                .reset_index()
            )
            sector_counts.columns = ["Sector", "Count"]

            fig_sector = go.Figure(go.Bar(
                x=sector_counts["Sector"],
                y=sector_counts["Count"],
                marker_color="#4fc3f7",
            ))
            fig_sector.update_layout(
                title="Stocks Passing Filters by Sector",
                template="plotly_dark",
                height=300,
                margin=dict(l=20, r=20, t=40, b=60),
                xaxis_tickangle=-35,
            )
            st.plotly_chart(fig_sector, use_container_width=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
