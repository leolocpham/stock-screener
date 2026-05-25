# =============================================================================
# exchange_data.py – Ticker lists for each supported exchange
#
# Strategy:
#   1. For NASDAQ/NYSE: attempt live download from NASDAQ's screener API.
#      Falls back to a large curated list if the API is unavailable.
#   2. For other exchanges: curated lists of the most liquid stocks.
#
# All lists contain raw yfinance-compatible symbols.
# =============================================================================

from __future__ import annotations
import logging
from typing import List

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Curated fallback / default ticker lists
# ---------------------------------------------------------------------------

_NASDAQ_DEFAULT: List[str] = [
    # Mega-cap / S&P 500 NASDAQ components
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","COST",
    "NFLX","AMD","ADBE","QCOM","TXN","INTC","CSCO","INTU","AMAT","MU",
    "LRCX","MRVL","KLAC","SNPS","CDNS","PANW","CRWD","FTNT","DDOG","ZS",
    "CRM","NOW","ORCL","ANSS","IDXX","VRSK","BIIB","REGN","AMGN","GILD",
    "VRTX","MRNA","ILMN","ISRG","DXCM","PODD","ALGN","ZBRA","FAST","ODFL",
    "PCAR","PAYX","ADP","EBAY","SBUX","PEP","MDLZ","CTSH","FISV","TMUS",
    "WBA","BKNG","ABNB","MELI","PYPL","NXPI","SWKS","MCHP","XLNX","WDAY",
    "TEAM","DOCU","COUP","ZM","OKTA","SPLK","VEEV","HUBS","NET","SNOW",
    "ROKU","TWLO","SHOP","SE","JD","BIDU","PDD","NTES","MKTX","ENPH",
    "FSLR","SEDG","PLUG","RIVN","LCID","XPEV","NIO","LI","GRAB","DKNG",
]

_NYSE_DEFAULT: List[str] = [
    # Blue-chip NYSE components
    "JPM","JNJ","UNH","V","XOM","WMT","MA","PG","CVX","HD",
    "LLY","MRK","KO","PFE","ABBV","BAC","DIS","MCD","PEP","ACN",
    "TMO","ABT","DHR","VZ","COP","NEE","PM","LIN","SCHW","LOW",
    "TGT","UPS","AMT","GS","BLK","C","DE","RTX","CAT","MMM",
    "IBM","GE","F","GM","T","MO","KHC","SO","D","DUK",
    "AEP","EXC","PCG","SRE","WEC","XEL","ES","ETR","AWK","CMS",
    "NI","LNT","EVRG","PNW","OGE","POR","SR","NWE","IDA","AVA",
    "BRK-B","CB","MET","PRU","AFL","AIG","AXP","SYF","COF","DFS",
    "WFC","USB","PNC","TFC","KEY","RF","HBAN","CFG","FITB","MTB",
    "SPG","PLD","AMT","CCI","EQIX","DLR","PSA","EQR","AVB","ESS",
]

_LSE_DEFAULT: List[str] = [
    # FTSE 100 representative sample (.L suffix required)
    "HSBA.L","BP.L","SHEL.L","AZN.L","ULVR.L","RIO.L","BATS.L","GSK.L",
    "LLOY.L","VOD.L","BARC.L","GLEN.L","AAL.L","BT-A.L","EXPN.L","REL.L",
    "LGEN.L","PRU.L","HL.L","WPP.L","IMB.L","NWG.L","SMT.L","STAN.L",
    "ABF.L","SSE.L","DGE.L","PSON.L","CPG.L","TSCO.L","SBRY.L","MKS.L",
    "JD.L","SPX.L","TUI.L","CNA.L","III.L","WTB.L","MNDI.L","RR.L",
    "RKT.L","SNX.L","SKG.L","EVR.L","FRAS.L","FLTR.L","PSN.L","TW.L",
]

_TSX_DEFAULT: List[str] = [
    # TSX 60 components (.TO suffix)
    "RY.TO","TD.TO","ENB.TO","CNQ.TO","BNS.TO","BMO.TO","CP.TO","TRP.TO",
    "CNR.TO","MFC.TO","SU.TO","SHOP.TO","BCE.TO","CM.TO","T.TO","ABX.TO",
    "WPM.TO","AGI.TO","CCO.TO","POW.TO","SNC.TO","BAM.TO","BIP-UN.TO",
    "BEP-UN.TO","AEM.TO","FNV.TO","K.TO","FM.TO","ERO.TO","LUN.TO",
    "ATD.TO","GIL.TO","MG.TO","CAR-UN.TO","REI-UN.TO","SIA.TO","AC.TO",
    "WN.TO","L.TO","DOL.TO","EMP-A.TO","MRU.TO","SAP.TO","X.TO","TECK-B.TO",
]

_ASX_DEFAULT: List[str] = [
    # ASX 200 top constituents (.AX suffix)
    "CBA.AX","BHP.AX","CSL.AX","ANZ.AX","NAB.AX","WBC.AX","WES.AX",
    "MQG.AX","RIO.AX","WOW.AX","TLS.AX","S32.AX","FMG.AX","STO.AX",
    "WTC.AX","XRO.AX","REA.AX","TCL.AX","IAG.AX","MPL.AX","QBE.AX",
    "SHL.AX","RMD.AX","COH.AX","CSR.AX","AMC.AX","ORI.AX","IPL.AX",
    "NXT.AX","WDS.AX","APA.AX","SKI.AX","JHX.AX","ALX.AX","SEK.AX",
    "CPU.AX","SUN.AX","AGL.AX","ORG.AX","OSH.AX","WOR.AX","NWS.AX",
]

_TOKYO_DEFAULT: List[str] = [
    # Nikkei 225 top constituents (.T suffix)
    "7203.T","6758.T","8306.T","6861.T","9432.T","7974.T","8316.T",
    "4063.T","6954.T","4661.T","9433.T","8035.T","6902.T","4502.T",
    "7267.T","8411.T","6367.T","4519.T","9984.T","6098.T","4568.T",
    "7751.T","6501.T","7011.T","6952.T","4523.T","6762.T","8766.T",
    "9022.T","9020.T","9021.T","8801.T","8802.T","6503.T","7270.T",
    "4543.T","4151.T","6724.T","4901.T","3382.T","7201.T","7261.T",
]

_EURONEXT_DEFAULT: List[str] = [
    # Euro Stoxx 50 — Paris (.PA) and Amsterdam (.AS)
    "MC.PA","TTE.PA","BNP.PA","SAN.PA","SAF.PA","OR.PA","AIR.PA",
    "SU.PA","CS.PA","ORA.PA","VIE.PA","SGO.PA","BN.PA","DG.PA",
    "CA.PA","RI.PA","SG.PA","STM.PA","TEP.PA","ATO.PA","KER.PA",
    "PUB.PA","RNO.PA","AC.PA","FP.PA","ENGI.PA","ALO.PA","EDEN.PA",
    "ASML.AS","INGA.AS","ABN.AS","PHIA.AS","AD.AS","WKL.AS",
    "UNA.AS","HEIA.AS","MT.AS","AH.AS","RAND.AS","AKZA.AS",
    "DSM.AS","IMCD.AS","NN.AS","BESI.AS","LIGHT.AS","VPK.AS",
]

# Map exchange key → curated list
_EXCHANGE_TICKERS = {
    "NASDAQ":   _NASDAQ_DEFAULT,
    "NYSE":     _NYSE_DEFAULT,
    "LSE":      _LSE_DEFAULT,
    "TSX":      _TSX_DEFAULT,
    "ASX":      _ASX_DEFAULT,
    "Tokyo":    _TOKYO_DEFAULT,
    "EuroNext": _EURONEXT_DEFAULT,
}


# ---------------------------------------------------------------------------
# Live download (NASDAQ / NYSE via NASDAQ's public screener API)
# ---------------------------------------------------------------------------

def _download_nasdaq_tickers(market: str = "nasdaq") -> List[str]:
    """
    Download live ticker list from NASDAQ's screener API.

    Args:
        market: "nasdaq" or "nyse"

    Returns:
        List of ticker symbols, or empty list on failure.
    """
    url = (
        f"https://api.nasdaq.com/api/screener/stocks"
        f"?tableonly=true&limit=5000&offset=0&market={market}&download=true"
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.nasdaq.com/",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        rows = resp.json().get("data", {}).get("rows", []) or []
        tickers = [r["symbol"].strip() for r in rows if r.get("symbol")]
        logger.info(f"Downloaded {len(tickers)} tickers from NASDAQ API ({market})")
        return tickers
    except Exception as exc:
        logger.warning(f"NASDAQ live download failed for {market}: {exc}")
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_exchange_tickers(
    exchanges: List[str],
    use_live_download: bool = False,
    limit: int | None = None,
) -> List[str]:
    """
    Return a deduplicated list of tickers for the selected exchanges.

    Args:
        exchanges:          Exchange keys from config.EXCHANGES (e.g. ["NASDAQ","LSE"]).
        use_live_download:  If True, attempt live download for NASDAQ/NYSE first.
        limit:              Cap total tickers returned (None = no cap).

    Returns:
        List of ticker symbols compatible with yfinance.
    """
    collected: List[str] = []
    seen: set = set()

    for exch in exchanges:
        if use_live_download and exch in ("NASDAQ", "NYSE"):
            live = _download_nasdaq_tickers(market=exch.lower())
            tickers = live if live else _EXCHANGE_TICKERS.get(exch, [])
        else:
            tickers = _EXCHANGE_TICKERS.get(exch, [])

        for t in tickers:
            if t not in seen:
                seen.add(t)
                collected.append(t)

    if limit:
        collected = collected[:limit]

    return collected


def parse_custom_tickers(raw: str) -> List[str]:
    """
    Parse a user-pasted ticker string.
    Accepts comma, newline, or space-separated symbols; strips whitespace.
    """
    import re
    tokens = re.split(r"[,\s\n]+", raw.strip())
    return [t.strip().upper() for t in tokens if t.strip()]
