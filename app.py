"""
╔══════════════════════════════════════════════════════════════════════╗
║  Commodities Options Analysis Dashboard v3.0 (GOLD & SILVER)        ║
║  Streamlit Edition — Deploy on Streamlit Community Cloud             ║
║  Data: Dhan API (primary) | Demo Mode (fallback)                    ║
║  v3 NEW: 3-Month Term Structure · Rollover Velocity · Carry Anomaly ║
║  v3 NEW: Gamma Regime Classifier · IV Smile Classifier (12 scenes)  ║
║  v3 NEW: Vol/OI Ratio · IV Smile History · Spot Fallback Fix         ║
║  v3 NEW: Plain-English metric explanations on every card             ║
║  Auto-expiry detection | Score: 0–100 | Auto-refresh every 60s      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os, json, time, warnings, csv as _csv, io, requests
from datetime import date, timedelta, datetime, timezone

_IST = timezone(timedelta(hours=5, minutes=30))
def now_ist() -> datetime:
    return datetime.now(_IST)
def strftime_ist(fmt: str) -> str:
    return now_ist().strftime(fmt)

import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import streamlit as st
import plotly.graph_objs as go

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Commodity Options Dashboard v3",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────
#  CREDENTIALS
# ─────────────────────────────────────────────────────────────────────
class CFG:
    try:
        DHAN_CLIENT_ID    = st.secrets.get("DHAN_CLIENT_ID",    os.getenv("DHAN_CLIENT_ID",    ""))
        DHAN_ACCESS_TOKEN = st.secrets.get("DHAN_ACCESS_TOKEN", os.getenv("DHAN_ACCESS_TOKEN", ""))
        OWNER_PASSWORD    = st.secrets.get("OWNER_PASSWORD",    os.getenv("OWNER_PASSWORD",    "12345"))
    except Exception:
        DHAN_CLIENT_ID    = os.getenv("DHAN_CLIENT_ID",    "")
        DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN", "")
        OWNER_PASSWORD    = os.getenv("OWNER_PASSWORD",    "12345")
    USE_DHAN      = bool(DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN)
    USE_DEMO_MODE = not USE_DHAN

# ─────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────
RISK_FREE_RATE       = 0.065
ATM_BAND             = 20
AUTO_REFRESH_SECONDS = 60

DHAN_SECURITY = {
    "GOLD":    {"id": 114, "seg": "MCX_COMM", "step": 100},
    "GOLDM":   {"id": 117, "seg": "MCX_COMM", "step": 100},
    "SILVER":  {"id": 115, "seg": "MCX_COMM", "step": 1000},
    "SILVERM": {"id": 122, "seg": "MCX_COMM", "step": 1000},
}
COMMODITY_SYMBOLS = ["GOLD", "GOLDM", "SILVER", "SILVERM"]

# ─────────────────────────────────────────────────────────────────────
#  AUTO-DETECT MONTHLY SCRIP IDs  (near / next / far — auto-refreshed daily)
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def get_dynamic_futures_ids():
    """
    Downloads Dhan's daily master CSV and resolves near, next, AND far-month
    futures security IDs for each commodity symbol.
    Plain English: finds the contract ID numbers we need to fetch live prices.
    """
    url = "https://images.dhan.co/api-data/api-scrip-master.csv"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df.columns = [c.upper() for c in df.columns]
        df_mc = df[(df['SEM_EXM_EXCH_ID'] == 'MCX_COMM') & (df['SEM_INSTRUMENT_NAME'] == 'FUTCOM')]
        id_map = {}
        for sym in COMMODITY_SYMBOLS:
            if sym == "GOLD":
                df_sym = df_mc[df_mc['SEM_TRADING_SYMBOL'].str.match(r'^GOLD\d')]
            elif sym == "SILVER":
                df_sym = df_mc[df_mc['SEM_TRADING_SYMBOL'].str.match(r'^SILVER\d')]
            else:
                df_sym = df_mc[df_mc['SEM_TRADING_SYMBOL'].str.startswith(sym)]
            df_sym = df_sym.dropna(subset=['SEM_EXPIRY_CODE']).sort_values('SEM_EXPIRY_CODE')
            if len(df_sym) >= 3:
                id_map[sym] = [int(df_sym.iloc[i]['SEM_SMST_SECURITY_ID']) for i in range(3)]
            elif len(df_sym) == 2:
                id_map[sym] = [int(df_sym.iloc[i]['SEM_SMST_SECURITY_ID']) for i in range(2)]
            elif len(df_sym) == 1:
                id_map[sym] = [int(df_sym.iloc[0]['SEM_SMST_SECURITY_ID'])]
        return id_map
    except Exception as e:
        print(f"[Auto-Scrips] Failed: {e}")
        return {}

# ─────────────────────────────────────────────────────────────────────
#  COLOURS
# ─────────────────────────────────────────────────────────────────────
BG         = "#FFFFFF"
CARD       = "#F8FAFC"
TEXT       = "#1E293B"
ACCENT     = "#B8960C"
MUTED      = "#64748B"
GOLD       = "#B8960C"
SILVER     = "#475569"
GREEN      = "#059669"
RED        = "#DC2626"
AMBER      = "#D97706"
BLUE       = "#2563EB"
CYAN       = "#0891B2"
PINK       = "#DB2777"
BORDER     = "#E2E8F0"
SECTION_BG = "#F1F5F9"
PURPLE     = "#7C3AED"

# ─────────────────────────────────────────────────────────────────────
#  METRIC EXPLANATIONS  (plain English, one line each)
# ─────────────────────────────────────────────────────────────────────
METRIC_EXPLAIN = {
    "EV Ratio":          "Call vs put time-value premium — above 1.2 means options traders are paying more for upside.",
    "Net Delta":         "Net directional lean from all open positions — positive means market leans bullish.",
    "GEX":               "Gamma Exposure — positive pins price in range; negative amplifies any move.",
    "Vanna":             "How much delta changes when IV moves — positive means rising fear helps bulls.",
    "Momentum":          "New money flowing into calls vs puts today — positive means fresh bullish bets.",
    "Vega Skew":         "Call IV-sensitivity vs put IV-sensitivity — above 1 means calls are more reactive.",
    "G/T Ratio":         "Gamma vs Theta — high value means the market is unstable and likely to trend.",
    "PCR":               "Put-Call Ratio — above 1 means more puts than calls are open (bearish lean).",
    "Max Pain":          "Strike where option sellers lose the least — price gravitates here near expiry.",
    "ATM Pressure":      "New put OI vs new call OI near spot — positive means support is building.",
    "Skew Slope":        "Put IV steepness vs call IV steepness — high means traders fear sharp drops.",
    "IV Rank":           "Where ATM IV sits today within the smile range — above 70 = expensive options.",
    "Gamma Flip":        "Level where dealer hedging flips from stabilising to amplifying price moves.",
    "Wall Width":        "Distance between the biggest put wall (support) and call wall (resistance).",
    "Near OI %":         "Share of all open interest sitting close to current price — high means strong pin.",
    "Roll Spread":       "Price difference between near and next futures — positive = contango (normal carry).",
    "Spread %":          "Roll spread as % of near-month price — measures carry cost in percentage terms.",
    "Rollover %":        "How much OI has shifted to next month — high % means expiry rollover well advanced.",
    "Term Structure":    "Shape of the futures curve across 3 months — steepening contango signals bullish carry.",
    "Carry Anomaly":     "Actual roll cost vs what IV implies it should cost — above 1.5 means futures pricing a big move.",
    "Rollover Velocity": "Rate of OI moving from near to next month — above 1.2 means longs are adding conviction.",
    "Near Vol/OI":       "Volume-to-OI ratio for near-month futures — above 0.3 means active fresh positioning.",
    "Slope Near→Next":   "Annualised carry from near to next month — positive = contango (bullish carry).",
    "Slope Next→Far":    "Annualised carry from next to far month — steeper than near slope = bullish acceleration.",
}

# ─────────────────────────────────────────────────────────────────────
#  HISTORY PERSISTENCE
# ─────────────────────────────────────────────────────────────────────
HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "commodity_oi_history.json")
LOG_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "commodity_decision_logs")
os.makedirs(LOG_DIR, exist_ok=True)

_CSV_COLUMNS = [
    "ts", "symbol", "spot", "expiry", "atm", "atm_iv", "iv_rank",
    "gex", "gamma_flip", "pcr", "net_delta", "momentum", "atm_pressure",
    "gt_ratio", "support", "resistance", "wall_width", "max_pain",
    "call_oi_total", "put_oi_total", "score",
    "put_wing_excess", "call_wing_excess",           # NEW v3
    "roll_spread_pct", "rollover_pct", "ts_bias",    # NEW v3
]

def _get_log_paths():
    base = os.path.join(LOG_DIR, f"decision_log_{date.today().isoformat()}")
    return base + ".jsonl", base + ".csv"

def _ensure_csv_header(csv_path):
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
            w.writeheader()

def write_decision_log(record: dict):
    try:
        jsonl_path, csv_path = _get_log_paths()
        _ensure_csv_header(csv_path)
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
            w.writerow({k: record.get(k, "") for k in _CSV_COLUMNS})
    except Exception as e:
        print(f"[DecisionLog] {e}")

def _prune_to_today(history: dict) -> dict:
    today = date.today().isoformat()
    return {
        sym: [t for t in ticks if isinstance(t, dict) and str(t.get("ts", "")).startswith(today)]
        for sym, ticks in history.items() if isinstance(ticks, list)
    }

def load_history_from_disk() -> dict:
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return _prune_to_today(raw)
    except Exception as e:
        print(f"[History] Load error: {e}")
    return {}

def save_history_to_disk(history: dict):
    try:
        clean = {k: v for k, v in history.items() if isinstance(v, list)}
        tmp   = HISTORY_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(clean, f)
        os.replace(tmp, HISTORY_FILE)
    except Exception as e:
        print(f"[History] Save error: {e}")

# ─────────────────────────────────────────────────────────────────────
#  UTILITY
# ─────────────────────────────────────────────────────────────────────
def safe_num(x, d=0.0):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)): return d
        return float(x)
    except Exception: return d

def _zscore(arr, window):
    if len(arr) < 2: return 0.0
    w = arr[-window:]; mean = w.mean(); std = w.std()
    if std < 1e-9: return 0.0
    return float((w[-1] - mean) / std)

def _extract_sym_history(history, symbol):
    if isinstance(history, dict):
        ticks = history.get(symbol, [])
        if ticks: return ticks
        sym_keys = [k for k in history if isinstance(history[k], list)]
        if sym_keys: return history[max(sym_keys, key=lambda k: len(history[k]))]
        return []
    if isinstance(history, list): return history
    return []

# ─────────────────────────────────────────────────────────────────────
#  BLACK-SCHOLES
# ─────────────────────────────────────────────────────────────────────
def _bs_price(S, K, T, r, sigma, opt):
    if T <= 0 or sigma <= 0: return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt == "CE": return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def _bs_greeks(S, K, T, r, sigma, opt):
    if T <= 0 or sigma <= 0: return 0, 0, 0, 0
    d1  = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2  = d1 - sigma * np.sqrt(T); nd1 = norm.pdf(d1)
    delta = norm.cdf(d1) if opt == "CE" else -norm.cdf(-d1)
    gamma = nd1 / (S * sigma * np.sqrt(T))
    theta = (-(S * nd1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2 if opt == "CE" else -d2)) / 365
    vega  = S * nd1 * np.sqrt(T) / 100
    return delta, gamma, theta, vega

def _solve_iv(mkt_price, S, K, T, r, opt):
    if T <= 0 or mkt_price <= 0: return 0.0
    try:
        return brentq(lambda v: (_bs_price(S, K, T, r, v, opt) - mkt_price), 1e-4, 5.0, xtol=1e-5, maxiter=100)
    except Exception: return 0.0

# ─────────────────────────────────────────────────────────────────────
#  DHAN FETCHERS
# ─────────────────────────────────────────────────────────────────────
def fetch_dhan_expiry_list(symbol: str = "GOLD"):
    """Auto-detects all future expiry dates for the selected commodity."""
    sec     = DHAN_SECURITY.get(symbol, DHAN_SECURITY["GOLD"])
    headers = {"access-token": CFG.DHAN_ACCESS_TOKEN, "client-id": str(CFG.DHAN_CLIENT_ID), "Content-Type": "application/json"}
    try:
        resp     = requests.post("https://api.dhan.co/v2/optionchain/expirylist", headers=headers,
                                 json={"UnderlyingScrip": sec["id"], "UnderlyingSeg": sec["seg"]}, timeout=15)
        expiries = resp.json().get("data", [])
        today    = date.today().isoformat()
        return [e for e in expiries if e >= today]
    except Exception as e:
        print(f"[Dhan] Expiry list error: {e}")
        return []

def fetch_dhan_option_chain(symbol: str = "GOLD", expiry: str = None):
    """
    Fetches the full option chain for the given symbol and expiry.
    Plain English: downloads all strike prices with their call/put prices and open interest.
    """
    if not CFG.USE_DHAN:
        return pd.DataFrame(), 0.0, ""
    sec     = DHAN_SECURITY.get(symbol, DHAN_SECURITY["GOLD"])
    headers = {"access-token": CFG.DHAN_ACCESS_TOKEN, "client-id": str(CFG.DHAN_CLIENT_ID), "Content-Type": "application/json"}
    if expiry is None:
        try:
            exp_resp = requests.post("https://api.dhan.co/v2/optionchain/expirylist", headers=headers,
                                     json={"UnderlyingScrip": sec["id"], "UnderlyingSeg": sec["seg"]}, timeout=15)
            expiries = exp_resp.json().get("data", [])
            today    = date.today().isoformat()
            future   = [e for e in expiries if e >= today]
            expiry   = future[0] if future else (expiries[0] if expiries else "")
        except Exception as e:
            print(f"[Dhan] Expiry list error: {e}")
            return pd.DataFrame(), 0.0, ""
    if not expiry:
        return pd.DataFrame(), 0.0, ""
    try:
        oc_resp = requests.post("https://api.dhan.co/v2/optionchain", headers=headers,
                                json={"UnderlyingScrip": sec["id"], "UnderlyingSeg": sec["seg"], "Expiry": expiry}, timeout=20)
        resp    = oc_resp.json()
    except Exception as e:
        print(f"[Dhan] Option chain error: {e}")
        return pd.DataFrame(), 0.0, expiry
    if resp.get("status") != "success":
        return pd.DataFrame(), 0.0, expiry
    data = resp.get("data", {})
    spot = float(data.get("last_price", 0))
    oc   = data.get("oc", {})
    rows = []
    for strike_str, chain in oc.items():
        K  = float(strike_str)
        ce = chain.get("ce", {}) or {}; pe = chain.get("pe", {}) or {}
        cg = ce.get("greeks", {}) or {}; pg = pe.get("greeks", {}) or {}
        rows.append({
            "strike": K,
            "call_ltp":      float(ce.get("last_price", 0) or 0),
            "call_oi":       int(ce.get("oi", 0) or 0),
            "call_prev_oi":  int(ce.get("previous_oi", 0) or 0),
            "call_oi_chg":   int(ce.get("oi", 0) or 0) - int(ce.get("previous_oi", 0) or 0),
            "call_vol":      int(ce.get("volume", 0) or 0),
            "call_bid":      float(ce.get("top_bid_price", 0) or 0),
            "call_ask":      float(ce.get("top_ask_price", 0) or 0),
            "call_iv":       float(ce.get("implied_volatility", 0) or 0),
            "call_delta":    float(cg.get("delta", 0) or 0),
            "call_gamma":    float(cg.get("gamma", 0) or 0),
            "call_theta":    float(cg.get("theta", 0) or 0),
            "call_vega":     float(cg.get("vega", 0) or 0),
            "put_ltp":       float(pe.get("last_price", 0) or 0),
            "put_oi":        int(pe.get("oi", 0) or 0),
            "put_prev_oi":   int(pe.get("previous_oi", 0) or 0),
            "put_oi_chg":    int(pe.get("oi", 0) or 0) - int(pe.get("previous_oi", 0) or 0),
            "put_vol":       int(pe.get("volume", 0) or 0),
            "put_bid":       float(pe.get("top_bid_price", 0) or 0),
            "put_ask":       float(pe.get("top_ask_price", 0) or 0),
            "put_iv":        float(pe.get("implied_volatility", 0) or 0),
            "put_delta":     float(pg.get("delta", 0) or 0),
            "put_gamma":     float(pg.get("gamma", 0) or 0),
            "put_theta":     float(pg.get("theta", 0) or 0),
            "put_vega":      float(pg.get("vega", 0) or 0),
        })
    df = pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)
    return df, spot, expiry

# ─────────────────────────────────────────────────────────────────────
#  FUTURES ROLL — 3-MONTH TERM STRUCTURE  (v3 upgrade)
# ─────────────────────────────────────────────────────────────────────
def fetch_futures_roll(symbol: str = "GOLD") -> dict:
    """
    Fetches near, next, AND far-month futures data.
    Plain English: compares prices across 3 contract months to read the market's
    view on future direction (contango = expecting higher prices; backwardation = delivery pressure).
    """
    if not CFG.USE_DHAN:
        return {}
    sec     = DHAN_SECURITY.get(symbol, DHAN_SECURITY["GOLD"])
    headers = {"access-token": CFG.DHAN_ACCESS_TOKEN, "client-id": str(CFG.DHAN_CLIENT_ID), "Content-Type": "application/json"}

    try:
        exp_resp = requests.post("https://api.dhan.co/v2/optionchain/expirylist", headers=headers,
                                 json={"UnderlyingScrip": sec["id"], "UnderlyingSeg": sec["seg"]}, timeout=15)
        expiries = exp_resp.json().get("data", [])
        today    = date.today().isoformat()
        future   = [e for e in expiries if e >= today]
        if len(future) < 2:
            print(f"[Roll] Fewer than 2 expiries for {symbol}")
            return {}
        near_expiry = future[0]
        next_expiry = future[1]
        far_expiry  = future[2] if len(future) >= 3 else None
    except Exception as e:
        print(f"[Roll] Expiry list error: {e}")
        return {}

    def _fetch_chain(expiry):
        try:
            r    = requests.post("https://api.dhan.co/v2/optionchain", headers=headers,
                                 json={"UnderlyingScrip": sec["id"], "UnderlyingSeg": sec["seg"], "Expiry": expiry}, timeout=20)
            data = r.json().get("data", {})
            ltp  = float(data.get("last_price", 0) or 0)
            oc   = data.get("oc", {})
            call_oi  = sum(int((v.get("ce") or {}).get("oi", 0) or 0) for v in oc.values())
            put_oi   = sum(int((v.get("pe") or {}).get("oi", 0) or 0) for v in oc.values())
            call_vol = sum(int((v.get("ce") or {}).get("volume", 0) or 0) for v in oc.values())
            put_vol  = sum(int((v.get("pe") or {}).get("volume", 0) or 0) for v in oc.values())
            return ltp, call_oi + put_oi, call_vol + put_vol
        except Exception as e:
            print(f"[Roll] Chain fetch error for {expiry}: {e}")
            return 0.0, 0, 0

    near_ltp, near_oi, near_vol = _fetch_chain(near_expiry)
    next_ltp, next_oi, next_vol = _fetch_chain(next_expiry)
    far_ltp,  far_oi,  far_vol  = _fetch_chain(far_expiry) if far_expiry else (0.0, 0, 0)

    if near_ltp == 0:
        print(f"[Roll] near_ltp=0 for {symbol} — market may be closed")
        return {}

    total_oi        = near_oi + next_oi
    roll_spread     = round(next_ltp - near_ltp, 2)
    roll_spread_pct = round((roll_spread / near_ltp * 100) if near_ltp else 0, 3)
    rollover_pct    = round((next_oi / total_oi * 100) if total_oi else 0, 1)

    # Near→Next slope (annualised %)
    # Plain English: how much extra you pay per year to hold next month vs near month
    try:
        near_dt = datetime.strptime(near_expiry, "%Y-%m-%d").date()
        next_dt = datetime.strptime(next_expiry, "%Y-%m-%d").date()
        days_nn = max((next_dt - near_dt).days, 1)
        slope_near_next = round((next_ltp - near_ltp) / near_ltp * (365 / days_nn) * 100, 2) if near_ltp > 0 else 0
    except Exception:
        slope_near_next = roll_spread_pct

    slope_next_far = 0.0
    if far_expiry and far_ltp > 0 and next_ltp > 0:
        try:
            far_dt  = datetime.strptime(far_expiry, "%Y-%m-%d").date()
            days_nf = max((far_dt - next_dt).days, 1)
            slope_next_far = round((far_ltp - next_ltp) / next_ltp * (365 / days_nf) * 100, 2)
        except Exception:
            slope_next_far = 0.0

    # Term structure shape classifier
    # Plain English: reads the 3-month curve to judge whether the market
    # expects prices to keep rising (steepening) or is losing conviction (flattening).
    if slope_near_next > 0 and slope_next_far > 0:
        if slope_next_far >= slope_near_next * 0.9:
            ts_shape, ts_bias, ts_color = "STEEPENING CONTANGO ▲▲", 2, "#00E676"
            ts_desc = "Far month more expensive than near — strong bullish carry signal"
        else:
            ts_shape, ts_bias, ts_color = "FLATTENING CONTANGO ▲", 1, "#69F0AE"
            ts_desc = "Contango losing steam — carry still positive but momentum slowing"
    elif slope_near_next > 0 and slope_next_far <= 0:
        ts_shape, ts_bias, ts_color = "HUMP / NEAR CARRY ONLY", 0, "#FFD600"
        ts_desc = "Near month in premium but far month flat — mixed / expiry-specific carry"
    elif slope_near_next <= 0 and slope_next_far <= 0:
        if slope_near_next <= slope_next_far * 1.1:
            ts_shape, ts_bias, ts_color = "STEEP BACKWARDATION ▼▼", -2, "#FF5252"
            ts_desc = "Full curve inverted — delivery pressure / strong short-term demand squeeze"
        else:
            ts_shape, ts_bias, ts_color = "MILD BACKWARDATION ▼", -1, "#FF6D00"
            ts_desc = "Near month cheaper than next — mild delivery pressure, watch for squeeze"
    else:
        ts_shape, ts_bias, ts_color = "INVERTED HUMP", 0, "#FFD600"
        ts_desc = "Unusual shape — near in backwardation, far in contango; transitional"

    # Near-month market structure bias
    if roll_spread > 0:   bias, bias_color = "CONTANGO ▲  Bullish Carry",          "#00E676"
    elif roll_spread < 0: bias, bias_color = "BACKWARDATION ▼  Delivery Pressure", "#FF5252"
    else:                 bias, bias_color = "FLAT",                                "#FFD600"

    # Vol/OI ratios — Plain English: how much activity per open contract (high = more speculation)
    near_vol_oi = round(near_vol / near_oi, 3) if near_oi > 0 else 0.0
    next_vol_oi = round(next_vol / next_oi, 3) if next_oi > 0 else 0.0
    far_vol_oi  = round(far_vol  / far_oi,  3) if far_oi  > 0 else 0.0

    return {
        "near_ltp": near_ltp, "next_ltp": next_ltp, "far_ltp": far_ltp,
        "near_oi":  near_oi,  "next_oi":  next_oi,  "far_oi":  far_oi,
        "near_vol": near_vol, "next_vol": next_vol,  "far_vol": far_vol,
        "near_vol_oi": near_vol_oi, "next_vol_oi": next_vol_oi, "far_vol_oi": far_vol_oi,
        "roll_spread": roll_spread, "roll_spread_pct": roll_spread_pct,
        "rollover_pct": rollover_pct,
        "bias": bias, "bias_color": bias_color,
        "slope_near_next": slope_near_next, "slope_next_far": slope_next_far,
        "ts_shape": ts_shape, "ts_bias": ts_bias, "ts_color": ts_color, "ts_desc": ts_desc,
        "near_expiry": near_expiry, "next_expiry": next_expiry, "far_expiry": far_expiry or "",
        "has_far": bool(far_expiry and far_ltp > 0),
    }

# ─────────────────────────────────────────────────────────────────────
#  DEMO MODE
# ─────────────────────────────────────────────────────────────────────
def fetch_demo_option_chain(symbol: str = "GOLD"):
    np.random.seed(int(time.time()) // 60)
    if "GOLD" in symbol:
        spot = 93500.0 + np.random.normal(0, 150)
        step = DHAN_SECURITY[symbol]["step"]
    else:
        spot = 96500.0 + np.random.normal(0, 300)
        step = DHAN_SECURITY[symbol]["step"]
    atm     = round(spot / step) * step
    strikes = np.arange(atm - 25 * step, atm + 26 * step, step)
    T = 10 / 365.0; r = RISK_FREE_RATE
    vix = 18.5 + np.random.normal(0, 1)
    rows = []
    for K in strikes:
        mono  = (K - spot) / spot
        iv_c  = max(0.05, (vix/100) + 0.015*mono**2 + abs(mono)*0.04 + np.random.normal(0, 0.005))
        iv_p  = max(0.05, (vix/100) + 0.025*mono**2 - mono*0.03     + np.random.normal(0, 0.005))
        cp    = _bs_price(spot, K, T, r, iv_c, "CE"); pp = _bs_price(spot, K, T, r, iv_p, "PE")
        cd,cg,ct,cv  = _bs_greeks(spot, K, T, r, iv_c, "CE")
        pd2,pg,pt,pv = _bs_greeks(spot, K, T, r, iv_p, "PE")
        cof = max(0, 2+mono*8)*np.random.lognormal(0,0.4)
        pof = max(0, 2-mono*9)*np.random.lognormal(0,0.4)
        rows.append({
            "strike": K,
            "call_ltp": round(max(0.05,cp+np.random.normal(0,0.3)),2),
            "call_oi": int(max(10,cof*800)), "call_oi_chg": int(np.random.normal(20,150)),
            "call_vol": int(abs(np.random.normal(300,150))),
            "call_bid": round(max(0.05,cp-2.0),2), "call_ask": round(cp+2.0,2),
            "call_iv": round(iv_c*100,2), "call_delta": round(cd,4),
            "call_gamma": round(cg,6), "call_theta": round(ct,4), "call_vega": round(cv,4),
            "put_ltp": round(max(0.05,pp+np.random.normal(0,0.3)),2),
            "put_oi": int(max(10,pof*1000)), "put_oi_chg": int(np.random.normal(-10,180)),
            "put_vol": int(abs(np.random.normal(350,200))),
            "put_bid": round(max(0.05,pp-2.0),2), "put_ask": round(pp+2.0,2),
            "put_iv": round(iv_p*100,2), "put_delta": round(pd2,4),
            "put_gamma": round(pg,6), "put_theta": round(pt,4), "put_vega": round(pv,4),
        })
    expiry = (date.today() + timedelta(days=10)).strftime("%Y-%m-%d")
    return pd.DataFrame(rows), round(spot, 2), expiry

def demo_futures_roll(symbol: str = "GOLD") -> dict:
    """Generates realistic demo futures roll data for all 3 months."""
    near_ltp = (93500.0 if "GOLD" in symbol else 96500.0) + np.random.normal(0, 80)
    spread1  = abs(np.random.normal(120, 40))
    spread2  = abs(np.random.normal(110, 35))
    next_ltp = near_ltp + spread1
    far_ltp  = next_ltp + spread2
    near_oi  = int(np.random.normal(18000, 2000))
    next_oi  = int(np.random.normal(4500,  800))
    far_oi   = int(np.random.normal(800,   200))
    total_oi = near_oi + next_oi
    near_vol = int(np.random.normal(5000, 500))
    next_vol = int(np.random.normal(900,  200))
    far_vol  = int(np.random.normal(180,   60))
    rsp      = round(next_ltp - near_ltp, 2)
    rsp_pct  = round(rsp / near_ltp * 100, 3)
    slope_nn = round(rsp / near_ltp * (365/30) * 100, 2)
    slope_nf = round((far_ltp - next_ltp) / next_ltp * (365/30) * 100, 2)
    near_expiry = (date.today() + timedelta(days=10)).strftime("%Y-%m-%d")
    next_expiry = (date.today() + timedelta(days=40)).strftime("%Y-%m-%d")
    far_expiry  = (date.today() + timedelta(days=70)).strftime("%Y-%m-%d")
    return {
        "near_ltp": round(near_ltp,2), "next_ltp": round(next_ltp,2), "far_ltp": round(far_ltp,2),
        "near_oi": near_oi, "next_oi": next_oi, "far_oi": far_oi,
        "near_vol": near_vol, "next_vol": next_vol, "far_vol": far_vol,
        "near_vol_oi": round(near_vol/near_oi,3) if near_oi>0 else 0,
        "next_vol_oi": round(next_vol/next_oi,3) if next_oi>0 else 0,
        "far_vol_oi":  round(far_vol/far_oi,3)  if far_oi>0  else 0,
        "roll_spread": rsp, "roll_spread_pct": rsp_pct,
        "rollover_pct": round(next_oi/total_oi*100,1),
        "bias": "CONTANGO ▲  Bullish Carry", "bias_color": "#00E676",
        "slope_near_next": slope_nn, "slope_next_far": slope_nf,
        "ts_shape": "STEEPENING CONTANGO ▲▲", "ts_bias": 2, "ts_color": "#00E676",
        "ts_desc": "Far month more expensive than near — strong bullish carry signal",
        "near_expiry": near_expiry, "next_expiry": next_expiry, "far_expiry": far_expiry,
        "has_far": True,
    }

def get_option_chain(symbol: str = "GOLD", expiry: str = None):
    if CFG.USE_DHAN:
        df, spot, exp = fetch_dhan_option_chain(symbol, expiry)
        if not df.empty:
            return df, spot, exp, "Dhan API (MCX)"
    df, spot, exp = fetch_demo_option_chain(symbol)
    return df, spot, exp, "DEMO MODE (Commodities)"

# ─────────────────────────────────────────────────────────────────────
#  WING EXCESS HELPER  (feeds IV Smile Classifier history)
# ─────────────────────────────────────────────────────────────────────
def compute_wing_excess(df_band, atm, atm_iv, symbol="GOLD"):
    """
    Measures how much the OTM put and call wings deviate above the ATM IV.
    Plain English: tells you if traders are paying extra to hedge against
    sharp drops (put skew) or sharp rises (call skew).
    """
    step = DHAN_SECURITY.get(symbol, DHAN_SECURITY["GOLD"])["step"]
    otm_put_iv = df_band.loc[
        df_band["strike"].between(atm - 6*step, atm - 2*step) & (df_band["put_iv"] > 0.5), "put_iv"
    ]
    otm_call_iv = df_band.loc[
        df_band["strike"].between(atm + 2*step, atm + 6*step) & (df_band["call_iv"] > 0.5), "call_iv"
    ]
    if len(otm_put_iv) < 2 or len(otm_call_iv) < 2:
        return None, None
    put_wing_excess  = round(float(otm_put_iv.mean())  - atm_iv, 2)
    call_wing_excess = round(float(otm_call_iv.mean()) - atm_iv, 2)
    return put_wing_excess, call_wing_excess

# ─────────────────────────────────────────────────────────────────────
#  GAMMA REGIME CLASSIFIER  (ported + adapted from Nifty app)
# ─────────────────────────────────────────────────────────────────────
def classify_gamma_regime(gex, wall_width, momentum, atm_iv, iv_rank, spot, gamma_flip, step):
    """
    Classifies the current market structure into 5 gamma regimes.
    Plain English: tells you whether the market is pinned in a range,
    trending, about to break out, or in an unstable flip zone.
    """
    flip_dist = abs(spot - gamma_flip) if gamma_flip is not None else 9999
    near_flip = flip_dist < max(3.0 * step, step * 3) if gamma_flip is not None else False
    if iv_rank >= 70:   vol_regime = "HIGH_VOL"
    elif iv_rank <= 30: vol_regime = "LOW_VOL"
    else:               vol_regime = "MID_VOL"
    narrow_wall  = wall_width < 10 * step   # tight range
    moderate_wall = wall_width < 20 * step  # moderate range
    if gex > 0 and narrow_wall and vol_regime == "LOW_VOL":
        regime, desc = "PINNED / RANGE", "Dealers long gamma — they'll sell rallies and buy dips, pinning price near ATM"
        color = GREEN
    elif gex > 0 and moderate_wall:
        regime, desc = "RANGE / PIN", "Positive GEX supports a range — directional moves get dampened by dealer hedging"
        color = CYAN
    elif gex < 0 and abs(momentum) > 300 and vol_regime in ("MID_VOL", "HIGH_VOL"):
        regime, desc = "TREND / EXPANSION", "Negative GEX + fresh momentum — dealers amplify moves, trend is self-reinforcing"
        color = AMBER
    elif near_flip:
        regime, desc = "FLIP ZONE / UNSTABLE", "Spot is near the gamma flip level — small moves can trigger large dealer rehedging"
        color = RED
    else:
        regime, desc = "TRANSITION", "Mixed signals — neither buyers nor sellers clearly in control, wait for clarity"
        color = MUTED
    return regime, desc, vol_regime, color

# ─────────────────────────────────────────────────────────────────────
#  IV SMILE CLASSIFIER  — 12 scenarios  (ported + adapted for commodities)
# ─────────────────────────────────────────────────────────────────────
def classify_iv_smile_scenario(df_band, m, spot, symbol="GOLD", iv_smile_history=None):
    """
    Classifies the live IV smile into one of 12 market scenarios.
    Plain English: reads the shape of implied volatility across strikes
    to identify what the options market is pricing in (fear, euphoria, breakout, etc.).
    Uses intraday history to detect trends in the smile shape for better accuracy.
    """
    if df_band is None or (hasattr(df_band, 'empty') and df_band.empty):
        return None
    df   = df_band.copy()
    step = DHAN_SECURITY.get(symbol, DHAN_SECURITY["GOLD"])["step"]
    atm     = safe_num(m.get("atm", spot))
    atm_iv  = safe_num(m.get("atm_iv", 0))
    iv_rank = safe_num(m.get("iv_rank", 50))
    if atm_iv <= 0:
        return None

    otm_put_iv = df.loc[
        df["strike"].between(atm - 6*step, atm - 2*step) & (df["put_iv"] > 0.5), "put_iv"
    ]
    otm_call_iv = df.loc[
        df["strike"].between(atm + 2*step, atm + 6*step) & (df["call_iv"] > 0.5), "call_iv"
    ]
    if len(otm_put_iv) < 2 or len(otm_call_iv) < 2:
        return None

    put_wing_excess  = float(otm_put_iv.mean())  - atm_iv
    call_wing_excess = float(otm_call_iv.mean()) - atm_iv
    skew_asymmetry   = put_wing_excess - call_wing_excess  # +ve = put skew, -ve = call skew

    # Intraday trend info
    trend_info = {"has_trend": False}
    if iv_smile_history and len(iv_smile_history) >= 3:
        lookback = min(len(iv_smile_history) - 1, 8)
        ref = iv_smile_history[-(lookback + 1)]
        d_atm  = atm_iv - safe_num(ref.get("atm_iv", atm_iv))
        d_put  = put_wing_excess  - safe_num(ref.get("put_wing_excess",  put_wing_excess))
        d_call = call_wing_excess - safe_num(ref.get("call_wing_excess", call_wing_excess))
        peak_atm = max(r.get("atm_iv", 0) for r in iv_smile_history)
        trend_info = {
            "has_trend":   True,
            "d_atm_iv":    round(d_atm, 2),
            "d_put_wing":  round(d_put, 2),
            "d_call_wing": round(d_call, 2),
            "peak_atm_iv": round(peak_atm, 2),
            "ticks":       len(iv_smile_history),
        }

    def _sc(sid, name, badge, bc, signals, strategies, confidence, description):
        return {
            "scenario_id": sid, "scenario_name": name,
            "badge": badge, "badge_color": bc,
            "atm_iv": round(atm_iv, 2),
            "put_wing_excess":  round(put_wing_excess,  2),
            "call_wing_excess": round(call_wing_excess, 2),
            "skew_asymmetry":   round(skew_asymmetry,   2),
            "iv_rank": round(iv_rank, 1),
            "signals": signals, "strategies": strategies,
            "confidence": confidence, "description": description,
            "trend": trend_info,
        }

    # Scenario 12: Inverted Smile (data artifact)
    if put_wing_excess < -2.0 and call_wing_excess < -2.0:
        return _sc(12, "Inverted Smile", "DATA ANOMALY", AMBER,
            ["ATM IV exceeds OTM wings on both sides — extremely rare in commodities",
             "Likely illiquid strikes or bid-ask spread distortion",
             "Do not trade options signals until verified"],
            ["VERIFY DATA", "CHECK LIQUIDITY"], 30,
            "ATM IV higher than both wings. Verify data before acting.")

    # Scenario 8: Compressed IV / Coiled Spring
    if iv_rank <= 20 and abs(put_wing_excess) < 2.0 and abs(call_wing_excess) < 2.0:
        if trend_info.get("has_trend") and trend_info.get("peak_atm_iv", 0) > atm_iv * 1.5:
            return _sc(11, "Post-Event IV Crush", "VOL COLLAPSE", MUTED,
                [f"ATM IV crushed from session peak {trend_info['peak_atm_iv']:.1f}% to {atm_iv:.1f}%",
                 "Entire surface deflated — the event has resolved",
                 "Options buyers lose even if direction was right (vega loss dominates)",
                 "Classic post-news / expiry IV deflation pattern"],
                ["SELL STRADDLES", "IRON CONDOR", "COLLECT THETA"], 90,
                f"Post-event IV crush confirmed. Peak {trend_info['peak_atm_iv']:.1f}% to now {atm_iv:.1f}%. Sell premium.")
        return _sc(8, "Compressed IV / Coiled Spring", "BREAKOUT ALERT", AMBER,
            [f"ATM IV at session low (rank {iv_rank:.0f}%ile) — market deeply compressed",
             "Both wings flat — full smile compression",
             "Historically this precedes sharp moves in either direction",
             "For Gold/Silver: often triggered by upcoming Fed/macro event"],
            ["BUY STRADDLE", "BUY STRANGLE", "AVOID SELLING VOL"], 85,
            f"IV rank {iv_rank:.0f}%ile with flat wings. Classic coiled spring — buy vol, manage theta carefully.")

    # Scenario 11: Post-Event IV Crush (snapshot only)
    if iv_rank <= 15 and put_wing_excess < 2.0 and call_wing_excess < 2.0:
        return _sc(11, "Post-Event IV Crush", "VOL COLLAPSE", MUTED,
            ["Entire surface deflated — major event has resolved",
             "Premium across all strikes compressed",
             "Best time to sell options for the next cycle"],
            ["SELL STRADDLES", "IRON CONDOR", "COLLECT THETA"], 85,
            f"Post-event IV crush (rank {iv_rank:.0f}%ile). Entire smile deflated. Sell premium going forward.")

    # Scenario 2: Crash Fear / Panic (steep put skew + high IV)
    if put_wing_excess > 5.0 and skew_asymmetry > 4.0 and iv_rank >= 65:
        return _sc(2, "Crash Fear / Panic", "EXTREME FEAR", RED,
            [f"Put wing excess: {put_wing_excess:.1f}pp above ATM — extreme downside hedging",
             f"IV rank {iv_rank:.0f}%ile — options extremely expensive across the board",
             "Institutional panic buying of OTM puts — hedging against sharp drop",
             "For Gold: typically occurs during USD surge or risk-off macro events"],
            ["BUY CALL SPREADS (fade panic)", "SELL PUT SPREADS (collect fear premium)",
             "AVOID LONG PUTS (too expensive)"], 82,
            f"Extreme put skew ({put_wing_excess:.1f}pp) + high IV ({iv_rank:.0f}%ile). Classic panic hedging — mean reversion trade opportunity.")

    # Scenario 3: Mild Put Skew (standard downside hedging)
    if put_wing_excess > 2.5 and skew_asymmetry > 1.5 and put_wing_excess > call_wing_excess:
        return _sc(3, "Mild Put Skew", "DEFENSIVE", AMBER,
            [f"Put wing excess: {put_wing_excess:.1f}pp — steady downside hedging active",
             "Participants paying a modest premium to protect long positions",
             "Normal for Gold during macro uncertainty periods",
             "Not panic — structured / portfolio hedging"],
            ["BULL PUT SPREAD", "SELL OTM PUTS (collect skew premium)",
             "CALL SPREAD (if bullish underlying)"], 75,
            f"Moderate put skew ({put_wing_excess:.1f}pp). Orderly downside hedging — not panic. Favour selling OTM puts.")

    # Scenario 5: Symmetric Wide Smile (high IV both sides — pre-event)
    if put_wing_excess > 3.0 and call_wing_excess > 3.0 and abs(skew_asymmetry) < 2.0:
        label = "Pre-Event / High Uncertainty" if iv_rank >= 55 else "Wide Symmetric Smile"
        return _sc(5, label, "HIGH UNCERTAINTY", PINK,
            [f"Both wings elevated: put +{put_wing_excess:.1f}pp, call +{call_wing_excess:.1f}pp",
             "No directional bias — market uncertain about the size AND direction of move",
             "IV rank {:.0f}%ile — premiums expensive on all strikes".format(iv_rank),
             "For Gold/Silver: typical before FOMC, US CPI, or geopolitical event"],
            ["SELL IRON CONDOR (sell both wings)", "AVOID LONG STRADDLE (too expensive)",
             "CALENDAR SPREAD (sell front, buy back)"], 78,
            f"Symmetric wide smile — market pricing a big move but unknown direction. Sell vol, not direction.")

    # Scenario 6: Call Skew / Upside Speculation
    if call_wing_excess > 2.5 and skew_asymmetry < -1.5 and call_wing_excess > put_wing_excess:
        return _sc(6, "Call Skew / Upside Speculation", "BULLISH PREMIUM", GREEN,
            [f"Call wing excess: {call_wing_excess:.1f}pp — traders paying up for upside calls",
             "Rare in commodities — signals strong speculative buying of OTM calls",
             "Often seen in Gold/Silver during supply shock or safe-haven breakout",
             "Smart money may be positioning for a breakout above resistance"],
            ["BUY ATM CALLS", "BULL CALL SPREAD", "SELL OTM PUTS (fade the skew)"], 77,
            f"Call skew active ({call_wing_excess:.1f}pp above ATM). Bullish positioning in options — calls in demand.")

    # Scenario 7: Strong Call Skew (aggressive upside)
    if call_wing_excess > 5.0 and skew_asymmetry < -3.0 and iv_rank >= 55:
        return _sc(7, "Strong Call Skew / Breakout Bet", "AGGRESSIVE BULLS", GREEN,
            [f"Extreme call wing excess: +{call_wing_excess:.1f}pp — aggressive upside positioning",
             f"IV rank {iv_rank:.0f}%ile with call-heavy skew — institutional breakout positioning",
             "Gold/Silver: potential run toward next major resistance level",
             "Follow the flow — but manage size as momentum plays can reverse fast"],
            ["BUY CALL SPREADS", "LONG FUTURES + BUY PUT PROTECTION",
             "AVOID SHORT CALLS"], 73,
            f"Very strong call skew (+{call_wing_excess:.1f}pp). Aggressive bullish bets active. Follow but hedge.")

    # Scenario 9: Smirk — structural put buyer dominance at low IV
    if put_wing_excess > 1.5 and call_wing_excess < 0.5 and iv_rank <= 40:
        return _sc(9, "Smirk / Structural Put Buyer", "CAUTIOUS", AMBER,
            [f"Put wing elevated ({put_wing_excess:.1f}pp) while calls flat — one-sided hedging",
             "Low-IV environment with persistent put demand — structural downside protection",
             "Longs hedging positions without paying up for calls",
             "Common in Gold during slow consolidation phases"],
            ["SELL PUT SPREADS (collect smirk premium)",
             "BULL PUT SPREAD (define risk / collect premium)",
             "BUY CALLS CHEAP (calls are at discount)"], 72,
            f"Put smirk at low IV — structural hedging, not panic. Sell OTM puts or buy cheap calls.")

    # Scenario 4: Normal / Balanced Smile
    if abs(skew_asymmetry) < 1.5 and put_wing_excess > 0 and call_wing_excess > 0:
        if iv_rank >= 45:
            return _sc(4, "Normal Smile — Elevated IV", "RANGE / NEUTRAL", CYAN,
                [f"Balanced smile with IV rank {iv_rank:.0f}%ile — no strong directional bias",
                 "Both wings present but neither dominant — market neutral positioning",
                 f"Skew asymmetry {skew_asymmetry:.2f} — very close to balanced",
                 "Wait for a clearer signal before taking directional options risk"],
                ["IRON CONDOR", "SHORT STRADDLE (sell ATM premium)",
                 "WAIT FOR SKEW TO DEVELOP"], 70,
                f"Normal balanced smile at elevated IV. No directional edge from smile. Sell premium in range.")
        else:
            return _sc(4, "Normal Smile — Low IV", "QUIET MARKET", MUTED,
                [f"Balanced, compressed smile with IV rank {iv_rank:.0f}%ile",
                 "Market in no-man's land — low vol, low conviction on direction",
                 "Options cheap but no catalyst visible — avoid straddles",
                 "For Gold/Silver: typical between major macro events"],
                ["WAIT FOR CATALYST", "AVOID LARGE OPTIONS POSITIONS",
                 "CONSIDER SMALL CALL SPREAD (cheap)"], 60,
                f"Normal low-IV balanced smile. Market quiet. Wait for a catalyst before positioning.")

    # Fallback: Transitional / Undefined
    return _sc(0, "Transitional / Evolving", "MONITORING", MUTED,
        ["Smile shape doesn't fit a clean scenario — likely in transition",
         f"Put wing: {put_wing_excess:.1f}pp, Call wing: {call_wing_excess:.1f}pp, IV rank: {iv_rank:.0f}%ile",
         "Continue monitoring — a clearer pattern typically emerges within 1-2 refreshes"],
        ["WAIT FOR CLARITY", "NO STRONG EDGE YET"], 45,
        "Smile in transition. Monitor for the next 1-2 refresh cycles.")

# ─────────────────────────────────────────────────────────────────────
#  CARRY ANOMALY  (futures roll cost vs what IV implies)
# ─────────────────────────────────────────────────────────────────────
def compute_carry_anomaly(roll: dict, atm_iv: float) -> float:
    """
    Compares the actual futures roll cost against what ATM implied volatility
    predicts it should cost.
    Plain English: above 1.5 means futures are pricing in a bigger move than
    the options market expects — a divergence signal worth watching.
    """
    near_ltp    = roll.get("near_ltp", 0)
    roll_spread = abs(roll.get("roll_spread", 0))
    if near_ltp <= 0 or atm_iv <= 0:
        return 1.0
    expected_weekly = (atm_iv / 100) * near_ltp / np.sqrt(52)
    if expected_weekly <= 0:
        return 1.0
    return round(roll_spread / expected_weekly, 2)

# ─────────────────────────────────────────────────────────────────────
#  ROLLOVER VELOCITY  (rate of OI moving from near to next month)
# ─────────────────────────────────────────────────────────────────────
def compute_rollover_velocity_zscore(oi_history: dict, symbol: str):
    """
    Measures how fast open interest is rolling from near to next month,
    standardised against today's session average.
    Plain English: above +1.5σ means longs are aggressively adding — bullish;
    below -1.5σ means positions are being liquidated — bearish.
    """
    hist = oi_history.get(symbol, [])
    if len(hist) < 3:
        return 0.0, "Collecting rollover data…", MUTED
    velocities = [h.get("rollover_velocity", 0.0) for h in hist if "rollover_velocity" in h]
    if len(velocities) < 3:
        return 0.0, "Collecting rollover data…", MUTED
    arr  = np.array(velocities[-20:], dtype=float)
    mean = arr.mean(); std = arr.std() if arr.std() > 1e-9 else 1.0
    z    = round((velocities[-1] - mean) / std, 2)
    last_v = velocities[-1]
    if last_v >= 1.3:
        interp, color = "CONVICTION ROLL ↑ — Longs rolling with conviction (bullish)", GREEN
    elif last_v >= 0.8:
        interp, color = "NORMAL ROLL — Typical pre-expiry rollover", CYAN
    elif last_v >= 0.3:
        interp, color = "SLOW ROLL — Partial rollover, some position closure", AMBER
    else:
        interp, color = "LIQUIDATION ↓ — OI leaving near, not entering next (bearish)", RED
    return z, interp, color

# ─────────────────────────────────────────────────────────────────────
#  METRICS ENGINE
# ─────────────────────────────────────────────────────────────────────
def select_atm_band(df, spot, symbol="GOLD"):
    step    = DHAN_SECURITY.get(symbol, DHAN_SECURITY["GOLD"])["step"]
    strikes = sorted(df["strike"].unique())
    atm     = min(strikes, key=lambda x: abs(x - spot))
    lo, hi  = atm - ATM_BAND * step, atm + ATM_BAND * step
    return df[df["strike"].between(lo, hi)].copy(), atm

def compute_max_pain(df):
    results = {}
    for K in df["strike"]:
        cl = (df[df["strike"] < K]["call_oi"] * (K - df[df["strike"] < K]["strike"])).sum()
        pl = (df[df["strike"] > K]["put_oi"]  * (df[df["strike"] > K]["strike"] - K)).sum()
        results[K] = cl + pl
    return min(results, key=results.get) if results else 0.0

def _fill_missing_greeks(df_band, spot, expiry=None):
    """Backfills greeks from Black-Scholes when Dhan returns zeros (common for MCX)."""
    if df_band.empty: return df_band
    if df_band["call_gamma"].abs().max() > 1e-9 and df_band["put_gamma"].abs().max() > 1e-9:
        return df_band
    try:
        T = max((datetime.strptime(expiry[:10], "%Y-%m-%d").date() - date.today()).days, 1) / 365.0 if expiry else 10/365.0
    except Exception:
        T = 10 / 365.0
    df2 = df_band.copy()
    for idx, row in df2.iterrows():
        K    = float(row["strike"])
        iv_c = float(row.get("call_iv", 0) or 0) / 100.0
        iv_p = float(row.get("put_iv",  0) or 0) / 100.0
        iv_c = iv_c if iv_c > 0.01 else 0.15
        iv_p = iv_p if iv_p > 0.01 else 0.15
        _, cg, _, _ = _bs_greeks(spot, K, T, RISK_FREE_RATE, iv_c, "CE")
        _, pg, _, _ = _bs_greeks(spot, K, T, RISK_FREE_RATE, iv_p, "PE")
        df2.at[idx, "call_gamma"] = cg
        df2.at[idx, "put_gamma"]  = pg
    return df2

def compute_gamma_flip(df_band, spot):
    if df_band.empty: return None
    strikes = sorted(df_band["strike"].unique()); cum_gex = 0.0
    for K in strikes:
        row = df_band[df_band["strike"] == K]
        if row.empty: continue
        gex_k    = (float(row["call_gamma"].values[0]) * float(row["call_oi"].values[0]) -
                    float(row["put_gamma"].values[0])  * float(row["put_oi"].values[0])) * spot**2 * 0.01
        prev_gex = cum_gex; cum_gex += gex_k
        if prev_gex > 0 and cum_gex <= 0: return K
        if prev_gex < 0 and cum_gex >= 0: return K
    return None

def compute_iv_rank(df_band):
    all_ivs = pd.concat([df_band["call_iv"], df_band["put_iv"]]).dropna()
    all_ivs = all_ivs[all_ivs > 0]
    if all_ivs.empty: return 50.0
    iv_min = all_ivs.min(); iv_max = all_ivs.max(); atm_iv = all_ivs.median()
    if iv_max == iv_min: return 50.0
    return round((atm_iv - iv_min) / (iv_max - iv_min) * 100, 1)

def compute_metrics(df, spot, symbol="GOLD", expiry=None, roll=None):
    """
    Master metrics function. Also uses futures near_ltp as spot fallback
    when the option chain returns zero (common for MCX commodities).
    Plain English: computes all the numbers that drive the dashboard signals.
    """
    if df.empty: return {}

    # ── Spot fallback: use near-month futures LTP if option chain spot is zero ──
    # Plain English: when the API doesn't return a cash spot, we use the nearest
    # futures contract price as a substitute — standard practice for MCX commodities.
    if spot == 0 and roll and roll.get("near_ltp", 0) > 0:
        spot = roll["near_ltp"]

    if spot == 0: return {}

    df_band, atm = select_atm_band(df, spot, symbol)
    df_band      = _fill_missing_greeks(df_band, spot, expiry)
    step         = DHAN_SECURITY.get(symbol, DHAN_SECURITY["GOLD"])["step"]

    df_band["intr_c"] = np.maximum(0, spot - df_band["strike"])
    df_band["ev_c"]   = np.maximum(0, df_band["call_ltp"] - df_band["intr_c"])
    df_band["intr_p"] = np.maximum(0, df_band["strike"] - spot)
    df_band["ev_p"]   = np.maximum(0, df_band["put_ltp"] - df_band["intr_p"])

    ev_sum_c = df_band["ev_c"].sum(); ev_sum_p = df_band["ev_p"].sum()
    ev_ratio = ev_sum_c / ev_sum_p if ev_sum_p > 0 else 1.0

    net_delta = ((df_band["call_oi"] * df_band["call_delta"]).sum() +
                 (df_band["put_oi"]  * df_band["put_delta"]).sum())
    net_gamma = ((df_band["call_oi"] * df_band["call_gamma"]).sum() +
                 (df_band["put_oi"]  * df_band["put_gamma"]).sum())
    net_theta = ((df_band["call_oi"] * df_band["call_theta"]).sum() +
                 (df_band["put_oi"]  * df_band["put_theta"]).sum())
    gex       = ((df_band["call_oi"] * df_band["call_gamma"]).sum() -
                 (df_band["put_oi"]  * df_band["put_gamma"]).sum()) * spot**2 * 0.01
    vanna     = ((df_band["call_oi"] * df_band["call_vega"] * df_band["call_delta"]).sum() +
                 (df_band["put_oi"]  * df_band["put_vega"]  * df_band["put_delta"]).sum()) / max(spot, 1)
    gt_ratio  = abs(net_gamma) / max(abs(net_theta), 1e-6)
    momentum  = ((df_band["call_oi_chg"] * df_band["call_delta"]).sum() +
                 (df_band["put_oi_chg"]  * df_band["put_delta"]).sum())
    sum_vega_c = (df_band["call_oi"] * df_band["call_vega"]).sum()
    sum_vega_p = (df_band["put_oi"]  * df_band["put_vega"]).sum()
    vega_skew  = sum_vega_c / sum_vega_p if sum_vega_p > 0 else 1.0

    total_coi = df["call_oi"].sum(); total_poi = df["put_oi"].sum()
    pcr       = total_poi / total_coi if total_coi > 0 else 1.0
    max_pain  = compute_max_pain(df)

    atm_row = df_band[df_band["strike"] == atm]
    atm_iv  = float(((atm_row["call_iv"].values[0] if not atm_row.empty else 0) +
                     (atm_row["put_iv"].values[0]  if not atm_row.empty else 0)) / 2)

    support    = df_band.loc[df_band["put_oi"].idxmax(),  "strike"] if not df_band.empty else 0
    resistance = df_band.loc[df_band["call_oi"].idxmax(), "strike"] if not df_band.empty else 0
    wall_width = float(resistance - support) if resistance > support else float(step * 4)

    near_band     = df_band[df_band["strike"].between(atm - 3*step, atm + 3*step)]
    total_oi_band = df_band["call_oi"].sum() + df_band["put_oi"].sum()
    near_oi_total = near_band["call_oi"].sum() + near_band["put_oi"].sum()
    near_oi_conc  = near_oi_total / total_oi_band if total_oi_band > 0 else 0.5
    near_oichg_total = abs(near_band["call_oi_chg"]).sum() + abs(near_band["put_oi_chg"]).sum()
    band_oichg_total = abs(df_band["call_oi_chg"]).sum() + abs(df_band["put_oi_chg"]).sum()
    near_oichg_conc  = near_oichg_total / band_oichg_total if band_oichg_total > 0 else 0.5
    atm_pressure     = float(near_band["put_oi_chg"].sum() - near_band["call_oi_chg"].sum())

    otm_puts  = df_band[df_band["strike"] < atm - step]
    otm_calls = df_band[df_band["strike"] > atm + step]
    if len(otm_puts) >= 2 and len(otm_calls) >= 2:
        put_slope  = float(np.polyfit(otm_puts["strike"],  otm_puts["put_iv"],   1)[0])
        call_slope = float(np.polyfit(otm_calls["strike"], otm_calls["call_iv"], 1)[0])
        skew_slope = round(put_slope - call_slope, 4)
    else:
        skew_slope = 0.0

    iv_rank    = compute_iv_rank(df_band)
    gamma_flip = compute_gamma_flip(df_band, spot)

    return {
        "ev_ratio":   round(ev_ratio,  3),
        "net_delta":  round(net_delta, 0),
        "net_gamma":  round(net_gamma, 6),
        "net_theta":  round(net_theta, 0),
        "gex":        round(gex, 0),
        "vanna":      round(vanna, 2),
        "gt_ratio":   round(gt_ratio, 4),
        "momentum":   round(momentum,  0),
        "vega_skew":  round(vega_skew, 3),
        "max_pain":   round(max_pain,  0),
        "pcr":        round(pcr, 2),
        "atm_iv":     round(atm_iv, 2),
        "atm":        atm, "spot": spot,
        "support":    support,
        "resistance": resistance,
        "wall_width": wall_width,
        "near_oi_concentration":    round(near_oi_conc, 3),
        "near_oichg_concentration": round(near_oichg_conc, 3),
        "atm_pressure": round(atm_pressure, 0),
        "skew_slope":   round(skew_slope, 4),
        "iv_rank":      iv_rank,
        "gamma_flip":   gamma_flip,
        "call_oi_total": int(total_coi),
        "put_oi_total":  int(total_poi),
        "df_band":       df_band,
    }

# ─────────────────────────────────────────────────────────────────────
#  SCORING ENGINE  (v3: includes futures term structure + carry anomaly)
# ─────────────────────────────────────────────────────────────────────
def compute_score(m, roll=None):
    """
    Combines options metrics AND futures signals into a single 0–100 score.
    Plain English: above 70 = bullish bias, 45–55 = neutral, below 30 = bearish bias.
    v3 adds futures carry, term structure, and rollover conviction as score components.
    """
    if not m: return 50.0
    score = 15  # base
    ev    = m["ev_ratio"]
    score += 15 if ev >= 1.2 else (0 if ev < 0.8 else 7.5)
    d     = m["net_delta"]
    score += 20 if d >= 1_000 else (0 if d < -1_000 else 10)
    mom   = m["momentum"]
    score += 15 if mom >= 500 else (0 if mom < -500 else 7.5)
    vega  = m["vega_skew"]
    score += 10 if vega >= 1.2 else (0 if vega < 0.8 else 5)
    vanna = m["vanna"]
    score += 10 if vanna >= 10 else (0 if vanna < -10 else 5)

    # ── v3: Futures term structure bias (±6 pts)
    # Plain English: bullish carry curve adds to score; backwardation subtracts.
    if roll:
        ts_bias = roll.get("ts_bias", 0)   # -2 to +2
        score  += ts_bias * 3               # adds -6 to +6

        # Carry anomaly contribution (±4 pts)
        # Plain English: when futures roll cost far exceeds what IV implies, a big move is coming.
        ca = roll.get("carry_anomaly", 1.0)
        if ca >= 1.5:   score += 4
        elif ca <= 0.5: score -= 3

        # Rollover conviction (±4 pts)
        # Plain English: longs rolling forward (conviction) adds; liquidation subtracts.
        rv = roll.get("rollover_velocity", 0.8)
        if rv >= 1.3:   score += 4
        elif rv <= 0.3: score -= 4

    return round(min(100, max(0, score)), 1)

def strategy_recommendation(score, m, symbol="GOLD"):
    support    = m.get("support", 0); resistance = m.get("resistance", 0)
    atm        = m.get("atm", 0);     step = DHAN_SECURITY.get(symbol, DHAN_SECURITY["GOLD"])["step"]
    if score >= 85:   name, color = "Long Call / Bull Call Spread", "#00C853"
    elif score >= 70: name, color = "Bull Call Spread",             "#69F0AE"
    elif score >= 55: name, color = "Bull Put Spread (High Prob)",  "#B2FF59"
    elif score >= 45: name, color = "Iron Condor",                  "#FFD600"
    elif score >= 31: name, color = "Bear Call Spread",             "#FF6D00"
    elif score >= 16: name, color = "Bear Put Spread",              "#F44336"
    else:             name, color = "Long Put",                     "#B71C1C"
    if score >= 85:   legs = f"Buy {int(atm)} CE  |  Sell {int(resistance)} CE"
    elif score >= 70: legs = f"Buy {int(atm)} CE  |  Sell {int(atm+2*step)} CE"
    elif score >= 55: legs = f"Sell {int(support+step)} PE  |  Buy {int(support-step)} PE"
    elif score >= 45: legs = (f"Sell {int(support+step)} PE / Buy {int(support-step)} PE  +  "
                               f"Sell {int(resistance-step)} CE / Buy {int(resistance+step)} CE")
    elif score >= 31: legs = f"Sell {int(atm)} CE  |  Buy {int(atm+2*step)} CE"
    elif score >= 16: legs = f"Buy {int(atm)} PE  |  Sell {int(atm-2*step)} PE"
    else:             legs = f"Buy {int(support-step)} PE"
    if score >= 55:   mode, mc = "TREND MODE — Bullish", "#00E676"
    elif score >= 45: mode, mc = "NEUTRAL / RANGE",      "#FFD740"
    else:             mode, mc = "TREND MODE — Bearish",  "#FF5252"
    return {"name": name, "legs": legs, "color": color, "market_mode": mode, "mode_color": mc}

# ─────────────────────────────────────────────────────────────────────
#  OI VELOCITY
# ─────────────────────────────────────────────────────────────────────
def compute_oi_velocity(history, symbol="GOLD"):
    sym_history = _extract_sym_history(history, symbol)
    if len(sym_history) < 3:
        return {"call_oi_velocity": 0, "put_oi_velocity": 0, "call_oi_accel": 0, "put_oi_accel": 0,
                "call_vel_zscore": 0, "put_vel_zscore": 0, "alert_level": "NONE",
                "alert_text": "Collecting data…", "n_ticks": 0}
    call_oi = np.array([safe_num(x.get("call_oi_total", 0)) for x in sym_history], dtype=float)
    put_oi  = np.array([safe_num(x.get("put_oi_total",  0)) for x in sym_history], dtype=float)
    if call_oi.max() == 0 and put_oi.max() == 0:
        nd_arr  = np.array([safe_num(x.get("net_delta",    0)) for x in sym_history], dtype=float)
        mom_arr = np.array([safe_num(x.get("oi_net_delta", 0)) for x in sym_history], dtype=float)
        call_oi = np.maximum(nd_arr, 0) + np.maximum(mom_arr, 0)
        put_oi  = np.maximum(-nd_arr, 0) + np.maximum(-mom_arr, 0)
    c_vel = np.diff(call_oi); p_vel = np.diff(put_oi)
    if len(c_vel) < 2:
        return {"call_oi_velocity": 0, "put_oi_velocity": 0, "call_oi_accel": 0, "put_oi_accel": 0,
                "call_vel_zscore": 0, "put_vel_zscore": 0, "alert_level": "NONE",
                "alert_text": "Collecting data…", "n_ticks": len(sym_history)}
    c_accel = float(c_vel[-1] - c_vel[-2]); p_accel = float(p_vel[-1] - p_vel[-2])
    window  = min(10, len(c_vel))
    c_vel_z = _zscore(c_vel, window); p_vel_z = _zscore(p_vel, window)
    max_z   = max(abs(c_vel_z), abs(p_vel_z))
    if max_z >= 2.0:
        alert_level = "DANGER"
        side        = "CALL" if abs(c_vel_z) > abs(p_vel_z) else "PUT"
        direction   = "surge" if (c_vel_z if side=="CALL" else p_vel_z) > 0 else "unwind"
        alert_text  = f"⚡ {side} OI {direction} detected — velocity {max_z:.1f}σ above norm. Expect directional move."
    elif max_z >= 1.2:
        alert_level = "WATCH"
        side        = "CALL" if abs(c_vel_z) > abs(p_vel_z) else "PUT"
        alert_text  = f"⚠ {side} OI velocity elevated ({max_z:.1f}σ). Monitor closely."
    else:
        alert_level = "NONE"
        alert_text  = "OI velocity within normal range."
    return {"call_oi_velocity": float(c_vel[-1]), "put_oi_velocity": float(p_vel[-1]),
            "call_oi_accel": c_accel, "put_oi_accel": p_accel,
            "call_vel_zscore": round(c_vel_z, 2), "put_vel_zscore": round(p_vel_z, 2),
            "alert_level": alert_level, "alert_text": alert_text, "n_ticks": len(sym_history)}

# ─────────────────────────────────────────────────────────────────────
#  OI REGIME + COMBINED BIAS PANEL
# ─────────────────────────────────────────────────────────────────────
def _bucket_oi_15min(sym_history):
    if len(sym_history) < 3: return [], [], []
    call_oi = np.array([safe_num(x.get("call_oi_total", 0)) for x in sym_history], dtype=float)
    put_oi  = np.array([safe_num(x.get("put_oi_total",  0)) for x in sym_history], dtype=float)
    if call_oi.max() == 0 and put_oi.max() == 0:
        nd  = np.array([safe_num(x.get("net_delta",    0)) for x in sym_history], dtype=float)
        mom = np.array([safe_num(x.get("oi_net_delta", 0)) for x in sym_history], dtype=float)
        call_oi = np.maximum(nd, 0) + np.maximum(mom, 0)
        put_oi  = np.maximum(-nd, 0) + np.maximum(-mom, 0)
    ts = [x.get("ts", "") for x in sym_history]
    c_vel = np.diff(call_oi); p_vel = np.diff(put_oi); ts_v = ts[1:]
    bc, bp = {}, {}
    for i, t in enumerate(ts_v):
        try:
            t_part = t.split("T")[-1] if "T" in t else t
            parts  = t_part.split(":"); hh, mm = int(parts[0]), int(parts[1])
            label  = f"{hh:02d}:{(mm//15)*15:02d}"
        except Exception:
            label = t
        bc[label] = bc.get(label, 0.0) + float(c_vel[i])
        bp[label] = bp.get(label, 0.0) + float(p_vel[i])
    labels = sorted(bc.keys())
    return labels, [bc[l] for l in labels], [bp.get(l, 0.0) for l in labels]

def _oi_regime_info(c_bkt, p_bkt):
    if not c_bkt or not p_bkt:
        return {"label": "Collecting OI data for regime detection…", "sub": "", "bg": CARD, "fg": MUTED, "border": MUTED}
    c_arr = np.array(c_bkt, dtype=float); p_arr = np.array(p_bkt, dtype=float)
    c_std = float(c_arr.std()) if c_arr.std() > 1e-9 else 1.0
    p_std = float(p_arr.std()) if p_arr.std() > 1e-9 else 1.0
    avg_c = abs(float(np.mean(list(c_arr[-3:])))) ; avg_p = abs(float(np.mean(list(p_arr[-3:]))))
    buyer  = (avg_c > 1.0*c_std) or (avg_p > 1.0*p_std)
    seller = (avg_c <= 0.8*c_std) and (avg_p <= 0.8*p_std)
    if buyer and not seller:
        return {"label": "OPTION BUYER'S REGIME", "sub": "OI velocity elevated — directional participants active. Premium expensive. Favour directional plays.", "bg": "#FFFBEB", "fg": "#D97706", "border": "#D97706"}
    elif seller:
        return {"label": "OPTION SELLER'S REGIME", "sub": "OI velocity subdued — writers in control. Range-bound / premium decay favoured. Sell spreads or iron condors.", "bg": "#F0FDF4", "fg": "#059669", "border": "#059669"}
    else:
        return {"label": "TRANSITIONAL REGIME", "sub": "Mixed OI signals — neither buyers nor sellers clearly dominant. Wait for clarity.", "bg": "#EFF6FF", "fg": "#0891B2", "border": "#0891B2"}

def _combined_bias_info(c_bkt, p_bkt):
    if not c_bkt or not p_bkt: return None
    c_arr = np.array(c_bkt, dtype=float); p_arr = np.array(p_bkt, dtype=float)
    mean_c = float(c_arr.mean()); std_c = float(c_arr.std()) if c_arr.std() > 1e-9 else 1.0
    mean_p = float(p_arr.mean()); std_p = float(p_arr.std()) if p_arr.std() > 1e-9 else 1.0
    comp_c = c_arr[:-1] if len(c_arr)>=2 else c_arr; comp_p = p_arr[:-1] if len(p_arr)>=2 else p_arr
    sig_c = float(np.mean(comp_c[-2:])) if len(comp_c)>=2 else (float(np.mean(comp_c)) if len(comp_c) else 0.0)
    sig_p = float(np.mean(comp_p[-2:])) if len(comp_p)>=2 else (float(np.mean(comp_p)) if len(comp_p) else 0.0)
    c_z = (sig_c - mean_c) / std_c; p_z = (sig_p - mean_p) / std_p
    STRONG, WEAK = 0.8, 0.3
    c_up=c_z>STRONG; c_down=c_z<-STRONG; c_flat=abs(c_z)<WEAK
    p_up=p_z>STRONG; p_down=p_z<-STRONG; p_flat=abs(p_z)<WEAK
    if   c_up and p_up:     bias, bc = "PINNED — Range / Max-Pain Bias",          BLUE
    elif c_up and p_down:   bias, bc = "BULLISH — Upside Bias from Writers",       GREEN
    elif c_down and p_up:   bias, bc = "BEARISH — Downside Bias from Writers",     RED
    elif c_down and p_down: bias, bc = "EXPANSION — Breakout / Breakdown Risk",    "#9333EA"
    elif c_up and p_flat:   bias, bc = "MILDLY BEARISH — Resistance Reinforcing",  AMBER
    elif p_up and c_flat:   bias, bc = "MILDLY BULLISH — Support Reinforcing",     "#10B981"
    else:                   bias, bc = "NEUTRAL — No Clear OI Signal",             MUTED
    return {"bias": bias, "bc": bc, "c_z": c_z, "p_z": p_z}

# ─────────────────────────────────────────────────────────────────────
#  INTRADAY OI RECORDER  (v3: adds rollover_velocity)
# ─────────────────────────────────────────────────────────────────────
def record_intraday_oi(symbol: str, roll: dict, oi_history: dict):
    """
    Records near/next/far OI each refresh cycle.
    v3: also tracks rollover_velocity = how fast OI moves from near to next month.
    Plain English: builds the intraday OI history curve shown in the chart below.
    """
    if not roll: return oi_history
    ts   = strftime_ist("%H:%M")
    hist = oi_history.setdefault(symbol, [])
    # Compute rollover velocity
    # Plain English: compares how much OI entered next month vs how much left near month
    rollover_velocity = 0.8  # default = normal
    if len(hist) >= 1:
        prev        = hist[-1]
        delta_near  = roll.get("near_oi", 0) - prev.get("near_oi", 0)
        delta_next  = roll.get("next_oi", 0) - prev.get("next_oi", 0)
        if abs(delta_near) > 10:
            rollover_velocity = round(delta_next / abs(delta_near), 3)
    entry = {
        "ts":                 ts,
        "near_oi":            roll.get("near_oi", 0),
        "next_oi":            roll.get("next_oi", 0),
        "far_oi":             roll.get("far_oi", 0),
        "total_oi":           roll.get("near_oi", 0) + roll.get("next_oi", 0),
        "rollover_velocity":  rollover_velocity,
    }
    if hist and hist[-1]["ts"] == ts:
        hist[-1] = entry
    else:
        hist.append(entry)
    if len(hist) > 600:
        oi_history[symbol] = hist[-600:]
    return oi_history

# ─────────────────────────────────────────────────────────────────────
#  CHARTS
# ─────────────────────────────────────────────────────────────────────
def chart_layout(**kw):
    return dict(paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC",
                font=dict(color=TEXT, size=11),
                xaxis=dict(gridcolor="#E2E8F0", linecolor="#CBD5E1"),
                yaxis=dict(gridcolor="#E2E8F0", linecolor="#CBD5E1"),
                margin=dict(l=40, r=20, t=55, b=38), height=340, **kw)

def score_gauge_fig(score):
    color = GREEN if score >= 70 else (AMBER if score >= 45 else RED)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, domain={"x":[0,1],"y":[0,1]},
        title={"text":"Market Score","font":{"color":TEXT,"size":12}},
        number={"font":{"color":color,"size":36}},
        gauge={"axis":{"range":[0,100],"tickcolor":"#444"},"bar":{"color":color},"bgcolor":CARD,
               "steps":[{"range":[0,30],"color":"#FEE2E2"},{"range":[30,45],"color":"#FFEDD5"},
                        {"range":[45,55],"color":"#FEF3C7"},{"range":[55,70],"color":"#DCFCE7"},
                        {"range":[70,100],"color":"#D1FAE5"}],
               "threshold":{"line":{"color":color,"width":3},"thickness":0.8,"value":score}},
    ))
    fig.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                      margin=dict(l=20,r=20,t=30,b=5), height=220)
    return fig

def build_iv_history_chart(sym_history):
    empty = go.Figure(); empty.update_layout(**chart_layout(title="Cumulative Δ ATM IV — 15-min buckets"))
    if len(sym_history) < 2: return empty
    buckets = {}
    for x in sym_history:
        ts = x.get("ts",""); iv = safe_num(x.get("atm_iv",0))
        if iv <= 0: continue
        try:
            t_part = ts.split("T")[-1] if "T" in ts else ts
            parts  = t_part.split(":"); hh, mm = int(parts[0]), int(parts[1])
            label  = f"{hh:02d}:{(mm//15)*15:02d}"
        except Exception:
            label = ts
        buckets[label] = iv
    if not buckets: return empty
    labels = sorted(buckets.keys()); vals = [buckets[l] for l in labels]
    base = vals[0]; cum_d = [v-base for v in vals]
    iv_delta  = cum_d[-1] if cum_d else 0
    direction = "IV RISING" if iv_delta>0.5 else ("IV FALLING" if iv_delta<-0.5 else "IV FLAT")
    line_color = "#059669" if iv_delta<-0.5 else ("#DC2626" if iv_delta>0.5 else CYAN)
    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color="#94A3B8", opacity=0.7, annotation_text="open baseline", annotation_font_size=10)
    fig.add_trace(go.Scatter(x=labels, y=cum_d, mode="lines+markers", name="Cumul Δ ATM IV",
                             line=dict(color=line_color, width=2.5), marker=dict(size=6, color=line_color),
                             hovertemplate="<b>%{x}</b><br>Cumul Δ IV: %{y:+.2f}pp<extra></extra>"))
    lk = chart_layout(title=f"Cumul Δ ATM IV — 15-min | {direction} ({iv_delta:+.2f}pp)")
    lk["yaxis"] = dict(title="Δ IV (pp)", gridcolor="#E2E8F0")
    lk["xaxis"] = dict(title="15-min bucket", gridcolor="#E2E8F0")
    fig.update_layout(**lk)
    return fig

def _build_oi_vel_chart(sym_history, side="CALL"):
    side_label = "Call" if side=="CALL" else "Put"
    line_color = CYAN if side=="CALL" else PINK
    band_color = "rgba(0,229,255,0.08)" if side=="CALL" else "rgba(255,64,129,0.08)"
    empty = go.Figure(); empty.update_layout(**chart_layout(title=f"Δ {side_label} OI Velocity Z-Score — 15-min"))
    if len(sym_history) < 3: return empty
    call_oi = np.array([safe_num(x.get("call_oi_total",0)) for x in sym_history], dtype=float)
    put_oi  = np.array([safe_num(x.get("put_oi_total", 0)) for x in sym_history], dtype=float)
    ts_raw  = [x.get("ts","") for x in sym_history]
    vel     = np.diff(call_oi if side=="CALL" else put_oi); ts_v = ts_raw[1:]
    def _bucket(t):
        try:
            parts = t.split("T")[-1].split(":"); hh, mm = int(parts[0]), int(parts[1])
            return f"{hh:02d}:{(mm//15)*15:02d}"
        except Exception: return t
    buckets: dict = {}
    for i, t in enumerate(ts_v):
        lbl = _bucket(t); buckets[lbl] = buckets.get(lbl, 0.0) + float(vel[i])
    if not buckets: return empty
    labels_c = sorted(buckets.keys()); arr_c = np.array([buckets[l] for l in labels_c], dtype=float)
    mean_val = float(arr_c.mean()); std_val = float(arr_c.std()) if arr_c.std()>1e-9 else 1.0
    z_arr    = (arr_c - mean_val) / std_val
    live_lbl = _bucket(ts_raw[-1]); live_vel = sum(float(vel[i]) for i,t in enumerate(ts_v) if _bucket(t)==live_lbl)
    live_z   = (live_vel - mean_val) / std_val
    all_lbl  = list(labels_c); all_z = list(z_arr); is_live = [False]*len(labels_c)
    if live_lbl not in labels_c:
        all_lbl.append(live_lbl); all_z.append(live_z); is_live.append(True)
    else:
        all_z[-1] = live_z; is_live[-1] = True
    latest_z = all_z[-1]
    alert = (f"⚡ SURGE +{latest_z:.1f}σ" if latest_z>=2.0 else
             f"⚠ ELEVATED +{latest_z:.1f}σ" if latest_z>=1.2 else
             f"⚡ UNWIND {latest_z:.1f}σ" if latest_z<=-2.0 else
             f"↘ EASING {latest_z:.1f}σ" if latest_z<=-1.2 else f"NORMAL {latest_z:+.1f}σ")
    n = len(all_lbl); fig = go.Figure()
    fig.add_trace(go.Scatter(x=all_lbl+all_lbl[::-1], y=[2.0]*n+[-2.0]*n,
                             fill="toself", fillcolor=band_color,
                             line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip", showlegend=False))
    closed_x=[l for l,lv in zip(all_lbl,is_live) if not lv]; closed_y=[z for z,lv in zip(all_z,is_live) if not lv]
    live_x=[l for l,lv in zip(all_lbl,is_live) if lv];       live_y=[z for z,lv in zip(all_z,is_live) if lv]
    if closed_x:
        fig.add_trace(go.Scatter(x=closed_x, y=closed_y, mode="lines+markers",
                                 name=f"{side_label} OI Vel Z", line=dict(color=line_color, width=2.5),
                                 marker=dict(size=6, color=line_color),
                                 hovertemplate="<b>%{x}</b><br>Z: %{y:+.2f}σ<extra></extra>"))
    if live_x:
        if closed_x:
            fig.add_trace(go.Scatter(x=[closed_x[-1],live_x[0]], y=[closed_y[-1],live_y[0]],
                                     mode="lines", line=dict(color=line_color, width=1.5, dash="dot"),
                                     showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=live_x, y=live_y, mode="markers", name="🔴 Live (forming)",
                                 marker=dict(size=10, color=line_color, opacity=0.5, symbol="circle-open",
                                             line=dict(width=2,color=line_color))))
    for y_val, dash, col, ann in [(0,"dash","#94A3B8","mean"),(2,"dot",RED,"+2σ"),
                                   (1,"dot",AMBER,"+1σ"),(-1,"dot",GREEN,"−1σ"),(-2,"dot",GREEN,"−2σ")]:
        fig.add_hline(y=y_val, line_dash=dash, line_color=col, opacity=0.5,
                      annotation_text=ann, annotation_font_size=10)
    lk = chart_layout(title=f"Δ {side_label} OI Vel Z-Score — 15-min | {alert}")
    lk["yaxis"] = dict(title="Z-score (σ)", gridcolor="#E2E8F0")
    lk["xaxis"] = dict(title="15-min bucket", gridcolor="#E2E8F0")
    fig.update_layout(**lk); return fig

def build_rollover_velocity_chart(oi_history, symbol):
    """
    Charts how fast OI rolls from near to next month each refresh cycle.
    Plain English: above 1 = more OI entering next month than leaving near (bullish);
    below 0.3 = liquidation (bearish).
    """
    hist = oi_history.get(symbol, [])
    fig  = go.Figure()
    if len(hist) < 3:
        fig.add_annotation(text="Collecting rollover velocity data… refresh a few times",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                           font=dict(color=MUTED, size=12))
        fig.update_layout(**chart_layout(title="Rollover Velocity (Near→Next OI Flow)"))
        return fig
    ts_v = [h["ts"] for h in hist]
    rv   = [h.get("rollover_velocity", 0.8) for h in hist]
    colors = [GREEN if v>=1.3 else (CYAN if v>=0.8 else (AMBER if v>=0.3 else RED)) for v in rv]
    fig.add_hline(y=1.3, line_dash="dot", line_color=GREEN, opacity=0.6,
                  annotation_text="Conviction Roll ≥1.3", annotation_font_size=9)
    fig.add_hline(y=0.3, line_dash="dot", line_color=RED, opacity=0.6,
                  annotation_text="Liquidation ≤0.3", annotation_font_size=9)
    fig.add_trace(go.Scatter(x=ts_v, y=rv, mode="lines+markers",
                             marker=dict(color=colors, size=7),
                             line=dict(color=CYAN, width=2),
                             hovertemplate="<b>%{x}</b><br>Roll Velocity: %{y:.3f}<extra></extra>"))
    lk = chart_layout(title="Rollover Velocity — Δ Next OI / |Δ Near OI|")
    lk["yaxis"] = dict(title="Velocity Ratio", gridcolor="#E2E8F0", zeroline=True, zerolinecolor="#94A3B8")
    lk["xaxis"] = dict(title="Time", gridcolor="#E2E8F0")
    fig.update_layout(**lk); return fig

def build_term_structure_chart(roll: dict):
    """
    Plots the 3-month futures curve (near, next, far).
    Plain English: an upward curve = contango (normal); downward = backwardation (delivery pressure).
    """
    fig = go.Figure()
    if not roll:
        fig.update_layout(**chart_layout(title="Futures Term Structure (3-Month Curve)"))
        return fig
    labels = ["Near", "Next"]
    prices = [roll["near_ltp"], roll["next_ltp"]]
    colors_bar = [GOLD, "#CE93D8"]
    if roll.get("has_far") and roll.get("far_ltp", 0) > 0:
        labels.append("Far"); prices.append(roll["far_ltp"]); colors_bar.append(CYAN)
    fig.add_trace(go.Bar(x=labels, y=prices, marker_color=colors_bar,
                         text=[f"₹{p:,.2f}" for p in prices],
                         textposition="outside",
                         hovertemplate="<b>%{x}</b><br>LTP: ₹%{y:,.2f}<extra></extra>"))
    ts_color = roll.get("ts_color", CYAN)
    fig.add_annotation(text=roll.get("ts_shape",""), xref="paper", yref="paper",
                       x=0.5, y=1.12, showarrow=False, font=dict(color=ts_color, size=11, family="monospace"))
    lk = chart_layout(title="Futures Term Structure — 3-Month Curve")
    lk["yaxis"] = dict(title="LTP (₹)", gridcolor="#E2E8F0", tickformat=",")
    fig.update_layout(**lk, showlegend=False)
    return fig

# ═════════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ═════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
    .stApp {{ background-color: {BG}; }}
    .main .block-container {{ padding-top: 1rem; max-width: 1400px; }}
    h1, h2, h3, h4 {{ color: {GOLD}; }}
    .stMarkdown, .stText {{ color: {TEXT}; }}
    div[data-testid="stMetric"] {{
        background-color: {CARD}; border: 1px solid {BORDER};
        border-radius: 8px; padding: 10px 14px;
    }}
    div[data-testid="stMetric"] label {{
        font-size: 10px; color: {MUTED}; text-transform: uppercase; letter-spacing: 0.5px;
    }}
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
        font-size: 19px; font-weight: 700;
    }}
    .section-header {{
        background-color: {SECTION_BG}; color: {ACCENT}; font-weight: 700; font-size: 13px;
        padding: 8px 16px; border-radius: 8px; margin-bottom: 10px; letter-spacing: 0.3px;
        border-left: 4px solid {ACCENT};
    }}
    .regime-banner {{ border-radius: 10px; padding: 12px 20px; margin-bottom: 10px; }}
    .regime-label  {{ font-weight: 800; font-size: 16px; letter-spacing: 0.5px; }}
    .regime-sub    {{ font-weight: 500; font-size: 12px; opacity: 0.85; margin-top: 3px; }}
    .strat-box {{ background-color: {CARD}; border-radius: 10px; padding: 14px;
                  border: 1px solid {BORDER}; border-left-width: 4px; border-left-style: solid; }}
    .bias-cell {{ border-radius: 8px; padding: 8px 10px; text-align: center; }}
    .alert-text {{ font-size: 12px; font-weight: 600; line-height: 1.5; }}
    .stDataFrame {{ background-color: {CARD}; }}
    .stSelectbox label, .stButton button {{ color: {TEXT}; }}
    .stButton button {{
        background-color: {ACCENT}; color: #FFFFFF; font-weight: bold;
        border: none; border-radius: 6px;
    }}
    hr {{ border-color: {BORDER}; }}
    [data-testid="stSidebar"] {{ background-color: #F8FAFC; border-right: 1px solid {BORDER}; }}
    .stTextInput input {{ background-color: #FFFFFF; border: 1px solid {BORDER}; color: {TEXT}; }}
    .explain-text {{ font-size: 9px; color: #888899; margin-top: 3px; font-style: italic; line-height: 1.4; }}
</style>
""", unsafe_allow_html=True)

# ── Session state init ─────────────────────────────────────────────────
if "history"           not in st.session_state: st.session_state["history"]           = load_history_from_disk()
if "oi_history"        not in st.session_state: st.session_state["oi_history"]        = {}
if "last_refresh"      not in st.session_state: st.session_state["last_refresh"]      = 0
if "is_owner"          not in st.session_state: st.session_state["is_owner"]          = False
if "owner_pw_attempt"  not in st.session_state: st.session_state["owner_pw_attempt"]  = ""
if "owner_login_error" not in st.session_state: st.session_state["owner_login_error"] = False
if "iv_smile_history"  not in st.session_state: st.session_state["iv_smile_history"]  = {}  # NEW v3

# ── SIDEBAR: Owner Login ───────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:8px 0 12px 0;">
        <span style="font-size:20px;">🌟</span>
        <div style="font-size:13px;font-weight:700;color:{GOLD};margin-top:4px;">Commodities v3</div>
    </div>""", unsafe_allow_html=True)
    st.divider()
    if st.session_state["is_owner"]:
        st.markdown(f"""
        <div style="background:#1a2e00;border:1.5px solid #00E676;border-radius:8px;
                    padding:10px 14px;text-align:center;margin-bottom:12px;">
            <div style="font-size:14px;font-weight:800;color:#00E676;">🔑 OWNER MODE</div>
            <div style="font-size:10px;color:#888;margin-top:2px;">Full controls unlocked</div>
        </div>""", unsafe_allow_html=True)
        if st.button("🔒 Log out", use_container_width=True):
            st.session_state["is_owner"] = False; st.session_state["owner_login_error"] = False; st.rerun()
    else:
        st.markdown(f"""
        <div style="background:#1a1a2e;border:1px solid {BORDER};border-radius:8px;
                    padding:10px 14px;text-align:center;margin-bottom:12px;">
            <div style="font-size:12px;font-weight:700;color:{MUTED};">👁 VIEW-ONLY MODE</div>
        </div>""", unsafe_allow_html=True)
        pw_input = st.text_input("Owner Password", type="password", key="pw_field", placeholder="Enter owner password…")
        if st.button("🔑 Unlock Owner Mode", use_container_width=True):
            if pw_input == CFG.OWNER_PASSWORD:
                st.session_state["is_owner"] = True; st.session_state["owner_login_error"] = False; st.rerun()
            else:
                st.session_state["owner_login_error"] = True
        if st.session_state["owner_login_error"]:
            st.error("Incorrect password.")
    st.divider()
    st.markdown(f"""
    <div style="font-size:10px;color:{MUTED};text-align:center;line-height:1.6;">
        Data source: {'Dhan API ✅' if CFG.USE_DHAN else 'DEMO MODE'}<br>
        Auto-refresh: {AUTO_REFRESH_SECONDS}s<br>
        v3: Term Structure · Smile Classifier
    </div>""", unsafe_allow_html=True)

# ── Auto-refresh ──────────────────────────────────────────────────────
time_since = time.time() - st.session_state["last_refresh"]
refresh_placeholder = st.empty()
is_owner = st.session_state["is_owner"]

# ── TITLE ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;margin-bottom:14px;border-bottom:2px solid {ACCENT};padding-bottom:10px;">
    <h1 style="margin:0;color:{GOLD};font-size:26px;font-weight:800;letter-spacing:1px;">
        🌟 Commodities Options Analysis v3
    </h1>
    <div style="font-size:11px;color:{MUTED};margin-top:3px;">
        MCX GOLD & SILVER · Term Structure · Rollover Velocity · IV Smile Classifier · Gamma Regime · Carry Anomaly
    </div>
</div>""", unsafe_allow_html=True)

# ── TOP CONTROLS ──────────────────────────────────────────────────────
col_ctrl1,col_ctrl2,col_ctrl3,col_ctrl4,col_ctrl5 = st.columns([1.5,1.5,2,1.5,1])

with col_ctrl1:
    symbol = st.selectbox("COMMODITY", COMMODITY_SYMBOLS, index=0)

with col_ctrl2:
    # Auto-detect expiries — no manual override needed unless owner
    if CFG.USE_DHAN:
        expiries = fetch_dhan_expiry_list(symbol)
    else:
        expiries = [(date.today() + timedelta(days=d)).strftime("%Y-%m-%d") for d in [10,40,70]]
    if is_owner:
        expiry = st.selectbox("EXPIRY", expiries, index=0 if expiries else None,
                              help="Auto-detected. Owner can override.")
        st.session_state["selected_expiry"] = expiry
    else:
        expiry = st.session_state.get("selected_expiry", expiries[0] if expiries else "")
        st.markdown(f"""
        <div style="padding-top:6px;">
            <div style="font-size:10px;color:{MUTED};text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">EXPIRY</div>
            <div style="font-size:14px;font-weight:700;color:#80CBC4;">{expiry}</div>
        </div>""", unsafe_allow_html=True)

with col_ctrl3:
    st.markdown(f"""
    <div style="padding-top:28px;font-size:11px;color:{MUTED};font-style:italic;">
        📡 {'Dhan API (MCX) ✅' if CFG.USE_DHAN else 'DEMO MODE — Add credentials in secrets'}
    </div>""", unsafe_allow_html=True)

with col_ctrl4:
    st.markdown(f"""
    <div style="padding-top:28px;font-size:11px;color:{MUTED};">
        🕐 {strftime_ist('%H:%M:%S')} IST
    </div>""", unsafe_allow_html=True)

with col_ctrl5:
    if is_owner:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        refresh_clicked = st.button("⟳ Refresh", use_container_width=True)
    else:
        refresh_clicked = False
        st.markdown(f"<div style='padding-top:32px;font-size:10px;color:{MUTED};text-align:center;'>🔒 View Only</div>", unsafe_allow_html=True)

if is_owner:
    auto_refresh = st.checkbox("Auto-refresh every 60s", value=True)
else:
    auto_refresh = True

if auto_refresh:
    if is_owner:
        remaining = max(0, AUTO_REFRESH_SECONDS - int(time_since))
        refresh_placeholder.markdown(f"<div style='text-align:center;font-size:11px;color:{MUTED};'>⏳ Auto-refresh in {remaining}s</div>", unsafe_allow_html=True)
    if time_since >= AUTO_REFRESH_SECONDS:
        st.session_state["last_refresh"] = time.time(); st.rerun()
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=AUTO_REFRESH_SECONDS*1000, key="autorefresh")
    except ImportError:
        pass

if refresh_clicked:
    st.session_state["last_refresh"] = time.time(); st.rerun()

# ── FETCH DATA ────────────────────────────────────────────────────────
# Fetch futures roll FIRST so spot fallback is available for compute_metrics
roll = fetch_futures_roll(symbol) if CFG.USE_DHAN else demo_futures_roll(symbol)
df, spot, exp, source = get_option_chain(symbol, expiry if expiry else None)

if df.empty:
    st.error("No data available. Check API credentials or try again.")
    st.stop()

# compute_metrics now receives roll for spot fallback
m      = compute_metrics(df, spot, symbol, expiry=exp, roll=roll)
if not m:
    st.error("Could not compute metrics — spot price unavailable and no futures data.")
    st.stop()

# Resolved spot (may have been substituted from near_ltp)
spot = m.get("spot", spot)

# ── v3: Compute wing excess and update IV smile history ───────────────
df_band = m.pop("df_band", df)
atm_iv  = m.get("atm_iv", 0)
atm     = m.get("atm", 0)

put_wing_excess, call_wing_excess = compute_wing_excess(df_band, atm, atm_iv, symbol)

smile_hist = st.session_state["iv_smile_history"].setdefault(symbol, [])
if put_wing_excess is not None:
    smile_tick = {
        "ts":               now_ist().isoformat(timespec="seconds"),
        "atm_iv":           atm_iv,
        "put_wing_excess":  put_wing_excess,
        "call_wing_excess": call_wing_excess,
    }
    if not smile_hist or smile_hist[-1].get("ts","")[:16] != smile_tick["ts"][:16]:
        smile_hist.append(smile_tick)
        if len(smile_hist) > 100:
            st.session_state["iv_smile_history"][symbol] = smile_hist[-100:]

# ── Carry anomaly & rollover velocity ────────────────────────────────
carry_anomaly = compute_carry_anomaly(roll, atm_iv) if roll else 1.0
roll_vel_z, roll_vel_interp, roll_vel_color = compute_rollover_velocity_zscore(
    st.session_state["oi_history"], symbol)

# Inject carry_anomaly + rollover_velocity into roll dict for score computation
if roll:
    roll["carry_anomaly"]       = carry_anomaly
    roll["rollover_velocity"]   = st.session_state["oi_history"].get(symbol, [{}])[-1].get("rollover_velocity", 0.8)

score = compute_score(m, roll)
strat = strategy_recommendation(score, m, symbol)

# ── Gamma Regime ──────────────────────────────────────────────────────
step   = DHAN_SECURITY.get(symbol, DHAN_SECURITY["GOLD"])["step"]
g_regime, g_regime_desc, vol_regime, g_regime_color = classify_gamma_regime(
    m.get("gex", 0), m.get("wall_width", 0), m.get("momentum", 0),
    atm_iv, m.get("iv_rank", 50), spot, m.get("gamma_flip"), step)

# ── IV Smile Scenario ─────────────────────────────────────────────────
iv_smile_result = classify_iv_smile_scenario(df_band, m, spot, symbol, smile_hist)

# ── Record history ────────────────────────────────────────────────────
st.session_state["oi_history"] = record_intraday_oi(symbol, roll, st.session_state["oi_history"])

ts_full = now_ist().isoformat(timespec="seconds")
tick = {
    "ts": ts_full, "symbol": symbol, "spot": spot, "atm_iv": atm_iv,
    "net_delta": m.get("net_delta",0), "oi_net_delta": m.get("momentum",0),
    "max_pain": m.get("max_pain",0), "support": m.get("support",0), "resistance": m.get("resistance",0),
    "gex": m.get("gex",0), "pcr": m.get("pcr",0), "atm_pressure": m.get("atm_pressure",0),
    "wall_width": m.get("wall_width",0), "gamma_flip": m.get("gamma_flip",None),
    "iv_rank": m.get("iv_rank",50), "gt_ratio": m.get("gt_ratio",0),
    "call_oi_total": m.get("call_oi_total",0), "put_oi_total": m.get("put_oi_total",0),
    "put_wing_excess":  put_wing_excess  if put_wing_excess  is not None else 0,   # NEW v3
    "call_wing_excess": call_wing_excess if call_wing_excess is not None else 0,   # NEW v3
    "roll_spread_pct":  roll.get("roll_spread_pct",0) if roll else 0,              # NEW v3
    "rollover_pct":     roll.get("rollover_pct",0)    if roll else 0,              # NEW v3
    "ts_bias":          roll.get("ts_bias",0)         if roll else 0,              # NEW v3
}
sym_hist = st.session_state["history"].setdefault(symbol, [])
if not sym_hist or sym_hist[-1].get("ts","")[:16] != ts_full[:16]:
    sym_hist.append(tick)
    if len(sym_hist) > 600: st.session_state["history"][symbol] = sym_hist[-600:]
    save_history_to_disk(st.session_state["history"])

log_rec = {k: m.get(k,0) for k in _CSV_COLUMNS if k in m}
log_rec.update({"ts":ts_full,"symbol":symbol,"spot":spot,"expiry":exp,"score":score,
                "atm":m.get("atm",0),"gamma_flip":m.get("gamma_flip",""),
                "put_wing_excess":tick["put_wing_excess"],"call_wing_excess":tick["call_wing_excess"],
                "roll_spread_pct":tick["roll_spread_pct"],"rollover_pct":tick["rollover_pct"],
                "ts_bias":tick["ts_bias"]})
write_decision_log(log_rec)

# ══════════════════════════════════════════════════════════════════════
#  HEADER CARDS
# ══════════════════════════════════════════════════════════════════════
gf            = m.get("gamma_flip")
theme_color   = SILVER if "SILVER" in symbol else GOLD
iv_rank_color = RED if m.get("iv_rank",50)>70 else (GREEN if m.get("iv_rank",50)<30 else AMBER)
pcr_color     = GREEN if m["pcr"]>0.8 else RED
gf_color      = RED if gf and spot<gf else GREEN
atp_color     = GREEN if m["atm_pressure"]>0 else RED

header_cols = st.columns(10)
header_data = [
    ("Commodity",    symbol,                         theme_color),
    ("Spot / LTP",   f'₹ {spot:,.2f}',               "#FFFFFF"),
    ("Expiry",       exp,                             MUTED),
    ("ATM IV",       f'{atm_iv:.2f} %',               "#80CBC4"),
    ("IV Rank",      f'{m.get("iv_rank",50):.0f}',    iv_rank_color),
    ("PCR",          f'{m["pcr"]}',                   pcr_color),
    ("Max Pain",     f'{int(m["max_pain"])}',          "#CE93D8"),
    ("Gamma Flip",   f'{int(gf)}' if gf else '—',     gf_color),
    ("ATM Pressure", f'{int(m["atm_pressure"]):+,}',  atp_color),
    ("Wall Width",   f'{int(m["wall_width"]):,}',      BLUE),
]
for col,(label,value,color) in zip(header_cols,header_data):
    col.metric(label,value)
    col.markdown(f"<div style='font-size:9px;color:{color};margin-top:-10px;'><b>{value}</b></div>",
                 unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════
#  GAMMA REGIME BANNER  (NEW v3)
# ══════════════════════════════════════════════════════════════════════
regime_bg = {GREEN:"#F0FDF4", CYAN:"#EFF6FF", AMBER:"#FFFBEB", RED:"#FFF1F2", MUTED:"#F8FAFC"}
st.markdown(f"""
<div style="background:{regime_bg.get(g_regime_color,CARD)};border:1.5px solid {g_regime_color};
            border-radius:10px;padding:12px 20px;margin-bottom:12px;">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
        <div>
            <div style="font-size:10px;color:{MUTED};font-weight:600;text-transform:uppercase;letter-spacing:0.5px;">
                ⚡ GAMMA REGIME  ·  {vol_regime.replace('_',' ')}
            </div>
            <div style="font-size:16px;font-weight:800;color:{g_regime_color};">{g_regime}</div>
            <div style="font-size:11px;color:{g_regime_color};opacity:0.85;margin-top:2px;">{g_regime_desc}</div>
        </div>
        <div style="font-size:10px;color:{MUTED};text-align:right;">
            GEX: <b style="color:{GREEN if m.get('gex',0)>0 else RED};">{m.get('gex',0):,.0f}</b> &nbsp;|&nbsp;
            Gamma Flip: <b style="color:{gf_color};">{int(gf) if gf else '—'}</b>
        </div>
    </div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  SCORE + STRATEGY + METRICS
# ══════════════════════════════════════════════════════════════════════
col_gauge, col_strat, col_metrics = st.columns([1, 1.2, 3])
with col_gauge:
    st.plotly_chart(score_gauge_fig(score), use_container_width=True, config={"displayModeBar":False})

with col_strat:
    st.markdown(f"""
    <div class="strat-box" style="border-left-color:{strat['color']};">
        <div style="font-size:10px;color:{MUTED};">MARKET MODE</div>
        <div style="font-size:16px;font-weight:700;color:{strat['mode_color']};margin-bottom:10px;">{strat['market_mode']}</div>
        <div style="font-size:10px;color:{MUTED};">STRATEGY</div>
        <div style="font-size:14px;font-weight:700;color:{strat['color']};margin-bottom:8px;">{strat['name']}</div>
        <div style="font-size:10px;color:{MUTED};">EXECUTION</div>
        <div style="font-size:12px;color:#888;font-family:monospace;">{strat['legs']}</div>
    </div>""", unsafe_allow_html=True)

with col_metrics:
    metric_items = [
        ("EV Ratio",    f'{m["ev_ratio"]}',          GREEN if m["ev_ratio"]>=1.2 else RED,  METRIC_EXPLAIN["EV Ratio"]),
        ("Net Delta",   f'{int(m["net_delta"]):,}',   GREEN if m["net_delta"]>0 else RED,    METRIC_EXPLAIN["Net Delta"]),
        ("Momentum",    f'{int(m["momentum"]):,}',    GREEN if m["momentum"]>0 else RED,     METRIC_EXPLAIN["Momentum"]),
        ("Vega Skew",   f'{m["vega_skew"]}',          GREEN if m["vega_skew"]>=1.2 else RED, METRIC_EXPLAIN["Vega Skew"]),
        ("GEX",         f'{m["gex"]:,.0f}',           GREEN if m["gex"]>0 else RED,          METRIC_EXPLAIN["GEX"]),
        ("Vanna",       f'{m["vanna"]:,.2f}',         GREEN if m["vanna"]>0 else RED,        METRIC_EXPLAIN["Vanna"]),
        ("G/T Ratio",   f'{m["gt_ratio"]}',           BLUE,                                  METRIC_EXPLAIN["G/T Ratio"]),
        ("Skew Slope",  f'{m["skew_slope"]:.4f}',     RED if m["skew_slope"]>0.15 else MUTED,METRIC_EXPLAIN["Skew Slope"]),
        ("Support",     f'{int(m["support"])}',        GREEN,                                 "Strongest put OI wall — acts as a price floor."),
        ("Resistance",  f'{int(m["resistance"])}',     RED,                                   "Strongest call OI wall — acts as a price ceiling."),
        ("Near OI %",   f'{m["near_oi_concentration"]*100:.0f}%', BLUE,                      METRIC_EXPLAIN["Near OI %"]),
        ("PCR",         f'{m["pcr"]}',                GREEN if m["pcr"]>0.8 else RED,        METRIC_EXPLAIN["PCR"]),
    ]
    for row_start in range(0, len(metric_items), 4):
        row_items = metric_items[row_start:row_start+4]
        cols = st.columns(4)
        for col,(label,value,color,explain) in zip(cols,row_items):
            col.markdown(f"""
            <div style="background-color:{CARD};border-radius:8px;padding:10px 14px;
                        border:1px solid {BORDER};min-height:90px;">
                <div style="font-size:10px;color:{MUTED};text-transform:uppercase;letter-spacing:0.5px;">{label}</div>
                <div style="font-size:19px;font-weight:700;color:{color};">{value}</div>
                <div class="explain-text">{explain}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  IV SMILE SCENARIO CARD  (NEW v3)
# ══════════════════════════════════════════════════════════════════════
if iv_smile_result:
    sr = iv_smile_result
    bc = sr["badge_color"]
    st.markdown("---")
    st.markdown('<div class="section-header">🎭 IV Smile Scenario — 12-Pattern Classifier</div>', unsafe_allow_html=True)
    sc_left, sc_right = st.columns([2, 1])
    with sc_left:
        signals_html = "".join([f"<div style='font-size:11px;color:{TEXT};padding:3px 0;border-bottom:1px solid {BORDER};'>• {s}</div>" for s in sr["signals"]])
        strats_html  = "".join([f"<span style='background:{bc}22;color:{bc};border:1px solid {bc};border-radius:4px;padding:2px 8px;font-size:10px;font-weight:700;margin:2px;display:inline-block;'>{s}</span>" for s in sr["strategies"]])
        trend_html = ""
        if sr["trend"].get("has_trend"):
            t = sr["trend"]
            trend_html = f"""
            <div style="font-size:10px;color:{MUTED};margin-top:8px;padding-top:6px;border-top:1px solid {BORDER};">
                📈 Intraday trend ({t['ticks']} ticks): ATM IV Δ={t['d_atm_iv']:+.2f}pp |
                Put wing Δ={t['d_put_wing']:+.2f}pp | Call wing Δ={t['d_call_wing']:+.2f}pp |
                Session peak IV: {t['peak_atm_iv']:.1f}%
            </div>"""
        st.markdown(f"""
        <div style="background:{CARD};border:1.5px solid {bc};border-left:5px solid {bc};
                    border-radius:10px;padding:16px 18px;">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                <span style="background:{bc};color:#fff;border-radius:6px;padding:3px 10px;
                              font-size:11px;font-weight:800;">{sr['badge']}</span>
                <span style="font-size:16px;font-weight:800;color:{bc};">{sr['scenario_name']}</span>
                <span style="font-size:11px;color:{MUTED};margin-left:auto;">
                    Confidence: <b style="color:{bc};">{sr['confidence']}%</b>
                </span>
            </div>
            <div style="font-size:11px;color:{TEXT};margin-bottom:10px;font-style:italic;">{sr['description']}</div>
            <div style="margin-bottom:10px;">{signals_html}</div>
            <div style="font-size:10px;color:{MUTED};font-weight:600;margin-bottom:4px;">STRATEGIES:</div>
            <div>{strats_html}</div>
            {trend_html}
        </div>""", unsafe_allow_html=True)
    with sc_right:
        wing_items = [
            ("ATM IV",          f'{sr["atm_iv"]:.2f}%',  CYAN,   "Implied vol at the nearest-to-spot strike."),
            ("Put Wing Excess", f'{sr["put_wing_excess"]:+.2f}pp', RED if sr["put_wing_excess"]>3 else AMBER,
             "How much extra OTM puts cost vs ATM — measures downside fear."),
            ("Call Wing Excess",f'{sr["call_wing_excess"]:+.2f}pp', GREEN if sr["call_wing_excess"]>3 else MUTED,
             "How much extra OTM calls cost vs ATM — measures upside speculation."),
            ("Skew Asymmetry",  f'{sr["skew_asymmetry"]:+.2f}',   bc,
             "Put wing minus call wing — positive = put-skewed (bearish hedging dominant)."),
            ("IV Rank",         f'{sr["iv_rank"]:.0f}%ile', iv_rank_color,
             "Where today's IV sits within the smile range — above 70 = expensive options."),
        ]
        for label,value,color,explain in wing_items:
            st.markdown(f"""
            <div style="background:{CARD};border-radius:8px;padding:8px 12px;
                        border:1px solid {BORDER};margin-bottom:6px;">
                <div style="font-size:10px;color:{MUTED};text-transform:uppercase;">{label}</div>
                <div style="font-size:16px;font-weight:700;color:{color};">{value}</div>
                <div class="explain-text">{explain}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  CHARTS ROW 1
# ══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-header">📊 Options Analysis Charts</div>', unsafe_allow_html=True)
chart_cols = st.columns(3)

def _vline(fig, val, color, label):
    if val:
        fig.add_vline(x=val, line_dash="dash", line_color=color, opacity=0.7,
                      annotation_text=label, annotation_font_size=10)

with chart_cols[0]:
    dv = df_band["call_oi"] - df_band["put_oi"]
    f1 = go.Figure(go.Bar(x=df_band["strike"], y=dv,
                          marker_color=[RED if v>0 else GREEN for v in dv], name="Net OI"))
    _vline(f1,spot,AMBER,"Spot"); _vline(f1,gf,PINK,"γ-Flip")
    f1.update_layout(**chart_layout(title="Net OI (Call − Put) · Positive=Call heavy, Negative=Put heavy"))
    st.plotly_chart(f1, use_container_width=True, config={"displayModeBar":False})

with chart_cols[1]:
    mv = df_band["call_oi_chg"] - df_band["put_oi_chg"]
    f2 = go.Figure(go.Bar(x=df_band["strike"], y=mv,
                          marker_color=[RED if v>0 else GREEN for v in mv], name="OI Change"))
    _vline(f2,spot,AMBER,"Spot")
    f2.update_layout(**chart_layout(title="OI Momentum · New call bets vs new put bets today"))
    st.plotly_chart(f2, use_container_width=True, config={"displayModeBar":False})

with chart_cols[2]:
    gv = (df_band["call_oi"]*df_band["call_gamma"] - df_band["put_oi"]*df_band["put_gamma"])*spot**2*0.01
    f3 = go.Figure(go.Bar(x=df_band["strike"], y=gv,
                          marker_color=[RED if v>0 else GREEN for v in gv], name="GEX"))
    _vline(f3,spot,AMBER,"Spot"); _vline(f3,gf,PINK,"γ-Flip")
    f3.update_layout(**chart_layout(title="Gamma Exposure · Positive pins price; negative amplifies moves"))
    st.plotly_chart(f3, use_container_width=True, config={"displayModeBar":False})

chart_cols2 = st.columns(2)
with chart_cols2[0]:
    f4 = go.Figure([go.Bar(x=df_band["strike"], y=df_band["call_oi"], name="Call OI", marker_color=GOLD),
                    go.Bar(x=df_band["strike"], y=df_band["put_oi"],  name="Put OI",  marker_color=SILVER)])
    _vline(f4,spot,AMBER,"Spot")
    f4.update_layout(**chart_layout(title="Call vs Put OI · Peaks = support/resistance walls", barmode="group",
                                    legend=dict(orientation="h",y=1.08,x=0)))
    st.plotly_chart(f4, use_container_width=True, config={"displayModeBar":False})

with chart_cols2[1]:
    f5 = go.Figure([go.Scatter(x=df_band["strike"], y=df_band["call_iv"], mode="lines+markers",
                               name="Call IV", line=dict(color=GOLD)),
                    go.Scatter(x=df_band["strike"], y=df_band["put_iv"],  mode="lines+markers",
                               name="Put IV",  line=dict(color=SILVER))])
    _vline(f5,spot,AMBER,"Spot")
    f5.update_layout(**chart_layout(title="IV Smile · Shape tells you where fear/speculation is priced in",
                                    yaxis_title="IV %", legend=dict(orientation="h",y=1.08,x=0)))
    st.plotly_chart(f5, use_container_width=True, config={"displayModeBar":False})

# ══════════════════════════════════════════════════════════════════════
#  OPTION CHAIN TABLE
# ══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 📋 MCX Option Chain — ATM Band")
table_rows = []
for _, r in df_band.sort_values("strike").iterrows():
    K = r["strike"]
    table_rows.append({
        "Call OI":   f"{int(r['call_oi']):,}",
        "Call ΔOI":  f"{int(r['call_oi_chg']):,}",
        "Call IV %": f"{r['call_iv']:.1f}",
        "Call Δ":    f"{r['call_delta']:.3f}",
        "STRIKE":    f"{int(K)} {'◀ ATM' if K==m['atm'] else ''}",
        "Put Δ":     f"{r['put_delta']:.3f}",
        "Put IV %":  f"{r['put_iv']:.1f}",
        "Put ΔOI":   f"{int(r['put_oi_chg']):,}",
        "Put OI":    f"{int(r['put_oi']):,}",
    })
st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════
#  KEY LEVELS
# ══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 📍 Key Price Levels")
level_items = [
    ("🧲 Max Pain",   int(m["max_pain"]),   "#CE93D8", "Market tends to close near this strike on expiry day."),
    ("🛡 Support",    int(m["support"]),    GREEN,     "Highest put OI — strongest price floor in the chain."),
    ("🚧 Resistance", int(m["resistance"]), RED,       "Highest call OI — strongest price ceiling in the chain."),
    ("⚡ ATM Strike", int(m["atm"]),        BLUE,      "Strike closest to current spot price."),
]
if gf:
    level_items.append(("🔀 Gamma Flip", int(gf), PINK, "Below this = dealer hedging amplifies moves (trend zone)."))
level_cols = st.columns(len(level_items))
for col,(lbl,val,c,tip) in zip(level_cols,level_items):
    col.markdown(f"""
    <div style="background-color:{CARD};border-radius:8px;padding:10px 18px;
                border:1px solid {BORDER};border-bottom:3px solid {c};">
        <div style="font-size:11px;color:{MUTED};">{lbl}</div>
        <div style="font-size:22px;font-weight:700;color:{c};">{val}</div>
        <div class="explain-text">{tip}</div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  FUTURES ROLL — v3: 3-MONTH TERM STRUCTURE + CARRY ANOMALY + VOL/OI
# ══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-header">📦 Futures Roll Analysis — 3-Month Term Structure + Carry Anomaly</div>', unsafe_allow_html=True)

if roll:
    # ── Row 1: LTP across 3 months ──
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    for col, label, value, color, explain in [
        (r1c1, "Near Month LTP",  f'₹{roll["near_ltp"]:,.2f}',  GOLD,    "Price of the nearest futures contract."),
        (r1c2, "Next Month LTP",  f'₹{roll["next_ltp"]:,.2f}',  "#CE93D8","Price of the second-month futures contract."),
        (r1c3, "Far Month LTP",   f'₹{roll["far_ltp"]:,.2f}' if roll.get("has_far") else "N/A",
                                                                   CYAN,    "Price of the third-month contract — confirms term structure."),
        (r1c4, "Roll Spread (₹)", f'{roll["roll_spread"]:+,.2f}', roll["bias_color"], "How much next month costs above near month."),
    ]:
        col.markdown(f"""
        <div style="background:{CARD};border-radius:8px;padding:10px 14px;border:1px solid {BORDER};margin-bottom:6px;">
            <div style="font-size:10px;color:{MUTED};text-transform:uppercase;">{label}</div>
            <div style="font-size:18px;font-weight:700;color:{color};">{value}</div>
            <div class="explain-text">{explain}</div>
        </div>""", unsafe_allow_html=True)

    # ── Row 2: Slopes + structure ──
    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    for col, label, value, color, explain in [
        (r2c1, "Slope Near→Next (ann.%)", f'{roll["slope_near_next"]:+.2f}%', roll["bias_color"],
         METRIC_EXPLAIN["Slope Near→Next"]),
        (r2c2, "Slope Next→Far (ann.%)",  f'{roll["slope_next_far"]:+.2f}%' if roll.get("has_far") else "N/A",
                                           roll.get("ts_color", MUTED), METRIC_EXPLAIN["Slope Next→Far"]),
        (r2c3, "Rollover %",              f'{roll["rollover_pct"]} %', BLUE, METRIC_EXPLAIN["Rollover %"]),
        (r2c4, "Market Structure",        roll["bias"],                roll["bias_color"],
         "Overall contango vs backwardation read from near and next month prices."),
    ]:
        col.markdown(f"""
        <div style="background:{CARD};border-radius:8px;padding:10px 14px;border:1px solid {BORDER};margin-bottom:6px;">
            <div style="font-size:10px;color:{MUTED};text-transform:uppercase;">{label}</div>
            <div style="font-size:15px;font-weight:700;color:{color};">{value}</div>
            <div class="explain-text">{explain}</div>
        </div>""", unsafe_allow_html=True)

    # ── Term Structure Banner ──
    ts_col = roll.get("ts_color", CYAN)
    st.markdown(f"""
    <div style="background:{ts_col}11;border:1.5px solid {ts_col};border-radius:10px;
                padding:12px 18px;margin:8px 0 12px 0;">
        <div style="font-size:14px;font-weight:800;color:{ts_col};">{roll.get('ts_shape','')}</div>
        <div style="font-size:11px;color:{TEXT};margin-top:4px;">{roll.get('ts_desc','')}</div>
    </div>""", unsafe_allow_html=True)

    # ── Row 3: Carry Anomaly + Vol/OI + Rollover Velocity ──
    r3c1, r3c2, r3c3, r3c4 = st.columns(4)
    ca_color = RED if carry_anomaly>=1.5 else (GREEN if carry_anomaly<=0.5 else AMBER)
    for col, label, value, color, explain in [
        (r3c1, "Carry Anomaly",  f'{carry_anomaly:.2f}×',          ca_color, METRIC_EXPLAIN["Carry Anomaly"]),
        (r3c2, "Near Vol/OI",    f'{roll["near_vol_oi"]:.3f}',      AMBER if roll["near_vol_oi"]>0.4 else MUTED,
         METRIC_EXPLAIN["Near Vol/OI"]),
        (r3c3, "Rollover Vel",   f'{roll.get("rollover_velocity",0.8):.3f}', roll_vel_color,
         METRIC_EXPLAIN["Rollover Velocity"]),
        (r3c4, "Near Month OI",  f'{roll["near_oi"]:,}',            GOLD, "Total open interest in the near-month contract."),
    ]:
        col.markdown(f"""
        <div style="background:{CARD};border-radius:8px;padding:10px 14px;border:1px solid {BORDER};margin-bottom:6px;">
            <div style="font-size:10px;color:{MUTED};text-transform:uppercase;">{label}</div>
            <div style="font-size:18px;font-weight:700;color:{color};">{value}</div>
            <div class="explain-text">{explain}</div>
        </div>""", unsafe_allow_html=True)

    # Rollover velocity interpretation
    st.markdown(f"""
    <div style="background:{roll_vel_color}11;border:1px solid {roll_vel_color};border-radius:8px;
                padding:10px 16px;margin-bottom:12px;">
        <span style="font-size:12px;font-weight:700;color:{roll_vel_color};">{roll_vel_interp}</span>
    </div>""", unsafe_allow_html=True)

    # ── Charts: Term Structure | OI Bar | Intraday OI | Rollover Velocity ──
    fc1, fc2 = st.columns(2)
    with fc1:
        st.plotly_chart(build_term_structure_chart(roll), use_container_width=True, config={"displayModeBar":False})
    with fc2:
        f_roll = go.Figure([
            go.Bar(name="Near OI",  x=["Near","Next","Far"],
                   y=[roll["near_oi"],roll["next_oi"],roll.get("far_oi",0)],
                   marker_color=[GOLD,"#CE93D8",CYAN]),
            go.Bar(name="Volume", x=["Near","Next","Far"],
                   y=[roll["near_vol"],roll["next_vol"],roll.get("far_vol",0)],
                   marker_color=["rgba(184,150,12,0.5)","rgba(206,147,216,0.5)","rgba(8,145,178,0.5)"]),
        ])
        f_roll.add_annotation(text=roll["bias"], xref="paper", yref="paper", x=0.5, y=1.12,
                              showarrow=False, font=dict(color=roll["bias_color"],size=12))
        f_roll.update_layout(**chart_layout(title="OI & Volume — Near / Next / Far Month", barmode="group"),
                             legend=dict(orientation="h",y=1.02,x=0))
        st.plotly_chart(f_roll, use_container_width=True, config={"displayModeBar":False})

    fc3, fc4 = st.columns(2)
    with fc3:
        oi_hist = st.session_state["oi_history"].get(symbol, [])
        if len(oi_hist) >= 2:
            ts_v   = [r["ts"]       for r in oi_hist]
            near_v = [r["near_oi"]  for r in oi_hist]
            nxt_v  = [r["next_oi"]  for r in oi_hist]
            far_v  = [r.get("far_oi",0) for r in oi_hist]
            total  = [r["total_oi"] for r in oi_hist]
            f_oi   = go.Figure([
                go.Scatter(x=ts_v, y=total,  mode="lines", name="Total OI",  line=dict(color=CYAN,   width=2)),
                go.Scatter(x=ts_v, y=near_v, mode="lines", name="Near OI",   line=dict(color=GOLD,   width=1.5, dash="dot")),
                go.Scatter(x=ts_v, y=nxt_v,  mode="lines", name="Next OI",   line=dict(color="#CE93D8", width=1.5, dash="dot")),
                go.Scatter(x=ts_v, y=far_v,  mode="lines", name="Far OI",    line=dict(color=CYAN,   width=1.5, dash="dash")),
            ])
            f_oi.update_layout(**chart_layout(title="Intraday OI Curve — Today's Session"),
                               legend=dict(orientation="h",y=1.08,x=0),
                               xaxis_title="Time", yaxis_title="Open Interest")
        else:
            f_oi = go.Figure()
            f_oi.add_annotation(text="Collecting OI history… refresh a few times",
                                xref="paper",yref="paper",x=0.5,y=0.5,showarrow=False,
                                font=dict(color=MUTED,size=13))
            f_oi.update_layout(**chart_layout(title="Intraday OI Curve (Building…)"))
        st.plotly_chart(f_oi, use_container_width=True, config={"displayModeBar":False})

    with fc4:
        st.plotly_chart(build_rollover_velocity_chart(st.session_state["oi_history"], symbol),
                        use_container_width=True, config={"displayModeBar":False})
else:
    st.warning("Roll data unavailable — market may be closed or outside MCX trading hours (9 AM – 11:30 PM IST).")

# ══════════════════════════════════════════════════════════════════════
#  SECTION 8: OI REGIME · OI VELOCITY · IV HISTORY · COMBINED BIAS
# ══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-header">⚡ Section 8 — OI Regime · OI Velocity · IV History · Combined Bias Panel</div>', unsafe_allow_html=True)

sym_history  = _extract_sym_history(st.session_state["history"], symbol)
labels, c_bkt, p_bkt = _bucket_oi_15min(sym_history)
regime_info  = _oi_regime_info(c_bkt, p_bkt)
bias_info    = _combined_bias_info(c_bkt, p_bkt)
oi_vel       = compute_oi_velocity(st.session_state["history"], symbol)

st.markdown(f"""
<div class="regime-banner" style="background:{regime_info['bg']};border:1.5px solid {regime_info['border']};">
    <div class="regime-label" style="color:{regime_info['fg']};">{regime_info['label']}</div>
    <div class="regime-sub"   style="color:{regime_info['fg']};">{regime_info['sub']}</div>
</div>""", unsafe_allow_html=True)

alert_colors = {"NONE":GREEN,"WATCH":AMBER,"DANGER":RED}
alert_col    = alert_colors.get(oi_vel["alert_level"], MUTED)
vel_cols     = st.columns([1,1,2])
for col,(label,vel,zscore) in zip(vel_cols[:2],[
    ("Call OI Vel / tick", oi_vel["call_oi_velocity"], oi_vel["call_vel_zscore"]),
    ("Put OI Vel / tick",  oi_vel["put_oi_velocity"],  oi_vel["put_vel_zscore"]),
]):
    col_color = RED if zscore>=2.0 else (AMBER if zscore>=1.2 else (GREEN if zscore<=-1.2 else MUTED))
    col.markdown(f"""
    <div style="background:{CARD};border-radius:8px;padding:10px 14px;border:1px solid {BORDER};">
        <div style="font-size:11px;font-weight:700;color:{TEXT};text-transform:uppercase;">{label}</div>
        <div style="font-size:18px;font-weight:700;color:{col_color};">{vel:+,.0f}</div>
        <div style="font-size:12px;font-weight:600;color:{col_color};">z={zscore:+.2f}σ</div>
        <div class="explain-text">Rate of change per refresh tick; z-score vs session average.</div>
    </div>""", unsafe_allow_html=True)

with vel_cols[2]:
    st.markdown(f"""
    <div style="background:{CARD};border-radius:8px;padding:10px 14px;border:1px solid {BORDER};min-height:80px;">
        <span class="alert-text" style="color:{alert_col};">{oi_vel['alert_text']}</span>
    </div>""", unsafe_allow_html=True)

sec8_cols = st.columns(3)
with sec8_cols[0]:
    st.plotly_chart(build_iv_history_chart(sym_history), use_container_width=True, config={"displayModeBar":False})
with sec8_cols[1]:
    st.plotly_chart(_build_oi_vel_chart(sym_history,"CALL"), use_container_width=True, config={"displayModeBar":False})
with sec8_cols[2]:
    st.plotly_chart(_build_oi_vel_chart(sym_history,"PUT"),  use_container_width=True, config={"displayModeBar":False})

if bias_info:
    MATRIX = [
        ("Call ↑  Put ↑","PINNED",       BLUE,    "Both walls building → pin / range / max-pain gravity"),
        ("Call ↑  Put ↓","BULLISH",      GREEN,   "Ceiling stays, floor gone → slow drift up"),
        ("Call ↓  Put ↑","BEARISH",      RED,     "Ceiling gone, floor stays → slow drift down"),
        ("Call ↓  Put ↓","EXPANSION",    "#9333EA","All walls dissolving → breakout/breakdown risk"),
        ("Call ↑  Put ~","MILD BEARISH", AMBER,   "Ceiling heavy, floor neutral → capped / mild bear"),
        ("Call ~  Put ↑","MILD BULLISH", "#10B981","Floor solid, ceiling neutral → lifted / mild bull"),
    ]
    bc = bias_info["bc"]; c_z = bias_info["c_z"]; p_z = bias_info["p_z"]
    def _z_badge(label, z):
        col = RED if z>1.5 else (AMBER if z>0.5 else (GREEN if z<-1.5 else (AMBER if z<-0.5 else MUTED)))
        return f'<span style="background:{col}33;color:{col};border:1px solid {col};border-radius:6px;padding:2px 8px;font-size:12px;font-weight:700;margin-right:8px;">{label}: {z:+.2f}σ</span>'
    st.markdown(f"""
    <div style="background:{CARD};border:1px solid {bc};border-left:4px solid {bc};
                border-radius:10px;padding:16px 18px;margin-top:12px;">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:10px;">
            <span style="font-size:15px;font-weight:800;color:{bc};">{bias_info['bias']}</span>
            <div>{_z_badge("Call OI",c_z)}{_z_badge("Put OI",p_z)}</div>
        </div>
        <div style="font-size:10px;color:{MUTED};font-weight:700;text-transform:uppercase;margin-bottom:8px;">
            📋 6-Scenario Reference — which box is currently active is highlighted
        </div>
    </div>""", unsafe_allow_html=True)
    bias_cols = st.columns(6)
    for col,(combo,blabel,bcolor,bdesc) in zip(bias_cols, MATRIX):
        is_active = blabel in bias_info["bias"]
        col.markdown(f"""
        <div class="bias-cell" style="background:{bcolor}{'22' if is_active else '11'};
                    border:{'2px' if is_active else '1px'} solid {bcolor};">
            <div style="font-size:11px;font-weight:700;color:{bcolor};font-family:monospace;margin-bottom:2px;">{combo}</div>
            <div style="font-size:12px;font-weight:800;color:{bcolor};margin-bottom:3px;">{blabel}</div>
            <div style="font-size:10px;color:{TEXT};line-height:1.4;">{bdesc}</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(f"""
<div style="text-align:center;font-size:10px;color:{MUTED};padding:10px;">
    Commodities Options Analysis Dashboard v3.0 · Streamlit Edition<br>
    Data: {'Dhan API (MCX)' if CFG.USE_DHAN else 'DEMO MODE'} ·
    Auto-refresh: {AUTO_REFRESH_SECONDS}s ·
    History ticks: {sum(len(v) for v in st.session_state['history'].values() if isinstance(v,list))} ·
    IV Smile ticks: {sum(len(v) for v in st.session_state['iv_smile_history'].values() if isinstance(v,list))}
</div>""", unsafe_allow_html=True)
