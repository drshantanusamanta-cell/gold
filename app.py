"""
╔══════════════════════════════════════════════════════════════════════╗
║  Commodities Options Analysis Dashboard v2.0 (GOLD & SILVER)        ║
║  Streamlit Edition — Deploy on Streamlit Community Cloud             ║
║  Data: Dhan API (primary) | Demo Mode (fallback)                    ║
║  NEW: Auto-Updating Monthly Scrips from Dhan Master CSV             ║
║  OI Velocity · Gamma Flip · IV Rank · ATM Pressure                  ║
║  IV History Chart · OI Regime · Bias Panel · History Persist        ║
║  Score: 0–100 | Strategy Engine | Auto-refresh every 60 s           ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os, json, time, warnings, csv as _csv, io, requests
from datetime import date, timedelta, datetime

import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import streamlit as st
import plotly.graph_objs as go

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Commodity Options Dashboard v2",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────
#  LOAD CREDENTIALS (Streamlit secrets → env vars → demo mode)
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
RISK_FREE_RATE        = 0.065
ATM_BAND              = 10
AUTO_REFRESH_SECONDS  = 60

DHAN_SECURITY = {
    "GOLD":    {"id": 114, "seg": "MCX_COMM", "step": 100},
    "GOLDM":   {"id": 117, "seg": "MCX_COMM", "step": 100},
    "SILVER":  {"id": 115, "seg": "MCX_COMM", "step": 1000},
    "SILVERM": {"id": 122, "seg": "MCX_COMM", "step": 1000},
}
COMMODITY_SYMBOLS = ["GOLD", "GOLDM", "SILVER", "SILVERM"]

# ─────────────────────────────────────────────────────────────────────
#  AUTOMATIC MONTHLY SCRIPS UPDATE
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def get_dynamic_futures_ids():
    """
    Automatically fetches the current month's near and next futures IDs 
    from Dhan's daily master scrip file.
    """
    url = "https://images.dhan.co/api-data/api-scrip-master.csv"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        
        # Standardize column names to uppercase just in case
        df.columns = [c.upper() for c in df.columns]
        
        # Filter for MCX Futures only
        df_mc = df[(df['SEM_EXM_EXCH_ID'] == 'MCX_COMM') & (df['SEM_INSTRUMENT_NAME'] == 'FUTCOM')]
        
        id_map = {}
        for sym in COMMODITY_SYMBOLS:
            # Find contracts that start with the symbol name exactly
            if sym == "GOLD":
                df_sym = df_mc[df_mc['SEM_TRADING_SYMBOL'].str.match(r'^GOLD\d')]
            elif sym == "SILVER":
                df_sym = df_mc[df_mc['SEM_TRADING_SYMBOL'].str.match(r'^SILVER\d')]
            else:
                df_sym = df_mc[df_mc['SEM_TRADING_SYMBOL'].str.startswith(sym)]
            
            # Drop any rows with missing expiry codes, sort by expiry date
            df_sym = df_sym.dropna(subset=['SEM_EXPIRY_CODE'])
            df_sym = df_sym.sort_values('SEM_EXPIRY_CODE')
            
            if len(df_sym) >= 2:
                # Grab the first two (Near and Next month)
                near_id = int(df_sym.iloc[0]['SEM_SMST_SECURITY_ID'])
                next_id = int(df_sym.iloc[1]['SEM_SMST_SECURITY_ID'])
                id_map[sym] = [near_id, next_id]
            elif len(df_sym) == 1:
                # Fallback if only one contract is available
                id_map[sym] = [int(df_sym.iloc[0]['SEM_SMST_SECURITY_ID'])]
                
        print(f"[Auto-Scrips] Successfully mapped futures IDs: {id_map}")
        return id_map
        
    except Exception as e:
        print(f"[Auto-Scrips] Failed to fetch dynamic IDs: {e}")
        return {}

# ─────────────────────────────────────────────────────────────────────
#  COLORS
# ─────────────────────────────────────────────────────────────────────
BG         = "#0D0D1A"
CARD       = "#1A1A2E"
TEXT       = "#E0E0E0"
ACCENT     = "#D4AF37"
MUTED      = "#666680"
GOLD       = "#FFD700"
SILVER     = "#C0C0C0"
GREEN      = "#00E676"
RED        = "#FF5252"
AMBER      = "#FFD600"
BLUE       = "#64B5F6"
CYAN       = "#00E5FF"
PINK       = "#FF4081"
BORDER     = "#2a2a3e"
SECTION_BG = "#1f1f38"

METRIC_EXPLAIN = {
    "EV Ratio":     "Call vs put time-value; >1.2 = bulls paying more (bullish), <0.8 = bears paying more (bearish).",
    "Net Delta":    "Overall directional bias from open positions; positive = net bullish, negative = net bearish.",
    "GEX":          "Gamma Exposure — positive GEX pins the market, negative GEX amplifies moves.",
    "Vanna":        "How delta changes when IV moves; positive = rising IV helps bulls.",
    "Momentum":     "Fresh money entering calls vs puts; positive = new bullish bets, negative = fresh bearish.",
    "Vega Skew":    "Call vega vs put vega; >1 = calls more IV-sensitive (bullish tone).",
    "G/T Ratio":    "Gamma-to-Theta ratio; high = market is unstable and trending.",
    "PCR":          "Put-Call Ratio; >1 = more puts, <0.7 = excessive calls (potential top).",
    "Max Pain":     "Strike where option writers lose least; market gravitates here near expiry.",
    "ATM Pressure": "Near-ATM put OI change vs call OI change; positive = support building.",
    "Skew Slope":   "Put IV slope vs call IV slope; high = fear of downside moves.",
    "IV Rank":      "ATM IV rank within smile range (0=low, 100=high). High IV favors premium selling.",
    "Gamma Flip":   "Strike where cumulative GEX crosses zero. Below flip = trend amplification zone.",
    "Wall Width":   "Distance between highest put OI strike (support) and highest call OI (resistance).",
    "Near OI %":    "Share of OI concentrated near ATM; high = strong pin, low = diffuse positioning.",
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
    out = {}
    for sym, ticks in history.items():
        if not isinstance(ticks, list):
            continue
        kept = [t for t in ticks if isinstance(t, dict) and str(t.get("ts", "")).startswith(today)]
        out[sym] = kept
    return out

def load_history_from_disk() -> dict:
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            pruned = _prune_to_today(raw)
            total  = sum(len(v) for v in pruned.values() if isinstance(v, list))
            print(f"[History] Loaded {total} ticks from disk")
            return pruned
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
#  UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────
def safe_num(x, d=0.0):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return d
        return float(x)
    except Exception:
        return d

def _zscore(arr, window):
    if len(arr) < 2:
        return 0.0
    w    = arr[-window:]
    mean = w.mean()
    std  = w.std()
    if std < 1e-9:
        return 0.0
    return float((w[-1] - mean) / std)

def _extract_sym_history(history, symbol):
    if isinstance(history, dict):
        ticks = history.get(symbol, [])
        if ticks:
            return ticks
        sym_keys = [k for k in history if isinstance(history[k], list)]
        if sym_keys:
            return history[max(sym_keys, key=lambda k: len(history[k]))]
        return []
    if isinstance(history, list):
        return history
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
    d1    = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2    = d1 - sigma * np.sqrt(T)
    nd1   = norm.pdf(d1)
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
    sec     = DHAN_SECURITY.get(symbol, DHAN_SECURITY["GOLD"])
    headers = {"access-token": CFG.DHAN_ACCESS_TOKEN, "client-id": str(CFG.DHAN_CLIENT_ID), "Content-Type": "application/json"}
    try:
        resp    = requests.post("https://api.dhan.co/v2/optionchain/expirylist", headers=headers,
                                json={"UnderlyingScrip": sec["id"], "UnderlyingSeg": sec["seg"]}, timeout=15)
        expiries = resp.json().get("data", [])
        today    = date.today().isoformat()
        return [e for e in expiries if e >= today]
    except Exception as e:
        print(f"[Dhan] Expiry list error: {e}")
        return []

def fetch_dhan_option_chain(symbol: str = "GOLD", expiry: str = None):
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
        ce = chain.get("ce", {}) or {}
        pe = chain.get("pe", {}) or {}
        cg = ce.get("greeks", {}) or {}
        pg = pe.get("greeks", {}) or {}
        rows.append({
            "strike": K,
            "call_ltp": float(ce.get("last_price", 0) or 0), "call_oi": int(ce.get("oi", 0) or 0),
            "call_prev_oi": int(ce.get("previous_oi", 0) or 0),
            "call_oi_chg": int(ce.get("oi", 0) or 0) - int(ce.get("previous_oi", 0) or 0),
            "call_vol": int(ce.get("volume", 0) or 0),
            "call_bid": float(ce.get("top_bid_price", 0) or 0), "call_ask": float(ce.get("top_ask_price", 0) or 0),
            "call_iv": float(ce.get("implied_volatility", 0) or 0),
            "call_delta": float(cg.get("delta", 0) or 0), "call_gamma": float(cg.get("gamma", 0) or 0),
            "call_theta": float(cg.get("theta", 0) or 0), "call_vega": float(cg.get("vega", 0) or 0),
            "put_ltp": float(pe.get("last_price", 0) or 0), "put_oi": int(pe.get("oi", 0) or 0),
            "put_prev_oi": int(pe.get("previous_oi", 0) or 0),
            "put_oi_chg": int(pe.get("oi", 0) or 0) - int(pe.get("previous_oi", 0) or 0),
            "put_vol": int(pe.get("volume", 0) or 0),
            "put_bid": float(pe.get("top_bid_price", 0) or 0), "put_ask": float(pe.get("top_ask_price", 0) or 0),
            "put_iv": float(pe.get("implied_volatility", 0) or 0),
            "put_delta": float(pg.get("delta", 0) or 0), "put_gamma": float(pg.get("gamma", 0) or 0),
            "put_theta": float(pg.get("theta", 0) or 0), "put_vega": float(pg.get("vega", 0) or 0),
        })
    df = pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)
    return df, spot, expiry

def fetch_futures_roll(symbol: str = "GOLD") -> dict:
    if not CFG.USE_DHAN:
        return {}
    
    # Fetch the dynamic IDs automatically
    futcom_ids = get_dynamic_futures_ids()
    ids = futcom_ids.get(symbol, [])
    
    if len(ids) < 2:
        print(f"[Dhan] Could not find 2 valid futures contracts for {symbol}")
        return {}
        
    near_id, next_id = ids[0], ids[1]
    
    headers = {"access-token": CFG.DHAN_ACCESS_TOKEN, "client-id": str(CFG.DHAN_CLIENT_ID), "Content-Type": "application/json"}
    try:
        resp     = requests.post("https://api.dhan.co/v2/marketquote", headers=headers,
                                 json={"securities": {"MCX_COMM": [str(near_id), str(next_id)]}}, timeout=10)
        d        = resp.json().get("data", {})
        near     = d.get(str(near_id), {})
        nxt      = d.get(str(next_id), {})
        near_ltp = float(near.get("last_price",    0) or 0)
        next_ltp = float(nxt.get("last_price",     0) or 0)
        near_oi  = int(near.get("open_interest",   0) or 0)
        next_oi  = int(nxt.get("open_interest",    0) or 0)
        near_vol = int(near.get("volume",           0) or 0)
        next_vol = int(nxt.get("volume",            0) or 0)
        total_oi        = near_oi + next_oi
        roll_spread     = round(next_ltp - near_ltp, 2)
        roll_spread_pct = round((roll_spread / near_ltp * 100) if near_ltp else 0, 3)
        rollover_pct    = round((next_oi / total_oi * 100) if total_oi else 0, 1)
        if roll_spread > 0:   bias, bias_color = "CONTANGO ▲  Bullish Carry", "#00E676"
        elif roll_spread < 0: bias, bias_color = "BACKWARDATION ▼  Delivery Pressure", "#FF5252"
        else:                 bias, bias_color = "FLAT", "#FFD600"
        return {"near_ltp": near_ltp, "next_ltp": next_ltp, "near_oi": near_oi, "next_oi": next_oi,
                "near_vol": near_vol, "next_vol": next_vol, "roll_spread": roll_spread,
                "roll_spread_pct": roll_spread_pct, "rollover_pct": rollover_pct,
                "bias": bias, "bias_color": bias_color}
    except Exception as e:
        print(f"[Dhan] Roll error: {e}")
        return {}

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
    strikes = np.arange(atm - 15 * step, atm + 16 * step, step)
    T       = 10 / 365.0
    r       = RISK_FREE_RATE
    vix     = 18.5 + np.random.normal(0, 1)
    rows = []
    for K in strikes:
        mono    = (K - spot) / spot
        iv_c    = max(0.05, (vix / 100) + 0.015 * mono**2 + abs(mono) * 0.04 + np.random.normal(0, 0.005))
        iv_p    = max(0.05, (vix / 100) + 0.025 * mono**2 - mono * 0.03 + np.random.normal(0, 0.005))
        c_price = _bs_price(spot, K, T, r, iv_c, "CE")
        p_price = _bs_price(spot, K, T, r, iv_p, "PE")
        cd, cg, ct, cv   = _bs_greeks(spot, K, T, r, iv_c, "CE")
        pd2, pg, pt, pv  = _bs_greeks(spot, K, T, r, iv_p, "PE")
        call_oi_fac = max(0, 2 + mono * 8) * np.random.lognormal(0, 0.4)
        put_oi_fac  = max(0, 2 - mono * 9) * np.random.lognormal(0, 0.4)
        rows.append({
            "strike": K, "call_ltp": round(max(0.05, c_price + np.random.normal(0, 0.3)), 2),
            "call_oi": int(max(10, call_oi_fac * 800)), "call_oi_chg": int(np.random.normal(20, 150)),
            "call_vol": int(abs(np.random.normal(300, 150))),
            "call_bid": round(max(0.05, c_price - 2.0), 2), "call_ask": round(c_price + 2.0, 2),
            "call_iv": round(iv_c * 100, 2), "call_delta": round(cd, 4),
            "call_gamma": round(cg, 6), "call_theta": round(ct, 4), "call_vega": round(cv, 4),
            "put_ltp": round(max(0.05, p_price + np.random.normal(0, 0.3)), 2),
            "put_oi": int(max(10, put_oi_fac * 1000)), "put_oi_chg": int(np.random.normal(-10, 180)),
            "put_vol": int(abs(np.random.normal(350, 200))),
            "put_bid": round(max(0.05, p_price - 2.0), 2), "put_ask": round(p_price + 2.0, 2),
            "put_iv": round(iv_p * 100, 2), "put_delta": round(pd2, 4),
            "put_gamma": round(pg, 6), "put_theta": round(pt, 4), "put_vega": round(pv, 4),
        })
    expiry = (date.today() + timedelta(days=10)).strftime("%Y-%m-%d")
    return pd.DataFrame(rows), round(spot, 2), expiry

def demo_futures_roll(symbol: str = "GOLD") -> dict:
    near_ltp = (93500.0 if "GOLD" in symbol else 96500.0) + np.random.normal(0, 80)
    spread   = abs(np.random.normal(120, 40))
    next_ltp = near_ltp + spread
    near_oi  = int(np.random.normal(18000, 2000))
    next_oi  = int(np.random.normal(4500, 800))
    total_oi = near_oi + next_oi
    return {"near_ltp": round(near_ltp, 2), "next_ltp": round(next_ltp, 2),
            "near_oi": near_oi, "next_oi": next_oi,
            "near_vol": int(np.random.normal(5000, 500)), "next_vol": int(np.random.normal(900, 200)),
            "roll_spread": round(next_ltp - near_ltp, 2),
            "roll_spread_pct": round((next_ltp - near_ltp) / near_ltp * 100, 3),
            "rollover_pct": round(next_oi / total_oi * 100, 1),
            "bias": "CONTANGO ▲  Bullish Carry", "bias_color": "#00E676"}

def get_option_chain(symbol: str = "GOLD", expiry: str = None):
    if CFG.USE_DHAN:
        df, spot, exp = fetch_dhan_option_chain(symbol, expiry)
        if not df.empty:
            return df, spot, exp, "Dhan API (MCX)"
    df, spot, exp = fetch_demo_option_chain(symbol)
    return df, spot, exp, "DEMO MODE (Commodities)"

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

def compute_gamma_flip(df_band, spot):
    if df_band.empty:
        return None
    strikes = sorted(df_band["strike"].unique())
    cum_gex = 0.0
    for K in strikes:
        row = df_band[df_band["strike"] == K]
        if row.empty:
            continue
        gex_k = (float(row["call_gamma"].values[0]) * float(row["call_oi"].values[0]) -
                 float(row["put_gamma"].values[0])  * float(row["put_oi"].values[0])) * spot**2 * 0.01
        prev_gex = cum_gex
        cum_gex += gex_k
        if prev_gex > 0 and cum_gex <= 0:
            return K
        if prev_gex < 0 and cum_gex >= 0:
            return K
    return None

def compute_iv_rank(df_band):
    all_ivs = pd.concat([df_band["call_iv"], df_band["put_iv"]]).dropna()
    all_ivs = all_ivs[all_ivs > 0]
    if all_ivs.empty:
        return 50.0
    iv_min = all_ivs.min()
    iv_max = all_ivs.max()
    atm_iv = all_ivs.median()
    if iv_max == iv_min:
        return 50.0
    return round((atm_iv - iv_min) / (iv_max - iv_min) * 100, 1)

def compute_metrics(df, spot, symbol="GOLD"):
    if df.empty:
        return {}
    df_band, atm = select_atm_band(df, spot, symbol)
    step         = DHAN_SECURITY.get(symbol, DHAN_SECURITY["GOLD"])["step"]

    df_band["intr_c"] = np.maximum(0, spot - df_band["strike"])
    df_band["ev_c"]   = np.maximum(0, df_band["call_ltp"] - df_band["intr_c"])
    df_band["intr_p"] = np.maximum(0, df_band["strike"] - spot)
    df_band["ev_p"]   = np.maximum(0, df_band["put_ltp"] - df_band["intr_p"])

    ev_sum_c  = df_band["ev_c"].sum()
    ev_sum_p  = df_band["ev_p"].sum()
    ev_ratio  = ev_sum_c / ev_sum_p if ev_sum_p > 0 else 1.0

    net_delta = ((df_band["call_oi"] * df_band["call_delta"]).sum() +
                 (df_band["put_oi"]  * df_band["put_delta"]).sum())
    net_gamma = ((df_band["call_oi"] * df_band["call_gamma"]).sum() +
                 (df_band["put_oi"]  * df_band["put_gamma"]).sum())
    net_theta = ((df_band["call_oi"] * df_band["call_theta"]).sum() +
                 (df_band["put_oi"]  * df_band["put_theta"]).sum())

    gex   = ((df_band["call_oi"] * df_band["call_gamma"]).sum() -
              (df_band["put_oi"]  * df_band["put_gamma"]).sum()) * spot**2 * 0.01
    vanna = ((df_band["call_oi"] * df_band["call_vega"] * df_band["call_delta"]).sum() +
             (df_band["put_oi"]  * df_band["put_vega"]  * df_band["put_delta"]).sum()) / max(spot, 1)

    gt_ratio = abs(net_gamma) / max(abs(net_theta), 1e-6)
    momentum = ((df_band["call_oi_chg"] * df_band["call_delta"]).sum() +
                (df_band["put_oi_chg"]  * df_band["put_delta"]).sum())

    sum_vega_c = (df_band["call_oi"] * df_band["call_vega"]).sum()
    sum_vega_p = (df_band["put_oi"]  * df_band["put_vega"]).sum()
    vega_skew  = sum_vega_c / sum_vega_p if sum_vega_p > 0 else 1.0

    total_coi = df["call_oi"].sum()
    total_poi = df["put_oi"].sum()
    pcr       = total_poi / total_coi if total_coi > 0 else 1.0
    max_pain  = compute_max_pain(df)

    atm_row = df_band[df_band["strike"] == atm]
    atm_iv  = float(((atm_row["call_iv"].values[0] if not atm_row.empty else 0) +
                     (atm_row["put_iv"].values[0]  if not atm_row.empty else 0)) / 2)

    support    = df_band.loc[df_band["put_oi"].idxmax(),  "strike"] if not df_band.empty else 0
    resistance = df_band.loc[df_band["call_oi"].idxmax(), "strike"] if not df_band.empty else 0
    wall_width = float(resistance - support) if resistance > support else float(step * 4)

    near_band      = df_band[df_band["strike"].between(atm - 3 * step, atm + 3 * step)]
    total_oi_band  = df_band["call_oi"].sum() + df_band["put_oi"].sum()
    near_oi_total  = near_band["call_oi"].sum() + near_band["put_oi"].sum()
    near_oi_conc   = near_oi_total / total_oi_band if total_oi_band > 0 else 0.5

    near_oichg_total = abs(near_band["call_oi_chg"]).sum() + abs(near_band["put_oi_chg"]).sum()
    band_oichg_total = abs(df_band["call_oi_chg"]).sum() + abs(df_band["put_oi_chg"]).sum()
    near_oichg_conc  = near_oichg_total / band_oichg_total if band_oichg_total > 0 else 0.5

    atm_pressure = float(near_band["put_oi_chg"].sum() - near_band["call_oi_chg"].sum())

    otm_puts  = df_band[df_band["strike"] < atm - step]
    otm_calls = df_band[df_band["strike"] > atm + step]
    if len(otm_puts) >= 2 and len(otm_calls) >= 2:
        put_slope  = float(np.polyfit(otm_puts["strike"], otm_puts["put_iv"], 1)[0])
        call_slope = float(np.polyfit(otm_calls["strike"], otm_calls["call_iv"], 1)[0])
        skew_slope = round(put_slope - call_slope, 4)
    else:
        skew_slope = 0.0

    iv_rank   = compute_iv_rank(df_band)
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
        "atm":        atm,
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
        "df_band":      df_band,
    }

# ─────────────────────────────────────────────────────────────────────
#  SCORING & STRATEGY ENGINE
# ─────────────────────────────────────────────────────────────────────
def compute_score(m):
    if not m: return 50.0
    score = 15
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
    return round(score, 1)

def strategy_recommendation(score, m, symbol="GOLD"):
    support    = m.get("support", 0)
    resistance = m.get("resistance", 0)
    atm        = m.get("atm", 0)
    step       = DHAN_SECURITY.get(symbol, DHAN_SECURITY["GOLD"])["step"]
    if score >= 85:   name, color = "Long Call / Bull Call Spread", "#00C853"
    elif score >= 70: name, color = "Bull Call Spread",             "#69F0AE"
    elif score >= 55: name, color = "Bull Put Spread (High Prob)",  "#B2FF59"
    elif score >= 45: name, color = "Iron Condor",                  "#FFD600"
    elif score >= 31: name, color = "Bear Call Spread",             "#FF6D00"
    elif score >= 16: name, color = "Bear Put Spread",              "#F44336"
    else:             name, color = "Long Put",                     "#B71C1C"
    if score >= 85:   legs = f"Buy {int(atm)} CE  |  Sell {int(resistance)} CE"
    elif score >= 70: legs = f"Buy {int(atm)} CE  |  Sell {int(atm + 2*step)} CE"
    elif score >= 55: legs = f"Sell {int(support + step)} PE  |  Buy {int(support - step)} PE"
    elif score >= 45: legs = (f"Sell {int(support+step)} PE / Buy {int(support-step)} PE  +  "
                               f"Sell {int(resistance-step)} CE / Buy {int(resistance+step)} CE")
    elif score >= 31: legs = f"Sell {int(atm)} CE  |  Buy {int(atm + 2*step)} CE"
    elif score >= 16: legs = f"Buy {int(atm)} PE  |  Sell {int(atm - 2*step)} PE"
    else:             legs = f"Buy {int(support - step)} PE"
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
    c_vel = np.diff(call_oi)
    p_vel = np.diff(put_oi)
    if len(c_vel) < 2:
        return {"call_oi_velocity": 0, "put_oi_velocity": 0, "call_oi_accel": 0, "put_oi_accel": 0,
                "call_vel_zscore": 0, "put_vel_zscore": 0, "alert_level": "NONE",
                "alert_text": "Collecting data…", "n_ticks": len(sym_history)}
    c_accel  = float(c_vel[-1] - c_vel[-2]) if len(c_vel) >= 2 else 0.0
    p_accel  = float(p_vel[-1] - p_vel[-2]) if len(p_vel) >= 2 else 0.0
    window   = min(10, len(c_vel))
    c_vel_z  = _zscore(c_vel, window)
    p_vel_z  = _zscore(p_vel, window)
    max_z    = max(abs(c_vel_z), abs(p_vel_z))
    if max_z >= 2.0:
        alert_level = "DANGER"
        side        = "CALL" if abs(c_vel_z) > abs(p_vel_z) else "PUT"
        direction   = "surge" if (c_vel_z if side == "CALL" else p_vel_z) > 0 else "unwind"
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
    if len(sym_history) < 3:
        return [], [], []
    call_oi = np.array([safe_num(x.get("call_oi_total", 0)) for x in sym_history], dtype=float)
    put_oi  = np.array([safe_num(x.get("put_oi_total",  0)) for x in sym_history], dtype=float)
    if call_oi.max() == 0 and put_oi.max() == 0:
        nd  = np.array([safe_num(x.get("net_delta",    0)) for x in sym_history], dtype=float)
        mom = np.array([safe_num(x.get("oi_net_delta", 0)) for x in sym_history], dtype=float)
        call_oi = np.maximum(nd, 0) + np.maximum(mom, 0)
        put_oi  = np.maximum(-nd, 0) + np.maximum(-mom, 0)
    ts    = [x.get("ts", "") for x in sym_history]
    c_vel = np.diff(call_oi)
    p_vel = np.diff(put_oi)
    ts_v  = ts[1:]
    bc, bp = {}, {}
    for i, t in enumerate(ts_v):
        try:
            t_part = t.split("T")[-1] if "T" in t else t
            parts  = t_part.split(":")
            hh, mm = int(parts[0]), int(parts[1])
            label  = f"{hh:02d}:{(mm // 15) * 15:02d}"
        except Exception:
            label = t
        bc[label] = bc.get(label, 0.0) + float(c_vel[i])
        bp[label] = bp.get(label, 0.0) + float(p_vel[i])
    labels = sorted(bc.keys())
    return labels, [bc[l] for l in labels], [bp.get(l, 0.0) for l in labels]

def _oi_regime_info(c_bkt, p_bkt):
    if not c_bkt or not p_bkt:
        return {"label": "Collecting OI data for regime detection…", "sub": "", "bg": CARD, "fg": MUTED, "border": MUTED}
    c_arr  = np.array(c_bkt, dtype=float)
    p_arr  = np.array(p_bkt, dtype=float)
    c_std  = float(c_arr.std()) if c_arr.std() > 1e-9 else 1.0
    p_std  = float(p_arr.std()) if p_arr.std() > 1e-9 else 1.0
    recent_c = list(c_arr[-3:])
    recent_p = list(p_arr[-3:])
    avg_c  = abs(float(np.mean(recent_c))) if recent_c else 0.0
    avg_p  = abs(float(np.mean(recent_p))) if recent_p else 0.0
    buyer  = (avg_c > 1.0 * c_std) or (avg_p > 1.0 * p_std)
    seller = (avg_c <= 0.8 * c_std) and (avg_p <= 0.8 * p_std)
    if buyer and not seller:
        return {"label": "OPTION BUYER'S REGIME", "sub": "OI velocity elevated — directional participants active. Premium is expensive. Favour directional plays.", "bg": "#1a1a00", "fg": "#FFD600", "border": "#FFD600"}
    elif seller:
        return {"label": "OPTION SELLER'S REGIME", "sub": "OI velocity subdued — writers in control. Range-bound / premium decay favoured. Sell spreads or iron condors.", "bg": "#001a0d", "fg": "#00E676", "border": "#00E676"}
    else:
        return {"label": "TRANSITIONAL REGIME", "sub": "Mixed OI signals — neither buyers nor sellers clearly dominant. Wait for clarity.", "bg": "#001a2e", "fg": "#80CBC4", "border": "#80CBC4"}

def _combined_bias_info(c_bkt, p_bkt):
    if not c_bkt or not p_bkt:
        return None
    c_arr = np.array(c_bkt, dtype=float)
    p_arr = np.array(p_bkt, dtype=float)
    mean_c = float(c_arr.mean()); std_c = float(c_arr.std()) if c_arr.std() > 1e-9 else 1.0
    mean_p = float(p_arr.mean()); std_p = float(p_arr.std()) if p_arr.std() > 1e-9 else 1.0
    completed_c = c_arr[:-1] if len(c_arr) >= 2 else c_arr
    completed_p = p_arr[:-1] if len(p_arr) >= 2 else p_arr
    sig_c = float(np.mean(completed_c[-2:])) if len(completed_c) >= 2 else float(np.mean(completed_c)) if len(completed_c) else 0.0
    sig_p = float(np.mean(completed_p[-2:])) if len(completed_p) >= 2 else float(np.mean(completed_p)) if len(completed_p) else 0.0
    c_z = (sig_c - mean_c) / std_c
    p_z = (sig_p - mean_p) / std_p
    STRONG, WEAK = 0.8, 0.3
    c_up   = c_z >  STRONG;  c_down = c_z < -STRONG;  c_flat = abs(c_z) < WEAK
    p_up   = p_z >  STRONG;  p_down = p_z < -STRONG;  p_flat = abs(p_z) < WEAK
    if   c_up and p_up:      bias, bc = "PINNED — Range / Max-Pain Bias",          BLUE
    elif c_up and p_down:    bias, bc = "BULLISH — Upside Bias from Writers",       GREEN
    elif c_down and p_up:    bias, bc = "BEARISH — Downside Bias from Writers",     RED
    elif c_down and p_down:  bias, bc = "EXPANSION — Breakout / Breakdown Risk",    "#9333EA"
    elif c_up and p_flat:    bias, bc = "MILDLY BEARISH — Resistance Reinforcing",  AMBER
    elif p_up and c_flat:    bias, bc = "MILDLY BULLISH — Support Reinforcing",     "#10B981"
    else:                    bias, bc = "NEUTRAL — No Clear OI Signal",             MUTED
    return {"bias": bias, "bc": bc, "c_z": c_z, "p_z": p_z}

# ─────────────────────────────────────────────────────────────────────
#  CHARTS
# ─────────────────────────────────────────────────────────────────────
def chart_layout(**kw):
    return dict(paper_bgcolor=CARD, plot_bgcolor=CARD, font=dict(color=TEXT, size=11),
                margin=dict(l=40, r=20, t=55, b=38), height=340, **kw)

def score_gauge_fig(score):
    color = GREEN if score >= 70 else (AMBER if score >= 45 else RED)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, domain={"x": [0,1], "y": [0,1]},
        title={"text": "Market Score", "font": {"color": TEXT, "size": 12}},
        number={"font": {"color": color, "size": 36}},
        gauge={"axis": {"range": [0, 100], "tickcolor": "#444"}, "bar": {"color": color}, "bgcolor": CARD,
               "steps": [{"range": [0,30], "color": "#1a0000"}, {"range": [30,45], "color": "#1a0f00"},
                         {"range": [45,55], "color": "#141400"}, {"range": [55,70], "color": "#001a00"},
                         {"range": [70,100], "color": "#002600"}],
               "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.8, "value": score}},
    ))
    fig.update_layout(paper_bgcolor="#111", plot_bgcolor="#111", margin=dict(l=20, r=20, t=30, b=5), height=220)
    return fig

def build_iv_history_chart(sym_history):
    empty = go.Figure()
    empty.update_layout(**chart_layout(title="Cumulative Δ ATM IV — 15-min buckets"))
    if len(sym_history) < 2:
        return empty
    buckets = {}
    for x in sym_history:
        ts = x.get("ts", ""); iv = safe_num(x.get("atm_iv", 0))
        if iv <= 0: continue
        try:
            t_part = ts.split("T")[-1] if "T" in ts else ts
            parts  = t_part.split(":")
            hh, mm = int(parts[0]), int(parts[1])
            label  = f"{hh:02d}:{(mm // 15) * 15:02d}"
        except Exception:
            label = ts
        buckets[label] = iv
    if len(buckets) < 1:
        return empty
    labels = sorted(buckets.keys())
    vals   = [buckets[l] for l in labels]
    base   = vals[0]
    cum_d  = [v - base for v in vals]
    iv_delta  = cum_d[-1] if cum_d else 0
    direction = ("IV RISING" if iv_delta > 0.5 else ("IV FALLING" if iv_delta < -0.5 else "IV FLAT"))
    line_color = ("#059669" if iv_delta < -0.5 else ("#DC2626" if iv_delta > 0.5 else CYAN))
    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color="#555", opacity=0.5, annotation_text="open baseline", annotation_font_size=10)
    fig.add_trace(go.Scatter(x=labels, y=cum_d, mode="lines+markers", name="Cumul Δ ATM IV",
                             line=dict(color=line_color, width=2.5), marker=dict(size=6, color=line_color),
                             hovertemplate="<b>%{x}</b><br>Cumul Δ IV: %{y:+.2f}pp<extra></extra>"))
    lk = chart_layout(title=f"Cumul Δ ATM IV — 15-min | {direction}  ({iv_delta:+.2f}pp)")
    lk["yaxis"] = dict(title="Δ IV (pp)", gridcolor="#222")
    lk["xaxis"] = dict(title="15-min bucket", gridcolor="#222")
    fig.update_layout(**lk)
    return fig

def _build_oi_vel_chart(sym_history, side="CALL"):
    side_label = "Call" if side == "CALL" else "Put"
    line_color = CYAN if side == "CALL" else PINK
    band_color = "rgba(0,229,255,0.08)" if side == "CALL" else "rgba(255,64,129,0.08)"
    empty = go.Figure()
    empty.update_layout(**chart_layout(title=f"Δ {side_label} OI Velocity Z-Score — 15-min"))
    if len(sym_history) < 3:
        return empty
    call_oi = np.array([safe_num(x.get("call_oi_total", 0)) for x in sym_history], dtype=float)
    put_oi  = np.array([safe_num(x.get("put_oi_total",  0)) for x in sym_history], dtype=float)
    ts_raw  = [x.get("ts", "") for x in sym_history]
    vel = np.diff(call_oi if side == "CALL" else put_oi)
    ts_v = ts_raw[1:]
    def _bucket(t):
        try:
            parts = t.split("T")[-1].split(":")
            hh, mm = int(parts[0]), int(parts[1])
            return f"{hh:02d}:{(mm // 15) * 15:02d}"
        except Exception:
            return t
    buckets: dict = {}
    for i, t in enumerate(ts_v):
        lbl = _bucket(t)
        buckets[lbl] = buckets.get(lbl, 0.0) + float(vel[i])
    if not buckets:
        return empty
    labels_c = sorted(buckets.keys())
    arr_c    = np.array([buckets[l] for l in labels_c], dtype=float)
    mean_val = float(arr_c.mean())
    std_val  = float(arr_c.std()) if arr_c.std() > 1e-9 else 1.0
    z_arr    = (arr_c - mean_val) / std_val
    live_lbl = _bucket(ts_raw[-1])
    live_vel = sum(float(vel[i]) for i, t in enumerate(ts_v) if _bucket(t) == live_lbl)
    live_z   = (live_vel - mean_val) / std_val
    all_lbl  = list(labels_c); all_z = list(z_arr); is_live = [False] * len(labels_c)
    if live_lbl not in labels_c:
        all_lbl.append(live_lbl); all_z.append(live_z); is_live.append(True)
    else:
        all_z[-1] = live_z; is_live[-1] = True
    latest_z = all_z[-1]
    alert = (f"⚡ SURGE +{latest_z:.1f}σ" if latest_z >= 2.0 else
             f"⚠ ELEVATED +{latest_z:.1f}σ" if latest_z >= 1.2 else
             f"⚡ UNWIND {latest_z:.1f}σ" if latest_z <= -2.0 else
             f"↘ EASING {latest_z:.1f}σ"  if latest_z <= -1.2 else f"NORMAL {latest_z:+.1f}σ")
    n   = len(all_lbl)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=all_lbl + all_lbl[::-1], y=[2.0]*n + [-2.0]*n,
                             fill="toself", fillcolor=band_color,
                             line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip", showlegend=False))
    closed_x = [l for l, lv in zip(all_lbl, is_live) if not lv]
    closed_y = [z for z, lv in zip(all_z,   is_live) if not lv]
    live_x   = [l for l, lv in zip(all_lbl, is_live) if lv]
    live_y   = [z for z, lv in zip(all_z,   is_live) if lv]
    if closed_x:
        fig.add_trace(go.Scatter(x=closed_x, y=closed_y, mode="lines+markers",
                                 name=f"{side_label} OI Vel Z",
                                 line=dict(color=line_color, width=2.5), marker=dict(size=6, color=line_color),
                                 hovertemplate="<b>%{x}</b><br>Z: %{y:+.2f}σ<extra></extra>"))
    if live_x:
        if closed_x:
            fig.add_trace(go.Scatter(x=[closed_x[-1], live_x[0]], y=[closed_y[-1], live_y[0]],
                                     mode="lines", line=dict(color=line_color, width=1.5, dash="dot"),
                                     showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=live_x, y=live_y, mode="markers", name="🔴 Live (forming)",
                                 marker=dict(size=10, color=line_color, opacity=0.5, symbol="circle-open",
                                             line=dict(width=2, color=line_color))))
    for y_val, dash, col, ann in [(0, "dash", "#555", "mean"), (2, "dot", RED, "+2σ"),
                                   (1, "dot", AMBER, "+1σ"), (-1, "dot", GREEN, "−1σ"), (-2, "dot", GREEN, "−2σ")]:
        fig.add_hline(y=y_val, line_dash=dash, line_color=col, opacity=0.5,
                      annotation_text=ann, annotation_font_size=10)
    lk = chart_layout(title=f"Δ {side_label} OI Vel Z-Score — 15-min | {alert}")
    lk["yaxis"] = dict(title="Z-score (σ)", gridcolor="#222")
    lk["xaxis"] = dict(title="15-min bucket", gridcolor="#222")
    fig.update_layout(**lk)
    return fig

# ─────────────────────────────────────────────────────────────────────
#  INTRADAY OI RECORDER
# ─────────────────────────────────────────────────────────────────────
def record_intraday_oi(symbol: str, roll: dict, oi_history: dict):
    if not roll: return oi_history
    ts    = time.strftime("%H:%M")
    entry = {"ts": ts, "near_oi": roll.get("near_oi", 0), "next_oi": roll.get("next_oi", 0),
             "total_oi": roll.get("near_oi", 0) + roll.get("next_oi", 0)}
    hist  = oi_history.setdefault(symbol, [])
    if hist and hist[-1]["ts"] == ts:
        hist[-1] = entry
    else:
        hist.append(entry)
    if len(hist) > 600:
        oi_history[symbol] = hist[-600:]
    return oi_history

# ═════════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ═════════════════════════════════════════════════════════════════════

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    .stApp {{ background-color: {BG}; }}
    .main .block-container {{ padding-top: 1rem; max-width: 1400px; }}
    h1, h2, h3, h4 {{ color: {GOLD}; }}
    .stMarkdown, .stText {{ color: {TEXT}; }}
    
    /* Metric cards */
    div[data-testid="stMetric"] {{
        background-color: {CARD};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 10px 14px;
    }}
    div[data-testid="stMetric"] label {{
        font-size: 10px;
        color: {MUTED};
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
        font-size: 19px;
        font-weight: 700;
    }}
    
    /* Section headers */
    .section-header {{
        background-color: {SECTION_BG};
        color: {ACCENT};
        font-weight: 700;
        font-size: 13px;
        padding: 8px 16px;
        border-radius: 8px;
        margin-bottom: 10px;
        letter-spacing: 0.3px;
        border-left: 4px solid {ACCENT};
    }}
    
    /* Regime banner */
    .regime-banner {{
        border-radius: 10px;
        padding: 12px 20px;
        margin-bottom: 10px;
    }}
    .regime-label {{
        font-weight: 800;
        font-size: 16px;
        letter-spacing: 0.5px;
    }}
    .regime-sub {{
        font-weight: 500;
        font-size: 12px;
        opacity: 0.85;
        margin-top: 3px;
    }}
    
    /* Strategy box */
    .strat-box {{
        background-color: {CARD};
        border-radius: 10px;
        padding: 14px;
        border: 1px solid {BORDER};
        border-left-width: 4px;
        border-left-style: solid;
    }}
    
    /* Bias panel */
    .bias-cell {{
        border-radius: 8px;
        padding: 8px 10px;
        text-align: center;
    }}
    
    /* Alert text */
    .alert-text {{
        font-size: 12px;
        font-weight: 600;
        line-height: 1.5;
    }}
    
    /* Dataframe */
    .stDataFrame {{ background-color: {CARD}; }}
    
    /* Selectbox and buttons */
    .stSelectbox label, .stButton button {{
        color: {TEXT};
    }}
    .stButton button {{
        background-color: {ACCENT};
        color: black;
        font-weight: bold;
        border: none;
        border-radius: 6px;
    }}
    
    /* Divider */
    hr {{
        border-color: {BORDER};
    }}
</style>
""", unsafe_allow_html=True)

# ── Initialize session state ──────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state["history"] = load_history_from_disk()
if "oi_history" not in st.session_state:
    st.session_state["oi_history"] = {}
if "last_refresh" not in st.session_state:
    st.session_state["last_refresh"] = 0
if "is_owner" not in st.session_state:
    st.session_state["is_owner"] = False
if "owner_pw_attempt" not in st.session_state:
    st.session_state["owner_pw_attempt"] = ""
if "owner_login_error" not in st.session_state:
    st.session_state["owner_login_error"] = False

# ── SIDEBAR: Owner Login ──────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding: 8px 0 12px 0;">
        <span style="font-size:20px;">🌟</span>
        <div style="font-size:13px; font-weight:700; color:{GOLD}; margin-top:4px;">
            Commodities Dashboard
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    if st.session_state["is_owner"]:
        st.markdown(f"""
        <div style="background:#1a2e00; border:1.5px solid #00E676; border-radius:8px;
                    padding:10px 14px; text-align:center; margin-bottom:12px;">
            <div style="font-size:14px; font-weight:800; color:#00E676;">🔑 OWNER MODE</div>
            <div style="font-size:10px; color:#888; margin-top:2px;">Full controls unlocked</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔒 Log out", use_container_width=True):
            st.session_state["is_owner"] = False
            st.session_state["owner_login_error"] = False
            st.rerun()
    else:
        st.markdown(f"""
        <div style="background:#1a1a2e; border:1px solid {BORDER}; border-radius:8px;
                    padding:10px 14px; text-align:center; margin-bottom:12px;">
            <div style="font-size:12px; font-weight:700; color:{MUTED};">👁 VIEW-ONLY MODE</div>
            <div style="font-size:10px; color:#555; margin-top:2px;">Enter password to unlock owner controls</div>
        </div>
        """, unsafe_allow_html=True)
        pw_input = st.text_input("Owner Password", type="password", key="pw_field",
                                 placeholder="Enter owner password…")
        if st.button("🔑 Unlock Owner Mode", use_container_width=True):
            if pw_input == CFG.OWNER_PASSWORD:
                st.session_state["is_owner"] = True
                st.session_state["owner_login_error"] = False
                st.rerun()
            else:
                st.session_state["owner_login_error"] = True
        if st.session_state["owner_login_error"]:
            st.error("Incorrect password.")

    st.divider()
    st.markdown(f"""
    <div style="font-size:10px; color:{MUTED}; text-align:center; line-height:1.6;">
        Data source: {'Dhan API ✅' if CFG.USE_DHAN else 'DEMO MODE'}<br>
        Auto-refresh: {AUTO_REFRESH_SECONDS}s
    </div>
    """, unsafe_allow_html=True)

# ── Auto-refresh (60 seconds) ─────────────────────────────────────────
time_since = time.time() - st.session_state["last_refresh"]

# Use a placeholder for auto-refresh countdown
refresh_placeholder = st.empty()

# ── TITLE ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align: center; margin-bottom: 14px; border-bottom: 2px solid {ACCENT}; padding-bottom: 10px;">
    <h1 style="margin: 0; color: {GOLD}; font-size: 26px; font-weight: 800; letter-spacing: 1px;">
        🌟 Commodities Options Analysis v2
    </h1>
    <div style="font-size: 11px; color: {MUTED}; margin-top: 3px;">
        MCX GOLD & SILVER Intelligence · OI Velocity · Gamma Flip · IV Rank · Regime Detection
    </div>
</div>
""", unsafe_allow_html=True)

# ── TOP CONTROLS ──────────────────────────────────────────────────────
is_owner = st.session_state["is_owner"]

col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4, col_ctrl5 = st.columns([1.5, 1.5, 2, 1.5, 1])

with col_ctrl1:
    symbol = st.selectbox("COMMODITY", COMMODITY_SYMBOLS, index=0)

with col_ctrl2:
    # Fetch expiry list
    if CFG.USE_DHAN:
        expiries = fetch_dhan_expiry_list(symbol)
    else:
        expiries = [(date.today() + timedelta(days=10)).strftime("%Y-%m-%d")]

    if is_owner:
        expiry = st.selectbox("EXPIRY", expiries, index=0 if expiries else None,
                              help="Select expiry date" if expiries else "No expiries available")
        # Persist owner's expiry choice
        st.session_state["selected_expiry"] = expiry
    else:
        # Viewer: show whichever expiry the owner last selected (or nearest)
        expiry = st.session_state.get("selected_expiry", expiries[0] if expiries else "")
        st.markdown(f"""
        <div style="padding-top: 6px;">
            <div style="font-size: 10px; color: {MUTED}; text-transform: uppercase;
                        letter-spacing: 0.5px; margin-bottom: 4px;">EXPIRY</div>
            <div style="font-size: 14px; font-weight: 700; color: #80CBC4;">{expiry}</div>
        </div>
        """, unsafe_allow_html=True)

with col_ctrl3:
    st.markdown(f"""
    <div style="padding-top: 28px; font-size: 11px; color: {MUTED}; font-style: italic;">
        📡 Source: {'Dhan API (MCX) ✅' if CFG.USE_DHAN else 'DEMO MODE (Add credentials in secrets)'}
    </div>
    """, unsafe_allow_html=True)

with col_ctrl4:
    st.markdown(f"""
    <div style="padding-top: 28px; font-size: 11px; color: {MUTED};">
        🕐 Last updated: {time.strftime('%H:%M:%S')}
    </div>
    """, unsafe_allow_html=True)

with col_ctrl5:
    if is_owner:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        refresh_clicked = st.button("⟳ Refresh", use_container_width=True)
    else:
        refresh_clicked = False
        st.markdown(f"""
        <div style="padding-top: 32px; font-size: 10px; color: {MUTED}; text-align:center;">
            🔒 View Only
        </div>
        """, unsafe_allow_html=True)

# ── Auto-refresh logic ────────────────────────────────────────────────
if is_owner:
    auto_refresh = st.checkbox("Auto-refresh every 60s", value=True)
else:
    auto_refresh = True  # always auto-refresh for viewers, silently

if auto_refresh:
    if is_owner:
        # Show countdown only to owner
        remaining = max(0, AUTO_REFRESH_SECONDS - int(time_since))
        refresh_placeholder.markdown(
            f"<div style='text-align:center; font-size:11px; color:{MUTED};'>"
            f"⏳ Auto-refresh in {remaining}s</div>",
            unsafe_allow_html=True
        )
    if time_since >= AUTO_REFRESH_SECONDS:
        st.session_state["last_refresh"] = time.time()
        st.rerun()
    # Use st_autorefresh as backup for auto-refresh
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=AUTO_REFRESH_SECONDS * 1000, key="autorefresh")
    except ImportError:
        pass

if refresh_clicked:
    st.session_state["last_refresh"] = time.time()
    st.rerun()

# ── FETCH DATA ────────────────────────────────────────────────────────
df, spot, exp, source = get_option_chain(symbol, expiry if expiry else None)

if df.empty:
    st.error("No data available. Check API credentials or try again.")
    st.stop()

m = compute_metrics(df, spot, symbol)
score = compute_score(m)
strat = strategy_recommendation(score, m, symbol)
df_band = m.pop("df_band", df)

# Fetch futures roll
roll = fetch_futures_roll(symbol) if CFG.USE_DHAN else demo_futures_roll(symbol)
st.session_state["oi_history"] = record_intraday_oi(symbol, roll, st.session_state["oi_history"])

# Build and save history tick
ts_full = datetime.now().isoformat(timespec="seconds")
tick = {
    "ts":            ts_full,
    "symbol":        symbol,
    "spot":          spot,
    "atm_iv":        m.get("atm_iv", 0),
    "net_delta":     m.get("net_delta", 0),
    "oi_net_delta":  m.get("momentum", 0),
    "max_pain":      m.get("max_pain", 0),
    "support":       m.get("support", 0),
    "resistance":    m.get("resistance", 0),
    "gex":           m.get("gex", 0),
    "pcr":           m.get("pcr", 0),
    "atm_pressure":  m.get("atm_pressure", 0),
    "wall_width":    m.get("wall_width", 0),
    "gamma_flip":    m.get("gamma_flip", None),
    "iv_rank":       m.get("iv_rank", 50),
    "gt_ratio":      m.get("gt_ratio", 0),
    "call_oi_total": m.get("call_oi_total", 0),
    "put_oi_total":  m.get("put_oi_total",  0),
}
sym_hist = st.session_state["history"].setdefault(symbol, [])
# Avoid duplicate ticks within same minute
if not sym_hist or sym_hist[-1].get("ts", "")[:16] != ts_full[:16]:
    sym_hist.append(tick)
    if len(sym_hist) > 600:
        st.session_state["history"][symbol] = sym_hist[-600:]
    save_history_to_disk(st.session_state["history"])

# Decision log
log_rec = {k: m.get(k, 0) for k in _CSV_COLUMNS if k in m}
log_rec.update({"ts": ts_full, "symbol": symbol, "spot": spot, "expiry": exp, "score": score,
                "atm": m.get("atm", 0), "gamma_flip": m.get("gamma_flip", "")})
write_decision_log(log_rec)

# ── HEADER CARDS ROW ──────────────────────────────────────────────────
gf = m.get("gamma_flip")
theme_color = SILVER if "SILVER" in symbol else GOLD
iv_rank_color = RED if m.get("iv_rank", 50) > 70 else (GREEN if m.get("iv_rank", 50) < 30 else AMBER)
pcr_color = GREEN if m["pcr"] > 0.8 else RED
gf_color = RED if gf and spot < gf else GREEN
atm_pressure_color = GREEN if m["atm_pressure"] > 0 else RED

header_cols = st.columns(10)
header_data = [
    ("Commodity", symbol, theme_color),
    ("Spot Price", f'₹ {spot:,.2f}', "#FFFFFF"),
    ("Expiry", exp, MUTED),
    ("ATM IV", f'{m.get("atm_iv", 0):.2f} %', "#80CBC4"),
    ("IV Rank", f'{m.get("iv_rank", 50):.0f}', iv_rank_color),
    ("PCR", f'{m["pcr"]}', pcr_color),
    ("Max Pain", f'{int(m["max_pain"])}', "#CE93D8"),
    ("Gamma Flip", f'{int(gf)}' if gf else '—', gf_color),
    ("ATM Pressure", f'{int(m["atm_pressure"]):+,}', atm_pressure_color),
    ("Wall Width", f'{int(m["wall_width"]):,}', BLUE),
]
for col, (label, value, color) in zip(header_cols, header_data):
    col.metric(label, value)
    col.markdown(f"""<div style='font-size:9px; color:{color}; margin-top:-10px;'>
        <span style='color:{MUTED};'>{label}:</span> <b style='color:{color};'>{value}</b></div>""",
        unsafe_allow_html=True)

st.markdown("---")

# ── SCORE GAUGE + STRATEGY + METRICS ──────────────────────────────────
col_gauge, col_strat, col_metrics = st.columns([1, 1.2, 3])

with col_gauge:
    st.plotly_chart(score_gauge_fig(score), use_container_width=True, config={"displayModeBar": False})

with col_strat:
    st.markdown(f"""
    <div class="strat-box" style="border-left-color: {strat['color']};">
        <div style="font-size: 10px; color: {MUTED};">MARKET MODE</div>
        <div style="font-size: 16px; font-weight: 700; color: {strat['mode_color']}; margin-bottom: 10px;">
            {strat['market_mode']}
        </div>
        <div style="font-size: 10px; color: {MUTED};">STRATEGY</div>
        <div style="font-size: 14px; font-weight: 700; color: {strat['color']}; margin-bottom: 8px;">
            {strat['name']}
        </div>
        <div style="font-size: 10px; color: {MUTED};">EXECUTION</div>
        <div style="font-size: 12px; color: #CCC; font-family: monospace;">
            {strat['legs']}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_metrics:
    metric_items = [
        ("EV Ratio", f'{m["ev_ratio"]}', GREEN if m["ev_ratio"] >= 1.2 else RED, METRIC_EXPLAIN["EV Ratio"]),
        ("Net Delta", f'{int(m["net_delta"]):,}', GREEN if m["net_delta"] > 0 else RED, METRIC_EXPLAIN["Net Delta"]),
        ("Momentum", f'{int(m["momentum"]):,}', GREEN if m["momentum"] > 0 else RED, METRIC_EXPLAIN["Momentum"]),
        ("Vega Skew", f'{m["vega_skew"]}', GREEN if m["vega_skew"] >= 1.2 else RED, METRIC_EXPLAIN["Vega Skew"]),
        ("GEX", f'{m["gex"]:,.0f}', GREEN if m["gex"] > 0 else RED, METRIC_EXPLAIN["GEX"]),
        ("Vanna", f'{m["vanna"]:,.2f}', GREEN if m["vanna"] > 0 else RED, METRIC_EXPLAIN["Vanna"]),
        ("G/T Ratio", f'{m["gt_ratio"]}', BLUE, METRIC_EXPLAIN["G/T Ratio"]),
        ("Skew Slope", f'{m["skew_slope"]:.4f}', RED if m["skew_slope"] > 0.15 else MUTED, METRIC_EXPLAIN["Skew Slope"]),
        ("Support", f'{int(m["support"])}', GREEN, ""),
        ("Resistance", f'{int(m["resistance"])}', RED, ""),
        ("Near OI %", f'{m["near_oi_concentration"]*100:.0f}%', BLUE, METRIC_EXPLAIN["Near OI %"]),
        ("PCR", f'{m["pcr"]}', GREEN if m["pcr"] > 0.8 else RED, METRIC_EXPLAIN["PCR"]),
    ]
    # Display in grid of 4 columns x 3 rows
    for row_start in range(0, len(metric_items), 4):
        row_items = metric_items[row_start:row_start+4]
        cols = st.columns(4)
        for col, (label, value, color, explain) in zip(cols, row_items):
            col.markdown(f"""
            <div style="background-color: {CARD}; border-radius: 8px; padding: 10px 14px;
                        border: 1px solid {BORDER}; min-height: 80px;">
                <div style="font-size: 10px; color: {MUTED}; text-transform: uppercase; letter-spacing: 0.5px;">{label}</div>
                <div style="font-size: 19px; font-weight: 700; color: {color};">{value}</div>
                <div style="font-size: 9px; color: #888899; margin-top: 4px; font-style: italic; line-height: 1.3;">{explain}</div>
            </div>
            """, unsafe_allow_html=True)

# ── CHARTS ROW 1 ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">📊 Options Analysis Charts</div>', unsafe_allow_html=True)

chart_cols = st.columns(3)

def _vline(fig, val, color, label):
    if val:
        fig.add_vline(x=val, line_dash="dash", line_color=color, opacity=0.7,
                      annotation_text=label, annotation_font_size=10)

# Chart 1: Net OI
with chart_cols[0]:
    dv = df_band["call_oi"] - df_band["put_oi"]
    f1 = go.Figure(go.Bar(x=df_band["strike"], y=dv,
                          marker_color=[RED if v > 0 else GREEN for v in dv], name="Net OI"))
    _vline(f1, spot, AMBER, "Spot"); _vline(f1, gf, PINK, "γ-Flip")
    f1.update_layout(**chart_layout(title="Net OI (Call − Put) [Delta Proxy]"))
    st.plotly_chart(f1, use_container_width=True, config={"displayModeBar": False})

# Chart 2: OI Momentum
with chart_cols[1]:
    mv = df_band["call_oi_chg"] - df_band["put_oi_chg"]
    f2 = go.Figure(go.Bar(x=df_band["strike"], y=mv,
                          marker_color=[RED if v > 0 else GREEN for v in mv], name="OI Change"))
    _vline(f2, spot, AMBER, "Spot")
    f2.update_layout(**chart_layout(title="OI Momentum (Change in OI)"))
    st.plotly_chart(f2, use_container_width=True, config={"displayModeBar": False})

# Chart 3: GEX
with chart_cols[2]:
    gv = (df_band["call_oi"] * df_band["call_gamma"] - df_band["put_oi"] * df_band["put_gamma"]) * spot**2 * 0.01
    f3 = go.Figure(go.Bar(x=df_band["strike"], y=gv,
                          marker_color=[RED if v > 0 else GREEN for v in gv], name="GEX"))
    _vline(f3, spot, AMBER, "Spot"); _vline(f3, gf, PINK, "γ-Flip")
    f3.update_layout(**chart_layout(title="Gamma Exposure (GEX) per Strike"))
    st.plotly_chart(f3, use_container_width=True, config={"displayModeBar": False})

chart_cols2 = st.columns(2)

with chart_cols2[0]:
    f4 = go.Figure([
        go.Bar(x=df_band["strike"], y=df_band["call_oi"], name="Call OI", marker_color=GOLD),
        go.Bar(x=df_band["strike"], y=df_band["put_oi"],  name="Put OI",  marker_color=SILVER)
    ])
    _vline(f4, spot, AMBER, "Spot")
    f4.update_layout(**chart_layout(title="Call vs Put OI", barmode="group",
                                    legend=dict(orientation="h", y=1.08, x=0)))
    st.plotly_chart(f4, use_container_width=True, config={"displayModeBar": False})

with chart_cols2[1]:
    f5 = go.Figure([
        go.Scatter(x=df_band["strike"], y=df_band["call_iv"], mode="lines+markers", name="Call IV",
                   line=dict(color=GOLD)),
        go.Scatter(x=df_band["strike"], y=df_band["put_iv"],  mode="lines+markers", name="Put IV",
                   line=dict(color=SILVER))
    ])
    _vline(f5, spot, AMBER, "Spot")
    f5.update_layout(**chart_layout(title="IV Smile", yaxis_title="IV %",
                                    legend=dict(orientation="h", y=1.08, x=0)))
    st.plotly_chart(f5, use_container_width=True, config={"displayModeBar": False})

# ── OPTION CHAIN TABLE ────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📋 MCX Option Chain — ATM Band")

table_rows = []
for _, r in df_band.sort_values("strike").iterrows():
    K = r["strike"]
    table_rows.append({
        "Call OI": f"{int(r['call_oi']):,}",
        "Call ΔOI": f"{int(r['call_oi_chg']):,}",
        "Call IV %": f"{r['call_iv']:.1f}",
        "Call Δ": f"{r['call_delta']:.3f}",
        "STRIKE": f"{int(K)} {'◀ ATM' if K == m['atm'] else ''}",
        "Put Δ": f"{r['put_delta']:.3f}",
        "Put IV %": f"{r['put_iv']:.1f}",
        "Put ΔOI": f"{int(r['put_oi_chg']):,}",
        "Put OI": f"{int(r['put_oi']):,}",
    })
st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

# ── KEY LEVELS ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📍 Key Price Levels")

level_items = [
    ("🧲 Max Pain", int(m["max_pain"]), "#CE93D8", "Market tends to close near this on expiry"),
    ("🛡 Support", int(m["support"]), GREEN, "Highest put OI — strong floor"),
    ("🚧 Resistance", int(m["resistance"]), RED, "Highest call OI — strong ceiling"),
    ("⚡ ATM Strike", int(m["atm"]), BLUE, "At-the-money strike"),
]
if gf:
    level_items.append(("🔀 Gamma Flip", int(gf), PINK, "Below = dealer short-gamma → trend amplification"))

level_cols = st.columns(len(level_items))
for col, (lbl, val, c, tip) in zip(level_cols, level_items):
    col.markdown(f"""
    <div style="background-color: {CARD}; border-radius: 8px; padding: 10px 18px;
                border: 1px solid {BORDER}; border-bottom: 3px solid {c};">
        <div style="font-size: 11px; color: {MUTED};">{lbl}</div>
        <div style="font-size: 22px; font-weight: 700; color: {c};">{val}</div>
        <div style="font-size: 9px; color: #888899; margin-top: 3px; font-style: italic;">{tip}</div>
    </div>
    """, unsafe_allow_html=True)

# ── FUTURES ROLL ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📦 Futures Roll Analysis & Intraday OI Curve")

if roll:
    roll_cols = st.columns(4)
    roll_items = [
        ("Near LTP", f'₹ {roll["near_ltp"]:,.2f}', GOLD),
        ("Next LTP", f'₹ {roll["next_ltp"]:,.2f}', SILVER),
        ("Roll Spread", f'₹ {roll["roll_spread"]:+,.2f}', roll["bias_color"]),
        ("Spread %", f'{roll["roll_spread_pct"]:+.3f} %', roll["bias_color"]),
        ("Near Month OI", f'{roll["near_oi"]:,}', "#80CBC4"),
        ("Next Month OI", f'{roll["next_oi"]:,}', "#CE93D8"),
        ("Rollover %", f'{roll["rollover_pct"]} %', BLUE),
        ("Structure", roll["bias"], roll["bias_color"]),
    ]
    for i, (label, value, color) in enumerate(roll_items):
        with roll_cols[i % 4]:
            st.markdown(f"""
            <div style="background-color: {CARD}; border-radius: 8px; padding: 8px 12px;
                        border: 1px solid {BORDER}; min-height: 60px;">
                <div style="font-size: 10px; color: {MUTED}; text-transform: uppercase;">{label}</div>
                <div style="font-size: 16px; font-weight: 700; color: {color};">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    roll_chart_cols = st.columns(2)
    with roll_chart_cols[0]:
        f_roll = go.Figure([
            go.Bar(name="Near Month OI",  x=["Near", "Next"],
                   y=[roll["near_oi"], roll["next_oi"]], marker_color=[GOLD, "#CE93D8"]),
            go.Bar(name="Volume", x=["Near", "Next"],
                   y=[roll.get("near_vol", 0), roll.get("next_vol", 0)],
                   marker_color=["rgba(212,175,55,0.5)", "rgba(206,147,216,0.5)"]),
        ])
        f_roll.add_annotation(text=roll["bias"], xref="paper", yref="paper", x=0.5, y=1.12,
                              showarrow=False, font=dict(color=roll["bias_color"], size=12))
        f_roll.update_layout(**chart_layout(title="Near vs Next Month OI & Volume", barmode="group"),
                             legend=dict(orientation="h", y=1.02, x=0))
        st.plotly_chart(f_roll, use_container_width=True, config={"displayModeBar": False})

    with roll_chart_cols[1]:
        oi_hist = st.session_state["oi_history"].get(symbol, [])
        if len(oi_hist) >= 2:
            ts_v  = [r["ts"]       for r in oi_hist]
            near  = [r["near_oi"]  for r in oi_hist]
            nxt   = [r["next_oi"]  for r in oi_hist]
            total = [r["total_oi"] for r in oi_hist]
            f_oi  = go.Figure([
                go.Scatter(x=ts_v, y=total, mode="lines", name="Total OI", line=dict(color=CYAN,   width=2)),
                go.Scatter(x=ts_v, y=near,  mode="lines", name="Near OI",  line=dict(color=GOLD,   width=1.5, dash="dot")),
                go.Scatter(x=ts_v, y=nxt,   mode="lines", name="Next OI",  line=dict(color="#CE93D8", width=1.5, dash="dot")),
            ])
            f_oi.update_layout(**chart_layout(title="Intraday OI Curve (Today's Session)"),
                               legend=dict(orientation="h", y=1.08, x=0),
                               xaxis_title="Time", yaxis_title="Open Interest")
        else:
            f_oi = go.Figure()
            f_oi.add_annotation(text="Collecting OI history… refresh a few times",
                                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                                font=dict(color=MUTED, size=13))
            f_oi.update_layout(**chart_layout(title="Intraday OI Curve (Building…)"))
        st.plotly_chart(f_oi, use_container_width=True, config={"displayModeBar": False})
else:
    st.warning("Roll data unavailable")

# ── SECTION 8: INTELLIGENCE DASHBOARD ─────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">⚡ Section 8 — OI Regime · OI Velocity · IV History · Combined Bias Panel</div>', unsafe_allow_html=True)

# Compute Section 8 data
sym_history = _extract_sym_history(st.session_state["history"], symbol)
labels, c_bkt, p_bkt = _bucket_oi_15min(sym_history)
regime_info = _oi_regime_info(c_bkt, p_bkt)
bias_info   = _combined_bias_info(c_bkt, p_bkt)
oi_vel      = compute_oi_velocity(st.session_state["history"], symbol)

# Regime Banner
st.markdown(f"""
<div class="regime-banner" style="background: {regime_info['bg']}; border: 1.5px solid {regime_info['border']};">
    <div class="regime-label" style="color: {regime_info['fg']};">{regime_info['label']}</div>
    <div class="regime-sub" style="color: {regime_info['fg']};">{regime_info['sub']}</div>
</div>
""", unsafe_allow_html=True)

# OI Velocity Cards
alert_colors = {"NONE": GREEN, "WATCH": AMBER, "DANGER": RED}
alert_col = alert_colors.get(oi_vel["alert_level"], MUTED)

vel_cols = st.columns([1, 1, 2])
vel_items = [
    ("Call OI Vel / tick", oi_vel["call_oi_velocity"], oi_vel["call_vel_zscore"]),
    ("Put OI Vel / tick",  oi_vel["put_oi_velocity"],  oi_vel["put_vel_zscore"]),
]
for col, (label, vel, zscore) in zip(vel_cols[:2], vel_items):
    col_color = RED if zscore >= 2.0 else (AMBER if zscore >= 1.2 else (GREEN if zscore <= -1.2 else MUTED))
    col.markdown(f"""
    <div style="background-color: {CARD}; border-radius: 8px; padding: 10px 14px;
                border: 1px solid {BORDER};">
        <div style="font-size: 11px; font-weight: 700; color: {TEXT};
                    text-transform: uppercase; letter-spacing: 0.5px;">{label}</div>
        <div style="font-size: 18px; font-weight: 700; color: {col_color};">
            {vel:+,.0f}" if abs(vel) > 1 else "0"
        </div>
        <div style="font-size: 12px; font-weight: 600; color: {col_color};">z={zscore:+.2f}σ</div>
    </div>
    """, unsafe_allow_html=True)

with vel_cols[2]:
    st.markdown(f"""
    <div style="background-color: {CARD}; border-radius: 8px; padding: 10px 14px;
                border: 1px solid {BORDER}; min-height: 80px; display: flex; align-items: center;">
        <span class="alert-text" style="color: {alert_col};">{oi_vel['alert_text']}</span>
    </div>
    """, unsafe_allow_html=True)

# IV History + OI Velocity Charts
sec8_cols = st.columns(3)
with sec8_cols[0]:
    st.plotly_chart(build_iv_history_chart(sym_history), use_container_width=True, config={"displayModeBar": False})
with sec8_cols[1]:
    st.plotly_chart(_build_oi_vel_chart(sym_history, side="CALL"), use_container_width=True, config={"displayModeBar": False})
with sec8_cols[2]:
    st.plotly_chart(_build_oi_vel_chart(sym_history, side="PUT"), use_container_width=True, config={"displayModeBar": False})

# Combined Bias Panel
if bias_info:
    MATRIX = [
        ("Call ↑  Put ↑", "PINNED",       BLUE,     "Both walls building → pin / range / max-pain gravity"),
        ("Call ↑  Put ↓", "BULLISH",      GREEN,    "Ceiling stays, floor gone → slow drift up"),
        ("Call ↓  Put ↑", "BEARISH",      RED,      "Ceiling gone, floor stays → slow drift down"),
        ("Call ↓  Put ↓", "EXPANSION",    "#9333EA","All walls dissolving → breakout/breakdown risk"),
        ("Call ↑  Put ~", "MILD BEARISH", AMBER,    "Ceiling heavy, floor neutral → capped / mild bear"),
        ("Call ~  Put ↑", "MILD BULLISH", "#10B981","Floor solid, ceiling neutral → lifted / mild bull"),
    ]
    bc = bias_info["bc"]
    c_z = bias_info["c_z"]
    p_z = bias_info["p_z"]
    
    # Z-score badges
    def _z_badge_html(label, z):
        col = RED if z > 1.5 else (AMBER if z > 0.5 else (GREEN if z < -1.5 else (AMBER if z < -0.5 else MUTED)))
        return f"""<span style="background: {col}33; color: {col}; border: 1px solid {col};
                    border-radius: 6px; padding: 2px 8px; font-size: 12px;
                    font-weight: 700; margin-right: 8px; white-space: nowrap;">
                    {label}: {z:+.2f}σ</span>"""
    
    st.markdown(f"""
    <div style="background-color: {CARD}; border: 1px solid {bc}; border-left: 4px solid {bc};
                border-radius: 10px; padding: 16px 18px; margin-top: 12px;">
        <div style="display: flex; justify-content: space-between; align-items: center;
                    flex-wrap: wrap; gap: 8px; margin-bottom: 10px;">
            <span style="font-size: 15px; font-weight: 800; color: {bc};">{bias_info['bias']}</span>
            <div>{_z_badge_html("Call OI", c_z)}{_z_badge_html("Put OI", p_z)}</div>
        </div>
        <div style="font-size: 11px; font-weight: 700; color: {MUTED};
                    text-transform: uppercase; margin-bottom: 8px;">📋 6-Scenario Reference</div>
    </div>
    """, unsafe_allow_html=True)
    
    bias_cols = st.columns(6)
    for col, (combo, blabel, bcolor, bdesc) in zip(bias_cols, MATRIX):
        is_active = blabel in bias_info["bias"]
        col.markdown(f"""
        <div class="bias-cell" style="background: {bcolor}{'22' if is_active else '11'};
                    border: {'2px' if is_active else '1px'} solid {bcolor};">
            <div style="font-size: 11px; font-weight: 700; color: {bcolor};
                        font-family: monospace; margin-bottom: 2px;">{combo}</div>
            <div style="font-size: 12px; font-weight: 800; color: {bcolor}; margin-bottom: 3px;">{blabel}</div>
            <div style="font-size: 10px; color: {TEXT}; line-height: 1.4;">{bdesc}</div>
        </div>
        """, unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; font-size: 10px; color: {MUTED}; padding: 10px;">
    Commodities Options Analysis Dashboard v2.0 · Streamlit Edition<br>
    Data: {'Dhan API (MCX)' if CFG.USE_DHAN else 'DEMO MODE'} · 
    Auto-refresh: {AUTO_REFRESH_SECONDS}s · 
    History ticks: {sum(len(v) for v in st.session_state['history'].values() if isinstance(v, list))}
</div>
""", unsafe_allow_html=True)
