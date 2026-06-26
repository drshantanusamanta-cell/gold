"""
╔══════════════════════════════════════════════════════════════════════╗
║  Commodities Options Analysis Dashboard v4.0 (GOLDM & SILVERM)     ║
║  Streamlit Edition — Deploy on Streamlit Community Cloud            ║
║  Data: Dhan API (primary) | Demo Mode (fallback)                   ║
║  v4: 3-Month Term Structure · Gamma Regime · IV Smile Classifier   ║
║  Carry Anomaly · Rollover Velocity · Spot Fallback · LTP Fix       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os, json, time, warnings, csv as _csv, io, requests
from datetime import date, timedelta, datetime, timezone

_IST = timezone(timedelta(hours=5, minutes=30))
def now_ist() -> datetime:   return datetime.now(_IST)
def strftime_ist(fmt: str):  return now_ist().strftime(fmt)

import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import streamlit as st
import plotly.graph_objs as go

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Commodity Options Dashboard v4",
                   page_icon="🌟", layout="wide", initial_sidebar_state="collapsed")

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

RISK_FREE_RATE       = 0.065
ATM_BAND             = 20
AUTO_REFRESH_SECONDS = 60

DHAN_SECURITY = {
    "GOLDM":   {"id": 117, "seg": "MCX_COMM", "step": 100},
    "SILVERM": {"id": 122, "seg": "MCX_COMM", "step": 1000},
}
COMMODITY_SYMBOLS = ["GOLDM", "SILVERM"]

@st.cache_data(ttl=86400, show_spinner=False)
def get_dynamic_futures_ids():
    """Downloads Dhan master CSV and resolves near/next/far futures IDs per symbol."""
    url = "https://images.dhan.co/api-data/api-scrip-master.csv"
    try:
        resp = requests.get(url, timeout=15); resp.raise_for_status()
        df   = pd.read_csv(io.StringIO(resp.text))
        df.columns = [c.upper() for c in df.columns]
        df_mc = df[(df['SEM_EXM_EXCH_ID'] == 'MCX_COMM') & (df['SEM_INSTRUMENT_NAME'] == 'FUTCOM')]
        id_map = {}
        for sym in COMMODITY_SYMBOLS:
            if sym == "GOLDM":
                df_sym = df_mc[df_mc['SEM_TRADING_SYMBOL'].str.match(r'^GOLDM')]
            elif sym == "SILVERM":
                df_sym = df_mc[df_mc['SEM_TRADING_SYMBOL'].str.match(r'^SILVERM')]
            else:
                df_sym = df_mc[df_mc['SEM_TRADING_SYMBOL'].str.startswith(sym)]
            df_sym = df_sym.dropna(subset=['SEM_EXPIRY_CODE']).sort_values('SEM_EXPIRY_CODE')
            if len(df_sym) >= 3:
                id_map[sym] = [int(df_sym.iloc[i]['SEM_SMST_SECURITY_ID']) for i in range(3)]
            elif len(df_sym) >= 2:
                id_map[sym] = [int(df_sym.iloc[i]['SEM_SMST_SECURITY_ID']) for i in range(2)]
            elif len(df_sym) == 1:
                id_map[sym] = [int(df_sym.iloc[0]['SEM_SMST_SECURITY_ID'])]
        print(f"[Auto-Scrips] {id_map}")
        return id_map
    except Exception as e:
        print(f"[Auto-Scrips] Failed: {e}"); return {}

BG         = "#FFFFFF"; CARD       = "#F8FAFC"; TEXT       = "#1E293B"
ACCENT     = "#B8960C"; MUTED      = "#64748B"; GOLD       = "#B8960C"
SILVER     = "#475569"; GREEN      = "#059669"; RED        = "#DC2626"
AMBER      = "#D97706"; BLUE       = "#2563EB"; CYAN       = "#0891B2"
PINK       = "#DB2777"; BORDER     = "#E2E8F0"; SECTION_BG = "#F1F5F9"
PURPLE     = "#7C3AED"

METRIC_EXPLAIN = {
    "EV Ratio":          "Call vs put time-value; >1.2 = bulls paying more (bullish), <0.8 = bears paying more (bearish).",
    "Net Delta":         "Overall directional bias from open positions; positive = net bullish, negative = net bearish.",
    "GEX":               "Gamma Exposure — positive GEX pins the market, negative GEX amplifies moves.",
    "Vanna":             "How delta changes when IV moves; positive = rising IV helps bulls.",
    "Momentum":          "Fresh money entering calls vs puts; positive = new bullish bets, negative = fresh bearish.",
    "Vega Skew":         "Call vega vs put vega; >1 = calls more IV-sensitive (bullish tone).",
    "G/T Ratio":         "Gamma-to-Theta ratio; high = market is unstable and trending.",
    "PCR":               "Put-Call Ratio; >1 = more puts, <0.7 = excessive calls (potential top).",
    "Max Pain":          "Strike where option writers lose least; market gravitates here near expiry.",
    "ATM Pressure":      "Near-ATM put OI change vs call OI change; positive = support building.",
    "Skew Slope":        "Put IV slope vs call IV slope; high = fear of downside moves.",
    "IV Rank":           "ATM IV rank within smile range (0=low, 100=high). High IV favors premium selling.",
    "Gamma Flip":        "Strike where cumulative GEX crosses zero. Below flip = trend amplification zone.",
    "Wall Width":        "Distance between highest put OI strike (support) and highest call OI (resistance).",
    "Near OI %":         "Share of OI concentrated near ATM; high = strong pin, low = diffuse positioning.",
    "Roll Spread":       "Price difference between near and next futures — positive = contango (normal carry).",
    "Spread %":          "Roll spread as % of near-month price — measures carry cost in percentage terms.",
    "Rollover %":        "How much OI has shifted to next month — high % means expiry rollover well advanced.",
    "Term Structure":    "Shape of the futures curve across 3 months — steepening contango signals bullish carry.",
    "Carry Anomaly":     "Actual roll cost vs what IV implies it should cost — above 1.5 means futures pricing a big move.",
    "Rollover Velocity": "Rate of OI moving from near to next month — above 1.2 means longs are adding conviction.",
    "Near Vol/OI":       "Volume-to-OI ratio for near-month — above 0.3 means active fresh positioning.",
    "Slope Near→Next":   "Annualised carry from near to next month — positive = contango (bullish carry).",
    "Slope Next→Far":    "Annualised carry from next to far month — steeper = bullish acceleration.",
    "Gamma Regime":      "5-state market structure: PINNED (low vol) → TREND/EXPANSION (high vol) → FLIP ZONE (danger).",
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
    "put_wing_excess", "call_wing_excess",
    "roll_spread_pct", "rollover_pct", "ts_bias",
]

def _get_log_paths():
    base = os.path.join(LOG_DIR, f"decision_log_{date.today().isoformat()}")
    return base + ".jsonl", base + ".csv"

def _ensure_csv_header(csv_path):
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            _csv.DictWriter(f, fieldnames=_CSV_COLUMNS, extrasaction="ignore").writeheader()

def write_decision_log(record: dict):
    try:
        jl, cp = _get_log_paths(); _ensure_csv_header(cp)
        with open(jl, "a", encoding="utf-8") as f: f.write(json.dumps(record) + "\n")
        with open(cp, "a", newline="", encoding="utf-8") as f:
            _csv.DictWriter(f, fieldnames=_CSV_COLUMNS, extrasaction="ignore").writerow(
                {k: record.get(k, "") for k in _CSV_COLUMNS})
    except Exception as e: print(f"[DecisionLog] {e}")

def _prune_to_today(history):
    today = date.today().isoformat()
    return {s: [t for t in tks if isinstance(t, dict) and str(t.get("ts","")).startswith(today)]
            for s, tks in history.items() if isinstance(tks, list)}

def load_history_from_disk():
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f: raw = json.load(f)
            p = _prune_to_today(raw)
            print(f"[History] Loaded {sum(len(v) for v in p.values())} ticks"); return p
    except Exception as e: print(f"[History] Load error: {e}")
    return {}

def save_history_to_disk(history):
    try:
        tmp = HISTORY_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in history.items() if isinstance(v, list)}, f)
        os.replace(tmp, HISTORY_FILE)
    except Exception as e: print(f"[History] Save error: {e}")

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
    w = arr[-window:]; std = w.std()
    return float((w[-1] - w.mean()) / std) if std > 1e-9 else 0.0

def _extract_sym_history(history, symbol):
    if isinstance(history, dict):
        ticks = history.get(symbol, [])
        if ticks: return ticks
        sym_keys = [k for k in history if isinstance(history[k], list)]
        return history[max(sym_keys, key=lambda k: len(history[k]))] if sym_keys else []
    return history if isinstance(history, list) else []

# ─────────────────────────────────────────────────────────────────────
#  BLACK-SCHOLES
# ─────────────────────────────────────────────────────────────────────
def _bs_price(S, K, T, r, sigma, opt):
    if T <= 0 or sigma <= 0: return 0.0
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T)); d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2) if opt=="CE" else K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def _bs_greeks(S, K, T, r, sigma, opt):
    if T <= 0 or sigma <= 0: return 0,0,0,0
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T)); d2 = d1 - sigma*np.sqrt(T)
    nd1   = norm.pdf(d1)
    delta = norm.cdf(d1) if opt=="CE" else -norm.cdf(-d1)
    gamma = nd1 / (S*sigma*np.sqrt(T))
    theta = (-(S*nd1*sigma)/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2 if opt=="CE" else -d2)) / 365
    vega  = S*nd1*np.sqrt(T) / 100
    return delta, gamma, theta, vega

def _solve_iv(mkt, S, K, T, r, opt):
    if T <= 0 or mkt <= 0: return 0.0
    try: return brentq(lambda v: _bs_price(S,K,T,r,v,opt) - mkt, 1e-4, 5.0, xtol=1e-5, maxiter=100)
    except Exception: return 0.0

# ─────────────────────────────────────────────────────────────────────
#  DHAN FETCHERS
#  Cache option chain calls for 55s (just under the 60s refresh cycle).
#  This ensures the roll's _fetch_oi calls for near-expiry hit the cache
#  rather than making a live API call, keeping total Dhan API calls ≤ 3.
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=55, show_spinner=False)
def fetch_dhan_expiry_list(symbol="GOLDM"):
    sec     = DHAN_SECURITY.get(symbol, DHAN_SECURITY["GOLDM"])
    headers = {"access-token": CFG.DHAN_ACCESS_TOKEN, "client-id": str(CFG.DHAN_CLIENT_ID), "Content-Type": "application/json"}
    try:
        resp     = requests.post("https://api.dhan.co/v2/optionchain/expirylist", headers=headers,
                                 json={"UnderlyingScrip": sec["id"], "UnderlyingSeg": sec["seg"]}, timeout=15)
        expiries = resp.json().get("data", [])
        today    = date.today().isoformat()
        return [e for e in expiries if e >= today]
    except Exception as e: print(f"[Dhan] Expiry list error: {e}"); return []

@st.cache_data(ttl=55, show_spinner=False)
def fetch_dhan_option_chain(symbol="GOLDM", expiry=None):
    if not CFG.USE_DHAN: return pd.DataFrame(), 0.0, ""
    sec     = DHAN_SECURITY.get(symbol, DHAN_SECURITY["GOLDM"])
    headers = {"access-token": CFG.DHAN_ACCESS_TOKEN, "client-id": str(CFG.DHAN_CLIENT_ID), "Content-Type": "application/json"}
    if expiry is None:
        try:
            er  = requests.post("https://api.dhan.co/v2/optionchain/expirylist", headers=headers,
                                json={"UnderlyingScrip": sec["id"], "UnderlyingSeg": sec["seg"]}, timeout=15)
            exl = er.json().get("data", []); today = date.today().isoformat()
            fut = [e for e in exl if e >= today]; expiry = fut[0] if fut else (exl[0] if exl else "")
        except Exception as e: print(f"[Dhan] Expiry list error: {e}"); return pd.DataFrame(), 0.0, ""
    if not expiry: return pd.DataFrame(), 0.0, ""
    try:
        r    = requests.post("https://api.dhan.co/v2/optionchain", headers=headers,
                             json={"UnderlyingScrip": sec["id"], "UnderlyingSeg": sec["seg"], "Expiry": expiry}, timeout=20)
        resp = r.json()
    except Exception as e: print(f"[Dhan] OC error: {e}"); return pd.DataFrame(), 0.0, expiry
    if resp.get("status") != "success": return pd.DataFrame(), 0.0, expiry
    data = resp.get("data", {}); spot = float(data.get("last_price", 0))
    rows = []
    for ks, chain in data.get("oc", {}).items():
        K  = float(ks)
        ce = chain.get("ce", {}) or {}; pe = chain.get("pe", {}) or {}
        cg = ce.get("greeks", {}) or {}; pg = pe.get("greeks", {}) or {}
        rows.append({
            "strike":       K,
            "call_ltp":     float(ce.get("last_price",0) or 0),
            "call_oi":      int(ce.get("oi",0) or 0),
            "call_prev_oi": int(ce.get("previous_oi",0) or 0),
            "call_oi_chg":  int(ce.get("oi",0) or 0) - int(ce.get("previous_oi",0) or 0),
            "call_vol":     int(ce.get("volume",0) or 0),
            "call_bid":     float(ce.get("top_bid_price",0) or 0),
            "call_ask":     float(ce.get("top_ask_price",0) or 0),
            "call_iv":      float(ce.get("implied_volatility",0) or 0),
            "call_delta":   float(cg.get("delta",0) or 0),
            "call_gamma":   float(cg.get("gamma",0) or 0),
            "call_theta":   float(cg.get("theta",0) or 0),
            "call_vega":    float(cg.get("vega",0) or 0),
            "put_ltp":      float(pe.get("last_price",0) or 0),
            "put_oi":       int(pe.get("oi",0) or 0),
            "put_prev_oi":  int(pe.get("previous_oi",0) or 0),
            "put_oi_chg":   int(pe.get("oi",0) or 0) - int(pe.get("previous_oi",0) or 0),
            "put_vol":      int(pe.get("volume",0) or 0),
            "put_bid":      float(pe.get("top_bid_price",0) or 0),
            "put_ask":      float(pe.get("top_ask_price",0) or 0),
            "put_iv":       float(pe.get("implied_volatility",0) or 0),
            "put_delta":    float(pg.get("delta",0) or 0),
            "put_gamma":    float(pg.get("gamma",0) or 0),
            "put_theta":    float(pg.get("theta",0) or 0),
            "put_vega":     float(pg.get("vega",0) or 0),
        })
    return pd.DataFrame(rows).sort_values("strike").reset_index(drop=True), spot, expiry

# ─────────────────────────────────────────────────────────────────────
#  FUTURES ROLL — 3-MONTH with LTP ENDPOINT FIX
#  Root cause of original bug: option-chain last_price returns the same
#  underlying near-month price regardless of which expiry you query.
#  Fix: use /v2/marketfeed/ltp with per-contract security IDs.
# ─────────────────────────────────────────────────────────────────────
def fetch_futures_roll(symbol="GOLDM") -> dict:
    if not CFG.USE_DHAN: return {}
    sec     = DHAN_SECURITY.get(symbol, DHAN_SECURITY["GOLDM"])
    headers = {"access-token": CFG.DHAN_ACCESS_TOKEN, "client-id": str(CFG.DHAN_CLIENT_ID), "Content-Type": "application/json"}

    # Step 1: expiry list
    try:
        er      = requests.post("https://api.dhan.co/v2/optionchain/expirylist", headers=headers,
                                json={"UnderlyingScrip": sec["id"], "UnderlyingSeg": sec["seg"]}, timeout=15)
        expiries = er.json().get("data", [])
        today    = date.today().isoformat()
        future   = [e for e in expiries if e >= today]
        if len(future) < 2: print(f"[Roll] < 2 expiries for {symbol}"); return {}
        near_expiry = future[0]; next_expiry = future[1]
        far_expiry  = future[2] if len(future) >= 3 else None
    except Exception as e: print(f"[Roll] Expiry error: {e}"); return {}

    # Step 2: per-contract LTPs via LTP endpoint (not option-chain last_price)
    fut_ids  = get_dynamic_futures_ids().get(symbol, [])
    near_id  = fut_ids[0] if len(fut_ids) >= 1 else None
    next_id  = fut_ids[1] if len(fut_ids) >= 2 else None
    far_id   = fut_ids[2] if len(fut_ids) >= 3 else None
    near_ltp = next_ltp = far_ltp = 0.0

    if near_id:
        ids_to_fetch = [i for i in [near_id, next_id, far_id] if i is not None]
        try:
            lr    = requests.post("https://api.dhan.co/v2/marketfeed/ltp", headers=headers,
                                  json={"MCX_COMM": ids_to_fetch}, timeout=10)
            ld    = lr.json().get("data", {}).get("MCX_COMM", {})
            near_ltp = float((ld.get(str(near_id)) or {}).get("last_price", 0) or 0)
            next_ltp = float((ld.get(str(next_id)) or {}).get("last_price", 0) or 0) if next_id else 0.0
            far_ltp  = float((ld.get(str(far_id))  or {}).get("last_price", 0) or 0) if far_id  else 0.0
            print(f"[Roll/{symbol}] near={near_ltp} next={next_ltp} far={far_ltp}")
        except Exception as e: print(f"[Roll/{symbol}] LTP endpoint failed: {e}")

    # Fallback: option-chain last_price for near only
    if near_ltp == 0:
        try:
            r2 = requests.post("https://api.dhan.co/v2/optionchain", headers=headers,
                               json={"UnderlyingScrip": sec["id"], "UnderlyingSeg": sec["seg"],
                                     "Expiry": near_expiry}, timeout=20)
            near_ltp = float(r2.json().get("data", {}).get("last_price", 0) or 0)
            print(f"[Roll/{symbol}] Fallback OC LTP: near={near_ltp}")
        except Exception as e: print(f"[Roll/{symbol}] Fallback failed: {e}")

    if near_ltp == 0:
        print(f"[Roll] near_ltp=0 for {symbol} — market closed or data unavailable"); return {}

    # Step 3: OI per expiry via option chain (options OI as positioning proxy)
    def _fetch_oi(exp):
        try:
            r  = requests.post("https://api.dhan.co/v2/optionchain", headers=headers,
                               json={"UnderlyingScrip": sec["id"], "UnderlyingSeg": sec["seg"],
                                     "Expiry": exp}, timeout=20)
            oc = r.json().get("data", {}).get("oc", {})
            oi  = sum(int((v.get("ce") or {}).get("oi",0) or 0) + int((v.get("pe") or {}).get("oi",0) or 0) for v in oc.values())
            vol = sum(int((v.get("ce") or {}).get("volume",0) or 0) + int((v.get("pe") or {}).get("volume",0) or 0) for v in oc.values())
            return oi, vol
        except Exception: return 0, 0

    near_oi, near_vol = _fetch_oi(near_expiry)
    next_oi, next_vol = _fetch_oi(next_expiry)
    far_oi,  far_vol  = _fetch_oi(far_expiry) if far_expiry else (0, 0)

    total_oi        = near_oi + next_oi
    roll_spread     = round(next_ltp - near_ltp, 2)
    roll_spread_pct = round((roll_spread / near_ltp * 100) if near_ltp else 0, 3)
    rollover_pct    = round((next_oi / total_oi * 100) if total_oi else 0, 1)

    # Annualised slopes
    try:
        nd  = datetime.strptime(near_expiry, "%Y-%m-%d").date()
        xtd = datetime.strptime(next_expiry, "%Y-%m-%d").date()
        days_nn = max((xtd - nd).days, 1)
        slope_near_next = round((next_ltp - near_ltp) / near_ltp * (365/days_nn) * 100, 2) if near_ltp > 0 else 0.0
    except Exception: slope_near_next = roll_spread_pct

    slope_next_far = 0.0
    if far_expiry and far_ltp > 0 and next_ltp > 0:
        try:
            fd      = datetime.strptime(far_expiry, "%Y-%m-%d").date()
            days_nf = max((fd - xtd).days, 1)
            slope_next_far = round((far_ltp - next_ltp) / next_ltp * (365/days_nf) * 100, 2)
        except Exception: pass

    # Term structure shape
    if slope_near_next > 0 and slope_next_far > 0:
        if slope_next_far >= slope_near_next * 0.9:
            ts_shape, ts_bias, ts_color = "STEEPENING CONTANGO ▲▲", 2, "#00E676"
            ts_desc = "Far month more expensive — strong bullish carry signal"
        else:
            ts_shape, ts_bias, ts_color = "FLATTENING CONTANGO ▲", 1, "#69F0AE"
            ts_desc = "Contango losing steam — carry still positive but momentum slowing"
    elif slope_near_next > 0 and slope_next_far <= 0:
        ts_shape, ts_bias, ts_color = "HUMP / NEAR CARRY ONLY", 0, "#FFD600"
        ts_desc = "Near month in premium but far flat — mixed / expiry-specific carry"
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

    if roll_spread > 0:   bias, bias_color = "CONTANGO ▲  Bullish Carry",          "#00E676"
    elif roll_spread < 0: bias, bias_color = "BACKWARDATION ▼  Delivery Pressure", "#FF5252"
    else:                 bias, bias_color = "FLAT",                                "#FFD600"

    near_vol_oi = round(near_vol / near_oi, 3) if near_oi > 0 else 0.0
    next_vol_oi = round(next_vol / next_oi, 3) if next_oi > 0 else 0.0
    far_vol_oi  = round(far_vol  / far_oi,  3) if far_oi  > 0 else 0.0

    return {
        "near_ltp": near_ltp, "next_ltp": next_ltp, "far_ltp": far_ltp,
        "near_oi":  near_oi,  "next_oi":  next_oi,  "far_oi":  far_oi,
        "near_vol": near_vol, "next_vol": next_vol,  "far_vol": far_vol,
        "near_vol_oi": near_vol_oi, "next_vol_oi": next_vol_oi, "far_vol_oi": far_vol_oi,
        "roll_spread": roll_spread, "roll_spread_pct": roll_spread_pct,
        "rollover_pct": rollover_pct, "bias": bias, "bias_color": bias_color,
        "slope_near_next": slope_near_next, "slope_next_far": slope_next_far,
        "ts_shape": ts_shape, "ts_bias": ts_bias, "ts_color": ts_color, "ts_desc": ts_desc,
        "near_expiry": near_expiry, "next_expiry": next_expiry, "far_expiry": far_expiry or "",
        "has_far": bool(far_expiry and far_ltp > 0),
    }
# ─────────────────────────────────────────────────────────────────────
#  DEMO MODE
# ─────────────────────────────────────────────────────────────────────
def fetch_demo_option_chain(symbol="GOLDM"):
    np.random.seed(int(time.time()) // 60)
    step = DHAN_SECURITY.get(symbol, DHAN_SECURITY["GOLDM"])["step"]
    spot = (93500.0 if "GOLD" in symbol else 96500.0) + np.random.normal(0, 150 if "GOLD" in symbol else 300)
    atm     = round(spot / step) * step
    strikes = np.arange(atm - 25*step, atm + 26*step, step)
    T = 10/365.0; r = RISK_FREE_RATE; vix = 18.5 + np.random.normal(0,1)
    rows = []
    for K in strikes:
        mono = (K - spot)/spot
        iv_c = max(0.05, (vix/100) + 0.015*mono**2 + abs(mono)*0.04 + np.random.normal(0,0.005))
        iv_p = max(0.05, (vix/100) + 0.025*mono**2 - mono*0.03     + np.random.normal(0,0.005))
        cp = _bs_price(spot,K,T,r,iv_c,"CE"); pp = _bs_price(spot,K,T,r,iv_p,"PE")
        cd,cg,ct,cv  = _bs_greeks(spot,K,T,r,iv_c,"CE")
        pd2,pg,pt,pv = _bs_greeks(spot,K,T,r,iv_p,"PE")
        cof = max(0, 2+mono*8)*np.random.lognormal(0,0.4)
        pof = max(0, 2-mono*9)*np.random.lognormal(0,0.4)
        rows.append({
            "strike": K,
            "call_ltp": round(max(0.05,cp+np.random.normal(0,0.3)),2),
            "call_oi": int(max(10,cof*800)), "call_oi_chg": int(np.random.normal(20,150)),
            "call_vol": int(abs(np.random.normal(300,150))),
            "call_bid": round(max(0.05,cp-2),2), "call_ask": round(cp+2,2),
            "call_iv": round(iv_c*100,2), "call_delta": round(cd,4),
            "call_gamma": round(cg,6), "call_theta": round(ct,4), "call_vega": round(cv,4),
            "put_ltp": round(max(0.05,pp+np.random.normal(0,0.3)),2),
            "put_oi": int(max(10,pof*1000)), "put_oi_chg": int(np.random.normal(-10,180)),
            "put_vol": int(abs(np.random.normal(350,200))),
            "put_bid": round(max(0.05,pp-2),2), "put_ask": round(pp+2,2),
            "put_iv": round(iv_p*100,2), "put_delta": round(pd2,4),
            "put_gamma": round(pg,6), "put_theta": round(pt,4), "put_vega": round(pv,4),
        })
    expiry = (date.today() + timedelta(days=10)).strftime("%Y-%m-%d")
    return pd.DataFrame(rows), round(spot,2), expiry

def demo_futures_roll(symbol="GOLDM") -> dict:
    near_ltp = (93500.0 if "GOLD" in symbol else 96500.0) + np.random.normal(0,80)
    spread1  = abs(np.random.normal(120,40)); spread2 = abs(np.random.normal(110,35))
    next_ltp = near_ltp + spread1; far_ltp = next_ltp + spread2
    near_oi  = int(np.random.normal(18000,2000)); next_oi = int(np.random.normal(4500,800))
    far_oi   = int(np.random.normal(800,200));    total_oi = near_oi + next_oi
    near_vol = int(np.random.normal(5000,500)); next_vol = int(np.random.normal(900,200)); far_vol = int(np.random.normal(200,50))
    roll_spread = round(next_ltp - near_ltp, 2)
    roll_spread_pct = round(roll_spread/near_ltp*100, 3)
    rollover_pct    = round(next_oi/total_oi*100, 1)
    near_expiry = date.today().strftime("%Y-%m-%d")
    next_expiry = (date.today() + timedelta(days=30)).strftime("%Y-%m-%d")
    far_expiry  = (date.today() + timedelta(days=60)).strftime("%Y-%m-%d")
    slope_near_next = round(roll_spread/near_ltp*(365/30)*100, 2)
    slope_next_far  = round((far_ltp-next_ltp)/next_ltp*(365/30)*100, 2)
    if slope_near_next > 0 and slope_next_far > 0:
        ts_shape, ts_bias, ts_color = "STEEPENING CONTANGO ▲▲", 2, "#00E676"
        ts_desc = "Far month more expensive — strong bullish carry signal"
    else:
        ts_shape, ts_bias, ts_color = "FLATTENING CONTANGO ▲", 1, "#69F0AE"
        ts_desc = "Contango present but losing steam"
    bias, bias_color = ("CONTANGO ▲  Bullish Carry", "#00E676") if roll_spread > 0 else ("BACKWARDATION ▼  Delivery Pressure", "#FF5252")
    return {
        "near_ltp": round(near_ltp,2), "next_ltp": round(next_ltp,2), "far_ltp": round(far_ltp,2),
        "near_oi": near_oi, "next_oi": next_oi, "far_oi": far_oi,
        "near_vol": near_vol, "next_vol": next_vol, "far_vol": far_vol,
        "near_vol_oi": round(near_vol/near_oi,3), "next_vol_oi": round(next_vol/next_oi,3), "far_vol_oi": round(far_vol/far_oi,3),
        "roll_spread": roll_spread, "roll_spread_pct": roll_spread_pct,
        "rollover_pct": rollover_pct, "bias": bias, "bias_color": bias_color,
        "slope_near_next": slope_near_next, "slope_next_far": slope_next_far,
        "ts_shape": ts_shape, "ts_bias": ts_bias, "ts_color": ts_color, "ts_desc": ts_desc,
        "near_expiry": near_expiry, "next_expiry": next_expiry, "far_expiry": far_expiry,
        "has_far": True,
    }

def get_option_chain(symbol="GOLDM", expiry=None):
    if CFG.USE_DHAN:
        df, spot, exp = fetch_dhan_option_chain(symbol, expiry)
        if not df.empty: return df, spot, exp, "Dhan API (MCX)"
    df, spot, exp = fetch_demo_option_chain(symbol)
    return df, spot, exp, "DEMO MODE (Commodities)"

# ─────────────────────────────────────────────────────────────────────
#  METRICS ENGINE
# ─────────────────────────────────────────────────────────────────────
def select_atm_band(df, spot, symbol="GOLDM"):
    step    = DHAN_SECURITY.get(symbol, DHAN_SECURITY["GOLDM"])["step"]
    strikes = sorted(df["strike"].unique())
    atm     = min(strikes, key=lambda x: abs(x - spot))
    return df[df["strike"].between(atm - ATM_BAND*step, atm + ATM_BAND*step)].copy(), atm

def compute_max_pain(df):
    results = {}
    for K in df["strike"]:
        cl = (df[df["strike"] < K]["call_oi"] * (K - df[df["strike"] < K]["strike"])).sum()
        pl = (df[df["strike"] > K]["put_oi"]  * (df[df["strike"] > K]["strike"] - K)).sum()
        results[K] = cl + pl
    return min(results, key=results.get) if results else 0.0

def _fill_missing_greeks(df_band, spot, expiry=None):
    """
    Exact port of the original app-5.py implementation.
    Only fills greeks when the API returned ALL-zero gammas (common with MCX).
    Does NOT overwrite IV — preserves Dhan's implied_volatility as-is.
    The key lesson: the per-strike delta==0 trigger was wrong — deep OTM options
    legitimately have delta=0.0000 (truncated by Dhan), and overwriting their IV
    with 15% corrupted the IV smile.
    """
    if df_band.empty: return df_band
    # Return unchanged if Dhan DID provide gammas (any non-zero = API has greeks)
    if df_band["call_gamma"].abs().max() > 1e-9 and df_band["put_gamma"].abs().max() > 1e-9:
        return df_band
    try:
        T = max((datetime.strptime(expiry[:10],"%Y-%m-%d").date() - date.today()).days, 1)/365.0 if expiry else 10/365.0
    except Exception: T = 10/365.0
    r   = RISK_FREE_RATE
    df2 = df_band.copy()
    for idx, row in df2.iterrows():
        K    = float(row["strike"])
        iv_c = float(row.get("call_iv", 0) or 0) / 100.0
        iv_p = float(row.get("put_iv",  0) or 0) / 100.0
        iv_c = iv_c if iv_c > 0.01 else 0.15
        iv_p = iv_p if iv_p > 0.01 else 0.15
        cd, cg, ct, cv  = _bs_greeks(spot, K, T, r, iv_c, "CE")
        pd2, pg, pt, pv = _bs_greeks(spot, K, T, r, iv_p, "PE")
        df2.at[idx, "call_delta"] = cd; df2.at[idx, "call_gamma"] = cg
        df2.at[idx, "call_theta"] = ct; df2.at[idx, "call_vega"]  = cv
        df2.at[idx, "put_delta"]  = pd2; df2.at[idx, "put_gamma"] = pg
        df2.at[idx, "put_theta"]  = pt;  df2.at[idx, "put_vega"]  = pv
    return df2

def compute_gamma_flip(df_band, spot):
    if df_band.empty: return None
    cum = 0.0
    for K in sorted(df_band["strike"].unique()):
        row = df_band[df_band["strike"]==K]
        if row.empty: continue
        gk = (float(row["call_gamma"].values[0])*float(row["call_oi"].values[0]) -
              float(row["put_gamma"].values[0]) *float(row["put_oi"].values[0])) * spot**2 * 0.01
        prev = cum; cum += gk
        if prev > 0 and cum <= 0: return K
        if prev < 0 and cum >= 0: return K
    return None

def compute_iv_rank(df_band):
    all_ivs = pd.concat([df_band["call_iv"], df_band["put_iv"]]).dropna()
    all_ivs = all_ivs[all_ivs > 0]
    if all_ivs.empty: return 50.0
    iv_min = all_ivs.min(); iv_max = all_ivs.max(); atm_iv = all_ivs.median()
    return round((atm_iv - iv_min)/(iv_max - iv_min)*100, 1) if iv_max != iv_min else 50.0

def compute_metrics(df, spot, symbol="GOLDM", expiry=None):
    if df.empty: return {}
    df_band, atm = select_atm_band(df, spot, symbol)
    df_band = _fill_missing_greeks(df_band, spot, expiry)
    step    = DHAN_SECURITY.get(symbol, DHAN_SECURITY["GOLDM"])["step"]

    df_band["intr_c"] = np.maximum(0, spot - df_band["strike"])
    df_band["ev_c"]   = np.maximum(0, df_band["call_ltp"] - df_band["intr_c"])
    df_band["intr_p"] = np.maximum(0, df_band["strike"] - spot)
    df_band["ev_p"]   = np.maximum(0, df_band["put_ltp"]  - df_band["intr_p"])

    ev_sum_c = df_band["ev_c"].sum(); ev_sum_p = df_band["ev_p"].sum()
    ev_ratio = ev_sum_c/ev_sum_p if ev_sum_p > 0 else 1.0
    net_delta = ((df_band["call_oi"]*df_band["call_delta"]).sum() + (df_band["put_oi"]*df_band["put_delta"]).sum())
    net_gamma = ((df_band["call_oi"]*df_band["call_gamma"]).sum() + (df_band["put_oi"]*df_band["put_gamma"]).sum())
    net_theta = ((df_band["call_oi"]*df_band["call_theta"]).sum() + (df_band["put_oi"]*df_band["put_theta"]).sum())
    gex   = ((df_band["call_oi"]*df_band["call_gamma"]).sum() - (df_band["put_oi"]*df_band["put_gamma"]).sum()) * spot**2 * 0.01
    vanna = ((df_band["call_oi"]*df_band["call_vega"]*df_band["call_delta"]).sum() +
             (df_band["put_oi"] *df_band["put_vega"] *df_band["put_delta"]).sum()) / max(spot,1)
    gt_ratio = abs(net_gamma)/max(abs(net_theta),1e-6)
    momentum = ((df_band["call_oi_chg"]*df_band["call_delta"]).sum() + (df_band["put_oi_chg"]*df_band["put_delta"]).sum())
    sv = (df_band["call_oi"]*df_band["call_vega"]).sum(); pv = (df_band["put_oi"]*df_band["put_vega"]).sum()
    vega_skew = sv/pv if pv > 0 else 1.0
    total_coi = df["call_oi"].sum(); total_poi = df["put_oi"].sum()
    pcr = total_poi/total_coi if total_coi > 0 else 1.0
    max_pain = compute_max_pain(df)
    atm_row = df_band[df_band["strike"]==atm]
    atm_iv  = float(((atm_row["call_iv"].values[0] if not atm_row.empty else 0) +
                     (atm_row["put_iv"].values[0]  if not atm_row.empty else 0))/2)
    support    = df_band.loc[df_band["put_oi"].idxmax(),  "strike"] if not df_band.empty else 0
    resistance = df_band.loc[df_band["call_oi"].idxmax(), "strike"] if not df_band.empty else 0
    wall_width = float(resistance - support) if resistance > support else float(step*4)
    near_band  = df_band[df_band["strike"].between(atm-3*step, atm+3*step)]
    tob        = df_band["call_oi"].sum() + df_band["put_oi"].sum()
    near_oi_conc   = (near_band["call_oi"].sum()+near_band["put_oi"].sum())/tob if tob > 0 else 0.5
    band_oichg     = abs(df_band["call_oi_chg"]).sum()+abs(df_band["put_oi_chg"]).sum()
    near_oichg_conc= (abs(near_band["call_oi_chg"]).sum()+abs(near_band["put_oi_chg"]).sum())/band_oichg if band_oichg > 0 else 0.5
    atm_pressure   = float(near_band["put_oi_chg"].sum() - near_band["call_oi_chg"].sum())
    otm_puts = df_band[df_band["strike"] < atm-step]; otm_calls = df_band[df_band["strike"] > atm+step]
    if len(otm_puts) >= 2 and len(otm_calls) >= 2:
        put_slope  = float(np.polyfit(otm_puts["strike"],  otm_puts["put_iv"],   1)[0])
        call_slope = float(np.polyfit(otm_calls["strike"], otm_calls["call_iv"], 1)[0])
        skew_slope = round(put_slope - call_slope, 4)
    else: skew_slope = 0.0
    iv_rank    = compute_iv_rank(df_band)
    gamma_flip = compute_gamma_flip(df_band, spot)
    return {
        "ev_ratio": round(ev_ratio,3), "net_delta": round(net_delta,0), "net_gamma": round(net_gamma,6),
        "net_theta": round(net_theta,0), "gex": round(gex,0), "vanna": round(vanna,2),
        "gt_ratio": round(gt_ratio,4), "momentum": round(momentum,0), "vega_skew": round(vega_skew,3),
        "max_pain": round(max_pain,0), "pcr": round(pcr,2), "atm_iv": round(atm_iv,2), "atm": atm,
        "support": support, "resistance": resistance, "wall_width": wall_width,
        "near_oi_concentration": round(near_oi_conc,3), "near_oichg_concentration": round(near_oichg_conc,3),
        "atm_pressure": round(atm_pressure,0), "skew_slope": round(skew_slope,4),
        "iv_rank": iv_rank, "gamma_flip": gamma_flip,
        "call_oi_total": int(total_coi), "put_oi_total": int(total_poi), "df_band": df_band,
    }

# ─────────────────────────────────────────────────────────────────────
#  SCORING — upgraded to accept roll signals
# ─────────────────────────────────────────────────────────────────────
def compute_score(m, roll=None):
    if not m: return 50.0
    score = 15
    ev = m["ev_ratio"]
    score += 15 if ev >= 1.2 else (0 if ev < 0.8 else 7.5)
    d = m["net_delta"]
    score += 20 if d >= 1_000 else (0 if d < -1_000 else 10)
    mom = m["momentum"]
    score += 15 if mom >= 500 else (0 if mom < -500 else 7.5)
    vega = m["vega_skew"]
    score += 10 if vega >= 1.2 else (0 if vega < 0.8 else 5)
    vanna = m["vanna"]
    score += 10 if vanna >= 10 else (0 if vanna < -10 else 5)
    if roll:
        ts_bias = roll.get("ts_bias", 0)
        score  += ts_bias * 3
        ca = roll.get("carry_anomaly", 1.0)
        if ca >= 1.5: score += 4
        elif ca <= 0.5: score -= 3
        rv = roll.get("rollover_velocity", 0.8)
        if rv >= 1.3: score += 4
        elif rv <= 0.3: score -= 4
    return round(min(max(score, 0), 100), 1)

def strategy_recommendation(score, m, symbol="GOLDM"):
    support = m.get("support",0); resistance = m.get("resistance",0)
    atm     = m.get("atm",0)
    step    = DHAN_SECURITY.get(symbol, DHAN_SECURITY["GOLDM"])["step"]
    if   score >= 85: name, color = "Long Call / Bull Call Spread", "#00C853"
    elif score >= 70: name, color = "Bull Call Spread",             "#69F0AE"
    elif score >= 55: name, color = "Bull Put Spread (High Prob)",  "#B2FF59"
    elif score >= 45: name, color = "Iron Condor",                  "#FFD600"
    elif score >= 31: name, color = "Bear Call Spread",             "#FF6D00"
    elif score >= 16: name, color = "Bear Put Spread",              "#F44336"
    else:             name, color = "Long Put",                     "#B71C1C"
    if   score >= 85: legs = f"Buy {int(atm)} CE  |  Sell {int(resistance)} CE"
    elif score >= 70: legs = f"Buy {int(atm)} CE  |  Sell {int(atm+2*step)} CE"
    elif score >= 55: legs = f"Sell {int(support+step)} PE  |  Buy {int(support-step)} PE"
    elif score >= 45: legs = (f"Sell {int(support+step)} PE / Buy {int(support-step)} PE  +  "
                               f"Sell {int(resistance-step)} CE / Buy {int(resistance+step)} CE")
    elif score >= 31: legs = f"Sell {int(atm)} CE  |  Buy {int(atm+2*step)} CE"
    elif score >= 16: legs = f"Buy {int(atm)} PE  |  Sell {int(atm-2*step)} PE"
    else:             legs = f"Buy {int(support-step)} PE"
    if   score >= 55: mode, mc = "TREND MODE — Bullish", "#00E676"
    elif score >= 45: mode, mc = "NEUTRAL / RANGE",      "#FFD740"
    else:             mode, mc = "TREND MODE — Bearish",  "#FF5252"
    return {"name": name, "legs": legs, "color": color, "market_mode": mode, "mode_color": mc}

# ─────────────────────────────────────────────────────────────────────
#  OI VELOCITY
# ─────────────────────────────────────────────────────────────────────
def compute_oi_velocity(history, symbol="GOLDM"):
    sym_history = _extract_sym_history(history, symbol)
    if len(sym_history) < 3:
        return {"call_oi_velocity":0,"put_oi_velocity":0,"call_oi_accel":0,"put_oi_accel":0,
                "call_vel_zscore":0,"put_vel_zscore":0,"alert_level":"NONE","alert_text":"Collecting data…","n_ticks":0}
    call_oi = np.array([safe_num(x.get("call_oi_total",0)) for x in sym_history], dtype=float)
    put_oi  = np.array([safe_num(x.get("put_oi_total", 0)) for x in sym_history], dtype=float)
    if call_oi.max()==0 and put_oi.max()==0:
        nd  = np.array([safe_num(x.get("net_delta",0)) for x in sym_history], dtype=float)
        mom = np.array([safe_num(x.get("oi_net_delta",0)) for x in sym_history], dtype=float)
        call_oi = np.maximum(nd,0)+np.maximum(mom,0); put_oi = np.maximum(-nd,0)+np.maximum(-mom,0)
    c_vel = np.diff(call_oi); p_vel = np.diff(put_oi)
    if len(c_vel) < 2:
        return {"call_oi_velocity":0,"put_oi_velocity":0,"call_oi_accel":0,"put_oi_accel":0,
                "call_vel_zscore":0,"put_vel_zscore":0,"alert_level":"NONE","alert_text":"Collecting data…","n_ticks":len(sym_history)}
    c_accel = float(c_vel[-1]-c_vel[-2]); p_accel = float(p_vel[-1]-p_vel[-2])
    window  = min(10, len(c_vel))
    c_vel_z = _zscore(c_vel, window); p_vel_z = _zscore(p_vel, window)
    max_z   = max(abs(c_vel_z), abs(p_vel_z))
    if max_z >= 2.0:
        side = "CALL" if abs(c_vel_z) > abs(p_vel_z) else "PUT"
        direction = "surge" if (c_vel_z if side=="CALL" else p_vel_z)>0 else "unwind"
        alert_level = "DANGER"; alert_text = f"⚡ {side} OI {direction} detected — velocity {max_z:.1f}σ above norm."
    elif max_z >= 1.2:
        side = "CALL" if abs(c_vel_z) > abs(p_vel_z) else "PUT"
        alert_level = "WATCH"; alert_text = f"⚠ {side} OI velocity elevated ({max_z:.1f}σ). Monitor closely."
    else:
        alert_level = "NONE"; alert_text = "OI velocity within normal range."
    return {"call_oi_velocity":float(c_vel[-1]),"put_oi_velocity":float(p_vel[-1]),
            "call_oi_accel":c_accel,"put_oi_accel":p_accel,
            "call_vel_zscore":round(c_vel_z,2),"put_vel_zscore":round(p_vel_z,2),
            "alert_level":alert_level,"alert_text":alert_text,"n_ticks":len(sym_history)}

# ─────────────────────────────────────────────────────────────────────
#  OI REGIME + COMBINED BIAS
# ─────────────────────────────────────────────────────────────────────
def _bucket_oi_15min(sym_history):
    if len(sym_history) < 3: return [], [], []
    call_oi = np.array([safe_num(x.get("call_oi_total",0)) for x in sym_history], dtype=float)
    put_oi  = np.array([safe_num(x.get("put_oi_total", 0)) for x in sym_history], dtype=float)
    if call_oi.max()==0 and put_oi.max()==0:
        nd  = np.array([safe_num(x.get("net_delta",0)) for x in sym_history], dtype=float)
        mom = np.array([safe_num(x.get("oi_net_delta",0)) for x in sym_history], dtype=float)
        call_oi = np.maximum(nd,0)+np.maximum(mom,0); put_oi = np.maximum(-nd,0)+np.maximum(-mom,0)
    ts = [x.get("ts","") for x in sym_history]; c_vel = np.diff(call_oi); p_vel = np.diff(put_oi); ts_v = ts[1:]
    bc, bp = {}, {}
    for i, t in enumerate(ts_v):
        try:
            p  = t.split("T")[-1].split(":"); hh, mm = int(p[0]), int(p[1]); label = f"{hh:02d}:{(mm//15)*15:02d}"
        except Exception: label = t
        bc[label] = bc.get(label,0.0)+float(c_vel[i]); bp[label] = bp.get(label,0.0)+float(p_vel[i])
    labels = sorted(bc.keys())
    return labels, [bc[l] for l in labels], [bp.get(l,0.0) for l in labels]

def _oi_regime_info(c_bkt, p_bkt):
    if not c_bkt or not p_bkt:
        return {"label":"Collecting OI data for regime detection…","sub":"","bg":CARD,"fg":MUTED,"border":MUTED}
    c_arr = np.array(c_bkt,dtype=float); p_arr = np.array(p_bkt,dtype=float)
    c_std = float(c_arr.std()) if c_arr.std()>1e-9 else 1.0; p_std = float(p_arr.std()) if p_arr.std()>1e-9 else 1.0
    rc = list(c_arr[-3:]); rp = list(p_arr[-3:])
    avg_c = abs(float(np.mean(rc))) if rc else 0.0; avg_p = abs(float(np.mean(rp))) if rp else 0.0
    buyer  = (avg_c > 1.0*c_std) or (avg_p > 1.0*p_std)
    seller = (avg_c <= 0.8*c_std) and (avg_p <= 0.8*p_std)
    if buyer and not seller:
        return {"label":"OPTION BUYER'S REGIME","sub":"OI velocity elevated — directional participants active. Premium is expensive. Favour directional plays.","bg":"#FFFBEB","fg":"#D97706","border":"#D97706"}
    elif seller:
        return {"label":"OPTION SELLER'S REGIME","sub":"OI velocity subdued — writers in control. Range-bound / premium decay favoured. Sell spreads or iron condors.","bg":"#F0FDF4","fg":"#059669","border":"#059669"}
    else:
        return {"label":"TRANSITIONAL REGIME","sub":"Mixed OI signals — neither buyers nor sellers clearly dominant. Wait for clarity.","bg":"#EFF6FF","fg":"#0891B2","border":"#0891B2"}

def _combined_bias_info(c_bkt, p_bkt):
    if not c_bkt or not p_bkt: return None
    c_arr = np.array(c_bkt,dtype=float); p_arr = np.array(p_bkt,dtype=float)
    mc = float(c_arr.mean()); sc = float(c_arr.std()) if c_arr.std()>1e-9 else 1.0
    mp = float(p_arr.mean()); sp = float(p_arr.std()) if p_arr.std()>1e-9 else 1.0
    cc = c_arr[:-1] if len(c_arr)>=2 else c_arr; cp = p_arr[:-1] if len(p_arr)>=2 else p_arr
    sc_v = float(np.mean(cc[-2:])) if len(cc)>=2 else float(np.mean(cc)) if len(cc) else 0.0
    sp_v = float(np.mean(cp[-2:])) if len(cp)>=2 else float(np.mean(cp)) if len(cp) else 0.0
    c_z = (sc_v-mc)/sc; p_z = (sp_v-mp)/sp
    STRONG, WEAK = 0.8, 0.3
    c_up = c_z>STRONG; c_down = c_z<-STRONG; c_flat = abs(c_z)<WEAK
    p_up = p_z>STRONG; p_down = p_z<-STRONG; p_flat = abs(p_z)<WEAK
    if   c_up and p_up:     bias, bc = "PINNED — Range / Max-Pain Bias",        BLUE
    elif c_up and p_down:   bias, bc = "BULLISH — Upside Bias from Writers",     GREEN
    elif c_down and p_up:   bias, bc = "BEARISH — Downside Bias from Writers",   RED
    elif c_down and p_down: bias, bc = "EXPANSION — Breakout / Breakdown Risk",  "#9333EA"
    elif c_up and p_flat:   bias, bc = "MILDLY BEARISH — Resistance Reinforcing",AMBER
    elif p_up and c_flat:   bias, bc = "MILDLY BULLISH — Support Reinforcing",   "#10B981"
    else:                   bias, bc = "NEUTRAL — No Clear OI Signal",           MUTED
    return {"bias": bias, "bc": bc, "c_z": c_z, "p_z": p_z}

# ─────────────────────────────────────────────────────────────────────
#  INTRADAY OI RECORDER — with rollover velocity
# ─────────────────────────────────────────────────────────────────────
def record_intraday_oi(symbol: str, roll: dict, oi_history: dict):
    if not roll: return oi_history
    ts     = strftime_ist("%H:%M")
    noi    = roll.get("near_oi", 0); xoi = roll.get("next_oi", 0)
    entry  = {"ts": ts, "near_oi": noi, "next_oi": xoi, "total_oi": noi + xoi}
    hist   = oi_history.setdefault(symbol, [])
    if len(hist) >= 1:
        prev = hist[-1]
        d_near = noi - prev.get("near_oi", 0); d_next = xoi - prev.get("next_oi", 0)
        entry["rollover_velocity"] = round(d_next/abs(d_near), 3) if abs(d_near) > 10 else prev.get("rollover_velocity", 0.8)
    else:
        entry["rollover_velocity"] = 0.8
    if hist and hist[-1]["ts"] == ts: hist[-1] = entry
    else: hist.append(entry)
    if len(hist) > 600: oi_history[symbol] = hist[-600:]
    return oi_history
# ─────────────────────────────────────────────────────────────────────
#  NEW v4 BIAS ENGINES
# ─────────────────────────────────────────────────────────────────────
def compute_wing_excess(df_band, atm, atm_iv, symbol="GOLDM"):
    """OTM wing IV average minus ATM IV — tells you what traders fear more."""
    if df_band.empty or atm_iv <= 0: return None, None
    step = DHAN_SECURITY.get(symbol, DHAN_SECURITY["GOLDM"])["step"]
    piv = df_band.loc[df_band["strike"].between(atm-6*step, atm-2*step) & (df_band["put_iv"]>0.5),  "put_iv"]
    civ = df_band.loc[df_band["strike"].between(atm+2*step, atm+6*step) & (df_band["call_iv"]>0.5), "call_iv"]
    if len(piv) < 2 or len(civ) < 2: return None, None
    return round(float(piv.mean())-atm_iv, 2), round(float(civ.mean())-atm_iv, 2)

def classify_iv_smile_scenario(df_band, m, spot, symbol="GOLDM", iv_smile_history=None):
    """12-scenario IV smile classifier — commodity-adapted for GOLDM/SILVERM step sizes."""
    if df_band.empty or spot <= 0:
        return {"scenario":"Insufficient data","color":MUTED,"desc":"Need option chain data","put_wing_excess":0,"call_wing_excess":0,"trend_put":0,"trend_call":0}
    atm    = m.get("atm", 0); atm_iv = m.get("atm_iv", 0)
    if atm_iv <= 0:
        return {"scenario":"No ATM IV","color":MUTED,"desc":"ATM IV not available","put_wing_excess":0,"call_wing_excess":0,"trend_put":0,"trend_call":0}
    pe, ce = compute_wing_excess(df_band, atm, atm_iv, symbol)
    if pe is None:
        return {"scenario":"Insufficient OTM data","color":MUTED,"desc":"Need more OTM strikes","put_wing_excess":0,"call_wing_excess":0,"trend_put":0,"trend_call":0}
    trend_put = trend_call = 0.0
    if iv_smile_history and len(iv_smile_history) >= 3:
        recent = iv_smile_history[-5:]
        trend_put  = float(np.mean([h.get("put_wing_excess",  0) for h in recent]))
        trend_call = float(np.mean([h.get("call_wing_excess", 0) for h in recent]))
    HIGH = atm_iv * 0.25; MED = atm_iv * 0.10
    if pe > HIGH and ce > HIGH:
        if abs(pe-ce) < MED:   scenario,color,desc = "SYMMETRIC WIDE / VOLATILITY BID",    PINK,  "Both wings elevated — large move priced in either direction. Favour long straddle / long gamma."
        elif pe > ce:           scenario,color,desc = "INVERTED SMILE / PUT SKEW HEAVY",     RED,   "Put wing dominates — crash fear / institutional hedging active. Sell call spreads."
        else:                   scenario,color,desc = "CALL SKEW DOMINANT",                  GREEN, "Call wing dominates — breakout speculation / short squeeze risk."
    elif pe > HIGH and ce < MED:
        if trend_put > MED:     scenario,color,desc = "CRASH FEAR / PANIC HEDGE",            RED,   "OTM put IV persistently elevated — strong fear; do not sell puts."
        else:                   scenario,color,desc = "SMIRK / STRUCTURAL PUT BUYER",        AMBER, "Moderate put skew, low call wing — structural hedgers active; mildly bearish."
    elif ce > HIGH and pe < MED:
        if ce > HIGH*1.5:       scenario,color,desc = "STRONG CALL SKEW / BREAKOUT BET",    "#00E676","Call wing very elevated — big upside being priced. Favour long calls / call spread."
        else:                   scenario,color,desc = "CALL SKEW / MILD UPSIDE BID",         "#69F0AE","Call wing moderately elevated — mild bullish speculation."
    elif pe < -MED and ce < -MED:
        scenario,color,desc = "IV CRUSH / POST-EVENT",    CYAN,  "Both wings below ATM — options cheap after event. Favour long gamma."
    elif abs(pe) < MED and abs(ce) < MED:
        if atm_iv > 25:         scenario,color,desc = "NORMAL / ELEVATED ATM IV",            AMBER, "Flat smile, high ATM IV — premium selling conditions. Iron condor / short strangle."
        else:                   scenario,color,desc = "NORMAL / LOW IV",                     GREEN, "Flat smile, low IV — options cheap; favour long gamma or long options."
    elif pe > MED and ce < MED:
        if trend_put > trend_call: scenario,color,desc = "MILD PUT SKEW / BUILDING",         AMBER, "Put wing slowly building — cautious market; avoid naked puts."
        else:                   scenario,color,desc = "COMPRESSED / COILED SPRING",          CYAN,  "Put wing present but call wing flat — tension building; watch for breakout."
    else:
        scenario,color,desc = "TRANSITIONAL / MIXED SIGNALS", MUTED, "Unclear smile shape — market in transition; wait for confirmation."
    return {"scenario":scenario,"color":color,"desc":desc,
            "put_wing_excess":pe,"call_wing_excess":ce,"trend_put":round(trend_put,2),"trend_call":round(trend_call,2)}

def classify_gamma_regime(gex, wall_width, momentum, atm_iv, iv_rank, spot, gamma_flip, step):
    """5-state gamma regime: PINNED → RANGE → TREND/EXPANSION → FLIP ZONE → TRANSITION."""
    if not step or step <= 0: step = 100
    flip_dist   = abs(spot - gamma_flip) if gamma_flip is not None else 9999
    near_flip   = flip_dist < max(3.0*step, step*3)
    narrow_wall = wall_width < 10*step
    mod_wall    = wall_width < 20*step
    high_iv     = iv_rank > 70
    positive    = gex > 0
    large_gex   = abs(gex) > 1e8
    if near_flip:
        return "FLIP ZONE / UNSTABLE", "Price near gamma flip — dealer hedging switches from stabilising to amplifying. High volatility risk.", "HIGH VOL RISK", RED
    elif positive and large_gex and narrow_wall:
        return "PINNED / RANGE", "Strong positive GEX + tight walls — dealers pin price; expect mean-reversion. Sell premium.", "LOW VOL / PIN", GREEN
    elif positive and mod_wall:
        return "RANGE / PIN", "Positive GEX with moderate range — price likely to oscillate. Range trades favoured.", "RANGE BOUND", CYAN
    elif not positive and large_gex and (abs(momentum) > 500 or high_iv):
        return "TREND / EXPANSION", "Negative GEX + high momentum or fear — dealers amplify moves. Favour directional plays.", "TREND / HIGH VOL", AMBER
    else:
        return "TRANSITION", "Mixed gamma signals — market changing regimes. Wait for confirmation before trading.", "NEUTRAL", MUTED

def compute_carry_anomaly(roll: dict, atm_iv: float) -> float:
    """Actual roll spread vs IV-implied weekly carry. >1.5 = futures pricing a big move."""
    if not roll or atm_iv <= 0: return 1.0
    near_ltp = roll.get("near_ltp", 0); roll_spread = roll.get("roll_spread", 0)
    if near_ltp <= 0: return 1.0
    expected_weekly = (atm_iv/100) * near_ltp / np.sqrt(52)
    return round(abs(roll_spread)/expected_weekly, 2) if expected_weekly > 0 else 1.0

def compute_rollover_velocity_zscore(oi_history, symbol):
    """Z-score of rollover velocity — shows whether today's roll pace is unusual."""
    hist = oi_history.get(symbol, [])
    rvs  = [h.get("rollover_velocity", 0.8) for h in hist if h.get("rollover_velocity") is not None]
    if len(rvs) < 3: return 0.0, "Collecting data…", MUTED
    arr  = np.array(rvs, dtype=float)
    std  = float(arr.std()) if arr.std() > 1e-9 else 1.0
    z    = float((arr[-1] - arr.mean()) / std)
    latest = float(arr[-1])
    if latest >= 1.3:   interp, color = f"Conviction roll — {latest:.2f} (longs adding)", GREEN
    elif latest >= 0.8: interp, color = f"Normal roll — {latest:.2f}", CYAN
    elif latest >= 0.3: interp, color = f"Slow roll — {latest:.2f} (caution)", AMBER
    else:               interp, color = f"Liquidation — {latest:.2f} (bearish unwind)", RED
    return round(z, 2), interp, color

# ─────────────────────────────────────────────────────────────────────
#  CHARTS — original + new v4
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
    fig.update_layout(paper_bgcolor="#FFFFFF",plot_bgcolor="#FFFFFF",margin=dict(l=20,r=20,t=30,b=5),height=220)
    return fig

def build_iv_history_chart(sym_history):
    empty = go.Figure(); empty.update_layout(**chart_layout(title="Cumulative Δ ATM IV — 15-min buckets"))
    if len(sym_history) < 2: return empty
    buckets = {}
    for x in sym_history:
        ts = x.get("ts",""); iv = safe_num(x.get("atm_iv",0))
        if iv <= 0: continue
        try:
            p  = ts.split("T")[-1].split(":") if "T" in ts else ts.split(":"); hh,mm = int(p[0]),int(p[1])
            label = f"{hh:02d}:{(mm//15)*15:02d}"
        except Exception: label = ts
        buckets[label] = iv
    if not buckets: return empty
    labels = sorted(buckets.keys()); vals = [buckets[l] for l in labels]
    base = vals[0]; cum_d = [v-base for v in vals]; iv_delta = cum_d[-1] if cum_d else 0
    direction = "IV RISING" if iv_delta>0.5 else ("IV FALLING" if iv_delta<-0.5 else "IV FLAT")
    lc = "#059669" if iv_delta<-0.5 else ("#DC2626" if iv_delta>0.5 else CYAN)
    fig = go.Figure()
    fig.add_hline(y=0,line_dash="dash",line_color="#94A3B8",opacity=0.7,annotation_text="open baseline",annotation_font_size=10)
    fig.add_trace(go.Scatter(x=labels,y=cum_d,mode="lines+markers",name="Cumul Δ ATM IV",
                             line=dict(color=lc,width=2.5),marker=dict(size=6,color=lc),
                             hovertemplate="<b>%{x}</b><br>Cumul Δ IV: %{y:+.2f}pp<extra></extra>"))
    lk = chart_layout(title=f"Cumul Δ ATM IV — 15-min | {direction}  ({iv_delta:+.2f}pp)")
    lk["yaxis"] = dict(title="Δ IV (pp)",gridcolor="#E2E8F0"); lk["xaxis"] = dict(title="15-min bucket",gridcolor="#E2E8F0")
    fig.update_layout(**lk); return fig

def _build_oi_vel_chart(sym_history, side="CALL"):
    sl = "Call" if side=="CALL" else "Put"; lc = CYAN if side=="CALL" else PINK
    bc = "rgba(0,229,255,0.08)" if side=="CALL" else "rgba(255,64,129,0.08)"
    empty = go.Figure(); empty.update_layout(**chart_layout(title=f"Δ {sl} OI Velocity Z-Score — 15-min"))
    if len(sym_history) < 3: return empty
    call_oi = np.array([safe_num(x.get("call_oi_total",0)) for x in sym_history],dtype=float)
    put_oi  = np.array([safe_num(x.get("put_oi_total", 0)) for x in sym_history],dtype=float)
    ts_raw  = [x.get("ts","") for x in sym_history]
    vel = np.diff(call_oi if side=="CALL" else put_oi); ts_v = ts_raw[1:]
    def _bkt(t):
        try: p=t.split("T")[-1].split(":"); return f"{int(p[0]):02d}:{(int(p[1])//15)*15:02d}"
        except Exception: return t
    bkts: dict = {}
    for i,t in enumerate(ts_v): lbl=_bkt(t); bkts[lbl]=bkts.get(lbl,0.0)+float(vel[i])
    if not bkts: return empty
    lbc = sorted(bkts.keys()); arc = np.array([bkts[l] for l in lbc],dtype=float)
    mv = float(arc.mean()); sv = float(arc.std()) if arc.std()>1e-9 else 1.0; za = (arc-mv)/sv
    ll = _bkt(ts_raw[-1]); lv = sum(float(vel[i]) for i,t in enumerate(ts_v) if _bkt(t)==ll); lz = (lv-mv)/sv
    al = list(lbc); az = list(za); is_live = [False]*len(lbc)
    if ll not in lbc: al.append(ll); az.append(lz); is_live.append(True)
    else: az[-1]=lz; is_live[-1]=True
    latest_z = az[-1]
    alert = (f"⚡ SURGE +{latest_z:.1f}σ" if latest_z>=2.0 else f"⚠ ELEVATED +{latest_z:.1f}σ" if latest_z>=1.2 else
             f"⚡ UNWIND {latest_z:.1f}σ" if latest_z<=-2.0 else f"↘ EASING {latest_z:.1f}σ" if latest_z<=-1.2 else f"NORMAL {latest_z:+.1f}σ")
    n = len(al); fig = go.Figure()
    fig.add_trace(go.Scatter(x=al+al[::-1],y=[2.0]*n+[-2.0]*n,fill="toself",fillcolor=bc,
                             line=dict(color="rgba(255,255,255,0)"),hoverinfo="skip",showlegend=False))
    cx=[l for l,lv2 in zip(al,is_live) if not lv2]; cy=[z for z,lv2 in zip(az,is_live) if not lv2]
    lx=[l for l,lv2 in zip(al,is_live) if lv2];     ly=[z for z,lv2 in zip(az,is_live) if lv2]
    if cx: fig.add_trace(go.Scatter(x=cx,y=cy,mode="lines+markers",name=f"{sl} OI Vel Z",
                                    line=dict(color=lc,width=2.5),marker=dict(size=6,color=lc),
                                    hovertemplate="<b>%{x}</b><br>Z: %{y:+.2f}σ<extra></extra>"))
    if lx:
        if cx: fig.add_trace(go.Scatter(x=[cx[-1],lx[0]],y=[cy[-1],ly[0]],mode="lines",
                                        line=dict(color=lc,width=1.5,dash="dot"),showlegend=False,hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=lx,y=ly,mode="markers",name="🔴 Live (forming)",
                                 marker=dict(size=10,color=lc,opacity=0.5,symbol="circle-open",line=dict(width=2,color=lc))))
    for yv,dash,col,ann in [(0,"dash","#94A3B8","mean"),(2,"dot",RED,"+2σ"),(1,"dot",AMBER,"+1σ"),(-1,"dot",GREEN,"−1σ"),(-2,"dot",GREEN,"−2σ")]:
        fig.add_hline(y=yv,line_dash=dash,line_color=col,opacity=0.5,annotation_text=ann,annotation_font_size=10)
    lk = chart_layout(title=f"Δ {sl} OI Vel Z-Score — 15-min | {alert}")
    lk["yaxis"]=dict(title="Z-score (σ)",gridcolor="#E2E8F0"); lk["xaxis"]=dict(title="15-min bucket",gridcolor="#E2E8F0")
    fig.update_layout(**lk); return fig

def build_term_structure_chart(roll: dict):
    """3-month futures curve. Upward = contango (normal); downward = backwardation (delivery pressure)."""
    fig = go.Figure()
    if not roll: fig.update_layout(**chart_layout(title="Futures Term Structure (3-Month Curve)")); return fig
    labels = ["Near","Next"]; prices = [roll["near_ltp"], roll["next_ltp"]]; colors_bar = [GOLD,"#CE93D8"]
    if roll.get("has_far") and roll.get("far_ltp",0)>0:
        labels.append("Far"); prices.append(roll["far_ltp"]); colors_bar.append(CYAN)
    fig.add_trace(go.Bar(x=labels,y=prices,marker_color=colors_bar,
                         text=[f"₹{p:,.0f}" for p in prices],textposition="outside",
                         hovertemplate="<b>%{x}</b><br>LTP: ₹%{y:,.0f}<extra></extra>"))
    fig.add_annotation(text=roll.get("ts_shape",""),xref="paper",yref="paper",x=0.5,y=1.12,
                       showarrow=False,font=dict(color=roll.get("ts_color",CYAN),size=11,family="monospace"))
    lk = chart_layout(title="Futures Term Structure — 3-Month Curve")
    lk["yaxis"]=dict(title="LTP (₹)",gridcolor="#E2E8F0",tickformat=",")
    fig.update_layout(**lk,showlegend=False); return fig

def build_rollover_velocity_chart(oi_history, symbol):
    """OI velocity from near→next month. Above 1.2 = conviction roll (bullish); below 0.3 = liquidation."""
    hist = oi_history.get(symbol, []); fig = go.Figure()
    if len(hist) < 3:
        fig.add_annotation(text="Collecting rollover velocity data… refresh a few times",
                           xref="paper",yref="paper",x=0.5,y=0.5,showarrow=False,font=dict(color=MUTED,size=12))
        fig.update_layout(**chart_layout(title="Rollover Velocity (Near→Next OI Flow)")); return fig
    ts_v = [h["ts"] for h in hist]; rv = [h.get("rollover_velocity",0.8) for h in hist]
    colors = [GREEN if v>=1.3 else (CYAN if v>=0.8 else (AMBER if v>=0.3 else RED)) for v in rv]
    fig.add_hline(y=1.3,line_dash="dot",line_color=GREEN,opacity=0.6,annotation_text="Conviction ≥1.3",annotation_font_size=9)
    fig.add_hline(y=0.3,line_dash="dot",line_color=RED,  opacity=0.6,annotation_text="Liquidation ≤0.3",annotation_font_size=9)
    fig.add_trace(go.Scatter(x=ts_v,y=rv,mode="lines+markers",
                             marker=dict(color=colors,size=7),line=dict(color=CYAN,width=2),
                             hovertemplate="<b>%{x}</b><br>Roll Velocity: %{y:.3f}<extra></extra>"))
    lk = chart_layout(title="Rollover Velocity — Δ Next OI / |Δ Near OI|")
    lk["yaxis"]=dict(title="Velocity Ratio",gridcolor="#E2E8F0",zeroline=True,zerolinecolor="#94A3B8")
    lk["xaxis"]=dict(title="Time",gridcolor="#E2E8F0"); fig.update_layout(**lk); return fig
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
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{ font-size: 19px; font-weight: 700; }}
    .section-header {{
        background-color: {SECTION_BG}; color: {ACCENT}; font-weight: 700; font-size: 13px;
        padding: 8px 16px; border-radius: 8px; margin-bottom: 10px;
        letter-spacing: 0.3px; border-left: 4px solid {ACCENT};
    }}
    .regime-banner {{ border-radius: 10px; padding: 12px 20px; margin-bottom: 10px; }}
    .regime-label {{ font-weight: 800; font-size: 16px; letter-spacing: 0.5px; }}
    .regime-sub   {{ font-weight: 500; font-size: 12px; opacity: 0.85; margin-top: 3px; }}
    .strat-box {{
        background-color: {CARD}; border-radius: 10px; padding: 14px;
        border: 1px solid {BORDER}; border-left-width: 4px; border-left-style: solid;
    }}
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
    .explain-text {{ font-size: 9px; color: #888899; margin-top: 3px; font-style: italic; line-height: 1.3; }}
</style>
""", unsafe_allow_html=True)

# Session state
for k, v in [("history", None), ("oi_history", {}), ("last_refresh", 0),
              ("is_owner", False), ("owner_pw_attempt", ""), ("owner_login_error", False),
              ("iv_smile_history", {})]:
    if k not in st.session_state:
        st.session_state[k] = load_history_from_disk() if k == "history" else v

# ── SIDEBAR ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:8px 0 12px 0;">
        <span style="font-size:20px;">🌟</span>
        <div style="font-size:13px;font-weight:700;color:{GOLD};margin-top:4px;">Commodities Dashboard</div>
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
            <div style="font-size:10px;color:#555;margin-top:2px;">Enter password to unlock owner controls</div>
        </div>""", unsafe_allow_html=True)
        pw_input = st.text_input("Owner Password", type="password", key="pw_field", placeholder="Enter owner password…")
        if st.button("🔑 Unlock Owner Mode", use_container_width=True):
            if pw_input == CFG.OWNER_PASSWORD:
                st.session_state["is_owner"] = True; st.session_state["owner_login_error"] = False; st.rerun()
            else: st.session_state["owner_login_error"] = True
        if st.session_state["owner_login_error"]: st.error("Incorrect password.")
    st.divider()
    st.markdown(f"""<div style="font-size:10px;color:{MUTED};text-align:center;line-height:1.6;">
        Data source: {'Dhan API ✅' if CFG.USE_DHAN else 'DEMO MODE'}<br>Auto-refresh: {AUTO_REFRESH_SECONDS}s
    </div>""", unsafe_allow_html=True)

# ── AUTO-REFRESH ──────────────────────────────────────────────────────
time_since = time.time() - st.session_state["last_refresh"]
refresh_placeholder = st.empty()
is_owner = st.session_state["is_owner"]

# ── TITLE ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;margin-bottom:14px;border-bottom:2px solid {ACCENT};padding-bottom:10px;">
    <h1 style="margin:0;color:{GOLD};font-size:26px;font-weight:800;letter-spacing:1px;">
        🌟 Commodities Options Analysis v4
    </h1>
    <div style="font-size:11px;color:{MUTED};margin-top:3px;">
        MCX GOLDM & SILVERM · Term Structure · Gamma Regime · IV Smile · OI Velocity · Regime Detection
    </div>
</div>""", unsafe_allow_html=True)

# ── TOP CONTROLS ──────────────────────────────────────────────────────
col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4, col_ctrl5 = st.columns([1.5,1.5,2,1.5,1])
with col_ctrl1:
    symbol = st.selectbox("COMMODITY", COMMODITY_SYMBOLS, index=0)
with col_ctrl2:
    expiries = fetch_dhan_expiry_list(symbol) if CFG.USE_DHAN else [(date.today()+timedelta(days=10)).strftime("%Y-%m-%d")]
    if is_owner:
        expiry = st.selectbox("EXPIRY", expiries, index=0 if expiries else None)
        st.session_state["selected_expiry"] = expiry
    else:
        expiry = st.session_state.get("selected_expiry", expiries[0] if expiries else "")
        st.markdown(f"""<div style="padding-top:6px;">
            <div style="font-size:10px;color:{MUTED};text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">EXPIRY</div>
            <div style="font-size:14px;font-weight:700;color:#80CBC4;">{expiry}</div>
        </div>""", unsafe_allow_html=True)
with col_ctrl3:
    st.markdown(f"""<div style="padding-top:28px;font-size:11px;color:{MUTED};font-style:italic;">
        📡 Source: {'Dhan API (MCX) ✅' if CFG.USE_DHAN else 'DEMO MODE (Add credentials in secrets)'}
    </div>""", unsafe_allow_html=True)
with col_ctrl4:
    st.markdown(f"""<div style="padding-top:28px;font-size:11px;color:{MUTED};">
        🕐 Last updated: {strftime_ist('%H:%M:%S')} IST
    </div>""", unsafe_allow_html=True)
with col_ctrl5:
    if is_owner:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        refresh_clicked = st.button("⟳ Refresh", use_container_width=True)
    else:
        refresh_clicked = False
        st.markdown(f"""<div style="padding-top:32px;font-size:10px;color:{MUTED};text-align:center;">🔒 View Only</div>""", unsafe_allow_html=True)

if is_owner: auto_refresh = st.checkbox("Auto-refresh every 60s", value=True)
else:        auto_refresh = True

if auto_refresh:
    if is_owner:
        remaining = max(0, AUTO_REFRESH_SECONDS - int(time_since))
        refresh_placeholder.markdown(f"<div style='text-align:center;font-size:11px;color:{MUTED};'>⏳ Auto-refresh in {remaining}s</div>", unsafe_allow_html=True)
    if time_since >= AUTO_REFRESH_SECONDS:
        st.session_state["last_refresh"] = time.time(); st.rerun()
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=AUTO_REFRESH_SECONDS*1000, key="autorefresh")
    except ImportError: pass
if refresh_clicked:
    st.session_state["last_refresh"] = time.time(); st.rerun()

# ── FETCH DATA ────────────────────────────────────────────────────────
# IMPORTANT: option chain FIRST (matches original app order).
# Fetching roll before the option chain adds 5 extra API calls — Dhan MCX
# rate-limits quickly and the 7th call (main option chain) returns partial data,
# causing jagged/wrong charts. Roll is fetched AFTER the main chain just like
# the original app-5.py.
df, spot, exp, source = get_option_chain(symbol, expiry if expiry else None)
if df.empty:
    st.error("No data available. Check API credentials or try again."); st.stop()

m        = compute_metrics(df, spot, symbol, expiry=exp)
atm_iv   = m.get("atm_iv", 0)
df_band  = m.pop("df_band", df)

# Spot fallback: if option chain returned last_price=0, use roll near_ltp
# Roll fetched AFTER option chain to avoid Dhan rate-limit on the main OC call
roll = fetch_futures_roll(symbol) if CFG.USE_DHAN else demo_futures_roll(symbol)
if spot == 0 and roll and roll.get("near_ltp", 0) > 0:
    spot = roll["near_ltp"]
    m    = compute_metrics(df, spot, symbol, expiry=exp)   # recompute with corrected spot
    atm_iv  = m.get("atm_iv", 0)
    df_band = m.pop("df_band", df)

# v4 computations
put_wing_excess, call_wing_excess = compute_wing_excess(df_band, m.get("atm",0), atm_iv, symbol)

smile_hist = st.session_state["iv_smile_history"].get(symbol, [])
iv_smile_result = classify_iv_smile_scenario(df_band, m, spot, symbol, smile_hist)
if put_wing_excess is not None and call_wing_excess is not None:
    smile_hist.append({"ts": now_ist().isoformat(timespec="seconds"),
                       "atm_iv": atm_iv, "put_wing_excess": put_wing_excess, "call_wing_excess": call_wing_excess})
    if len(smile_hist) > 200: smile_hist = smile_hist[-200:]
    st.session_state["iv_smile_history"][symbol] = smile_hist

step = DHAN_SECURITY.get(symbol, DHAN_SECURITY["GOLDM"])["step"]
g_regime, g_regime_desc, vol_regime, g_regime_color = classify_gamma_regime(
    m.get("gex",0), m.get("wall_width",0), m.get("momentum",0),
    atm_iv, m.get("iv_rank",50), spot, m.get("gamma_flip"), step)

carry_anomaly = compute_carry_anomaly(roll, atm_iv) if roll else 1.0
roll_vel_z, roll_vel_interp, roll_vel_color = compute_rollover_velocity_zscore(st.session_state["oi_history"], symbol)

if roll:
    roll["carry_anomaly"]     = carry_anomaly
    roll["rollover_velocity"] = st.session_state["oi_history"].get(symbol,[{}])[-1].get("rollover_velocity", 0.8)

score = compute_score(m, roll)
strat = strategy_recommendation(score, m, symbol)

# Record history
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
    "put_wing_excess":  put_wing_excess  if put_wing_excess  is not None else 0,
    "call_wing_excess": call_wing_excess if call_wing_excess is not None else 0,
    "roll_spread_pct":  roll.get("roll_spread_pct",0) if roll else 0,
    "rollover_pct":     roll.get("rollover_pct",0)    if roll else 0,
    "ts_bias":          roll.get("ts_bias",0)          if roll else 0,
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
                "roll_spread_pct":tick["roll_spread_pct"],"rollover_pct":tick["rollover_pct"],"ts_bias":tick["ts_bias"]})
write_decision_log(log_rec)

# ── HEADER CARDS ──────────────────────────────────────────────────────
gf = m.get("gamma_flip")
theme_color   = SILVER if "SILVER" in symbol else GOLD
iv_rank_color = RED if m.get("iv_rank",50)>70 else (GREEN if m.get("iv_rank",50)<30 else AMBER)
pcr_color     = GREEN if m["pcr"]>0.8 else RED
gf_color      = RED if gf and spot<gf else GREEN
atp_color     = GREEN if m["atm_pressure"]>0 else RED

header_cols = st.columns(10)
header_data = [
    ("Commodity",    symbol,                                theme_color),
    ("Spot Price",   f'₹ {spot:,.2f}',                     "#FFFFFF"),
    ("Expiry",       exp,                                   MUTED),
    ("ATM IV",       f'{m.get("atm_iv",0):.2f} %',         "#80CBC4"),
    ("IV Rank",      f'{m.get("iv_rank",50):.0f}',          iv_rank_color),
    ("PCR",          f'{m["pcr"]}',                         pcr_color),
    ("Max Pain",     f'{int(m["max_pain"])}',               "#CE93D8"),
    ("Gamma Flip",   f'{int(gf)}' if gf else '—',           gf_color),
    ("ATM Pressure", f'{int(m["atm_pressure"]):+,}',        atp_color),
    ("Wall Width",   f'{int(m["wall_width"]):,}',           BLUE),
]
for col, (label, value, color) in zip(header_cols, header_data):
    col.metric(label, value)
    col.markdown(f"""<div style='font-size:9px;color:{color};margin-top:-10px;'>
        <span style='color:{MUTED};'>{label}:</span> <b style='color:{color};'>{value}</b></div>""",
        unsafe_allow_html=True)

st.markdown("---")

# ── SCORE + STRATEGY + METRICS ────────────────────────────────────────
col_gauge, col_strat, col_metrics = st.columns([1,1.2,3])
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
        <div style="font-size:12px;color:#CCC;font-family:monospace;">{strat['legs']}</div>
    </div>""", unsafe_allow_html=True)
with col_metrics:
    metric_items = [
        ("EV Ratio",    f'{m["ev_ratio"]}',              GREEN if m["ev_ratio"]>=1.2 else RED,         METRIC_EXPLAIN["EV Ratio"]),
        ("Net Delta",   f'{int(m["net_delta"]):,}',      GREEN if m["net_delta"]>0 else RED,           METRIC_EXPLAIN["Net Delta"]),
        ("Momentum",    f'{int(m["momentum"]):,}',       GREEN if m["momentum"]>0 else RED,            METRIC_EXPLAIN["Momentum"]),
        ("Vega Skew",   f'{m["vega_skew"]}',             GREEN if m["vega_skew"]>=1.2 else RED,        METRIC_EXPLAIN["Vega Skew"]),
        ("GEX",         f'{m["gex"]:,.0f}',              GREEN if m["gex"]>0 else RED,                 METRIC_EXPLAIN["GEX"]),
        ("Vanna",       f'{m["vanna"]:,.2f}',            GREEN if m["vanna"]>0 else RED,               METRIC_EXPLAIN["Vanna"]),
        ("G/T Ratio",   f'{m["gt_ratio"]}',              BLUE,                                         METRIC_EXPLAIN["G/T Ratio"]),
        ("Skew Slope",  f'{m["skew_slope"]:.4f}',        RED if m["skew_slope"]>0.15 else MUTED,       METRIC_EXPLAIN["Skew Slope"]),
        ("Support",     f'{int(m["support"])}',          GREEN, ""),
        ("Resistance",  f'{int(m["resistance"])}',       RED,   ""),
        ("Near OI %",   f'{m["near_oi_concentration"]*100:.0f}%', BLUE,                                METRIC_EXPLAIN["Near OI %"]),
        ("PCR",         f'{m["pcr"]}',                   GREEN if m["pcr"]>0.8 else RED,               METRIC_EXPLAIN["PCR"]),
    ]
    for row_start in range(0, len(metric_items), 4):
        row_items = metric_items[row_start:row_start+4]
        cols = st.columns(4)
        for col, (label, value, color, explain) in zip(cols, row_items):
            col.markdown(f"""
            <div style="background-color:{CARD};border-radius:8px;padding:10px 14px;
                        border:1px solid {BORDER};min-height:80px;">
                <div style="font-size:10px;color:{MUTED};text-transform:uppercase;letter-spacing:0.5px;">{label}</div>
                <div style="font-size:19px;font-weight:700;color:{color};">{value}</div>
                <div class="explain-text">{explain}</div>
            </div>""", unsafe_allow_html=True)

# ── CHARTS ROW 1 ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">📊 Options Analysis Charts</div>', unsafe_allow_html=True)
chart_cols = st.columns(3)

def _vline(fig, val, color, label):
    if val: fig.add_vline(x=val, line_dash="dash", line_color=color, opacity=0.7,
                          annotation_text=label, annotation_font_size=10)

with chart_cols[0]:
    dv = df_band["call_oi"] - df_band["put_oi"]
    f1 = go.Figure(go.Bar(x=df_band["strike"],y=dv,marker_color=[RED if v>0 else GREEN for v in dv],name="Net OI"))
    _vline(f1,spot,AMBER,"Spot"); _vline(f1,gf,PINK,"γ-Flip")
    f1.update_layout(**chart_layout(title="Net OI (Call − Put) [Delta Proxy]"))
    st.plotly_chart(f1, use_container_width=True, config={"displayModeBar":False})
with chart_cols[1]:
    mv = df_band["call_oi_chg"] - df_band["put_oi_chg"]
    f2 = go.Figure(go.Bar(x=df_band["strike"],y=mv,marker_color=[RED if v>0 else GREEN for v in mv],name="OI Change"))
    _vline(f2,spot,AMBER,"Spot")
    f2.update_layout(**chart_layout(title="OI Momentum (Change in OI)"))
    st.plotly_chart(f2, use_container_width=True, config={"displayModeBar":False})
with chart_cols[2]:
    gv = (df_band["call_oi"]*df_band["call_gamma"]-df_band["put_oi"]*df_band["put_gamma"])*spot**2*0.01
    f3 = go.Figure(go.Bar(x=df_band["strike"],y=gv,marker_color=[RED if v>0 else GREEN for v in gv],name="GEX"))
    _vline(f3,spot,AMBER,"Spot"); _vline(f3,gf,PINK,"γ-Flip")
    f3.update_layout(**chart_layout(title="Gamma Exposure (GEX) per Strike"))
    st.plotly_chart(f3, use_container_width=True, config={"displayModeBar":False})

chart_cols2 = st.columns(2)
with chart_cols2[0]:
    f4 = go.Figure([
        go.Bar(x=df_band["strike"],y=df_band["call_oi"],name="Call OI",marker_color=GOLD),
        go.Bar(x=df_band["strike"],y=df_band["put_oi"], name="Put OI", marker_color=SILVER),
    ])
    _vline(f4,spot,AMBER,"Spot")
    f4.update_layout(**chart_layout(title="Call vs Put OI",barmode="group",
                                    legend=dict(orientation="h",y=1.08,x=0)))
    st.plotly_chart(f4, use_container_width=True, config={"displayModeBar":False})
with chart_cols2[1]:
    # IV Smile — filter to liquid strikes (OI > 0) to avoid zero-IV zigzag
    iv_df = df_band[df_band["call_oi"] + df_band["put_oi"] > 0].copy()
    # Smooth noisy API IVs with a 3-point rolling average for display
    iv_df = iv_df.sort_values("strike")
    civ_smooth = iv_df["call_iv"].rolling(3, center=True, min_periods=1).mean()
    piv_smooth = iv_df["put_iv"].rolling(3, center=True, min_periods=1).mean()
    f5 = go.Figure([
        go.Scatter(x=iv_df["strike"], y=civ_smooth, mode="lines+markers", name="Call IV (smoothed)",
                   line=dict(color=GOLD, width=2), marker=dict(size=5)),
        go.Scatter(x=iv_df["strike"], y=piv_smooth, mode="lines+markers", name="Put IV (smoothed)",
                   line=dict(color=SILVER, width=2), marker=dict(size=5)),
        go.Scatter(x=iv_df["strike"], y=iv_df["call_iv"], mode="markers", name="Call IV (raw)",
                   marker=dict(color=GOLD, size=4, opacity=0.35, symbol="circle-open")),
        go.Scatter(x=iv_df["strike"], y=iv_df["put_iv"],  mode="markers", name="Put IV (raw)",
                   marker=dict(color=SILVER, size=4, opacity=0.35, symbol="circle-open")),
    ])
    _vline(f5,spot,AMBER,"Spot")
    f5.update_layout(**chart_layout(title="IV Smile (liquid strikes, 3pt smoothed)",yaxis_title="IV %",
                                    legend=dict(orientation="h",y=1.08,x=0)))
    st.plotly_chart(f5, use_container_width=True, config={"displayModeBar":False})

# ── OPTION CHAIN TABLE ────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📋 MCX Option Chain — ATM Band")
table_rows = []
for _, r in df_band.sort_values("strike").iterrows():
    K = r["strike"]
    table_rows.append({
        "Call OI": f"{int(r['call_oi']):,}", "Call ΔOI": f"{int(r['call_oi_chg']):,}",
        "Call IV %": f"{r['call_iv']:.1f}", "Call Δ": f"{r['call_delta']:.3f}",
        "STRIKE": f"{int(K)} {'◀ ATM' if K==m.get('atm',0) else ''}",
        "Put Δ": f"{r['put_delta']:.3f}", "Put IV %": f"{r['put_iv']:.1f}",
        "Put ΔOI": f"{int(r['put_oi_chg']):,}", "Put OI": f"{int(r['put_oi']):,}",
    })
st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

# ── KEY PRICE LEVELS ──────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📍 Key Price Levels")
level_items = [
    ("🧲 Max Pain",   int(m["max_pain"]),   "#CE93D8", "Market tends to close near this on expiry"),
    ("🛡 Support",    int(m["support"]),    GREEN,     "Highest put OI — strong floor"),
    ("🚧 Resistance", int(m["resistance"]), RED,       "Highest call OI — strong ceiling"),
    ("⚡ ATM Strike", int(m.get("atm",0)), BLUE,       "At-the-money strike"),
]
if gf: level_items.append(("🔀 Gamma Flip", int(gf), PINK, "Below = dealer short-gamma → trend amplification"))
level_cols = st.columns(len(level_items))
for col, (lbl, val, c, tip) in zip(level_cols, level_items):
    col.markdown(f"""
    <div style="background-color:{CARD};border-radius:8px;padding:10px 18px;
                border:1px solid {BORDER};border-bottom:3px solid {c};">
        <div style="font-size:11px;color:{MUTED};">{lbl}</div>
        <div style="font-size:22px;font-weight:700;color:{c};">{val}</div>
        <div class="explain-text">{tip}</div>
    </div>""", unsafe_allow_html=True)

# ── FUTURES ROLL ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📦 Futures Roll Analysis & Intraday OI Curve")
if roll:
    roll_cols = st.columns(4)
    roll_items = [
        ("Near LTP",       f'₹ {roll["near_ltp"]:,.2f}',      GOLD),
        ("Next LTP",       f'₹ {roll["next_ltp"]:,.2f}',      SILVER),
        ("Roll Spread",    f'₹ {roll["roll_spread"]:+,.2f}',   roll["bias_color"]),
        ("Spread %",       f'{roll["roll_spread_pct"]:+.3f} %',roll["bias_color"]),
        ("Near Month OI",  f'{roll["near_oi"]:,}',             "#80CBC4"),
        ("Next Month OI",  f'{roll["next_oi"]:,}',             "#CE93D8"),
        ("Rollover %",     f'{roll["rollover_pct"]} %',        BLUE),
        ("Structure",      roll["bias"],                        roll["bias_color"]),
    ]
    for i, (label, value, color) in enumerate(roll_items):
        with roll_cols[i % 4]:
            st.markdown(f"""
            <div style="background-color:{CARD};border-radius:8px;padding:8px 12px;
                        border:1px solid {BORDER};min-height:60px;">
                <div style="font-size:10px;color:{MUTED};text-transform:uppercase;">{label}</div>
                <div style="font-size:16px;font-weight:700;color:{color};">{value}</div>
            </div>""", unsafe_allow_html=True)

    roll_chart_cols = st.columns(2)
    with roll_chart_cols[0]:
        f_roll = go.Figure([
            go.Bar(name="Near Month OI",  x=["Near","Next"],y=[roll["near_oi"],roll["next_oi"]],  marker_color=[GOLD,"#CE93D8"]),
            go.Bar(name="Volume",         x=["Near","Next"],y=[roll.get("near_vol",0),roll.get("next_vol",0)],
                   marker_color=["rgba(212,175,55,0.5)","rgba(206,147,216,0.5)"]),
        ])
        f_roll.add_annotation(text=roll["bias"],xref="paper",yref="paper",x=0.5,y=1.12,
                              showarrow=False,font=dict(color=roll["bias_color"],size=12))
        f_roll.update_layout(**chart_layout(title="Near vs Next Month OI & Volume",barmode="group"),
                             legend=dict(orientation="h",y=1.02,x=0))
        st.plotly_chart(f_roll, use_container_width=True, config={"displayModeBar":False})
    with roll_chart_cols[1]:
        oi_hist = st.session_state["oi_history"].get(symbol, [])
        if len(oi_hist) >= 2:
            ts_v  = [r["ts"] for r in oi_hist]; near = [r["near_oi"] for r in oi_hist]
            nxt   = [r["next_oi"] for r in oi_hist]; total = [r["total_oi"] for r in oi_hist]
            f_oi  = go.Figure([
                go.Scatter(x=ts_v,y=total,mode="lines",name="Total OI",  line=dict(color=CYAN,width=2)),
                go.Scatter(x=ts_v,y=near, mode="lines",name="Near OI",   line=dict(color=GOLD,width=1.5,dash="dot")),
                go.Scatter(x=ts_v,y=nxt,  mode="lines",name="Next OI",   line=dict(color="#CE93D8",width=1.5,dash="dot")),
            ])
            f_oi.update_layout(**chart_layout(title="Intraday OI Curve (Today's Session)"),
                               legend=dict(orientation="h",y=1.08,x=0),xaxis_title="Time",yaxis_title="Open Interest")
        else:
            f_oi = go.Figure()
            f_oi.add_annotation(text="Collecting OI history… refresh a few times",xref="paper",yref="paper",
                                x=0.5,y=0.5,showarrow=False,font=dict(color=MUTED,size=13))
            f_oi.update_layout(**chart_layout(title="Intraday OI Curve (Building…)"))
        st.plotly_chart(f_oi, use_container_width=True, config={"displayModeBar":False})
else:
    st.warning("Roll data unavailable — market may be closed or outside MCX trading hours (9 AM – 11:30 PM IST).")

# ── SECTION 8: INTELLIGENCE DASHBOARD ─────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">⚡ Section 8 — OI Regime · OI Velocity · IV History · Combined Bias Panel</div>', unsafe_allow_html=True)

sym_history = _extract_sym_history(st.session_state["history"], symbol)
labels, c_bkt, p_bkt = _bucket_oi_15min(sym_history)
regime_info = _oi_regime_info(c_bkt, p_bkt)
bias_info   = _combined_bias_info(c_bkt, p_bkt)
oi_vel      = compute_oi_velocity(st.session_state["history"], symbol)

st.markdown(f"""
<div class="regime-banner" style="background:{regime_info['bg']};border:1.5px solid {regime_info['border']};">
    <div class="regime-label" style="color:{regime_info['fg']};">{regime_info['label']}</div>
    <div class="regime-sub" style="color:{regime_info['fg']};">{regime_info['sub']}</div>
</div>""", unsafe_allow_html=True)

alert_col = {"NONE":GREEN,"WATCH":AMBER,"DANGER":RED}.get(oi_vel["alert_level"],MUTED)
vel_cols = st.columns([1,1,2])
vel_items = [
    ("Call OI Vel / tick", oi_vel["call_oi_velocity"], oi_vel["call_vel_zscore"]),
    ("Put OI Vel / tick",  oi_vel["put_oi_velocity"],  oi_vel["put_vel_zscore"]),
]
for col, (label, vel, zscore) in zip(vel_cols[:2], vel_items):
    cc = RED if zscore>=2.0 else (AMBER if zscore>=1.2 else (GREEN if zscore<=-1.2 else MUTED))
    col.markdown(f"""
    <div style="background-color:{CARD};border-radius:8px;padding:10px 14px;border:1px solid {BORDER};">
        <div style="font-size:11px;font-weight:700;color:{TEXT};text-transform:uppercase;letter-spacing:0.5px;">{label}</div>
        <div style="font-size:18px;font-weight:700;color:{cc};">{vel:+,.0f}</div>
        <div style="font-size:12px;font-weight:600;color:{cc};">z={zscore:+.2f}σ</div>
    </div>""", unsafe_allow_html=True)
with vel_cols[2]:
    st.markdown(f"""
    <div style="background-color:{CARD};border-radius:8px;padding:10px 14px;
                border:1px solid {BORDER};min-height:80px;display:flex;align-items:center;">
        <span class="alert-text" style="color:{alert_col};">{oi_vel['alert_text']}</span>
    </div>""", unsafe_allow_html=True)

sec8_cols = st.columns(3)
with sec8_cols[0]: st.plotly_chart(build_iv_history_chart(sym_history),use_container_width=True,config={"displayModeBar":False})
with sec8_cols[1]: st.plotly_chart(_build_oi_vel_chart(sym_history,side="CALL"),use_container_width=True,config={"displayModeBar":False})
with sec8_cols[2]: st.plotly_chart(_build_oi_vel_chart(sym_history,side="PUT"),use_container_width=True,config={"displayModeBar":False})

if bias_info:
    MATRIX = [
        ("Call ↑  Put ↑","PINNED",      BLUE,      "Both walls building → pin / range / max-pain gravity"),
        ("Call ↑  Put ↓","BULLISH",     GREEN,     "Ceiling stays, floor gone → slow drift up"),
        ("Call ↓  Put ↑","BEARISH",     RED,       "Ceiling gone, floor stays → slow drift down"),
        ("Call ↓  Put ↓","EXPANSION",   "#9333EA", "All walls dissolving → breakout/breakdown risk"),
        ("Call ↑  Put ~","MILD BEARISH",AMBER,     "Ceiling heavy, floor neutral → capped / mild bear"),
        ("Call ~  Put ↑","MILD BULLISH","#10B981", "Floor solid, ceiling neutral → lifted / mild bull"),
    ]
    bc_c = bias_info["bc"]; c_z = bias_info["c_z"]; p_z = bias_info["p_z"]
    def _z_badge(label, z):
        col_b = RED if z>1.5 else (AMBER if z>0.5 else (GREEN if z<-1.5 else (AMBER if z<-0.5 else MUTED)))
        return f"""<span style="background:{col_b}33;color:{col_b};border:1px solid {col_b};border-radius:6px;
                   padding:2px 8px;font-size:12px;font-weight:700;margin-right:8px;white-space:nowrap;">
                   {label}: {z:+.2f}σ</span>"""
    st.markdown(f"""
    <div style="background-color:{CARD};border:1px solid {bc_c};border-left:4px solid {bc_c};
                border-radius:10px;padding:16px 18px;margin-top:12px;">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:10px;">
            <span style="font-size:15px;font-weight:800;color:{bc_c};">{bias_info['bias']}</span>
            <div>{_z_badge("Call OI",c_z)}{_z_badge("Put OI",p_z)}</div>
        </div>
        <div style="font-size:11px;font-weight:700;color:{MUTED};text-transform:uppercase;margin-bottom:8px;">📋 6-Scenario Reference</div>
    </div>""", unsafe_allow_html=True)
    bias_cols = st.columns(6)
    for col, (combo,blabel,bcolor,bdesc) in zip(bias_cols,MATRIX):
        is_active = blabel in bias_info["bias"]
        col.markdown(f"""
        <div class="bias-cell" style="background:{bcolor}{'22' if is_active else '11'};
                    border:{'2px' if is_active else '1px'} solid {bcolor};">
            <div style="font-size:11px;font-weight:700;color:{bcolor};font-family:monospace;margin-bottom:2px;">{combo}</div>
            <div style="font-size:12px;font-weight:800;color:{bcolor};margin-bottom:3px;">{blabel}</div>
            <div style="font-size:10px;color:{TEXT};line-height:1.4;">{bdesc}</div>
        </div>""", unsafe_allow_html=True)

# ── SECTION 9: FUTURES INTELLIGENCE ──────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">🔮 Section 9 — Futures Intelligence · Gamma Regime · IV Smile · Carry Anomaly</div>', unsafe_allow_html=True)

# Gamma Regime Banner
st.markdown(f"""
<div style="background:{g_regime_color}22;border:1.5px solid {g_regime_color};border-radius:10px;
            padding:12px 20px;margin-bottom:12px;">
    <div style="font-size:14px;font-weight:800;color:{g_regime_color};letter-spacing:0.5px;">
        ⚡ GAMMA REGIME: {g_regime} — {vol_regime}
    </div>
    <div style="font-size:12px;color:{g_regime_color};opacity:0.85;margin-top:3px;">{g_regime_desc}</div>
    <div class="explain-text" style="margin-top:4px;">{METRIC_EXPLAIN["Gamma Regime"]}</div>
</div>""", unsafe_allow_html=True)

# IV Smile Scenario
if iv_smile_result:
    isc = iv_smile_result.get("color", MUTED)
    st.markdown(f"""
    <div style="background:{isc}15;border:1.5px solid {isc};border-radius:10px;
                padding:12px 18px;margin-bottom:12px;">
        <div style="font-size:13px;font-weight:800;color:{isc};">
            📐 IV SMILE: {iv_smile_result.get('scenario','N/A')}
        </div>
        <div style="font-size:11px;color:{isc};margin-top:3px;opacity:0.85;">{iv_smile_result.get('desc','')}</div>
        <div style="font-size:10px;color:{MUTED};margin-top:4px;">
            Put wing excess: {iv_smile_result.get('put_wing_excess',0):+.2f} pp  |  
            Call wing excess: {iv_smile_result.get('call_wing_excess',0):+.2f} pp  |  
            Session trend put: {iv_smile_result.get('trend_put',0):+.2f}  |  
            Session trend call: {iv_smile_result.get('trend_call',0):+.2f}
        </div>
    </div>""", unsafe_allow_html=True)

# Term Structure + Rollover Velocity
sec9_cols = st.columns(2)
with sec9_cols[0]:
    st.plotly_chart(build_term_structure_chart(roll), use_container_width=True, config={"displayModeBar":False})
    if roll:
        sc_nn = roll.get('slope_near_next',0); sc_nf = roll.get('slope_next_far',0)
        st.markdown(f"""
        <div style="background:{CARD};border-radius:8px;padding:8px 12px;border:1px solid {BORDER};margin-top:4px;font-size:11px;">
            <span style="color:{roll.get('ts_color',MUTED)};font-weight:700;">{roll.get('ts_shape','—')}</span>
            <span style="color:{MUTED};margin-left:8px;">{roll.get('ts_desc','')}</span><br>
            <span style="color:{MUTED};">Near→Next: </span>
            <span style="color:{GREEN if sc_nn>0 else RED};font-weight:600;">{sc_nn:+.2f}% p.a.</span>
            <span style="color:{MUTED};margin-left:10px;">Next→Far: </span>
            <span style="color:{GREEN if sc_nf>0 else RED};font-weight:600;">{sc_nf:+.2f}% p.a.</span>
        </div>""", unsafe_allow_html=True)
with sec9_cols[1]:
    st.plotly_chart(build_rollover_velocity_chart(st.session_state["oi_history"], symbol),
                    use_container_width=True, config={"displayModeBar":False})

# Carry Anomaly + metrics row
sec9_mcols = st.columns(5)
ca_color = RED if carry_anomaly >= 1.5 else (GREEN if carry_anomaly <= 0.5 else MUTED)
with sec9_mcols[0]:
    st.markdown(f"""
    <div style="background:{CARD};border-radius:8px;padding:10px 14px;border:1px solid {BORDER};">
        <div style="font-size:10px;color:{MUTED};text-transform:uppercase;">Carry Anomaly</div>
        <div style="font-size:20px;font-weight:700;color:{ca_color};">{carry_anomaly:.2f}×</div>
        <div class="explain-text">{METRIC_EXPLAIN['Carry Anomaly']}</div>
    </div>""", unsafe_allow_html=True)
with sec9_mcols[1]:
    rv_zc = GREEN if roll_vel_z>=1.0 else (RED if roll_vel_z<=-1.0 else MUTED)
    st.markdown(f"""
    <div style="background:{CARD};border-radius:8px;padding:10px 14px;border:1px solid {BORDER};">
        <div style="font-size:10px;color:{MUTED};text-transform:uppercase;">Roll Vel Z-Score</div>
        <div style="font-size:20px;font-weight:700;color:{rv_zc};">{roll_vel_z:+.2f}σ</div>
        <div class="explain-text" style="color:{roll_vel_color};">{roll_vel_interp}</div>
    </div>""", unsafe_allow_html=True)
if roll:
    nvo = roll.get("near_vol_oi",0)
    with sec9_mcols[2]:
        nvo_c = GREEN if nvo>0.3 else (AMBER if nvo>0.1 else MUTED)
        st.markdown(f"""
        <div style="background:{CARD};border-radius:8px;padding:10px 14px;border:1px solid {BORDER};">
            <div style="font-size:10px;color:{MUTED};text-transform:uppercase;">Near Vol/OI</div>
            <div style="font-size:20px;font-weight:700;color:{nvo_c};">{nvo:.3f}</div>
            <div class="explain-text">{METRIC_EXPLAIN['Near Vol/OI']}</div>
        </div>""", unsafe_allow_html=True)
    with sec9_mcols[3]:
        fl = roll.get("far_ltp",0)
        st.markdown(f"""
        <div style="background:{CARD};border-radius:8px;padding:10px 14px;border:1px solid {BORDER};">
            <div style="font-size:10px;color:{MUTED};text-transform:uppercase;">Far Month LTP</div>
            <div style="font-size:20px;font-weight:700;color:{CYAN if fl>0 else MUTED};">
                {'₹{:,.0f}'.format(fl) if fl>0 else '—'}
            </div>
            <div class="explain-text">3rd month futures — part of the term structure curve</div>
        </div>""", unsafe_allow_html=True)
    tsb = roll.get("ts_bias",0)
    with sec9_mcols[4]:
        st.markdown(f"""
        <div style="background:{CARD};border-radius:8px;padding:10px 14px;border:1px solid {BORDER};">
            <div style="font-size:10px;color:{MUTED};text-transform:uppercase;">Term Structure Bias</div>
            <div style="font-size:20px;font-weight:700;color:{GREEN if tsb>0 else (RED if tsb<0 else MUTED)};">{tsb:+d}</div>
            <div class="explain-text">−2 to +2: steepening contango = +2, steep backwardation = −2</div>
        </div>""", unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style="text-align:center;font-size:10px;color:{MUTED};padding:10px;">
    Commodities Options Analysis Dashboard v4.0 · Streamlit Edition<br>
    Data: {'Dhan API (MCX)' if CFG.USE_DHAN else 'DEMO MODE'} · 
    Auto-refresh: {AUTO_REFRESH_SECONDS}s · 
    History ticks: {sum(len(v) for v in st.session_state['history'].values() if isinstance(v,list))}
</div>""", unsafe_allow_html=True)
