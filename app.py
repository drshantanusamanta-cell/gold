"""
╔══════════════════════════════════════════════════════════════════════╗
║  Commodities FUTURES-ONLY Analysis Dashboard v6.1                    ║
║  (GOLDM · SILVERM · GOLDPETAL)                                       ║
║  Streamlit Edition — Deploy on Streamlit Community Cloud            ║
║  Data: Dhan API (primary) | Demo Mode (fallback)                    ║
║  v6.0 — surgical conversion from v5 (options+futures) to futures-only:║
║   • REMOVED: option chain fetch/analysis, Black-Scholes/Greeks,     ║
║     IV solver, GEX, Vanna, Gamma Regime, IV Smile, Max Pain, PCR,   ║
║     Net Delta/Momentum, option-based OI velocity & regime bias,     ║
║     Carry Anomaly (required option-implied ATM IV — no longer       ║
║     available without an option chain), option chain table.         ║
║   • KEPT UNCHANGED: futures roll fetch (near/next/far), term        ║
║     structure, rollover %, roll spread, rollover velocity + z-score, ║
║     intraday OI recorder, term-structure & rollover-velocity charts. ║
║   • Final bias scoring, decision matrix and strategy recommendation  ║
║     now run on FUTURES SIGNALS ONLY.                                 ║
║   • ADDED: GOLD PETAL (GOLDPETAL) as a selectable MCX asset class.   ║
║  v6.1 — practical follow-up improvements:                            ║
║   • Symbol resolution now has a loose fallback match + an owner-only ║
║     diagnostics panel showing whether each symbol (esp. GOLDPETAL)   ║
║     actually resolved real contracts from Dhan's master CSV.         ║
║   • oi_history (rollover-velocity z-score baseline) now persists to  ║
║     disk, pruned to the current day, so it survives an app restart   ║
║     instead of resetting to "Collecting data…" every time.           ║
║   • Directional thresholds are now symbol-aware via get_thresh() /   ║
║     SYMBOL_THRESH_OVERRIDES (currently identical defaults — override ║
║     per symbol once real history is collected).                      ║
║   • A visible "DEMO DATA" banner now appears above the score/verdict ║
║     whenever the live feed fell back to simulated roll data.         ║
║   • Added a "Score Trend — Today's Session" chart.                   ║
║   • Added an owner-only Score Calibration Check that reads the       ║
║     already-logged decision-log CSVs and buckets forward price       ║
║     moves by score range (a sanity check, not a rigorous backtest).  ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os, json, time, warnings, csv as _csv, io, requests
from datetime import date, timedelta, datetime, timezone

_IST = timezone(timedelta(hours=5, minutes=30))
def now_ist() -> datetime:   return datetime.now(_IST)
def strftime_ist(fmt: str):  return now_ist().strftime(fmt)

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objs as go

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Commodity Futures Dashboard v6",
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

AUTO_REFRESH_SECONDS = 60

# ─────────────────────────────────────────────────────────────────────
#  ASSET CLASSES — MCX commodity futures
#  GOLDPETAL added as a selectable asset class alongside GOLDM/SILVERM.
# ─────────────────────────────────────────────────────────────────────
COMMODITY_SYMBOLS = ["GOLDM", "SILVERM", "GOLDPETAL"]
SYMBOL_LABELS = {
    "GOLDM":     "Gold Mini (GOLDM)",
    "SILVERM":   "Silver Mini (SILVERM)",
    "GOLDPETAL": "Gold Petal (GOLDPETAL)",
}

# Populated by _resolve_symbol_df() each time the master CSV is parsed —
# lets the owner-only "Data Source Diagnostics" panel show, at runtime,
# whether each symbol (especially the newly-added GOLDPETAL) actually
# resolved to real MCX contracts instead of silently returning nothing.
_SYMBOL_MATCH_INFO = {}

def _resolve_symbol_df(df_mc, sym):
    """Match a symbol's futures contracts by trading-symbol prefix.
    Tries a strict prefix match first (as before); if that returns nothing
    — e.g. Dhan uses a slightly different prefix than expected for a symbol
    that was just added — falls back to a loose 'contains' match instead of
    silently returning zero contracts. Records what happened in
    _SYMBOL_MATCH_INFO so it can be surfaced in the UI."""
    patterns = {"GOLDM": r'^GOLDM', "SILVERM": r'^SILVERM', "GOLDPETAL": r'^GOLDPETAL'}
    pat = patterns.get(sym)
    if pat:
        df_sym = df_mc[df_mc['SEM_TRADING_SYMBOL'].str.match(pat, na=False)]
        used = f"strict prefix '{pat}'"
        if df_sym.empty:
            df_sym = df_mc[df_mc['SEM_TRADING_SYMBOL'].str.contains(sym, case=False, na=False)]
            used = f"fallback contains '{sym}' (strict prefix matched nothing)"
    else:
        df_sym = df_mc[df_mc['SEM_TRADING_SYMBOL'].str.startswith(sym)]
        used = f"startswith '{sym}'"
    _SYMBOL_MATCH_INFO[sym] = {
        "pattern_used": used,
        "matched": not df_sym.empty,
        "count": int(len(df_sym)),
        "sample": df_sym['SEM_TRADING_SYMBOL'].unique().tolist()[:5] if not df_sym.empty else [],
    }
    return df_sym

# ─────────────────────────────────────────────────────────────────────
#  CENTRALISED THRESHOLDS — futures-only signals
#  Single source of truth for all metric thresholds.
# ─────────────────────────────────────────────────────────────────────
class THRESH:
    # Rollover velocity (ratio of Δ next OI to |Δ near OI|)
    ROLL_VEL_CONVICTION  = 1.3   # ≥ → bullish conviction roll
    ROLL_VEL_NORMAL      = 0.8   # ≥ → normal pace
    ROLL_VEL_SLOW        = 0.3   # ≥ → slow / cautious
    # negative-velocity buckets (liquidation side)
    ROLL_VEL_LIQUID_MILD = -0.3  # ≥ → mild liquidation
    ROLL_VEL_LIQUID_HARD = -1.0  # < → hard liquidation

    # Rollover-velocity z-score
    ZSCORE_BULL          = 1.0   # ≥ → bullish (above norm)
    ZSCORE_BEAR          = -1.0  # ≤ → bearish (below norm)

    # Roll spread %
    ROLL_SPREAD_STRONG_CONTANGO  = 0.2
    ROLL_SPREAD_STRONG_BACKWARD  = -0.2

    # Rollover %
    ROLLOVER_PCT_ADVANCED = 40
    ROLLOVER_PCT_NORMAL   = 20

    # Z-score rolling window for rollover velocity
    ROLL_VEL_ZSCORE_WINDOW = 20   # ticks (≈ 20 min at 60s refresh)
    ROLL_VEL_ZSCORE_MIN    = 5    # need at least this many samples

    # History retention
    HISTORY_RETENTION_DAYS = 7
    HISTORY_MAX_TICKS       = 600   # per symbol

# ─────────────────────────────────────────────────────────────────────
#  SYMBOL-AWARE THRESHOLD OVERRIDES
#  The THRESH values above are shared defaults, originally tuned on
#  GOLDM/SILVERM. GOLDPETAL is a much smaller (1 gram) contract and may
#  show different noise characteristics in roll spread % / rollover
#  velocity — these overrides are UNCALIBRATED placeholders (currently
#  identical to the defaults) until real GOLDPETAL history is collected.
#  Add per-symbol keys here once you've logged a few weeks of data — the
#  owner-only "Threshold Calibration" panel further down shows the
#  effective thresholds currently in force for the selected symbol.
# ─────────────────────────────────────────────────────────────────────
SYMBOL_THRESH_OVERRIDES = {
    # Example — uncomment / adjust once you have real GOLDPETAL data:
    # "GOLDPETAL": {"ROLL_SPREAD_STRONG_CONTANGO": 0.15, "ROLLOVER_PCT_ADVANCED": 35},
}

def get_thresh(symbol, name):
    """Symbol-aware threshold lookup. Falls back to the shared THRESH class
    default when no override is defined for this symbol."""
    override = SYMBOL_THRESH_OVERRIDES.get(symbol, {})
    return override.get(name, getattr(THRESH, name))

@st.cache_data(ttl=86400, show_spinner=False)
def get_dynamic_futures_ids():
    """Downloads Dhan master CSV and resolves near/next/far futures IDs per symbol."""
    url = "https://images.dhan.co/api-data/api-scrip-master.csv"
    try:
        resp = requests.get(url, timeout=15); resp.raise_for_status()
        df   = pd.read_csv(io.StringIO(resp.text))
        df.columns = [c.upper() for c in df.columns]
        df_mc = df[(df['SEM_EXM_EXCH_ID'] == 'MCX') & (df['SEM_INSTRUMENT_NAME'] == 'FUTCOM')]
        id_map = {}
        for sym in COMMODITY_SYMBOLS:
            df_sym = _resolve_symbol_df(df_mc, sym)
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

@st.cache_data(ttl=86400, show_spinner=False)
def get_futures_contracts():
    """
    Returns {symbol: [{"id", "expiry", "tsym"}, ...]} for near/next/far FUTCOM contracts.
    Expiry dates are parsed from the master CSV — no option-chain API call needed.
    """
    url = "https://images.dhan.co/api-data/api-scrip-master.csv"
    try:
        resp = requests.get(url, timeout=15); resp.raise_for_status()
        df   = pd.read_csv(io.StringIO(resp.text))
        df.columns = [c.upper() for c in df.columns]
        df_mc = df[(df['SEM_EXM_EXCH_ID'] == 'MCX') & (df['SEM_INSTRUMENT_NAME'] == 'FUTCOM')]
        today   = date.today().isoformat()
        result  = {}

        def _parse_exp(raw):
            raw = str(raw or '').strip()
            try:
                if len(raw) == 10 and raw[4] == '-':   return raw              # YYYY-MM-DD
                if len(raw) >= 10 and raw[4] == '-':   return raw[:10]            # YYYY-MM-DD HH:MM:SS
                if len(raw) == 8  and raw.isdigit():   return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
                return datetime.strptime(raw, "%d-%b-%Y").strftime("%Y-%m-%d")
            except Exception: return raw

        for sym in COMMODITY_SYMBOLS:
            df_sym = _resolve_symbol_df(df_mc, sym)
            exp_col = 'SEM_EXPIRY_DATE' if 'SEM_EXPIRY_DATE' in df_sym.columns else 'SEM_EXPIRY_CODE'
            df_sym  = df_sym.dropna(subset=[exp_col]).copy()
            df_sym['_exp'] = df_sym[exp_col].apply(_parse_exp)
            df_sym  = df_sym[df_sym['_exp'] >= today].sort_values('_exp')
            contracts = []
            for _, row in df_sym.iterrows():
                contracts.append({
                    "id":    int(row['SEM_SMST_SECURITY_ID']),
                    "expiry": row['_exp'],
                    "tsym":  str(row.get('SEM_TRADING_SYMBOL', '') or ''),
                })
                if len(contracts) >= 3: break
            result[sym] = contracts
        print(f"[FutContracts] {result}")
        return result
    except Exception as e:
        print(f"[FutContracts] Failed: {e}"); return {}

BG         = "#FFFFFF"; CARD       = "#F8FAFC"; TEXT       = "#1E293B"
ACCENT     = "#B8960C"; MUTED      = "#64748B"; GOLD       = "#B8960C"
SILVER     = "#475569"; GREEN      = "#059669"; RED        = "#DC2626"
AMBER      = "#D97706"; BLUE       = "#2563EB"; CYAN       = "#0891B2"
PINK       = "#DB2777"; BORDER     = "#E2E8F0"; SECTION_BG = "#F1F5F9"
PURPLE     = "#7C3AED"

METRIC_EXPLAIN = {
    "Roll Spread":       "Price difference between near and next futures — positive = contango (normal carry).",
    "Spread %":          "Roll spread as % of near-month price — measures carry cost in percentage terms.",
    "Rollover %":        "How much OI has shifted to next month — high % means expiry rollover well advanced.",
    "Term Structure":    "Shape of the futures curve across 3 months — steepening contango signals bullish carry.",
    "Rollover Velocity": "Rate of OI moving from near to next month — above 1.3 means longs are adding conviction.",
    "Near Vol/OI":       "Volume-to-OI ratio for near-month — above 0.3 means active fresh positioning.",
    "Slope Near→Next":   "Annualised carry from near to next month — positive = contango (bullish carry).",
    "Slope Next→Far":    "Annualised carry from next to far month — steeper = bullish acceleration.",
}
# ─────────────────────────────────────────────────────────────────────
#  HISTORY PERSISTENCE
# ─────────────────────────────────────────────────────────────────────
HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "commodity_futures_history.json")
LOG_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "commodity_decision_logs")
os.makedirs(LOG_DIR, exist_ok=True)

_CSV_COLUMNS = [
    "ts", "symbol", "near_ltp", "next_ltp", "far_ltp",
    "roll_spread_pct", "rollover_pct", "ts_bias",
    "rollover_velocity", "roll_vel_z", "score",
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

def _prune_to_history_window(history):
    """Keep HISTORY_RETENTION_DAYS of ticks (not just today) so z-scores
    have a meaningful baseline. Returns dict {symbol: [tick,...]} with old ticks pruned."""
    cutoff = (date.today() - timedelta(days=THRESH.HISTORY_RETENTION_DAYS)).isoformat()
    pruned = {}
    for s, tks in history.items():
        if not isinstance(tks, list): continue
        kept = []
        for t in tks:
            if not isinstance(t, dict): continue
            ts = str(t.get("ts",""))
            # ts may be ISO with T separator (e.g., "2026-06-27T14:30:00+05:30")
            # or "YYYY-MM-DD HH:MM:SS" — extract date prefix
            date_part = ts.split("T")[0].split(" ")[0] if ts else ""
            if date_part >= cutoff:
                kept.append(t)
        # also cap to max ticks per symbol to bound memory
        if len(kept) > THRESH.HISTORY_MAX_TICKS:
            kept = kept[-THRESH.HISTORY_MAX_TICKS:]
        pruned[s] = kept
    return pruned

def load_history_from_disk():
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f: raw = json.load(f)
            p = _prune_to_history_window(raw)
            print(f"[History] Loaded {sum(len(v) for v in p.values())} ticks (window={THRESH.HISTORY_RETENTION_DAYS}d)"); return p
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
#  OI-HISTORY PERSISTENCE
#  Previously oi_history (which drives rollover velocity + its z-score —
#  a core signal in the bias matrix) lived only in Streamlit's in-memory
#  session state: it reset to "Collecting data…" on every server restart,
#  which matters in practice because Streamlit Community Cloud sleeps and
#  restarts idle apps. Persisting it the same way as `history` means the
#  z-score baseline survives a restart instead of re-warming from scratch.
#  Ticks are pruned to TODAY ONLY on load (not the 7-day window used for
#  `history`) because near/next OI levels reset around rollover and mixing
#  yesterday's OI base into today's velocity calc would be wrong.
# ─────────────────────────────────────────────────────────────────────
OI_HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "commodity_futures_oi_history.json")

def _disp_hm(ts: str) -> str:
    """Extract just the HH:MM portion for chart x-axis labels, regardless of
    whether `ts` is 'YYYY-MM-DD HH:MM' (new format) or bare 'HH:MM' (old
    format, kept for backward compatibility with any already-saved file)."""
    ts = str(ts)
    return ts.split(" ")[-1] if " " in ts else ts

def _prune_oi_to_today(oi_history):
    """Keep only today's ticks — OI velocity should never be computed across
    a day boundary (rollover / fresh session), so stale ticks from a
    previous day are dropped on load rather than silently blended in."""
    today = date.today().isoformat()
    pruned = {}
    for s, ticks in oi_history.items():
        if not isinstance(ticks, list): continue
        pruned[s] = [t for t in ticks if isinstance(t, dict) and str(t.get("ts", "")).startswith(today)]
    return pruned

def load_oi_history_from_disk():
    try:
        if os.path.exists(OI_HISTORY_FILE):
            with open(OI_HISTORY_FILE, "r", encoding="utf-8") as f: raw = json.load(f)
            p = _prune_oi_to_today(raw)
            print(f"[OIHistory] Loaded {sum(len(v) for v in p.values())} ticks (today only)"); return p
    except Exception as e: print(f"[OIHistory] Load error: {e}")
    return {}

def save_oi_history_to_disk(oi_history):
    try:
        tmp = OI_HISTORY_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in oi_history.items() if isinstance(v, list)}, f)
        os.replace(tmp, OI_HISTORY_FILE)
    except Exception as e: print(f"[OIHistory] Save error: {e}")

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

# ─────────────────────────────────────────────────────────────────────
#  FUTURES ROLL — 3-MONTH  (UNCHANGED FROM v5)
#  Uses /v2/marketfeed/quote (ONE call) for LTP + futures OI + volume.
#  Falls back to /v2/marketfeed/ltp if quote gives no prices.
#  Expiry dates come from master CSV via get_futures_contracts() — no
#  option-chain expiry-list or _fetch_oi calls, so zero extra API hits.
#  Cached for 55s so concurrent viewers share one quote call per 55s
#  window instead of hitting Dhan per visitor.
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=55, show_spinner=False)
def fetch_futures_roll(symbol="GOLDM") -> dict:
    if not CFG.USE_DHAN: return {}

    contracts = get_futures_contracts().get(symbol, [])
    if len(contracts) < 2:
        print(f"[Roll] < 2 contracts in master CSV for {symbol}"); return {}

    near_c = contracts[0]; next_c = contracts[1]
    far_c  = contracts[2] if len(contracts) >= 3 else None

    near_id     = near_c["id"];    next_id     = next_c["id"];    far_id     = far_c["id"]    if far_c else None
    near_expiry = near_c["expiry"]; next_expiry = next_c["expiry"]; far_expiry = far_c["expiry"] if far_c else ""
    near_tsym   = near_c["tsym"];  next_tsym   = next_c["tsym"];  far_tsym   = far_c["tsym"]  if far_c else ""

    headers      = {"access-token": CFG.DHAN_ACCESS_TOKEN, "client-id": str(CFG.DHAN_CLIENT_ID), "Content-Type": "application/json"}
    ids_to_fetch = [i for i in [near_id, next_id, far_id] if i is not None]

    near_ltp = next_ltp = far_ltp = 0.0
    near_oi  = next_oi  = far_oi  = 0
    near_vol = next_vol = far_vol = 0

    # ── Try /v2/marketfeed/quote — gives LTP + OI + Volume in ONE call ──
    try:
        qr = requests.post("https://api.dhan.co/v2/marketfeed/quote", headers=headers,
                           json={"MCX_COMM": ids_to_fetch}, timeout=12)
        resp_json = qr.json()
        raw_data  = resp_json.get("data", {})
        print(f"[Roll/Quote/{symbol}] status={resp_json.get('status','')} data_type={type(raw_data).__name__} keys={list(raw_data.keys())[:5] if isinstance(raw_data,dict) else 'N/A'}")

        # Build a unified lookup dict keyed by security ID string
        _quotes = {}
        if isinstance(raw_data, dict):
            # Expected: {"MCX_COMM": {"510764": {"ltp": ..., "openInterest": ...}}}
            segment_data = raw_data.get("MCX_COMM", raw_data)
            if isinstance(segment_data, dict):
                _quotes = {str(k): v for k, v in segment_data.items() if v and isinstance(v, dict)}
            # If segment_data is a list, handle array format
            elif isinstance(segment_data, list):
                for item in segment_data:
                    sid = str(item.get("symbolId", item.get("securityId", item.get("id", ""))))
                    if sid: _quotes[sid] = item
        elif isinstance(raw_data, list):
            for item in raw_data:
                sid = str(item.get("symbolId", item.get("securityId", item.get("id", ""))))
                if sid: _quotes[sid] = item

        def _q(sid):
            if sid is None: return 0.0, 0, 0
            d   = _quotes.get(str(sid), {}) or {}
            ltp = float(d.get("ltp", d.get("lastPrice", d.get("last_price", d.get("lastTradedPrice", 0)))) or 0)
            oi  = int(d.get("openInterest", d.get("oi", d.get("open_interest", d.get("netOI", 0)))) or 0)
            vol = int(d.get("totalVolume", d.get("volume", d.get("total_volume", 0)))) or 0
            return ltp, oi, vol

        near_ltp, near_oi, near_vol = _q(near_id)
        next_ltp, next_oi, next_vol = _q(next_id)
        far_ltp,  far_oi,  far_vol  = _q(far_id)
        print(f"[Roll/Quote/{symbol}] near={near_ltp},oi={near_oi} next={next_ltp},oi={next_oi} far={far_ltp},oi={far_oi}")
    except Exception as e:
        print(f"[Roll/{symbol}] Quote endpoint failed: {e}")

    # ── Fallback: /v2/marketfeed/ltp (price only, no OI) ──
    if near_ltp == 0:
        try:
            lr    = requests.post("https://api.dhan.co/v2/marketfeed/ltp", headers=headers,
                                  json={"MCX_COMM": ids_to_fetch}, timeout=10)
            ld    = lr.json().get("data", {}).get("MCX_COMM", {})
            near_ltp = float((ld.get(str(near_id)) or {}).get("last_price", 0) or 0)
            next_ltp = float((ld.get(str(next_id)) or {}).get("last_price", 0) or 0) if next_id else 0.0
            far_ltp  = float((ld.get(str(far_id))  or {}).get("last_price", 0) or 0) if far_id  else 0.0
            print(f"[Roll/LTP/{symbol}] near={near_ltp} next={next_ltp} far={far_ltp}")
        except Exception as e:
            print(f"[Roll/{symbol}] LTP fallback failed: {e}")

    if near_ltp == 0:
        print(f"[Roll] near_ltp=0 for {symbol} — market closed or data unavailable"); return {}

    total_oi        = near_oi + next_oi
    roll_spread     = round(next_ltp - near_ltp, 2)
    roll_spread_pct = round((roll_spread / near_ltp * 100) if near_ltp else 0, 3)
    rollover_pct    = round((next_oi / total_oi * 100) if total_oi else 0, 1)

    # Annualised term-structure slopes
    slope_near_next = slope_next_far = 0.0
    try:
        nd      = datetime.strptime(near_expiry, "%Y-%m-%d").date()
        xtd     = datetime.strptime(next_expiry, "%Y-%m-%d").date()
        days_nn = max((xtd - nd).days, 1)
        slope_near_next = round((next_ltp - near_ltp) / near_ltp * (365/days_nn) * 100, 2) if near_ltp > 0 else 0.0
        if far_expiry and far_ltp > 0 and next_ltp > 0:
            fd      = datetime.strptime(far_expiry, "%Y-%m-%d").date()
            days_nf = max((fd - xtd).days, 1)
            slope_next_far = round((far_ltp - next_ltp) / next_ltp * (365/days_nf) * 100, 2)
    except Exception as e:
        print(f"[Roll/{symbol}] Slope calc error: {e}")
        slope_near_next = roll_spread_pct

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
        "near_expiry": near_expiry, "next_expiry": next_expiry, "far_expiry": far_expiry,
        "near_tsym":   near_tsym,   "next_tsym":   next_tsym,   "far_tsym":   far_tsym,
        "has_far": bool(far_expiry and far_ltp > 0),
    }

# ─────────────────────────────────────────────────────────────────────
#  DEMO MODE — futures roll only. Base prices scaled per symbol so the
#  Gold Petal (1 gram lot, ~1/10th GOLDM price) demo data looks realistic.
# ─────────────────────────────────────────────────────────────────────
def demo_futures_roll(symbol="GOLDM") -> dict:
    _base = {"GOLDM": 93500.0, "SILVERM": 96500.0, "GOLDPETAL": 9350.0}.get(
        symbol, 93500.0 if "GOLD" in symbol else 96500.0)
    near_ltp = _base + np.random.normal(0, _base*0.00085)
    spread1  = abs(np.random.normal(_base*0.00128, _base*0.00043))
    spread2  = abs(np.random.normal(_base*0.00118, _base*0.00037))
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
    # Demo trading symbols
    from calendar import month_abbr
    _mn = lambda d: month_abbr[datetime.strptime(d,"%Y-%m-%d").month].upper() + datetime.strptime(d,"%Y-%m-%d").strftime("%y")
    near_tsym = f"{symbol}{_mn(near_expiry)}"; next_tsym = f"{symbol}{_mn(next_expiry)}"; far_tsym = f"{symbol}{_mn(far_expiry)}"
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
        "near_tsym": near_tsym, "next_tsym": next_tsym, "far_tsym": far_tsym,
        "has_far": True,
    }

# ─────────────────────────────────────────────────────────────────────
#  SCORING — futures signals only
# ─────────────────────────────────────────────────────────────────────
def _linear_score(value, lo, hi, weight, invert=False):
    """Linear interpolation between lo (score=0) and hi (score=weight).
    If invert=True, the polarity is flipped (high value → low score).
    Values outside [lo, hi] are clipped."""
    if hi == lo: return weight / 2
    if invert:
        v = max(lo, min(hi, value))
        return weight * (hi - v) / (hi - lo)
    else:
        v = max(lo, min(hi, value))
        return weight * (v - lo) / (hi - lo)

def compute_futures_score(roll, roll_vel_z=None, symbol="GOLDM"):
    """Futures-only composite score (0-100). Combines term-structure bias,
    roll spread %, rollover velocity, rollover %, and rollover-velocity
    z-score into a single directional score. Neutral baseline = 15,
    ceiling = 100 when every futures signal is maximally bullish.
    Thresholds are looked up per-symbol via get_thresh() (see
    SYMBOL_THRESH_OVERRIDES) so GOLDPETAL can eventually get its own
    calibration without touching GOLDM/SILVERM."""
    if not roll: return 50.0
    score = 15  # neutral baseline
    tsb = safe_num(roll.get("ts_bias", 0))
    score += _linear_score(tsb, -2, 2, 25)                                          # term structure: 0..25
    rsp = safe_num(roll.get("roll_spread_pct", 0))
    score += _linear_score(rsp, -0.3, 0.3, 20)                                       # roll spread %: 0..20
    rv  = safe_num(roll.get("rollover_velocity", get_thresh(symbol, "ROLL_VEL_NORMAL")))
    score += _linear_score(rv, get_thresh(symbol, "ROLL_VEL_LIQUID_HARD"), get_thresh(symbol, "ROLL_VEL_CONVICTION"), 20)  # rollover velocity: 0..20
    rp  = safe_num(roll.get("rollover_pct", get_thresh(symbol, "ROLLOVER_PCT_NORMAL")))
    score += _linear_score(rp, 0, get_thresh(symbol, "ROLLOVER_PCT_ADVANCED"), 10)   # rollover %: 0..10
    if roll_vel_z is not None:
        score += _linear_score(safe_num(roll_vel_z, 0), get_thresh(symbol, "ZSCORE_BEAR")*2, get_thresh(symbol, "ZSCORE_BULL")*2, 10)  # roll vel z: 0..10
    else:
        score += 5   # half credit while the z-score baseline is still building
    return round(min(max(score, 0), 100), 1)

def futures_recommendation(score, roll, symbol="GOLDM"):
    """Futures-only market mode + directional recommendation (no option legs)."""
    near_tsym = (roll or {}).get("near_tsym", symbol)
    next_tsym = (roll or {}).get("next_tsym", symbol)
    if   score >= 85: name, color, action = "Strong Long Futures",           "#00C853", f"BUY {near_tsym} — ride the trend, trail stop below near-month support"
    elif score >= 70: name, color, action = "Long Futures / Bullish Carry",  "#69F0AE", f"Accumulate {near_tsym} on dips; roll into {next_tsym} as expiry nears"
    elif score >= 55: name, color, action = "Mildly Long / Accumulate",      "#B2FF59", f"Small long in {near_tsym}; wait for confirmation before adding"
    elif score >= 45: name, color, action = "Neutral / Range-Bound",         "#FFD600", f"No strong edge — range-trade {near_tsym} or stay flat"
    elif score >= 31: name, color, action = "Mildly Short / Reduce",         "#FF6D00", f"Trim longs in {near_tsym}; avoid fresh buying"
    elif score >= 16: name, color, action = "Short Futures / Bearish Carry", "#F44336", f"Sell {near_tsym} on rallies; watch backwardation for squeeze risk"
    else:             name, color, action = "Strong Short Futures",         "#B71C1C", f"SELL {near_tsym} — delivery-pressure / backwardation squeeze risk"

    if   score >= 55: mode, mc = "TREND MODE — Bullish", "#00E676"
    elif score >= 45: mode, mc = "NEUTRAL / RANGE",      "#FFD740"
    else:             mode, mc = "TREND MODE — Bearish", "#FF5252"
    return {"name": name, "action": action, "color": color, "market_mode": mode, "mode_color": mc}

# ─────────────────────────────────────────────────────────────────────
#  INTRADAY OI RECORDER — with rollover velocity (UNCHANGED FROM v5)
# ─────────────────────────────────────────────────────────────────────
def record_intraday_oi(symbol: str, roll: dict, oi_history: dict):
    """Stable denominator with OI-relative floor, None for low-activity periods.

    v = d_next / max(|d_near|, 0.5% × near_oi)
      - Floor scales with near OI (5 lots minimum).
      - Sign comes from d_next only: positive = next OI growing (bullish rollover),
        negative = next OI shrinking (bearish liquidation).
      - When BOTH d_near and d_next are below floor, mark velocity as None so the
        z-score baseline excludes the tick instead of inheriting a stale value.
      - Clip to ±10 to prevent extreme outliers polluting the z-score baseline.

    ts is stored as full 'YYYY-MM-DD HH:MM' (not bare 'HH:MM') so that
    persisting oi_history to disk across a restart can't accidentally
    splice today's ticks onto a same-minute tick from a previous day.
    """
    if not roll: return oi_history
    ts     = strftime_ist("%Y-%m-%d %H:%M")
    noi    = roll.get("near_oi", 0); xoi = roll.get("next_oi", 0)
    entry  = {"ts": ts, "near_oi": noi, "next_oi": xoi, "total_oi": noi + xoi}
    hist   = oi_history.setdefault(symbol, [])
    if len(hist) >= 1:
        prev = hist[-1]
        d_near = noi - prev.get("near_oi", 0)
        d_next = xoi - prev.get("next_oi", 0)
        # OI-relative floor — 0.5 % of current near OI (minimum 5 lots)
        floor = max(5.0, 0.005 * float(noi))
        if abs(d_near) < floor and abs(d_next) < floor:
            # Too quiet to measure — mark as None so z-score skips it
            entry["rollover_velocity"] = None
        else:
            denom = max(abs(d_near), floor)
            v = d_next / denom if denom != 0 else 0.0
            entry["rollover_velocity"] = round(max(-10.0, min(10.0, v)), 3)
    else:
        entry["rollover_velocity"] = None  # first tick — no baseline yet
    if hist and hist[-1]["ts"] == ts: hist[-1] = entry
    else: hist.append(entry)
    if len(hist) > THRESH.HISTORY_MAX_TICKS: oi_history[symbol] = hist[-THRESH.HISTORY_MAX_TICKS:]
    return oi_history

def compute_rollover_velocity_zscore(oi_history, symbol):
    """Rolling-window z-score of rollover velocity.
    z = (last - mean(last N ticks)) / std(last N ticks)
    Returns None / "Collecting data…" when insufficient — UI renders "—".
    Bucket thresholds are symbol-aware via get_thresh()."""
    hist = oi_history.get(symbol, [])
    rvs_all = [h.get("rollover_velocity") for h in hist if h.get("rollover_velocity") is not None]
    if len(rvs_all) < THRESH.ROLL_VEL_ZSCORE_MIN:
        return None, "Collecting data…", MUTED

    window = min(THRESH.ROLL_VEL_ZSCORE_WINDOW, len(rvs_all))
    arr = np.array(rvs_all[-window:], dtype=float)
    std = float(arr.std()) if arr.std() > 1e-9 else 1.0
    z = float((arr[-1] - arr.mean()) / std)

    latest = float(arr[-1])
    if   latest >= get_thresh(symbol, "ROLL_VEL_CONVICTION"):
        interp, color = f"Conviction roll — {latest:.2f} (longs adding)", GREEN
    elif latest >= get_thresh(symbol, "ROLL_VEL_NORMAL"):
        interp, color = f"Normal roll — {latest:.2f}", CYAN
    elif latest >= get_thresh(symbol, "ROLL_VEL_SLOW"):
        interp, color = f"Slow roll — {latest:.2f} (caution)", AMBER
    elif latest >= get_thresh(symbol, "ROLL_VEL_LIQUID_MILD"):
        interp, color = f"Mild liquidation — {latest:.2f}", "#FF6D00"
    else:
        interp, color = f"Hard liquidation — {latest:.2f} (bearish unwind)", RED
    return round(z, 2), interp, color

# ─────────────────────────────────────────────────────────────────────
#  FUTURES BIAS DECISION MATRIX
#  Aggregates the futures signals above into a single decision matrix.
#  (Options-based signals and Carry Anomaly — which required option-
#  implied ATM IV — have been removed; this reads futures-only inputs.)
# ─────────────────────────────────────────────────────────────────────
def _bias_tag(bias, strength, color):
    """Render a small colour-coded chip for inline bias explanations."""
    if not bias: return ""
    label = bias.title()
    if strength and strength not in ("—", ""):
        label = f"{label} ({strength})"
    return (f"<span style='background:{color}22;color:{color};border:1px solid {color};"
            f"border-radius:5px;padding:1px 7px;font-size:10px;font-weight:700;"
            f"letter-spacing:0.3px;white-space:nowrap;'>{label}</span>")

def compute_futures_market_bias(roll, roll_vel_z, score, symbol="GOLDM"):
    """Aggregate futures signals into a single decision matrix.
    Thresholds are looked up per-symbol via get_thresh()."""
    signals = []

    def _add(name, value, bias, strength, color):
        signals.append({"name": name, "value": value, "bias": bias,
                        "strength": strength, "color": color})

    if roll:
        rsp = safe_num(roll.get("roll_spread_pct", 0)); rs = safe_num(roll.get("roll_spread", 0))
        if   rsp >  get_thresh(symbol, "ROLL_SPREAD_STRONG_CONTANGO"): _add("Roll Spread", f"₹{rs:+,.2f} ({rsp:+.3f}%)", "BULLISH","Strong",GREEN)
        elif rsp >  0:                                                 _add("Roll Spread", f"₹{rs:+,.2f} ({rsp:+.3f}%)", "BULLISH","Mild",  "#69F0AE")
        elif rsp <  get_thresh(symbol, "ROLL_SPREAD_STRONG_BACKWARD"): _add("Roll Spread", f"₹{rs:+,.2f} ({rsp:+.3f}%)", "BEARISH","Strong",RED)
        elif rsp <  0:                                                 _add("Roll Spread", f"₹{rs:+,.2f} ({rsp:+.3f}%)", "BEARISH","Mild",  "#FF6D00")
        else:                                                          _add("Roll Spread", f"₹{rs:+,.2f}",               "SIDEWAYS","—",   MUTED)

        tsb = safe_num(roll.get("ts_bias", 0)); tss = roll.get("ts_shape", "—")
        if   tsb >=  2: _add("Term Structure", tss, "BULLISH","Strong",GREEN)
        elif tsb ==  1: _add("Term Structure", tss, "BULLISH","Mild",  "#69F0AE")
        elif tsb <= -2: _add("Term Structure", tss, "BEARISH","Strong",RED)
        elif tsb == -1: _add("Term Structure", tss, "BEARISH","Mild",  "#FF6D00")
        else:           _add("Term Structure", tss, "SIDEWAYS","—",    MUTED)

        rv = safe_num(roll.get("rollover_velocity", get_thresh(symbol, "ROLL_VEL_NORMAL")))
        if   rv >= get_thresh(symbol, "ROLL_VEL_CONVICTION"): _add("Rollover Velocity", f"{rv:.2f}", "BULLISH","Conviction", GREEN)
        elif rv >= get_thresh(symbol, "ROLL_VEL_NORMAL"):      _add("Rollover Velocity", f"{rv:.2f}", "SIDEWAYS","Normal",   CYAN)
        elif rv >= get_thresh(symbol, "ROLL_VEL_SLOW"):        _add("Rollover Velocity", f"{rv:.2f}", "BEARISH","Slow",      AMBER)
        else:                                                   _add("Rollover Velocity", f"{rv:.2f}", "BEARISH","Liquidation",RED)

        rp = safe_num(roll.get("rollover_pct", 0))
        if   rp >= get_thresh(symbol, "ROLLOVER_PCT_ADVANCED"): _add("Rollover %", f"{rp:.0f}%", "BULLISH","Advanced", GREEN)
        elif rp >= get_thresh(symbol, "ROLLOVER_PCT_NORMAL"):    _add("Rollover %", f"{rp:.0f}%", "SIDEWAYS","Normal",  CYAN)
        else:                                                     _add("Rollover %", f"{rp:.0f}%", "SIDEWAYS","Early",   MUTED)

        if roll_vel_z is not None:
            rz = safe_num(roll_vel_z, 0)
            if   rz >= get_thresh(symbol, "ZSCORE_BULL"): _add("Roll Vel Z", f"{rz:+.2f}σ", "BULLISH","Above norm", GREEN)
            elif rz <= get_thresh(symbol, "ZSCORE_BEAR"): _add("Roll Vel Z", f"{rz:+.2f}σ", "BEARISH","Below norm", RED)
            else:                                          _add("Roll Vel Z", f"{rz:+.2f}σ", "SIDEWAYS","—",         MUTED)

    def _w(s): return 2 if s["strength"] == "Strong" else (1 if s["strength"] not in ("—","") else 0)
    bull_score     = sum(_w(s) for s in signals if s["bias"] == "BULLISH")
    bear_score     = sum(_w(s) for s in signals if s["bias"] == "BEARISH")
    sideways_score = sum(_w(s) for s in signals if s["bias"] == "SIDEWAYS")
    sideways_cnt   = sum(1 for s in signals if s["bias"] == "SIDEWAYS")
    net   = bull_score - bear_score
    total = bull_score + bear_score + sideways_score
    sideways_ratio = sideways_score / max(total, 1)

    if   net >=  4: verdict, v_color = "STRONG BULLISH 🚀",     GREEN
    elif net >=  2: verdict, v_color = "BULLISH 📈",            "#69F0AE"
    elif net <= -4: verdict, v_color = "STRONG BEARISH 📉",     RED
    elif net <= -2: verdict, v_color = "BEARISH ⚠",             "#FF6D00"
    elif sideways_ratio >= 0.5: verdict, v_color = "SIDEWAYS / RANGE-BOUND ↔", BLUE
    else: verdict, v_color = "NEUTRAL / TRANSITIONAL", MUTED

    return {
        "signals": signals,
        "verdict": verdict,
        "verdict_color": v_color,
        "bull_score": bull_score,
        "bear_score": bear_score,
        "sideways_count": sideways_cnt,
        "net_score": net,
    }

# ─────────────────────────────────────────────────────────────────────
#  CHARTS — futures only
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
        title={"text":"Futures Bias Score","font":{"color":TEXT,"size":12}},
        number={"font":{"color":color,"size":36}},
        gauge={"axis":{"range":[0,100],"tickcolor":"#444"},"bar":{"color":color},"bgcolor":CARD,
               "steps":[{"range":[0,30],"color":"#FEE2E2"},{"range":[30,45],"color":"#FFEDD5"},
                        {"range":[45,55],"color":"#FEF3C7"},{"range":[55,70],"color":"#DCFCE7"},
                        {"range":[70,100],"color":"#D1FAE5"}],
               "threshold":{"line":{"color":color,"width":3},"thickness":0.8,"value":score}},
    ))
    fig.update_layout(paper_bgcolor="#FFFFFF",plot_bgcolor="#FFFFFF",margin=dict(l=20,r=20,t=30,b=5),height=220)
    return fig

def build_term_structure_chart(roll: dict):
    """3-month futures curve. Upward = contango (normal); downward = backwardation (delivery pressure).
    (UNCHANGED FROM v5)"""
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
    """5-bucket colour scheme covering both positive (bullish) and negative
    (bearish) velocities, with explicit reference lines. Bucket thresholds
    are symbol-aware via get_thresh(); ts values may now carry a date
    prefix (oi_history is persisted to disk), so x-axis labels are
    shortened back to HH:MM via _disp_hm()."""
    hist = oi_history.get(symbol, []); fig = go.Figure()
    if len(hist) < 3:
        fig.add_annotation(text="Collecting rollover velocity data… refresh a few times",
                           xref="paper",yref="paper",x=0.5,y=0.5,showarrow=False,font=dict(color=MUTED,size=12))
        fig.update_layout(**chart_layout(title="Rollover Velocity (Near→Next OI Flow)")); return fig

    # Filter out None entries (low-activity ticks) for plotting
    plot_pts = [(h["ts"], h["rollover_velocity"]) for h in hist if h.get("rollover_velocity") is not None]
    if len(plot_pts) < 3:
        fig.add_annotation(text="Not enough non-None rollover velocity samples yet",
                           xref="paper",yref="paper",x=0.5,y=0.5,showarrow=False,font=dict(color=MUTED,size=12))
        fig.update_layout(**chart_layout(title="Rollover Velocity (Near→Next OI Flow)")); return fig

    ts_v = [_disp_hm(p[0]) for p in plot_pts]
    rv   = [p[1] for p in plot_pts]

    conviction  = get_thresh(symbol, "ROLL_VEL_CONVICTION")
    normal      = get_thresh(symbol, "ROLL_VEL_NORMAL")
    slow        = get_thresh(symbol, "ROLL_VEL_SLOW")
    liquid_mild = get_thresh(symbol, "ROLL_VEL_LIQUID_MILD")

    def _color_for(v):
        if v >=  conviction:    return GREEN       # strong conviction roll
        if v >=  normal:        return CYAN        # normal roll
        if v >=  slow:          return AMBER       # slow / cautious
        if v >=  liquid_mild:   return "#FF6D00"   # mild liquidation
        return RED                                    # hard liquidation
    colors = [_color_for(v) for v in rv]

    for y, col, ann in [
        ( conviction,  GREEN,    f"Conviction ≥{conviction}"),
        ( normal,      CYAN,     f"Normal ≥{normal}"),
        ( slow,        AMBER,    f"Slow ≥{slow}"),
        ( liquid_mild, "#FF6D00", f"Mild liq ≥{liquid_mild}"),
        (-liquid_mild, "#FF6D00", f"Hard liq ≤{liquid_mild}"),
    ]:
        fig.add_hline(y=y, line_dash="dot", line_color=col, opacity=0.6,
                      annotation_text=ann, annotation_font_size=9)
    fig.add_hline(y=0, line_dash="solid", line_color=MUTED, opacity=0.4)

    fig.add_trace(go.Scatter(x=ts_v, y=rv, mode="lines+markers",
                             marker=dict(color=colors, size=7), line=dict(color=CYAN, width=2),
                             hovertemplate="<b>%{x}</b><br>Roll Velocity: %{y:.3f}<extra></extra>"))
    lk = chart_layout(title="Rollover Velocity — Δ Next OI / max(|Δ Near OI|, floor)")
    lk["yaxis"] = dict(title="Velocity Ratio", gridcolor="#E2E8F0", zeroline=True, zerolinecolor="#94A3B8")
    lk["xaxis"] = dict(title="Time", gridcolor="#E2E8F0"); fig.update_layout(**lk); return fig

def build_score_history_chart(sym_history):
    """Line chart of the futures bias score across today's session, built
    from the same per-tick history already being logged for `history`.
    Cheap to add since the score is already recorded on every tick — this
    just plots it, giving a sense of whether the bias is strengthening or
    fading intraday instead of only showing a single point-in-time number."""
    fig = go.Figure()
    if len(sym_history) < 2:
        fig.add_annotation(text="Collecting score history… refresh a few times",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(color=MUTED, size=12))
        fig.update_layout(**chart_layout(title="Futures Bias Score — Today's Session")); return fig

    def _hm(ts):
        ts = str(ts)
        return ts[11:16] if len(ts) >= 16 and ts[10] == "T" else _disp_hm(ts)

    ts_v = [_hm(h.get("ts", "")) for h in sym_history]
    sc   = [safe_num(h.get("score", 50)) for h in sym_history]

    fig.add_hrect(y0=70, y1=100, fillcolor=GREEN, opacity=0.06, line_width=0)
    fig.add_hrect(y0=0,  y1=30,  fillcolor=RED,   opacity=0.06, line_width=0)
    fig.add_hline(y=50, line_dash="dash", line_color=MUTED, opacity=0.5,
                  annotation_text="neutral", annotation_font_size=10)
    fig.add_trace(go.Scatter(x=ts_v, y=sc, mode="lines+markers",
                             line=dict(color=CYAN, width=2), marker=dict(size=5),
                             hovertemplate="<b>%{x}</b><br>Score: %{y:.1f}<extra></extra>"))
    lk = chart_layout(title="Futures Bias Score — Today's Session")
    lk["yaxis"] = dict(title="Score (0-100)", range=[0, 100], gridcolor="#E2E8F0")
    lk["xaxis"] = dict(title="Time", gridcolor="#E2E8F0")
    fig.update_layout(**lk); return fig

# ─────────────────────────────────────────────────────────────────────
#  SCORE CALIBRATION CHECK (owner-only)
#  Reads the decision-log CSVs this app has already been writing to disk
#  and checks whether higher scores have actually preceded near-month
#  price increases — using the app's own logged history rather than a
#  synthetic backtest. This is a sanity check on the scoring weights,
#  not a rigorous walk-forward backtest (no slippage/costs modelled, and
#  early on the sample size will be small).
# ─────────────────────────────────────────────────────────────────────
def load_decision_log_history(symbol, max_files=60):
    """Reads up to `max_files` most recent decision_log_*.csv files from
    LOG_DIR, filters to `symbol`, and returns a combined, time-sorted
    DataFrame. Returns an empty DataFrame if nothing has been logged yet."""
    rows = []
    try:
        fnames = sorted(f for f in os.listdir(LOG_DIR) if f.startswith("decision_log_") and f.endswith(".csv"))
        for fn in fnames[-max_files:]:
            try:
                df = pd.read_csv(os.path.join(LOG_DIR, fn))
            except Exception:
                continue
            if "symbol" in df.columns:
                df = df[df["symbol"] == symbol]
            if not df.empty:
                rows.append(df)
    except Exception as e:
        print(f"[Calibration] Failed reading logs: {e}")
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    if "ts" in out.columns:
        out = out.sort_values("ts").reset_index(drop=True)
    return out

def score_calibration_check(log_df, forward_ticks=15):
    """Buckets historical ticks by score range and computes the average
    forward % change in near-month price `forward_ticks` rows later
    (≈ forward_ticks minutes at a 1-tick-per-minute refresh cadence).
    Returns None if there isn't enough logged history yet."""
    required = {"near_ltp", "score"}
    if log_df.empty or not required.issubset(log_df.columns) or len(log_df) < forward_ticks + 5:
        return None
    df = log_df.copy()
    df["near_ltp"] = pd.to_numeric(df["near_ltp"], errors="coerce")
    df["score"]    = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["near_ltp", "score"])
    df["fwd_near_ltp"]   = df["near_ltp"].shift(-forward_ticks)
    df["fwd_return_pct"] = (df["fwd_near_ltp"] - df["near_ltp"]) / df["near_ltp"] * 100
    df = df.dropna(subset=["fwd_return_pct"])
    if df.empty: return None
    bins   = [0, 30, 45, 55, 70, 100]
    labels = ["0-30 (Bearish)", "30-45", "45-55 (Neutral)", "55-70", "70-100 (Bullish)"]
    df["score_bucket"] = pd.cut(df["score"], bins=bins, labels=labels, include_lowest=True)
    summary = df.groupby("score_bucket", observed=True).agg(
        n=("fwd_return_pct", "size"),
        avg_fwd_return_pct=("fwd_return_pct", "mean"),
        pct_positive=("fwd_return_pct", lambda x: round((x > 0).mean() * 100, 1)),
    ).reset_index()
    summary["avg_fwd_return_pct"] = summary["avg_fwd_return_pct"].round(4)
    return summary

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
# NOTE: oi_history is loaded from disk (not just an empty dict) so the
# rollover-velocity z-score baseline survives an app restart — see the
# "OI-HISTORY PERSISTENCE" section above for why this matters in practice.
for k, v in [("history", None), ("oi_history", None), ("last_refresh", 0),
              ("is_owner", False), ("owner_pw_attempt", ""), ("owner_login_error", False)]:
    if k not in st.session_state:
        if k == "history":
            st.session_state[k] = load_history_from_disk()
        elif k == "oi_history":
            st.session_state[k] = load_oi_history_from_disk()
        else:
            st.session_state[k] = v

# ── SIDEBAR ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:8px 0 12px 0;">
        <span style="font-size:20px;">🌟</span>
        <div style="font-size:13px;font-weight:700;color:{GOLD};margin-top:4px;">Commodities Futures Dashboard</div>
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
        🌟 Shantanu's Commodity Futures Analysis Dashboard
    </h1>
    <div style="font-size:11px;color:{MUTED};margin-top:3px;">
        MCX GOLDM · SILVERM · GOLDPETAL &nbsp;·&nbsp; Term Structure · Rollover Velocity · Carry & Roll Analysis (Futures Only)
    </div>
</div>""", unsafe_allow_html=True)

# ── TOP CONTROLS ──────────────────────────────────────────────────────
col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns([2,2,2,1])
with col_ctrl1:
    symbol = st.selectbox("ASSET CLASS", COMMODITY_SYMBOLS, index=0,
                          format_func=lambda s: SYMBOL_LABELS.get(s, s))
with col_ctrl2:
    st.markdown(f"""<div style="padding-top:28px;font-size:11px;color:{MUTED};font-style:italic;">
        📡 Source: {'Dhan API (MCX) ✅' if CFG.USE_DHAN else 'DEMO MODE (Add credentials in secrets)'}
    </div>""", unsafe_allow_html=True)
with col_ctrl3:
    st.markdown(f"""<div style="padding-top:28px;font-size:11px;color:{MUTED};">
        🕐 Last updated: {strftime_ist('%H:%M:%S')} IST
    </div>""", unsafe_allow_html=True)
with col_ctrl4:
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

# ── FETCH FUTURES DATA ────────────────────────────────────────────────
roll = fetch_futures_roll(symbol) if CFG.USE_DHAN else demo_futures_roll(symbol)
_roll_is_demo = False
if not roll and CFG.USE_DHAN:
    roll = demo_futures_roll(symbol)
    _roll_is_demo = True
if not roll:
    st.error("No futures data available. Check API credentials or try again."); st.stop()

spot = roll.get("near_ltp", 0)

if _roll_is_demo:
    st.markdown(f"""<div style="background:{AMBER}22;border:1.5px solid {AMBER};border-radius:8px;
                padding:8px 16px;margin-bottom:10px;font-size:12px;font-weight:700;color:{AMBER};">
                ⚠ DEMO DATA IN USE — the live Dhan feed returned no prices (market likely closed
                or API issue). The score, verdict and every metric below are simulated, not real
                market signals.</div>""", unsafe_allow_html=True)

# Record intraday OI FIRST, then compute derived roll metrics so the
# dashboard reflects the *current* tick's velocity and z-score.
st.session_state["oi_history"] = record_intraday_oi(symbol, roll, st.session_state["oi_history"])
save_oi_history_to_disk(st.session_state["oi_history"])
roll_vel_z, roll_vel_interp, roll_vel_color = compute_rollover_velocity_zscore(st.session_state["oi_history"], symbol)

_latest_oi_ticks = st.session_state["oi_history"].get(symbol, [{}])
_latest_rv = _latest_oi_ticks[-1].get("rollover_velocity", None) if _latest_oi_ticks else None
roll["rollover_velocity"] = _latest_rv if _latest_rv is not None else get_thresh(symbol, "ROLL_VEL_NORMAL")

score = compute_futures_score(roll, roll_vel_z, symbol)
strat = futures_recommendation(score, roll, symbol)

ts_full = now_ist().isoformat(timespec="seconds")
tick = {
    "ts": ts_full, "symbol": symbol,
    "near_ltp": roll.get("near_ltp",0), "next_ltp": roll.get("next_ltp",0), "far_ltp": roll.get("far_ltp",0),
    "roll_spread_pct": roll.get("roll_spread_pct",0), "rollover_pct": roll.get("rollover_pct",0),
    "ts_bias": roll.get("ts_bias",0), "rollover_velocity": roll.get("rollover_velocity",0),
    "roll_vel_z": roll_vel_z if roll_vel_z is not None else 0, "score": score,
}
sym_hist = st.session_state["history"].setdefault(symbol, [])
if not sym_hist or sym_hist[-1].get("ts","")[:16] != ts_full[:16]:
    sym_hist.append(tick)
    if len(sym_hist) > THRESH.HISTORY_MAX_TICKS: st.session_state["history"][symbol] = sym_hist[-THRESH.HISTORY_MAX_TICKS:]
    save_history_to_disk(st.session_state["history"])
write_decision_log(dict(tick))

# ── HEADER CARDS ──────────────────────────────────────────────────────
theme_color = SILVER if "SILVER" in symbol else GOLD
rsp_val   = safe_num(roll.get("roll_spread_pct", 0))
rsp_color = GREEN if rsp_val > 0 else (RED if rsp_val < 0 else MUTED)
rp_val    = safe_num(roll.get("rollover_pct", 0))
rp_color  = GREEN if rp_val >= get_thresh(symbol, "ROLLOVER_PCT_ADVANCED") else (CYAN if rp_val >= get_thresh(symbol, "ROLLOVER_PCT_NORMAL") else MUTED)
tsb_val   = safe_num(roll.get("ts_bias", 0))
tsb_color = GREEN if tsb_val > 0 else (RED if tsb_val < 0 else MUTED)
rv_val    = safe_num(roll.get("rollover_velocity", 0))
rv_color  = GREEN if rv_val >= get_thresh(symbol, "ROLL_VEL_CONVICTION") else (RED if rv_val < get_thresh(symbol, "ROLL_VEL_SLOW") else CYAN)

header_cols = st.columns(8)
header_data = [
    ("Asset Class",     symbol,                                     theme_color),
    ("Near LTP",        f'₹ {spot:,.2f}',                          "#FFFFFF"),
    ("Near Expiry",     roll.get("near_expiry","—"),                 MUTED),
    ("Next Expiry",     roll.get("next_expiry","—"),                 MUTED),
    ("Roll Spread %",   f'{rsp_val:+.3f} %',                        rsp_color),
    ("Rollover %",      f'{rp_val:.0f} %',                          rp_color),
    ("Term Structure",  f'{int(tsb_val):+d}',                       tsb_color),
    ("Rollover Vel.",   f'{rv_val:.2f}',                            rv_color),
]
for col, (label, value, color) in zip(header_cols, header_data):
    col.metric(label, value)
    col.markdown(f"""<div style='font-size:9px;color:{color};margin-top:-10px;'>
        <span style='color:{MUTED};'>{label}:</span> <b style='color:{color};'>{value}</b></div>""",
        unsafe_allow_html=True)

st.markdown("---")

# ── FUTURES BIAS DECISION MATRIX ──────────────────────────────────────
combined_bias = compute_futures_market_bias(roll, roll_vel_z, score, symbol)
_cb = combined_bias
st.markdown('<div class="section-header">🧭 Futures Bias Decision Matrix</div>',
            unsafe_allow_html=True)

_bull_pct = max(0, _cb["bull_score"]) / max(1, _cb["bull_score"] + _cb["bear_score"] + _cb["sideways_count"]) * 100
_bear_pct = max(0, _cb["bear_score"]) / max(1, _cb["bull_score"] + _cb["bear_score"] + _cb["sideways_count"]) * 100
_side_pct = 100 - _bull_pct - _bear_pct

st.markdown(f"""
<div style="background:{_cb['verdict_color']}15;border:1.5px solid {_cb['verdict_color']};
            border-radius:10px;padding:14px 20px;margin-bottom:10px;">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
        <div style="font-size:18px;font-weight:800;color:{_cb['verdict_color']};letter-spacing:0.5px;">
            🎯 {_cb['verdict']}
        </div>
        <div style="display:flex;gap:8px;font-size:11px;font-weight:700;">
            <span style="background:{GREEN}22;color:{GREEN};border:1px solid {GREEN};border-radius:6px;padding:3px 10px;">
                🐂 Bull: {_cb['bull_score']}
            </span>
            <span style="background:{RED}22;color:{RED};border:1px solid {RED};border-radius:6px;padding:3px 10px;">
                🐻 Bear: {_cb['bear_score']}
            </span>
            <span style="background:{BLUE}22;color:{BLUE};border:1px solid {BLUE};border-radius:6px;padding:3px 10px;">
                ↔ Sideways: {_cb['sideways_count']}
            </span>
        </div>
    </div>
    <div style="font-size:11px;color:{MUTED};margin-top:6px;">
        Net directional score: <b style="color:{_cb['verdict_color']};">{_cb['net_score']:+d}</b>
        &nbsp;·&nbsp; Aggregated from {len(_cb['signals'])} live futures signals.
        Strength qualifier ("Strong" / "Mild") scales each signal's weight in the net score.
    </div>
    <div style="display:flex;height:8px;border-radius:4px;overflow:hidden;margin-top:10px;border:1px solid {BORDER};">
        <div style="width:{_bull_pct:.1f}%;background:{GREEN};"></div>
        <div style="width:{_side_pct:.1f}%;background:{BLUE};opacity:0.5;"></div>
        <div style="width:{_bear_pct:.1f}%;background:{RED};"></div>
    </div>
</div>""", unsafe_allow_html=True)

def _render_signal_cell(s):
    return f"""
    <div style="background:{CARD};border:1px solid {BORDER};border-left:3px solid {s['color']};
                border-radius:6px;padding:8px 10px;display:flex;justify-content:space-between;
                align-items:center;gap:6px;min-height:48px;">
        <div style="flex:1;min-width:0;">
            <div style="font-size:10px;color:{MUTED};text-transform:uppercase;letter-spacing:0.4px;
                        white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{s['name']}</div>
            <div style="font-size:13px;font-weight:700;color:{TEXT};white-space:nowrap;
                        overflow:hidden;text-overflow:ellipsis;">{s['value']}</div>
        </div>
        <div style="flex-shrink:0;">{_bias_tag(s['bias'], s['strength'], s['color'])}</div>
    </div>"""

_signals = _cb["signals"]
_half = (len(_signals) + 1) // 2
_mx_cols = st.columns(2)
with _mx_cols[0]:
    for s in _signals[:_half]:
        st.markdown(_render_signal_cell(s), unsafe_allow_html=True)
with _mx_cols[1]:
    for s in _signals[_half:]:
        st.markdown(_render_signal_cell(s), unsafe_allow_html=True)

st.markdown(f"""
<div style="background:{SECTION_BG};border-radius:8px;padding:8px 14px;margin-top:8px;
            font-size:11px;color:{MUTED};line-height:1.5;">
    <b>How to read this matrix:</b> Each cell shows a live futures metric with a colour-coded bias chip —
    <span style="color:{GREEN};font-weight:700;">🟢 BULLISH</span>,
    <span style="color:{RED};font-weight:700;">🔴 BEARISH</span>, or
    <span style="color:{BLUE};font-weight:700;">🔵 SIDEWAYS</span> (range / normal).
    "Strong" signals carry 2× the weight of "Mild" signals. The verdict at the top is the
    net of all bullish minus bearish scores; ties fall back to a sideways or neutral verdict.
</div>""", unsafe_allow_html=True)

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
        <div style="font-size:10px;color:{MUTED};">BIAS</div>
        <div style="font-size:14px;font-weight:700;color:{strat['color']};margin-bottom:8px;">{strat['name']}</div>
        <div style="font-size:10px;color:{MUTED};">RECOMMENDED ACTION</div>
        <div style="font-size:12px;color:#CCC;font-family:monospace;">{strat['action']}</div>
    </div>""", unsafe_allow_html=True)
with col_metrics:
    metric_items = [
        ("Roll Spread (₹)",   f'{roll.get("roll_spread",0):+,.2f}',        GREEN if roll.get("roll_spread",0)>0 else RED, METRIC_EXPLAIN["Roll Spread"]),
        ("Roll Spread %",     f'{roll.get("roll_spread_pct",0):+.3f}%',    rsp_color,                                     METRIC_EXPLAIN["Spread %"]),
        ("Rollover %",        f'{roll.get("rollover_pct",0)}%',            BLUE,                                          METRIC_EXPLAIN["Rollover %"]),
        ("Rollover Velocity", f'{roll.get("rollover_velocity",0):.2f}',    rv_color,                                      METRIC_EXPLAIN["Rollover Velocity"]),
        ("Roll Vel Z-Score",  f'{roll_vel_z:+.2f}σ' if roll_vel_z is not None else "—", MUTED if roll_vel_z is None else (GREEN if roll_vel_z>=get_thresh(symbol,"ZSCORE_BULL") else (RED if roll_vel_z<=get_thresh(symbol,"ZSCORE_BEAR") else MUTED)), "Z-score of latest rollover velocity vs its recent rolling baseline."),
        ("Near Vol/OI",       f'{roll.get("near_vol_oi",0):.3f}',          GREEN if roll.get("near_vol_oi",0)>0.3 else AMBER, METRIC_EXPLAIN["Near Vol/OI"]),
        ("Slope Near→Next",   f'{roll.get("slope_near_next",0):+.2f}% p.a.', GREEN if roll.get("slope_near_next",0)>0 else RED, METRIC_EXPLAIN["Slope Near→Next"]),
        ("Slope Next→Far",    f'{roll.get("slope_next_far",0):+.2f}% p.a.' if roll.get("has_far") else "—", GREEN if roll.get("slope_next_far",0)>0 else RED, METRIC_EXPLAIN["Slope Next→Far"]),
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

# ── SCORE TREND — TODAY'S SESSION ──────────────────────────────────────
# The score is already recorded on every tick (see `tick` above); this
# just plots it, so you can see whether the bias is strengthening or
# fading intraday instead of only seeing a single point-in-time number.
st.markdown("---")
st.markdown('<div class="section-header">📈 Score Trend — Today\'s Session</div>', unsafe_allow_html=True)
st.plotly_chart(build_score_history_chart(st.session_state["history"].get(symbol, [])),
                use_container_width=True, config={"displayModeBar":False})

# ── FUTURES ROLL — CONTRACT CARDS, SPREAD/ROLLOVER, CHARTS ────────────
# (UNCHANGED FROM v5 — no options logic here)
st.markdown("---")
st.markdown("### 📦 Futures Roll Analysis & Intraday OI Curve")
if roll:
    if _roll_is_demo:
        st.caption("⚠ **Demo roll data** — live Dhan feed returned no prices (market likely closed). Roll metrics use simulated values.")
    _has_far = roll.get("has_far") and roll.get("far_ltp", 0) > 0
    _ncols   = 3 if _has_far else 2
    _contract_cols = st.columns(_ncols)

    def _contract_card(col, label, color, ltp, oi, vol, vol_oi, expiry, tsym, bias_html):
        oi_str   = f"{oi:,}"  if oi  > 0 else "—"
        vol_str  = f"{vol:,}" if vol > 0 else "—"
        voi_str  = f"{vol_oi:.3f}" if vol_oi > 0 else "—"
        disp_sym = tsym[-7:] if len(tsym) > 7 else tsym
        col.markdown(f"""
        <div style="background:{CARD};border-radius:10px;padding:12px 16px;
                    border:1px solid {BORDER};border-top:3px solid {color};">
            <div style="font-size:10px;color:{MUTED};text-transform:uppercase;letter-spacing:0.5px;">{label}</div>
            <div style="font-size:13px;font-weight:700;color:{color};margin:2px 0;">{disp_sym}</div>
            <div style="font-size:10px;color:{MUTED};">Expiry: {expiry}</div>
            <div style="display:flex;gap:20px;margin-top:8px;">
                <div>
                    <div style="font-size:9px;color:{MUTED};">LTP (₹)</div>
                    <div style="font-size:20px;font-weight:800;color:{color};">{ltp:,.2f}</div>
                </div>
                <div>
                    <div style="font-size:9px;color:{MUTED};">Futures OI</div>
                    <div style="font-size:16px;font-weight:700;color:{TEXT};">{oi_str}</div>
                </div>
                <div>
                    <div style="font-size:9px;color:{MUTED};">Volume</div>
                    <div style="font-size:16px;font-weight:700;color:{TEXT};">{vol_str}</div>
                </div>
                <div>
                    <div style="font-size:9px;color:{MUTED};">Vol/OI</div>
                    <div style="font-size:16px;font-weight:700;color:{TEXT};">{voi_str}</div>
                </div>
            </div>
            <div style="margin-top:8px;padding-top:6px;border-top:1px dashed {BORDER};
                        font-size:10px;color:{MUTED};line-height:1.4;">
                {bias_html}
            </div>
        </div>""", unsafe_allow_html=True)

    _near_voi    = safe_num(roll.get("near_vol_oi", 0))
    _near_voi_b  = ("BULLISH" if _near_voi > 0.3 else "SIDEWAYS") if _near_voi > 0 else "SIDEWAYS"
    _near_voi_s  = "Active fresh positioning" if _near_voi > 0.3 else ("Normal" if _near_voi > 0 else "—")
    _near_voi_c  = GREEN if _near_voi > 0.3 else (CYAN if _near_voi > 0 else MUTED)

    _next_voi    = safe_num(roll.get("next_vol_oi", 0))
    _rollover_pct_val = safe_num(roll.get("rollover_pct", 0))
    if   _rollover_pct_val >= get_thresh(symbol, "ROLLOVER_PCT_ADVANCED"): _rollp_b, _rollp_s, _rollp_c = "BULLISH", "Advanced rollover", GREEN
    elif _rollover_pct_val >= get_thresh(symbol, "ROLLOVER_PCT_NORMAL"):    _rollp_b, _rollp_s, _rollp_c = "SIDEWAYS", "Normal pace",     CYAN
    else:                                                                   _rollp_b, _rollp_s, _rollp_c = "SIDEWAYS", "Early / cautious", MUTED

    _near_bias_html = (
        f"{_bias_tag(_near_voi_b, _near_voi_s, _near_voi_c)} "
        f"<span style='color:{TEXT};'>Near-month carries the most liquidity — high Vol/OI means fresh directional bets are being placed.</span>"
    )
    _next_bias_html = (
        f"{_bias_tag(_rollp_b, _rollp_s, _rollp_c)} "
        f"<span style='color:{TEXT};'>Next-month OI shows rollover conviction — high % means longs are rolling forward with conviction.</span>"
    )
    _far_bias_html = (
        f"{_bias_tag('SIDEWAYS', '—', MUTED)} "
        f"<span style='color:{TEXT};'>Far-month is the long-term view — thin OI is normal; rising OI here signals strategic positioning.</span>"
    )

    with _contract_cols[0]:
        _contract_card(_contract_cols[0], "NEAR MONTH", GOLD,
                       roll["near_ltp"], roll["near_oi"], roll["near_vol"], roll["near_vol_oi"],
                       roll["near_expiry"], roll.get("near_tsym","NEAR"), _near_bias_html)
    with _contract_cols[1]:
        _contract_card(_contract_cols[1], "NEXT MONTH", "#CE93D8",
                       roll["next_ltp"], roll["next_oi"], roll["next_vol"], roll["next_vol_oi"],
                       roll["next_expiry"], roll.get("next_tsym","NEXT"), _next_bias_html)
    if _has_far:
        with _contract_cols[2]:
            _contract_card(_contract_cols[2], "FAR MONTH", CYAN,
                           roll["far_ltp"], roll["far_oi"], roll["far_vol"], roll["far_vol_oi"],
                           roll["far_expiry"], roll.get("far_tsym","FAR"), _far_bias_html)

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    _spr_cols = st.columns(4)

    def _spread_bias(pct, _sym=symbol):
        strong_up   = get_thresh(_sym, "ROLL_SPREAD_STRONG_CONTANGO")
        strong_down = get_thresh(_sym, "ROLL_SPREAD_STRONG_BACKWARD")
        if   pct >  strong_up:   return "BULLISH", "Strong contango", GREEN
        elif pct >  0:           return "BULLISH", "Mild contango",   "#69F0AE"
        elif pct < strong_down:  return "BEARISH", "Strong backwardation", RED
        elif pct <  0:           return "BEARISH", "Mild backwardation",   "#FF6D00"
        else:                    return "SIDEWAYS", "Flat / no carry",      MUTED

    _rsp = safe_num(roll.get("roll_spread_pct", 0))
    _rs_b, _rs_s, _rs_c = _spread_bias(_rsp)
    _rp_b, _rp_s, _rp_c = (_rollp_b, _rollp_s, _rollp_c)
    _st_b = "BULLISH" if "CONTANGO" in (roll.get("bias","") or "").upper() else ("BEARISH" if "BACKWARDATION" in (roll.get("bias","") or "").upper() else "SIDEWAYS")
    _st_s = "Carry positive" if _st_b == "BULLISH" else ("Delivery pressure" if _st_b == "BEARISH" else "Indecisive")
    _st_c = roll.get("bias_color", MUTED)

    _spr_items = [
        ("Near→Next Spread (₹)", f'{roll["roll_spread"]:+,.2f}', roll["bias_color"],
         _bias_tag(_rs_b, _rs_s, _rs_c) + f" <span style='color:{TEXT};'>Next month pricier than near = cost-of-carry, bullish for trend.</span>"),
        ("Near→Next Spread (%)", f'{roll["roll_spread_pct"]:+.3f} %', roll["bias_color"],
         _bias_tag(_rs_b, _rs_s, _rs_c) + f" <span style='color:{TEXT};'>Same spread as % of price — annualised carry signal.</span>"),
        ("Rollover %",           f'{roll["rollover_pct"]} %', BLUE,
         _bias_tag(_rp_b, _rp_s, _rp_c) + f" <span style='color:{TEXT};'>Share of OI already in next month — high = conviction roll.</span>"),
        ("Structure",            roll["bias"], roll["bias_color"],
         _bias_tag(_st_b, _st_s, _st_c) + f" <span style='color:{TEXT};'>Curve shape: contango = normal carry, backwardation = supply squeeze.</span>"),
    ]
    if _has_far:
        nn_s = roll.get("next_ltp",0) - roll.get("near_ltp",0)
        nf_s = roll.get("far_ltp",0)  - roll.get("next_ltp",0)
        nf_pct = round(nf_s / roll["next_ltp"] * 100, 3) if roll.get("next_ltp",0) > 0 else 0
        _nf_b, _nf_s, _nf_c = _spread_bias(nf_pct)
        _spr_items = [
            ("Near→Next (₹)", f'{nn_s:+,.2f}', roll["bias_color"],
             _bias_tag(_rs_b, _rs_s, _rs_c) + f" <span style='color:{TEXT};'>Front of curve — positive = contango (bullish carry).</span>"),
            ("Next→Far (₹)",  f'{nf_s:+,.2f}',
             "#00E676" if nf_s > 0 else ("#FF5252" if nf_s < 0 else "#FFD600"),
             _bias_tag(_nf_b, _nf_s, _nf_c) + f" <span style='color:{TEXT};'>Back of curve — positive = bullish long-dated view.</span>"),
            ("Rollover %",    f'{roll["rollover_pct"]} %', BLUE,
             _bias_tag(_rp_b, _rp_s, _rp_c) + f" <span style='color:{TEXT};'>OI migration to next month — high = conviction roll.</span>"),
            ("Structure",     roll["bias"], roll["bias_color"],
             _bias_tag(_st_b, _st_s, _st_c) + f" <span style='color:{TEXT};'>Curve shape verdict — contango vs backwardation.</span>"),
        ]
    for col, (lbl, val, clr, bias_html) in zip(_spr_cols, _spr_items):
        col.markdown(f"""
        <div style="background:{CARD};border-radius:8px;padding:8px 12px;border:1px solid {BORDER};margin-top:4px;">
            <div style="font-size:10px;color:{MUTED};text-transform:uppercase;">{lbl}</div>
            <div style="font-size:16px;font-weight:700;color:{clr};">{val}</div>
            <div style="font-size:10px;margin-top:4px;line-height:1.4;">{bias_html}</div>
        </div>""", unsafe_allow_html=True)

    roll_chart_cols = st.columns(2)
    with roll_chart_cols[0]:
        _bar_x     = [roll.get("near_tsym","Near")[-7:], roll.get("next_tsym","Next")[-7:]]
        _bar_oi    = [roll["near_oi"], roll["next_oi"]]
        _bar_vol   = [roll.get("near_vol",0), roll.get("next_vol",0)]
        _bar_color = [GOLD, "#CE93D8"]
        _bar_vcol  = ["rgba(212,175,55,0.5)", "rgba(206,147,216,0.5)"]
        if _has_far:
            _bar_x.append(roll.get("far_tsym","Far")[-7:])
            _bar_oi.append(roll["far_oi"]); _bar_vol.append(roll.get("far_vol",0))
            _bar_color.append(CYAN); _bar_vcol.append("rgba(0,229,255,0.4)")
        f_roll = go.Figure([
            go.Bar(name="Futures OI", x=_bar_x, y=_bar_oi,  marker_color=_bar_color,
                   hovertemplate="<b>%{x}</b><br>OI: %{y:,}<extra></extra>"),
            go.Bar(name="Volume",     x=_bar_x, y=_bar_vol, marker_color=_bar_vcol,
                   hovertemplate="<b>%{x}</b><br>Vol: %{y:,}<extra></extra>"),
        ])
        f_roll.add_annotation(text=roll["bias"], xref="paper", yref="paper", x=0.5, y=1.12,
                              showarrow=False, font=dict(color=roll["bias_color"], size=12))
        f_roll.update_layout(**chart_layout(title="Futures OI & Volume by Contract", barmode="group"),
                             legend=dict(orientation="h", y=1.02, x=0))
        st.plotly_chart(f_roll, use_container_width=True, config={"displayModeBar":False})

    with roll_chart_cols[1]:
        oi_hist = st.session_state["oi_history"].get(symbol, [])
        if len(oi_hist) >= 2:
            ts_v  = [r["ts"] for r in oi_hist]
            near  = [r["near_oi"]  for r in oi_hist]
            nxt   = [r["next_oi"]  for r in oi_hist]
            total = [r["total_oi"] for r in oi_hist]
            f_oi  = go.Figure([
                go.Scatter(x=ts_v, y=total, mode="lines", name="Total OI",
                           line=dict(color=CYAN,  width=2)),
                go.Scatter(x=ts_v, y=near,  mode="lines", name=roll.get("near_tsym","Near")[-7:],
                           line=dict(color=GOLD,  width=1.5, dash="dot")),
                go.Scatter(x=ts_v, y=nxt,   mode="lines", name=roll.get("next_tsym","Next")[-7:],
                           line=dict(color="#CE93D8", width=1.5, dash="dot")),
            ])
            f_oi.update_layout(**chart_layout(title="Intraday Futures OI Curve (Today's Session)"),
                               legend=dict(orientation="h", y=1.08, x=0),
                               xaxis_title="Time", yaxis_title="Open Interest")
        else:
            f_oi = go.Figure()
            f_oi.add_annotation(text="Collecting OI history… refresh a few times",
                                xref="paper", yref="paper", x=0.5, y=0.5,
                                showarrow=False, font=dict(color=MUTED, size=13))
            f_oi.update_layout(**chart_layout(title="Intraday OI Curve (Building…)"))
        st.plotly_chart(f_oi, use_container_width=True, config={"displayModeBar":False})
else:
    st.warning("Roll data unavailable — market may be closed or outside MCX trading hours (9 AM – 11:30 PM IST).")

# ── FUTURES INTELLIGENCE — TERM STRUCTURE & ROLLOVER VELOCITY ────────
st.markdown("---")
st.markdown('<div class="section-header">🔮 Futures Intelligence — Term Structure · Rollover Velocity</div>', unsafe_allow_html=True)

sec9_cols = st.columns(2)
with sec9_cols[0]:
    st.plotly_chart(build_term_structure_chart(roll), use_container_width=True, config={"displayModeBar":False})
    if roll:
        sc_nn = roll.get('slope_near_next',0); sc_nf = roll.get('slope_next_far',0)
        _nn_b = "BULLISH" if sc_nn > 0 else ("BEARISH" if sc_nn < 0 else "SIDEWAYS")
        _nn_s = "Contango" if sc_nn > 0 else ("Backwardation" if sc_nn < 0 else "Flat")
        _nn_c = GREEN if sc_nn > 0 else (RED if sc_nn < 0 else MUTED)
        _nf_b = "BULLISH" if sc_nf > 0 else ("BEARISH" if sc_nf < 0 else "SIDEWAYS")
        _nf_s = "Steepening" if sc_nf > 0 else ("Inverting" if sc_nf < 0 else "Flat")
        _nf_c = GREEN if sc_nf > 0 else (RED if sc_nf < 0 else MUTED)
        _tsb_val = safe_num(roll.get('ts_bias', 0))
        _ts_card_b = ("BULLISH" if _tsb_val >= 1 else ("BEARISH" if _tsb_val <= -1 else "SIDEWAYS"))
        _ts_card_s = ("Strong" if abs(_tsb_val) >= 2 else ("Mild" if abs(_tsb_val) == 1 else "—"))
        _ts_card_c = roll.get('ts_color', MUTED)
        st.markdown(f"""
        <div style="background:{CARD};border-radius:8px;padding:8px 12px;border:1px solid {BORDER};margin-top:4px;font-size:11px;">
            <span style="color:{roll.get('ts_color',MUTED)};font-weight:700;">{roll.get('ts_shape','—')}</span>
            <span style="color:{MUTED};margin-left:8px;">{roll.get('ts_desc','')}</span><br>
            <span style="color:{MUTED};">Near→Next: </span>
            <span style="color:{GREEN if sc_nn>0 else RED};font-weight:600;">{sc_nn:+.2f}% p.a.</span>
            {_bias_tag(_nn_b, _nn_s, _nn_c)}
            <span style="color:{MUTED};margin-left:10px;">Next→Far: </span>
            <span style="color:{GREEN if sc_nf>0 else RED};font-weight:600;">{sc_nf:+.2f}% p.a.</span>
            {_bias_tag(_nf_b, _nf_s, _nf_c)}
            <div style="margin-top:5px;padding-top:4px;border-top:1px dashed {BORDER};font-size:10px;color:{TEXT};line-height:1.4;">
                {_bias_tag(_ts_card_b, _ts_card_s, _ts_card_c)}
                <span>Curve shape tells you the carry trend — upward = bulls paying to hold, downward = delivery pressure.</span>
            </div>
        </div>""", unsafe_allow_html=True)
with sec9_cols[1]:
    st.plotly_chart(build_rollover_velocity_chart(st.session_state["oi_history"], symbol),
                    use_container_width=True, config={"displayModeBar":False})

# Roll Vel Z / Near Vol/OI / Far Month LTP / Term Structure Bias cards
sec9_mcols = st.columns(4)
with sec9_mcols[0]:
    if roll_vel_z is None:
        rv_zc = MUTED
        _rvz_b, _rvz_s = "—", "Collecting data"
        _rvz_display = "—"
    else:
        rv_zc = GREEN if roll_vel_z >= get_thresh(symbol,"ZSCORE_BULL") else (RED if roll_vel_z <= get_thresh(symbol,"ZSCORE_BEAR") else MUTED)
        if   roll_vel_z >= get_thresh(symbol,"ZSCORE_BULL"): _rvz_b, _rvz_s = "BULLISH", "Above norm"
        elif roll_vel_z <= get_thresh(symbol,"ZSCORE_BEAR"): _rvz_b, _rvz_s = "BEARISH", "Below norm"
        else:                                                _rvz_b, _rvz_s = "SIDEWAYS", "—"
        _rvz_display = f"{roll_vel_z:+.2f}σ"
    st.markdown(f"""
    <div style="background:{CARD};border-radius:8px;padding:10px 14px;border:1px solid {BORDER};">
        <div style="font-size:10px;color:{MUTED};text-transform:uppercase;">Roll Vel Z-Score</div>
        <div style="font-size:20px;font-weight:700;color:{rv_zc};">{_rvz_display}</div>
        <div class="explain-text" style="color:{roll_vel_color};">{roll_vel_interp}</div>
        <div style="margin-top:4px;font-size:10px;line-height:1.4;">{_bias_tag(_rvz_b, _rvz_s, rv_zc)}
            <span style="color:{TEXT};">+σ = longs adding conviction; −σ = longs unwinding.</span>
        </div>
    </div>""", unsafe_allow_html=True)
if roll:
    nvo = roll.get("near_vol_oi",0)
    with sec9_mcols[1]:
        nvo_c = GREEN if nvo>0.3 else (AMBER if nvo>0.1 else MUTED)
        if   nvo > 0.3:  _nvo_b, _nvo_s = "BULLISH",  "Active flow"
        elif nvo > 0.1:  _nvo_b, _nvo_s = "SIDEWAYS", "Normal"
        else:            _nvo_b, _nvo_s = "SIDEWAYS", "Quiet"
        st.markdown(f"""
        <div style="background:{CARD};border-radius:8px;padding:10px 14px;border:1px solid {BORDER};">
            <div style="font-size:10px;color:{MUTED};text-transform:uppercase;">Near Vol/OI</div>
            <div style="font-size:20px;font-weight:700;color:{nvo_c};">{nvo:.3f}</div>
            <div class="explain-text">{METRIC_EXPLAIN['Near Vol/OI']}</div>
            <div style="margin-top:4px;font-size:10px;line-height:1.4;">{_bias_tag(_nvo_b, _nvo_s, nvo_c)}
                <span style="color:{TEXT};">High = fresh bets in near month; low = stale positioning.</span>
            </div>
        </div>""", unsafe_allow_html=True)
    with sec9_mcols[2]:
        fl = roll.get("far_ltp",0)
        _nl = roll.get("near_ltp", 0); _xt = roll.get("next_ltp", 0)
        if   fl > 0 and _xt > 0 and fl > _xt: _fl_b, _fl_s, _fl_c = "BULLISH",  "Steepening", GREEN
        elif fl > 0 and _xt > 0 and fl < _xt: _fl_b, _fl_s, _fl_c = "BEARISH",  "Inverting",  RED
        else:                                  _fl_b, _fl_s, _fl_c = "SIDEWAYS", "—",          CYAN
        st.markdown(f"""
        <div style="background:{CARD};border-radius:8px;padding:10px 14px;border:1px solid {BORDER};">
            <div style="font-size:10px;color:{MUTED};text-transform:uppercase;">Far Month LTP</div>
            <div style="font-size:20px;font-weight:700;color:{CYAN if fl>0 else MUTED};">
                {'₹{:,.0f}'.format(fl) if fl>0 else '—'}
            </div>
            <div class="explain-text">3rd month futures — part of the term structure curve</div>
            <div style="margin-top:4px;font-size:10px;line-height:1.4;">{_bias_tag(_fl_b, _fl_s, _fl_c)}
                <span style="color:{TEXT};">Far > Next = long-dated bullish view; Far < Next = bearish inversion.</span>
            </div>
        </div>""", unsafe_allow_html=True)
    tsb = roll.get("ts_bias",0)
    with sec9_mcols[3]:
        if   tsb >=  2: _tsb_b, _tsb_s, _tsb_c = "BULLISH", "Strong",   GREEN
        elif tsb ==  1: _tsb_b, _tsb_s, _tsb_c = "BULLISH", "Mild",     "#69F0AE"
        elif tsb <= -2: _tsb_b, _tsb_s, _tsb_c = "BEARISH", "Strong",   RED
        elif tsb == -1: _tsb_b, _tsb_s, _tsb_c = "BEARISH", "Mild",     "#FF6D00"
        else:           _tsb_b, _tsb_s, _tsb_c = "SIDEWAYS","—",        MUTED
        st.markdown(f"""
        <div style="background:{CARD};border-radius:8px;padding:10px 14px;border:1px solid {BORDER};">
            <div style="font-size:10px;color:{MUTED};text-transform:uppercase;">Term Structure Bias</div>
            <div style="font-size:20px;font-weight:700;color:{_tsb_c};">{tsb:+d}</div>
            <div class="explain-text">−2 to +2: steepening contango = +2, steep backwardation = −2</div>
            <div style="margin-top:4px;font-size:10px;line-height:1.4;">{_bias_tag(_tsb_b, _tsb_s, _tsb_c)}
                <span style="color:{TEXT};">+2 = strong contango (bullish carry); −2 = steep backwardation (supply squeeze).</span>
            </div>
        </div>""", unsafe_allow_html=True)

# ── OWNER TOOLS: DIAGNOSTICS & CALIBRATION ────────────────────────────
if is_owner:
    st.markdown("---")
    st.markdown('<div class="section-header">🔧 Owner Tools — Data Source Diagnostics & Score Calibration</div>',
                unsafe_allow_html=True)

    with st.expander("📡 Data Source Diagnostics — verify symbol resolution against Dhan's master CSV", expanded=False):
        st.caption("Shows what get_futures_contracts() actually matched the last time the master CSV was "
                   "parsed (cached for 24h). Use this to confirm GOLDPETAL — or any symbol — is resolving "
                   "real contracts instead of silently falling back to demo data.")
        if not _SYMBOL_MATCH_INFO:
            st.info("No resolution attempt recorded yet for this session — this populates once "
                    "get_futures_contracts() runs against the live Dhan API (requires DHAN credentials).")
        else:
            _diag_rows = []
            for sym, info in _SYMBOL_MATCH_INFO.items():
                _diag_rows.append({
                    "Symbol": sym,
                    "Pattern used": info.get("pattern_used", "—"),
                    "Matched?": "✅ Yes" if info.get("matched") else "❌ No",
                    "Contracts found": info.get("count", 0),
                    "Sample trading symbols": ", ".join(info.get("sample", [])) or "—",
                })
            st.dataframe(pd.DataFrame(_diag_rows), use_container_width=True, hide_index=True)
            if not _SYMBOL_MATCH_INFO.get(symbol, {}).get("matched", True):
                st.error(f"⚠ {symbol} did not resolve to any MCX FUTCOM contracts — check the trading-symbol "
                        f"prefix in SYMBOL_THRESH_OVERRIDES / _resolve_symbol_df() against Dhan's live master CSV.")

    with st.expander("🔬 Score Calibration Check — does a higher score actually precede a price rise?", expanded=False):
        st.caption("Reads the decision-log CSVs this app has been writing to disk and buckets historical "
                   "ticks by score range, then checks the average forward move in near-month price ~15 "
                   "ticks later. This is a sanity check using the app's OWN logged history — not a "
                   "rigorous backtest (no slippage/costs, small early sample, single-server clock) — "
                   "treat it as a way to catch obviously mis-calibrated weights, not as proof the score works.")
        _log_df = load_decision_log_history(symbol)
        if _log_df.empty:
            st.info(f"No logged history yet for {symbol} — check back after the app has been running for a while.")
        else:
            _summary = score_calibration_check(_log_df)
            if _summary is None or _summary.empty:
                st.info(f"Only {len(_log_df)} logged ticks so far for {symbol} — need more history "
                        f"(at least ~20 ticks) before a forward-return check is meaningful.")
            else:
                st.dataframe(_summary, use_container_width=True, hide_index=True)
                st.caption(f"Based on {len(_log_df)} logged ticks for {symbol}. 'avg_fwd_return_pct' = average "
                          f"% change in near-month price ~15 ticks later; 'pct_positive' = share of those "
                          f"forward windows that were positive. If the scoring weights are sane, both should "
                          f"generally trend upward from the bearish bucket to the bullish bucket — if they "
                          f"don't (or are flat/inverted), that's a signal to revisit the weights in "
                          f"compute_futures_score().")

# ── FOOTER ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style="text-align:center;font-size:10px;color:{MUTED};padding:10px;">
    Shantanu's Commodity Futures Analysis Dashboard v6.1 (Futures Only) · Streamlit Edition<br>
    Data: {'Dhan API (MCX)' if CFG.USE_DHAN else 'DEMO MODE'} ·
    Auto-refresh: {AUTO_REFRESH_SECONDS}s ·
    History ticks: {sum(len(v) for v in st.session_state['history'].values() if isinstance(v,list))} ·
    OI ticks (today): {sum(len(v) for v in st.session_state['oi_history'].values() if isinstance(v,list))}
</div>""", unsafe_allow_html=True)
