"""
signal_engine.py
================
Orchestrator for Buddy VIP Signals.

scan_and_emit():
  1. Iterate over the watchlist (crypto only).
  2. Fetch 1h (primary) + 4h (confirmation) candles.
  3. Detect candle patterns on primary.
  4. Detect HMM regime on both timeframes (with cache).
  5. Compute confluence (indicators + patterns + regime).
  6. If score >= threshold and direction != none AND cooldown passed,
     build signal dict.
  7. Caller (api_server) is responsible for persistence + push.

No Firestore calls happen here — keeps this module portable / testable.
"""

from __future__ import annotations
import time
from typing import Dict, List, Optional

from data_fetcher import fetch_all_timeframes_universal
from hmm_engine import RegimeHMM
from pattern_detector import detect_patterns, aggregate_bias
from confluence_scorer import calc_confluence

# ----- config --------------------------------------------------------------

WATCHLIST = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
    "XRP/USDT", "DOGE/USDT", "ADA/USDT", "AVAX/USDT",
]
PRIMARY_TF = "1h"
HIGHER_TF = "4h"
SCORE_THRESHOLD = 7.5
PER_ASSET_COOLDOWN_SEC = 4 * 3600       # 4h between signals per asset
GLOBAL_DAILY_LIMIT = 4                   # max 4 signals / day total

_hmm_cache: Dict[str, tuple] = {}        # "symbol_tf" -> (model, ts)
_last_sent: Dict[str, float] = {}        # "symbol" -> ts of last signal
_day_count: Dict[str, int] = {}          # "YYYY-MM-DD" -> count


def _hmm_regime(symbol: str, tf: str, df) -> str:
    key = f"{symbol}_{tf}"
    now = time.time()
    cached = _hmm_cache.get(key)
    if cached and (now - cached[1]) < 1800:
        model = cached[0]
    else:
        model = RegimeHMM(n_states=7)
        model.fit(df, auto_select=False)
        _hmm_cache[key] = (model, now)
    try:
        info = model.current_regime(df)
        return info.get("regime", "Lateral")
    except Exception:
        return "Lateral"


def _suggest_leverage(score: float, direction: str) -> int:
    """Conservative leverage suggestion based on score."""
    if score >= 9:
        return 5
    if score >= 8:
        return 4
    if score >= 7.5:
        return 3
    return 2


def _can_emit(symbol: str) -> bool:
    now = time.time()
    last = _last_sent.get(symbol, 0)
    if now - last < PER_ASSET_COOLDOWN_SEC:
        return False
    day_key = time.strftime("%Y-%m-%d", time.gmtime(now))
    if _day_count.get(day_key, 0) >= GLOBAL_DAILY_LIMIT:
        return False
    return True


def _mark_emitted(symbol: str) -> None:
    now = time.time()
    _last_sent[symbol] = now
    day_key = time.strftime("%Y-%m-%d", time.gmtime(now))
    _day_count[day_key] = _day_count.get(day_key, 0) + 1


def analyse_symbol(symbol: str) -> Optional[Dict]:
    """Run full analysis for a single symbol. Return a signal dict or None."""
    try:
        frames = fetch_all_timeframes_universal(symbol, timeframes=[PRIMARY_TF, HIGHER_TF])
    except Exception as e:
        return {"symbol": symbol, "error": f"fetch: {e}"}

    df_primary = frames.get(PRIMARY_TF)
    df_higher = frames.get(HIGHER_TF)
    if df_primary is None or len(df_primary) < 60:
        return None

    patterns = detect_patterns(df_primary)
    pat_bias = aggregate_bias(patterns)

    regime_p = _hmm_regime(symbol, PRIMARY_TF, df_primary)
    regime_h = _hmm_regime(symbol, HIGHER_TF, df_higher) if df_higher is not None and len(df_higher) >= 60 else None

    conf = calc_confluence(df_primary, df_higher, pat_bias, regime_p, regime_h)

    if conf["direction"] == "none":
        return None
    if conf["score"] < SCORE_THRESHOLD:
        return None

    leverage = _suggest_leverage(conf["score"], conf["direction"])

    return {
        "symbol": symbol,
        "direction": conf["direction"],
        "entry": conf["entry"],
        "sl": conf["sl"],
        "tp1": conf["tp1"],
        "tp2": conf["tp2"],
        "rr": conf["rr"],
        "score": conf["score"],
        "leverage": leverage,
        "risk_pct": 1.5,             # fixed 1.5% per trade suggestion
        "timeframe": PRIMARY_TF,
        "higher_tf": HIGHER_TF,
        "regime": regime_p,
        "higher_regime": regime_h,
        "patterns": pat_bias["patterns"],
        "reasons": conf["reasons"],
        "generated_at": int(time.time()),
    }


def scan_and_emit(watchlist: Optional[List[str]] = None) -> List[Dict]:
    """
    Scan the watchlist and return *new* signals to emit
    (respects cooldowns and daily limit).
    """
    symbols = watchlist or WATCHLIST
    emitted: List[Dict] = []
    for sym in symbols:
        if not _can_emit(sym):
            continue
        sig = analyse_symbol(sym)
        if not sig or "error" in (sig or {}):
            continue
        if "direction" not in sig:
            continue
        emitted.append(sig)
        _mark_emitted(sym)
    return emitted


def debug_scan(watchlist: Optional[List[str]] = None) -> List[Dict]:
    """Like scan_and_emit but returns every analysis (even below threshold)."""
    symbols = watchlist or WATCHLIST
    out: List[Dict] = []
    for sym in symbols:
        try:
            frames = fetch_all_timeframes_universal(sym, timeframes=[PRIMARY_TF, HIGHER_TF])
            df_p = frames.get(PRIMARY_TF)
            df_h = frames.get(HIGHER_TF)
            if df_p is None or len(df_p) < 60:
                out.append({"symbol": sym, "skipped": "no data"})
                continue
            patterns = detect_patterns(df_p)
            pat_bias = aggregate_bias(patterns)
            regime_p = _hmm_regime(sym, PRIMARY_TF, df_p)
            regime_h = _hmm_regime(sym, HIGHER_TF, df_h) if df_h is not None else None
            conf = calc_confluence(df_p, df_h, pat_bias, regime_p, regime_h)
            out.append({
                "symbol": sym,
                "score": conf["score"],
                "direction": conf["direction"],
                "regime": regime_p,
                "higher_regime": regime_h,
                "patterns": pat_bias["patterns"],
            })
        except Exception as e:
            out.append({"symbol": sym, "error": str(e)})
    return out
