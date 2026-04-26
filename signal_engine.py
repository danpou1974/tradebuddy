"""
signal_engine.py  v3.2
=======================
Orchestrator for Buddy VIP Signals.

CAMBIOS v3.2:
  - WATCHLIST actualizada: 9 pares (ETH,BNB,ADA,DOT,ATOM,NEAR,ARB,PAXG + BTC mean reversion)
  - Eliminados: SOL, XRP, DOGE (no funciona trend-following), AVAX (35% WR), LINK (39% WR)
  - Scorer reemplazado: scorer_v3.py (port exacto del signalScoringEngine.js v3.2)
  - Threshold: 8.5 (era 7.5)
  - Cooldown: 12h (era 4h)
  - Sin límite diario global (era 4/día)
  - PAIR_DIR_FILTER: ATOM/PAXG/ADA solo LONG, NEAR solo SHORT (backtest 1 año)
  - BTC: estrategia mean reversion (threshold 6.0, RR 1:2, lev max 5x)
  - HMM: se mantiene para contexto de régimen (no como filtro bloqueante)

scan_and_emit():
  1. Iterar watchlist.
  2. Fetch 1h + 4h candles.
  3. HMM regime (caché 30min).
  4. scorer_v3.score_signal() — v3.2 scoring.
  5. Si señal válida y cooldown OK → emitir.
"""

from __future__ import annotations
import time
from typing import Dict, List, Optional

from data_fetcher import fetch_all_timeframes_universal
from hmm_engine import RegimeHMM
from scorer_v3 import score_signal, PAIR_STRATEGIES, PAIR_DIR_FILTER

# ── Config ───────────────────────────────────────────────────────────────────

# Pares activos v3.2 — backtest 1 año confirmado
# ETH 66.7%WR·PF3.08 | BNB 60%·1.97 | ADA 57.9%·1.80 | DOT 55.6%·1.81
# ATOM 53.1%·1.44    | NEAR 52%·2.04 | ARB 60.9%·2.23 | PAXG 51.2%·1.90
# BTC mean reversion: WR 42%, PF 1.40, RR 1:2
WATCHLIST = [
    "BTC/USDT",
    "ETH/USDT", "BNB/USDT", "ADA/USDT",
    "DOT/USDT", "ATOM/USDT",
    "NEAR/USDT", "ARB/USDT", "PAXG/USDT",
]

PRIMARY_TF = "1h"
HIGHER_TF  = "4h"
PER_ASSET_COOLDOWN_SEC = 12 * 3600   # 12h entre señales por par (backtest calibrado)

_hmm_cache: Dict[str, tuple] = {}   # "symbol_tf" -> (model, ts)
_last_sent: Dict[str, float] = {}   # "symbol" -> timestamp última señal


# ── HMM regime (caché 30 min) ────────────────────────────────────────────────

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


# ── Cooldown ─────────────────────────────────────────────────────────────────

def _can_emit(symbol: str) -> bool:
    now = time.time()
    last = _last_sent.get(symbol, 0)
    return (now - last) >= PER_ASSET_COOLDOWN_SEC


def _mark_emitted(symbol: str) -> None:
    _last_sent[symbol] = time.time()


# ── Single symbol analysis ────────────────────────────────────────────────────

def analyse_symbol(symbol: str) -> Optional[Dict]:
    """Run full analysis for a single symbol. Returns a signal dict or None."""
    try:
        frames = fetch_all_timeframes_universal(symbol, timeframes=[PRIMARY_TF, HIGHER_TF])
    except Exception as e:
        return {"symbol": symbol, "error": f"fetch: {e}"}

    df_1h = frames.get(PRIMARY_TF)
    df_4h = frames.get(HIGHER_TF)

    if df_1h is None or len(df_1h) < 60:
        return None

    # HMM regime — solo para contexto, no bloquea señales
    try:
        regime_1h = _hmm_regime(symbol, PRIMARY_TF, df_1h)
        regime_4h = _hmm_regime(symbol, HIGHER_TF, df_4h) if df_4h is not None and len(df_4h) >= 60 else None
    except Exception:
        regime_1h = "Lateral"
        regime_4h = None

    # Score v3.2
    sig = score_signal(symbol, df_1h, df_4h, regime=regime_1h, higher_regime=regime_4h)
    if sig is None:
        return None

    # Enriquecer con régimen HMM
    sig["regime"]        = regime_1h
    sig["higher_regime"] = regime_4h
    sig["timeframe"]     = PRIMARY_TF
    sig["higher_tf"]     = HIGHER_TF
    sig["generated_at"]  = int(time.time())

    return sig


# ── Scan & emit ───────────────────────────────────────────────────────────────

def scan_and_emit(watchlist: Optional[List[str]] = None) -> List[Dict]:
    """
    Scan the watchlist and return new signals (respects cooldowns).
    """
    symbols = watchlist or WATCHLIST
    emitted: List[Dict] = []
    for sym in symbols:
        if not _can_emit(sym):
            continue
        sig = analyse_symbol(sym)
        if not sig or "error" in sig:
            continue
        if not sig.get("direction"):
            continue
        emitted.append(sig)
        _mark_emitted(sym)
    return emitted


def debug_scan(watchlist: Optional[List[str]] = None) -> List[Dict]:
    """Like scan_and_emit but returns all analyses (even below threshold)."""
    symbols = watchlist or WATCHLIST
    out: List[Dict] = []
    for sym in symbols:
        try:
            frames = fetch_all_timeframes_universal(sym, timeframes=[PRIMARY_TF, HIGHER_TF])
            df_1h = frames.get(PRIMARY_TF)
            df_4h = frames.get(HIGHER_TF)
            if df_1h is None or len(df_1h) < 60:
                out.append({"symbol": sym, "skipped": "no data"})
                continue
            regime_1h = _hmm_regime(sym, PRIMARY_TF, df_1h)
            regime_4h = _hmm_regime(sym, HIGHER_TF, df_4h) if df_4h is not None else None
            sig = score_signal(sym, df_1h, df_4h, regime=regime_1h, higher_regime=regime_4h)
            out.append({
                "symbol": sym,
                "score":     sig["score"] if sig else 0,
                "direction": sig["direction"] if sig else "none",
                "strategy":  sig.get("strategy", "trend") if sig else "trend",
                "regime":    regime_1h,
                "higher_regime": regime_4h,
                "signal_generated": sig is not None,
                "dir_filter": PAIR_DIR_FILTER.get(sym),
                "breakdown": sig.get("breakdown") if sig else None,
            })
        except Exception as e:
            out.append({"symbol": sym, "error": str(e)})
    return out
