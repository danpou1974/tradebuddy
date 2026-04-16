"""
confluence_scorer.py
====================
Combines multiple signal sources into one confluence score (0-10).
- Technical indicators (RSI, MACD, EMAs, Stoch)
- Candlestick patterns (via pattern_detector.aggregate_bias)
- HMM regime
- Support / resistance proximity

A signal is only emitted when score >= threshold (default 7.5) AND
higher-timeframe regime does not contradict the lower-timeframe setup.
"""

from __future__ import annotations
from typing import Dict, Optional, List
import numpy as np
import pandas as pd

from indicators import rsi, macd, ema, stochastic, atr


def _safe(val, default=0.0) -> float:
    try:
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return default
        return float(val)
    except Exception:
        return default


def score_indicators(df: pd.DataFrame) -> Dict:
    """Return bullish/bearish points (-10..+10) from classic indicators."""
    close = df["close"]
    score = 0
    reasons: List[str] = []

    # RSI
    r = _safe(rsi(close, 14).iloc[-1], 50)
    if r < 30:
        score += 2; reasons.append(f"RSI oversold ({r:.1f})")
    elif r < 40:
        score += 1; reasons.append(f"RSI low ({r:.1f})")
    elif r > 70:
        score -= 2; reasons.append(f"RSI overbought ({r:.1f})")
    elif r > 60:
        score -= 1; reasons.append(f"RSI high ({r:.1f})")

    # MACD histogram direction
    _, _, hist = macd(close)
    h_now = _safe(hist.iloc[-1])
    h_prev = _safe(hist.iloc[-2]) if len(hist) >= 2 else 0
    if h_now > 0 and h_now > h_prev:
        score += 2; reasons.append("MACD bullish momentum")
    elif h_now < 0 and h_now < h_prev:
        score -= 2; reasons.append("MACD bearish momentum")
    elif h_now > 0:
        score += 1
    elif h_now < 0:
        score -= 1

    # EMA trend (20 vs 50)
    e20 = _safe(ema(close, 20).iloc[-1])
    e50 = _safe(ema(close, 50).iloc[-1])
    if e20 > e50:
        score += 1; reasons.append("EMA20 > EMA50 (uptrend)")
    elif e20 < e50:
        score -= 1; reasons.append("EMA20 < EMA50 (downtrend)")

    # Price above/below EMA50
    price = _safe(close.iloc[-1])
    if price > e50:
        score += 1
    else:
        score -= 1

    # Stochastic
    k, d = stochastic(df)
    k_now = _safe(k.iloc[-1], 50)
    d_now = _safe(d.iloc[-1], 50)
    if k_now < 20 and k_now > d_now:
        score += 2; reasons.append("Stoch oversold crossover")
    elif k_now > 80 and k_now < d_now:
        score -= 2; reasons.append("Stoch overbought crossover")

    score = max(-10, min(10, score))
    return {"score": score, "reasons": reasons, "rsi": r, "macd_hist": h_now}


def score_regime(regime: str) -> int:
    """Translate HMM regime into bias contribution."""
    mapping = {
        "Alcista Fuerte": 3, "Strong Bullish": 3,
        "Alcista": 2, "Bullish": 2,
        "Acumulación": 1, "Accumulation": 1,
        "Lateral": 0, "Sideways": 0,
        "Distribución": -1, "Distribution": -1,
        "Bajista": -2, "Bearish": -2,
        "Bajista Fuerte": -3, "Strong Bearish": -3,
    }
    return mapping.get(regime, 0)


def find_support_resistance(df: pd.DataFrame, lookback: int = 50) -> Dict:
    """Simple rolling S/R from recent swing highs/lows."""
    d = df.tail(lookback)
    support = float(d["low"].min())
    resistance = float(d["high"].max())
    price = float(df["close"].iloc[-1])
    return {
        "support": round(support, 6),
        "resistance": round(resistance, 6),
        "price": round(price, 6),
        "dist_to_support_pct": (price - support) / price * 100,
        "dist_to_resistance_pct": (resistance - price) / price * 100,
    }


def calc_confluence(
    df_primary: pd.DataFrame,
    df_higher: Optional[pd.DataFrame],
    patterns_bias: Dict,
    regime_primary: str,
    regime_higher: Optional[str] = None,
) -> Dict:
    """
    Combine everything into a 0-10 confluence score + direction.

    Returns dict:
      { score: 0-10, direction: 'long'|'short'|'none',
        entry, sl, tp1, tp2, reasons: [...], sr: {...} }
    """
    ind = score_indicators(df_primary)
    pat = patterns_bias or {"score": 0, "bias": "neutral", "patterns": []}
    reg = score_regime(regime_primary)
    reg_hi = score_regime(regime_higher) if regime_higher else 0

    # Raw aggregate (-16..+16) roughly
    raw = ind["score"] + pat["score"] + reg + (reg_hi * 0.5)

    # Normalise to 0-10 on absolute side
    score10 = min(10.0, abs(raw) * 0.65)

    direction = "none"
    if raw >= 4 and (reg_hi >= 0):
        direction = "long"
    elif raw <= -4 and (reg_hi <= 0):
        direction = "short"

    # Higher timeframe contradicts => kill signal
    if direction == "long" and reg_hi < 0:
        direction = "none"
    if direction == "short" and reg_hi > 0:
        direction = "none"

    sr = find_support_resistance(df_primary, lookback=50)
    atr_series = atr(df_primary, 14)
    atr_now = _safe(atr_series.iloc[-1], sr["price"] * 0.01)

    entry = sr["price"]
    if direction == "long":
        sl = round(min(sr["support"], entry - 1.5 * atr_now), 6)
        tp1 = round(entry + 2.0 * atr_now, 6)
        tp2 = round(entry + 3.5 * atr_now, 6)
    elif direction == "short":
        sl = round(max(sr["resistance"], entry + 1.5 * atr_now), 6)
        tp1 = round(entry - 2.0 * atr_now, 6)
        tp2 = round(entry - 3.5 * atr_now, 6)
    else:
        sl = tp1 = tp2 = None

    reasons = list(ind["reasons"])
    if pat["patterns"]:
        reasons.append("Patterns: " + ", ".join(pat["patterns"]))
    reasons.append(f"Regime {regime_primary}" + (f" / HTF {regime_higher}" if regime_higher else ""))

    # Risk / reward
    rr = None
    if sl and tp1 and entry != sl:
        rr = round(abs(tp1 - entry) / abs(entry - sl), 2)

    return {
        "score": round(score10, 2),
        "direction": direction,
        "entry": round(entry, 6),
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "rr": rr,
        "atr": round(atr_now, 6),
        "sr": sr,
        "reasons": reasons,
        "raw_contributions": {
            "indicators": ind["score"],
            "patterns": pat["score"],
            "regime": reg,
            "higher_regime": reg_hi,
        },
    }
