"""
pattern_detector.py
===================
Candlestick pattern detection for Buddy VIP Signals.
Input: pandas DataFrame with columns open/high/low/close (OHLC).
Output: list of detected patterns with bullish/bearish bias.

All functions analyse the LAST fully-closed candle (and prior context
when required). Return None when no pattern detected.
"""

from __future__ import annotations
from typing import Optional, Dict, List
import pandas as pd


# ---------- helpers ---------------------------------------------------------

def _body(c) -> float:
    return abs(c["close"] - c["open"])

def _range(c) -> float:
    r = c["high"] - c["low"]
    return r if r > 0 else 1e-9

def _upper_shadow(c) -> float:
    return c["high"] - max(c["close"], c["open"])

def _lower_shadow(c) -> float:
    return min(c["close"], c["open"]) - c["low"]

def _is_bull(c) -> bool:
    return c["close"] > c["open"]

def _is_bear(c) -> bool:
    return c["close"] < c["open"]


# ---------- individual patterns --------------------------------------------

def detect_doji(c) -> Optional[Dict]:
    """Body <= 10% of range => indecision."""
    if _body(c) <= _range(c) * 0.10:
        return {"name": "Doji", "bias": "neutral", "strength": 1}
    return None


def detect_hammer(c) -> Optional[Dict]:
    """Small body top, long lower shadow >= 2x body. Bullish reversal."""
    body = _body(c)
    if body < _range(c) * 0.05:
        return None
    if _lower_shadow(c) >= 2 * body and _upper_shadow(c) <= body * 0.5:
        return {"name": "Hammer", "bias": "bullish", "strength": 2}
    return None


def detect_inverted_hammer(c) -> Optional[Dict]:
    body = _body(c)
    if body < _range(c) * 0.05:
        return None
    if _upper_shadow(c) >= 2 * body and _lower_shadow(c) <= body * 0.5:
        return {"name": "Inverted Hammer", "bias": "bullish", "strength": 2}
    return None


def detect_shooting_star(c, prev) -> Optional[Dict]:
    """Same shape as inverted hammer but after an uptrend => bearish."""
    body = _body(c)
    if body < _range(c) * 0.05:
        return None
    if _upper_shadow(c) >= 2 * body and _lower_shadow(c) <= body * 0.5:
        if prev is not None and prev["close"] > prev["open"]:
            return {"name": "Shooting Star", "bias": "bearish", "strength": 2}
    return None


def detect_hanging_man(c, prev) -> Optional[Dict]:
    """Hammer shape after an uptrend => bearish."""
    body = _body(c)
    if body < _range(c) * 0.05:
        return None
    if _lower_shadow(c) >= 2 * body and _upper_shadow(c) <= body * 0.5:
        if prev is not None and prev["close"] > prev["open"]:
            return {"name": "Hanging Man", "bias": "bearish", "strength": 2}
    return None


def detect_bullish_engulfing(prev, c) -> Optional[Dict]:
    if prev is None:
        return None
    if _is_bear(prev) and _is_bull(c):
        if c["open"] <= prev["close"] and c["close"] >= prev["open"]:
            return {"name": "Bullish Engulfing", "bias": "bullish", "strength": 3}
    return None


def detect_bearish_engulfing(prev, c) -> Optional[Dict]:
    if prev is None:
        return None
    if _is_bull(prev) and _is_bear(c):
        if c["open"] >= prev["close"] and c["close"] <= prev["open"]:
            return {"name": "Bearish Engulfing", "bias": "bearish", "strength": 3}
    return None


def detect_morning_star(a, b, c) -> Optional[Dict]:
    """Three-candle bullish reversal: big bear / small body / big bull."""
    if a is None or b is None:
        return None
    if _is_bear(a) and _body(a) > _range(a) * 0.5:
        if _body(b) < _range(b) * 0.35:
            if _is_bull(c) and c["close"] > (a["open"] + a["close"]) / 2:
                return {"name": "Morning Star", "bias": "bullish", "strength": 3}
    return None


def detect_evening_star(a, b, c) -> Optional[Dict]:
    if a is None or b is None:
        return None
    if _is_bull(a) and _body(a) > _range(a) * 0.5:
        if _body(b) < _range(b) * 0.35:
            if _is_bear(c) and c["close"] < (a["open"] + a["close"]) / 2:
                return {"name": "Evening Star", "bias": "bearish", "strength": 3}
    return None


def detect_three_white_soldiers(a, b, c) -> Optional[Dict]:
    if a is None or b is None:
        return None
    if all(_is_bull(x) for x in (a, b, c)):
        if b["close"] > a["close"] and c["close"] > b["close"]:
            if b["open"] > a["open"] and c["open"] > b["open"]:
                return {"name": "Three White Soldiers", "bias": "bullish", "strength": 3}
    return None


def detect_three_black_crows(a, b, c) -> Optional[Dict]:
    if a is None or b is None:
        return None
    if all(_is_bear(x) for x in (a, b, c)):
        if b["close"] < a["close"] and c["close"] < b["close"]:
            if b["open"] < a["open"] and c["open"] < b["open"]:
                return {"name": "Three Black Crows", "bias": "bearish", "strength": 3}
    return None


def detect_piercing_line(prev, c) -> Optional[Dict]:
    if prev is None:
        return None
    if _is_bear(prev) and _is_bull(c):
        mid = (prev["open"] + prev["close"]) / 2
        if c["open"] < prev["low"] and c["close"] > mid and c["close"] < prev["open"]:
            return {"name": "Piercing Line", "bias": "bullish", "strength": 2}
    return None


def detect_dark_cloud_cover(prev, c) -> Optional[Dict]:
    if prev is None:
        return None
    if _is_bull(prev) and _is_bear(c):
        mid = (prev["open"] + prev["close"]) / 2
        if c["open"] > prev["high"] and c["close"] < mid and c["close"] > prev["open"]:
            return {"name": "Dark Cloud Cover", "bias": "bearish", "strength": 2}
    return None


# ---------- orchestrator ----------------------------------------------------

def detect_patterns(df: pd.DataFrame) -> List[Dict]:
    """
    Run every detector on the last few candles.
    Returns list of detected patterns (empty list if none).
    """
    if df is None or len(df) < 3:
        return []

    # lowercase columns once
    df = df.rename(columns={c: c.lower() for c in df.columns})
    needed = {"open", "high", "low", "close"}
    if not needed.issubset(df.columns):
        return []

    c = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else None
    prev2 = df.iloc[-3] if len(df) >= 3 else None

    found: List[Dict] = []

    for fn in (detect_doji, detect_hammer, detect_inverted_hammer):
        r = fn(c)
        if r:
            found.append(r)

    for fn in (detect_shooting_star, detect_hanging_man):
        r = fn(c, prev)
        if r:
            found.append(r)

    for fn in (detect_bullish_engulfing, detect_bearish_engulfing,
               detect_piercing_line, detect_dark_cloud_cover):
        r = fn(prev, c)
        if r:
            found.append(r)

    for fn in (detect_morning_star, detect_evening_star,
               detect_three_white_soldiers, detect_three_black_crows):
        r = fn(prev2, prev, c)
        if r:
            found.append(r)

    return found


def aggregate_bias(patterns: List[Dict]) -> Dict:
    """
    Summarise a list of patterns into a single net bias & score.
    score range: -10 (very bearish) to +10 (very bullish).
    """
    score = 0
    for p in patterns:
        s = p.get("strength", 1)
        if p["bias"] == "bullish":
            score += s
        elif p["bias"] == "bearish":
            score -= s
    score = max(-10, min(10, score))
    if score >= 2:
        bias = "bullish"
    elif score <= -2:
        bias = "bearish"
    else:
        bias = "neutral"
    return {"score": score, "bias": bias, "patterns": [p["name"] for p in patterns]}
