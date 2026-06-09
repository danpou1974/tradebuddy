"""
scorer_v3.py  — TradeBuddy Signal Scoring Engine v3.2 (Python port)
=====================================================================
Port exacto del signalScoringEngine.js v3.2.

ESTRATEGIA ESTÁNDAR — trend-following (8 pares)
  1. Tendencia  (EMAs + ADX + EMA Pullback)  → máx 3.5
  2. Momentum   (RSI + MACD + Divergencia)   → máx 3.0
  3. Volumen    (ratio + MFI + CMF)          → máx 2.0
  4. MTF 4h     (confirmación 4h)            → máx 1.5
  5. Price Act. (BB + VWAP + Fibonacci)      → máx 1.5
  6. Patrones   (velas japonesas)            → máx 2.0 (gate: s1≥1.5 + core≥6.5)
  Threshold: 8.5

ESTRATEGIA MEAN REVERSION — BTC + ETH (contrarian en extremos)
  GATES (todos obligatorios): RSI<30/70 · BB<0.20/0.80 · Stoch<35/65 · VolRatio>1.2
  1. RSI extremo (<30 long / >70 short)     → máx 2.5
  2. Posición en BB (cerca de banda)        → máx 2.0
  3. Divergencia RSI ×2 (agotamiento)       → máx 2.0
  4. Vol + Stoch + CCI + 4h RSI + Fib      → máx 1.8
  Threshold: 6.0  ·  TP: 38.2% retrace · SL: 27.2% ext · RR ~1.4  ·  MaxLev: 5x  ·  Objetivo WR: 50-58%

FILTRO DIRECCIONAL (PAIR_DIR_FILTER):
  ATOM/USDT → solo LONG  (89% WR long vs 42% short)
  NEAR/USDT → solo SHORT (33% WR long vs 56% short)
  PAXG/USDT → solo LONG  (55% WR long vs 44% short)
  ADA/USDT  → solo LONG  (100% WR long vs 42% short)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple

# ── Config ──────────────────────────────────────────────────────────────────

SIGNAL_THRESHOLD = 8.5
BTC_MR_THRESHOLD = 6.0      # balance calidad/frecuencia — gates de calidad, threshold 6.0

PAIR_STRATEGIES: Dict[str, str] = {
    # RSI Pullback — ganadores backtest 12m: LINK +196% | BNB +408%
    "LINK/USDT": "pullback",
    "BNB/USDT":  "pullback",
    # PAXG y ARB usan 'trend' (default) — ganadores backtest 12m (+126% / +50%)
    # SACADOS BTC/ETH mean_reversion: perdían -260% / -138% (atajaban cuchillos en bear)
}

PAIR_DIR_FILTER: Dict[str, str] = {
    "ATOM/USDT": "long",
    "NEAR/USDT": "short",
    "PAXG/USDT": "long",
    "ADA/USDT":  "long",
}

# ── Math helpers ─────────────────────────────────────────────────────────────

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _r2(n: float) -> float:
    return round(n, 2)

def _safe(val, default=0.0) -> float:
    try:
        f = float(val)
        return default if (np.isnan(f) or np.isinf(f)) else f
    except Exception:
        return default

def _ema_series(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()

def _sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(period).mean()

def _std_series(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(period).std(ddof=0)

def _atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ── Raw indicator calculation ─────────────────────────────────────────────────

def calc_raw(df: pd.DataFrame) -> Optional[Dict]:
    """
    Compute all raw indicators from a DataFrame of candles (columns: open,high,low,close,volume).
    Returns a dict equivalent to the JS `r` object.
    """
    if df is None or len(df) < 50:
        return None

    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"].fillna(0)

    price = _safe(close.iloc[-1])

    # EMAs
    ema9   = _safe(_ema_series(close, 9).iloc[-1])
    ema21  = _safe(_ema_series(close, 21).iloc[-1])
    ema50  = _safe(_ema_series(close, 50).iloc[-1])
    ema200 = _safe(_ema_series(close, min(200, len(close))).iloc[-1])

    # MACD
    ema12 = _ema_series(close, 12)
    ema26 = _ema_series(close, 26)
    macd_line = ema12 - ema26
    macd_sig  = _ema_series(macd_line, 9)
    macd_hist = macd_line - macd_sig
    macd_val      = _safe(macd_line.iloc[-1])
    macd_sig_val  = _safe(macd_sig.iloc[-1])
    macd_hist_val = _safe(macd_hist.iloc[-1])
    macd_hist_prev = _safe(macd_hist.iloc[-2]) if len(macd_hist) >= 2 else 0.0

    # RSI (14)
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-10)
    rsi_val = _safe((100 - 100 / (1 + rs)).iloc[-1], 50)

    # ADX
    adx_val = _calc_adx(df)

    # Stochastic (14,3)
    lo14 = low.rolling(14).min()
    hi14 = high.rolling(14).max()
    stoch_k_s = 100 * (close - lo14) / (hi14 - lo14 + 1e-10)
    stoch_d_s = stoch_k_s.rolling(3).mean()
    stoch_k   = _safe(stoch_k_s.iloc[-1], 50)
    stoch_d   = _safe(stoch_d_s.iloc[-1], 50)

    # CCI (20)
    tp   = (high + low + close) / 3
    tp_ma  = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci_val = _safe(((tp - tp_ma) / (0.015 * tp_mad + 1e-10)).iloc[-1], 0)

    # MFI (14)
    mfi_val = _calc_mfi(df, 14)

    # Bollinger Bands (20, 2)
    bb_sma = _sma(close, 20)
    bb_std = _std_series(close, 20)
    bb_upper = bb_sma + 2 * bb_std
    bb_lower = bb_sma - 2 * bb_std
    bb_u   = _safe(bb_upper.iloc[-1])
    bb_l   = _safe(bb_lower.iloc[-1])
    bb_sma_val = _safe(bb_sma.iloc[-1], price)

    # ATR (14)
    atr_val = _safe(_atr_series(df, 14).iloc[-1], price * 0.01)

    # VWAP
    tp_vwap = (high + low + close) / 3
    vwap_val = _safe((tp_vwap * vol).cumsum().iloc[-1] / (vol.cumsum().iloc[-1] + 1e-10))

    # Volume ratio (avg last 3 / sma20)
    vol_sma20 = _safe(vol.rolling(20).mean().iloc[-1], 1)
    vol_avg3  = _safe(vol.iloc[-3:].mean(), 0)
    vol_ratio = vol_avg3 / (vol_sma20 + 1e-10)

    # Chaikin Money Flow (20)
    hl_range = (high - low).replace(0, np.nan)
    clv = ((close - low) - (high - close)) / hl_range
    cmf_val = _safe((clv * vol).rolling(20).sum().iloc[-1] / (vol.rolling(20).sum().iloc[-1] + 1e-10))

    return {
        "price": price,
        "rsi": rsi_val,
        "macd": macd_val,
        "macdSig": macd_sig_val,
        "macdHist": macd_hist_val,
        "macdHistPrev": macd_hist_prev,
        "ema9": ema9, "ema21": ema21, "ema50": ema50, "ema200": ema200,
        "adx": adx_val,
        "stochK": stoch_k, "stochD": stoch_d,
        "cci": cci_val,
        "mfi": mfi_val,
        "bbU": bb_u, "bbL": bb_l, "bbSma": bb_sma_val,
        "atr": atr_val,
        "vwap": vwap_val,
        "volRatio": vol_ratio,
        "chaikin": cmf_val,
    }


def _calc_adx(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period * 2:
        return 12.0
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    dmp, dmm, trs = [], [], []
    for i in range(1, len(h)):
        up = h[i] - h[i-1]; dn = l[i-1] - l[i]
        dmp.append(up if up > dn and up > 0 else 0)
        dmm.append(dn if dn > up and dn > 0 else 0)
        tr = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
        trs.append(tr)
    def _smma(arr, p):
        r = sum(arr[:p]) / p
        for v in arr[p:]:
            r = (r * (p-1) + v) / p
        return r
    s_tr  = _smma(trs,  period) + 1e-10
    s_dmp = _smma(dmp,  period)
    s_dmm = _smma(dmm,  period)
    di_p  = s_dmp / s_tr * 100
    di_m  = s_dmm / s_tr * 100
    adx   = abs(di_p - di_m) / (di_p + di_m + 1e-10) * 100
    return _r2(adx)


def _calc_mfi(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period + 1:
        return 50.0
    tp  = (df["high"] + df["low"] + df["close"]) / 3
    mf  = tp * df["volume"]
    pos = 0.0; neg = 0.0
    for i in range(len(tp) - period, len(tp)):
        if tp.iloc[i] > tp.iloc[i-1]:
            pos += mf.iloc[i]
        else:
            neg += mf.iloc[i]
    return _safe(100 - 100 / (1 + pos / (neg + 1e-10)), 50)


# ── Direction detection ───────────────────────────────────────────────────────

def detect_direction(r: Dict) -> Optional[str]:
    bull = bear = 0
    if r["ema9"]  > r["ema21"]:  bull += 2
    else:                         bear += 2
    if r["ema21"] > r["ema50"]:  bull += 1
    else:                         bear += 1
    if r["price"] > r["ema200"]: bull += 1
    else:                         bear += 1
    if r["macd"]  > 0:           bull += 2
    else:                         bear += 2
    if r["macdHist"] > 0:        bull += 1
    else:                         bear += 1
    if r["rsi"]   < 50:          bull += 1
    else:                         bear += 1
    if r["stochK"] < 50:         bull += 1
    else:                         bear += 1
    if bull == bear:
        return None
    return "long" if bull > bear else "short"


# ── RSI Divergence ────────────────────────────────────────────────────────────

def rsi_div_score(df: pd.DataFrame, direction: str) -> float:
    if df is None or len(df) < 30:
        return 0.0
    sl = df.tail(40)
    close_vals = sl["close"].values
    rsi_ser = []
    for i in range(15, len(close_vals)):
        d = np.diff(close_vals[:i+1])
        g = np.where(d > 0, d, 0)
        l = np.where(d < 0, -d, 0)
        ag = g[-14:].mean() if len(g) >= 14 else g.mean() + 1e-10
        al = l[-14:].mean() + 1e-10
        rsi_ser.append(100 - 100 / (1 + ag / al))

    lows_hi = sl["high"].values[-len(rsi_ser):]
    lows_lo = sl["low"].values[-len(rsi_ser):]
    half = len(rsi_ser) // 2
    a_rsi, b_rsi = rsi_ser[:half], rsi_ser[half:]
    a_lo,  b_lo  = lows_lo[:half], lows_lo[half:]
    a_hi,  b_hi  = lows_hi[:half], lows_hi[half:]

    if direction == "long":
        if (b_lo.min() < a_lo.min() and
                min(b_rsi) > min(a_rsi) + 3 and
                min(b_rsi) < 55):
            return 0.5
    else:
        if (b_hi.max() > a_hi.max() and
                max(b_rsi) < max(a_rsi) - 3 and
                max(b_rsi) > 45):
            return 0.5
    return 0.0


# ── Fibonacci levels ──────────────────────────────────────────────────────────

def fib_score(df: pd.DataFrame, price: float, direction: str) -> float:
    if df is None or len(df) < 20:
        return 0.0
    sl = df.tail(50)
    hi = float(sl["high"].max()); lo = float(sl["low"].min())
    rng = hi - lo
    if rng <= 0:
        return 0.0
    tol = rng * 0.015
    if direction == "long":
        f618 = hi - rng * 0.618; f500 = hi - rng * 0.500; f382 = hi - rng * 0.382
    else:
        f618 = lo + rng * 0.618; f500 = lo + rng * 0.500; f382 = lo + rng * 0.382
    if abs(price - f618) < tol: return 0.5
    if abs(price - f500) < tol: return 0.4
    if abs(price - f382) < tol: return 0.3
    return 0.0


# ── Candle pattern scoring ────────────────────────────────────────────────────

def candle_score(df: pd.DataFrame, direction: str, vol_ratio: float = 1.0) -> float:
    if df is None or len(df) < 3 or vol_ratio < 1.2:
        return 0.0
    rows = df.tail(3)
    c0, c1, c2 = rows.iloc[2], rows.iloc[1], rows.iloc[0]

    def body(c): return abs(c["close"] - c["open"])
    def rng(c):  return (c["high"] - c["low"]) or 0.0001
    def bull(c): return c["close"] >= c["open"]

    b0, b1, b2 = body(c0), body(c1), body(c2)
    r0, r1, r2 = rng(c0),  rng(c1),  rng(c2)
    bull0, bull1, bull2 = bull(c0), bull(c1), bull(c2)
    uw0 = c0["high"] - max(c0["open"], c0["close"])
    lw0 = min(c0["open"], c0["close"]) - c0["low"]
    uw1 = c1["high"] - max(c1["open"], c1["close"])
    s = 0.0

    if direction == "long":
        # Engulfing
        if not bull1 and bull0 and b0 >= b1*1.5 and c0["open"] <= c1["close"] and c0["close"] >= c1["open"]:
            s += 1.0
        # Morning Star
        if not bull2 and b2 > r2*0.55 and b1 < r1*0.30 and max(c1["open"],c1["close"]) < min(c2["open"],c2["close"]) and bull0 and b0 > r0*0.55 and c0["close"] > (c2["open"]+c2["close"])/2:
            s += 1.2
        # Three soldiers
        if bull0 and bull1 and bull2 and c0["close"] > c1["close"] > c2["close"] and b0 > r0*0.60 and b1 > r1*0.60 and b2 > r2*0.60 and uw0 < b0*0.3 and uw1 < b1*0.3:
            s += 1.0
        # Hammer / Pin bar
        if lw0 >= b0*3.0 and uw0 <= b0*0.3 and b0 > 0 and b0 < r0*0.25 and not bull1:
            s += 0.8
        elif lw0 > r0*0.67 and b0 < r0*0.25:
            s += 0.7
    else:
        # Bearish engulfing
        if bull1 and not bull0 and b0 >= b1*1.5 and c0["open"] >= c1["close"] and c0["close"] <= c1["open"]:
            s += 1.0
        # Evening Star
        if bull2 and b2 > r2*0.55 and b1 < r1*0.30 and min(c1["open"],c1["close"]) > max(c2["open"],c2["close"]) and not bull0 and b0 > r0*0.55 and c0["close"] < (c2["open"]+c2["close"])/2:
            s += 1.2
        # Three crows
        if not bull0 and not bull1 and not bull2 and c0["close"] < c1["close"] < c2["close"] and b0 > r0*0.60 and b1 > r1*0.60 and b2 > r2*0.60:
            s += 1.0
        # Shooting star / Pin bar
        if uw0 >= b0*3.0 and lw0 <= b0*0.3 and b0 > 0 and b0 < r0*0.25 and bull1:
            s += 0.8
        elif uw0 > r0*0.67 and b0 < r0*0.25:
            s += 0.7

    return _clamp(s, 0.0, 1.5)


# ── Score categories ──────────────────────────────────────────────────────────

def score_trend(r: Dict, direction: str) -> float:
    is_bull = direction == "long"
    s = 0.0
    if is_bull:
        if r["ema9"] > r["ema21"] and r["ema21"] > r["ema50"]: s += 1.5
        elif r["ema9"] > r["ema21"]: s += 0.8
        elif r["ema21"] > r["ema50"]: s += 0.4
    else:
        if r["ema9"] < r["ema21"] and r["ema21"] < r["ema50"]: s += 1.5
        elif r["ema9"] < r["ema21"]: s += 0.8
        elif r["ema21"] < r["ema50"]: s += 0.4
    if is_bull and r["price"] > r["ema200"]: s += 0.5
    elif not is_bull and r["price"] < r["ema200"]: s += 0.5
    # ADX
    adx = r["adx"]
    if adx >= 35: s += 1.0
    elif adx >= 25: s += 0.7
    elif adx >= 18: s += 0.3
    # EMA Pullback
    tol = r["price"] * 0.005
    if is_bull:
        if r["ema9"] > r["ema21"] > r["ema50"] and abs(r["price"] - r["ema9"]) < tol: s += 0.5
        elif r["ema9"] > r["ema21"] and abs(r["price"] - r["ema21"]) < tol * 1.5: s += 0.5
    else:
        if r["ema9"] < r["ema21"] < r["ema50"] and abs(r["price"] - r["ema9"]) < tol: s += 0.5
        elif r["ema9"] < r["ema21"] and abs(r["price"] - r["ema21"]) < tol * 1.5: s += 0.5
    return _clamp(s, 0.0, 3.5)


def score_momentum(r: Dict, direction: str, df: pd.DataFrame) -> float:
    is_bull = direction == "long"
    s = 0.0
    rsi = r["rsi"]
    if is_bull:
        if 30 <= rsi <= 50: s += 1.0
        elif 50 < rsi <= 60: s += 0.6
        elif rsi < 30: s += 0.4
    else:
        if 50 <= rsi <= 70: s += 1.0
        elif 40 <= rsi < 50: s += 0.6
        elif rsi > 70: s += 0.4
    if is_bull and r["macd"] > 0: s += 0.5
    elif not is_bull and r["macd"] < 0: s += 0.5
    hist_acc = (is_bull and r["macdHist"] > 0 and r["macdHist"] > r["macdHistPrev"]) or \
               (not is_bull and r["macdHist"] < 0 and r["macdHist"] < r["macdHistPrev"])
    if hist_acc: s += 0.5
    elif (is_bull and r["macdHist"] > 0) or (not is_bull and r["macdHist"] < 0): s += 0.25
    if is_bull and r["stochK"] < 30 and r["stochK"] >= r["stochD"]: s += 0.5
    elif not is_bull and r["stochK"] > 70 and r["stochK"] <= r["stochD"]: s += 0.5
    s += rsi_div_score(df, direction)
    return _clamp(s, 0.0, 3.0)


def score_volume(r: Dict, direction: str) -> float:
    is_bull = direction == "long"
    s = 0.0
    vr = r["volRatio"]
    if vr >= 2.5: s += 1.0
    elif vr >= 1.8: s += 0.7
    elif vr >= 1.3: s += 0.4
    elif vr < 0.7: s -= 0.5
    mfi = r["mfi"]
    if is_bull:
        if mfi >= 60: s += 0.5
        elif mfi >= 50: s += 0.25
    else:
        if mfi <= 40: s += 0.5
        elif mfi < 50: s += 0.25
    cmf = r["chaikin"]
    if is_bull:
        if cmf > 0.12: s += 0.5
        elif cmf > 0.04: s += 0.25
    else:
        if cmf < -0.12: s += 0.5
        elif cmf < -0.04: s += 0.25
    return _clamp(s, 0.0, 2.0)


def score_mtf(r4h: Optional[Dict], direction: str) -> float:
    if r4h is None:
        return 0.0
    is_bull = direction == "long"
    ok1 = (r4h["rsi"] > 45 and r4h["rsi"] < 72) if is_bull else (r4h["rsi"] < 55 and r4h["rsi"] > 28)
    ok2 = r4h["macd"] > 0 if is_bull else r4h["macd"] < 0
    ok3 = r4h["ema9"] > r4h["ema21"] if is_bull else r4h["ema9"] < r4h["ema21"]
    cf = sum([ok1, ok2, ok3])
    if cf == 3: return 1.5
    if cf == 2: return 1.0
    if cf == 1: return 0.5
    return -0.5


def score_price_action(r: Dict, direction: str, df: pd.DataFrame) -> float:
    is_bull = direction == "long"
    s = 0.0
    bb_range = r["bbU"] - r["bbL"]
    bb_pos = (r["price"] - r["bbL"]) / (bb_range + 1e-10) if bb_range > 0 else 0.5
    bb_w   = bb_range / (r["bbSma"] + 1e-10) if r["bbSma"] > 0 else 0
    if is_bull:
        if bb_pos <= 0.15: s += 0.5
        elif bb_pos <= 0.30: s += 0.2
    else:
        if bb_pos >= 0.85: s += 0.5
        elif bb_pos >= 0.70: s += 0.2
    if bb_w < 0.025: s += 0.3
    if is_bull and r["price"] > r["vwap"]: s += 0.3
    elif not is_bull and r["price"] < r["vwap"]: s += 0.3
    s += fib_score(df, r["price"], direction)
    return _clamp(s, 0.0, 1.5)


# ── Fibonacci SL / TP ────────────────────────────────────────────────────────

def calc_fib_tp_sl(
    df: pd.DataFrame,
    entry: float,
    direction: str,
    strategy: str = "trend",
    lookback: int = 60,
) -> tuple:
    """
    Calcula SL y TP usando niveles de Fibonacci.

    mean_reversion:
      Busca el swing previo al extremo actual (ventana [-lookback:-8]).
      LONG  → TP = 61.8% retracement hacia swing high previo
              SL = 127.2% extensión (precio invalida el extremo)
      SHORT → espejo

    trend:
      Busca la pata principal del swing (base + pico pre-pullback).
      LONG  → TP = extensión 161.8% (objetivo onda 3 clásico)
              SL = None → el caller mantiene ATR-based
      SHORT → espejo

    Returns (sl, tp1) donde sl puede ser None para trend.
    Retorna (None, None) si no hay suficientes datos o los niveles no tienen sentido.
    """
    if df is None or len(df) < max(lookback // 2, 20):
        return None, None

    is_bull = direction == "long"

    # ── Mean Reversion ───────────────────────────────────────────────────────
    if strategy == "mean_reversion":
        EXCLUDE = 8  # excluir velas del movimiento extremo actual
        window  = min(lookback, len(df) - EXCLUDE)
        if window < 15:
            return None, None

        pre_move = df.iloc[-(window + EXCLUDE):-EXCLUDE]

        if is_bull:
            swing_ref = float(pre_move["high"].max())   # pico antes de la caída
            move      = swing_ref - entry
        else:
            swing_ref = float(pre_move["low"].min())    # suelo antes de la subida
            move      = entry - swing_ref

        if move <= 0 or (move / entry) < 0.005:         # move < 0.5% — irrelevante
            return None, None

        # TP: 38.2% retracement (más conservador → WR más alto vs 61.8%)
        # SL: 27.2% extensión (invalida el swing extremo)
        # RR = 0.382 / 0.272 ≈ 1.40
        fib_tp = entry + 0.382 * move if is_bull else entry - 0.382 * move
        fib_sl = entry - 0.272 * move if is_bull else entry + 0.272 * move

        sl_dist = abs(entry - fib_sl)
        tp_dist = abs(fib_tp - entry)
        rr = tp_dist / (sl_dist + 1e-10)

        tp_ok = fib_tp > entry if is_bull else fib_tp < entry
        sl_ok = fib_sl < entry if is_bull else fib_sl > entry

        if tp_ok and sl_ok and rr >= 1.0:   # mínimo RR 1.0 (era 1.5)
            return round(fib_sl, 6), round(fib_tp, 6)
        return None, None

    # ── Trend Following ──────────────────────────────────────────────────────
    if strategy == "trend":
        lb   = min(lookback * 2, len(df))
        full = df.tail(lb)
        half = lb // 2

        base      = full.head(half)
        pre_pull  = full.iloc[half:lb - 5]   # pico/suelo antes del pullback actual
        if len(pre_pull) < 5:
            return None, None

        if is_bull:
            swing_low  = float(base["low"].min())
            swing_high = float(pre_pull["high"].max())
            leg  = swing_high - swing_low
            fib_tp = swing_high + 0.618 * leg       # extensión 161.8%
        else:
            swing_high = float(base["high"].max())
            swing_low  = float(pre_pull["low"].min())
            leg  = swing_high - swing_low
            fib_tp = swing_low - 0.618 * leg        # extensión 161.8% hacia abajo

        if leg <= 0:
            return None, None

        tp_ok = fib_tp > entry if is_bull else fib_tp < entry
        if not tp_ok:
            return None, None

        return None, round(fib_tp, 6)   # SL = None en trend (ATR-based del caller)

    return None, None


# ── Main scoring functions ────────────────────────────────────────────────────

def score_trend_following(
    symbol: str,
    df_1h: pd.DataFrame,
    df_4h: Optional[pd.DataFrame] = None,
    regime: str = "Lateral",
    higher_regime: Optional[str] = None,
) -> Optional[Dict]:
    """Standard trend-following scorer. Returns signal dict or None."""
    r = calc_raw(df_1h)
    if r is None:
        return None

    direction = detect_direction(r)
    if direction is None:
        return None

    # Direction filter
    allowed = PAIR_DIR_FILTER.get(symbol)
    if allowed and direction != allowed:
        return None

    # Macro filter: long only above EMA200, short only below
    macro_bull = r["price"] > r["ema200"]
    if direction == "long"  and not macro_bull: return None
    if direction == "short" and macro_bull:     return None

    r4h = calc_raw(df_4h) if df_4h is not None and len(df_4h) >= 50 else None

    s1 = score_trend(r, direction)
    s2 = score_momentum(r, direction, df_1h)
    s3 = score_volume(r, direction)
    s4 = score_mtf(r4h, direction)
    s5 = score_price_action(r, direction, df_1h)
    core = _clamp(_r2(s1 + s2 + s3 + s4 + s5), 0, 10)
    s6 = candle_score(df_1h, direction, r["volRatio"]) if (s1 >= 1.5 and core >= 6.5) else 0.0
    score = _clamp(_r2(s1 + s2 + s3 + s4 + s5 + s6), 0, 10)

    if score < SIGNAL_THRESHOLD:
        return None

    atr_v = r["atr"]
    price = r["price"]
    atr_pct = (atr_v / price) * 100 if price > 0 else 2.0
    sl_mult = 3.0 if atr_pct > 4 else (2.5 if atr_pct > 2 else 2.0)
    sl_dist = max(atr_v * sl_mult, price * 0.030)   # mínimo 3% — grid search óptimo (WR 58%, PF 2.10)
    sl      = price - sl_dist if direction == "long" else price + sl_dist
    tp1     = price + sl_dist * 1.5 if direction == "long" else price - sl_dist * 1.5  # RR 1.5
    rr_val  = 1.5
    lev     = 3 if atr_pct > 4 else (4 if atr_pct > 2.5 else (5 if atr_pct > 1.5 else 6))

    reasons = _build_reasons(r, direction, s1, s2, s3, s4, s5, s6, regime, higher_regime)

    be_level = round(price + sl_dist * 0.8, 6) if direction == "long" else round(price - sl_dist * 0.8, 6)

    return {
        "symbol": symbol,
        "direction": direction,
        "strategy": "trend",
        "entry": round(price, 6),
        "sl": round(sl, 6),
        "tp1": round(tp1, 6),
        "breakeven_level": be_level,
        "leverage": lev,
        "score": score,
        "rr": rr_val,
        "risk_pct": 1.0,
        "breakdown": {"s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5, "s6": s6},
        "reasons": reasons,
    }


def score_mean_reversion(
    symbol: str,
    df_1h: pd.DataFrame,
    df_4h: Optional[pd.DataFrame] = None,
) -> Optional[Dict]:
    """
    Mean reversion scorer v2 — BTC + ETH (y cualquier par en PAIR_STRATEGIES).

    GATES (todos obligatorios — filtran correcciones normales):
      1. RSI 1h < 30 (long) / > 70 (short)           — era 35/65
      2. BB position < 0.20 (long) / > 0.80 (short)  — era 0.25/0.75
      3. Stochastic K < 35 (long) / > 65 (short)     — NUEVO obligatorio
      4. Volume ratio > 1.2                            — NUEVO: requiere volumen de pánico

    Threshold: 6.0 — balance entre frecuencia (~5-10 señales/mes BTC+ETH) y calidad.
    Objetivo WR: 52-58% (vs 42% original).
    """
    r = calc_raw(df_1h)
    if r is None:
        return None

    price    = r["price"]
    rsi      = r["rsi"]
    bb_range = r["bbU"] - r["bbL"]
    bb_pos   = (price - r["bbL"]) / (bb_range + 1e-10) if bb_range > 0 else 0.5

    # ── Gate 1+2: RSI + BB ───────────────────────────────────────────────────
    if rsi < 30 and bb_pos < 0.20:
        direction = "long"
    elif rsi > 70 and bb_pos > 0.80:
        direction = "short"
    else:
        return None

    is_bull = direction == "long"

    # ── Gate 3: Stochastic confirma extremo ──────────────────────────────────
    if not ((is_bull and r["stochK"] < 35) or (not is_bull and r["stochK"] > 65)):
        return None

    # ── Gate 4: Volumen de pánico/euforia ─────────────────────────────────────
    if r["volRatio"] < 1.2:
        return None

    # ── Scoring ───────────────────────────────────────────────────────────────

    # 1. RSI extreme (0-2.5)
    if is_bull:
        if   rsi < 20: s_rsi = 2.5
        elif rsi < 23: s_rsi = 2.2
        elif rsi < 25: s_rsi = 2.0
        elif rsi < 28: s_rsi = 1.7
        else:          s_rsi = 1.3   # 28-30
    else:
        if   rsi > 80: s_rsi = 2.5
        elif rsi > 77: s_rsi = 2.2
        elif rsi > 75: s_rsi = 2.0
        elif rsi > 72: s_rsi = 1.7
        else:          s_rsi = 1.3   # 70-72
    s_rsi = _clamp(s_rsi, 0, 2.5)

    # 2. BB position (0-2.0)
    if is_bull:
        if   bb_pos <= 0.00: s_bb = 2.0   # por debajo banda inferior
        elif bb_pos <= 0.03: s_bb = 1.8
        elif bb_pos <= 0.07: s_bb = 1.5
        elif bb_pos <= 0.12: s_bb = 1.0
        else:                s_bb = 0.6   # 0.12-0.20 (zona ampliada)
    else:
        if   bb_pos >= 1.00: s_bb = 2.0
        elif bb_pos >= 0.97: s_bb = 1.8
        elif bb_pos >= 0.93: s_bb = 1.5
        elif bb_pos >= 0.88: s_bb = 1.0
        else:                s_bb = 0.6   # 0.80-0.88
    s_bb = _clamp(s_bb, 0, 2.0)

    # 3. RSI divergencia ×2 (0-2.0) — agotamiento del movimiento
    div   = rsi_div_score(df_1h, direction) * 4.0
    s_div = _clamp(div, 0, 2.0)

    # 4. Extra: volumen + stoch profundidad + CCI + 4h + fib (0-1.8)
    vr    = r["volRatio"]
    s_vol = 0.8 if vr >= 3.0 else (0.6 if vr >= 2.0 else (0.4 if vr >= 1.5 else 0.2))

    # Stoch: ya pasó gate (< 35 / > 65), bonus por profundidad
    if   (is_bull and r["stochK"] < 10) or (not is_bull and r["stochK"] > 90): s_stoch = 0.5
    elif (is_bull and r["stochK"] < 20) or (not is_bull and r["stochK"] > 80): s_stoch = 0.3
    elif (is_bull and r["stochK"] < 25) or (not is_bull and r["stochK"] > 75): s_stoch = 0.2
    else:                                                                        s_stoch = 0.1

    # CCI confirma sobrecompra/sobreventa
    cci = r["cci"]
    if   (is_bull and cci < -150) or (not is_bull and cci > 150): s_cci = 0.4
    elif (is_bull and cci < -100) or (not is_bull and cci > 100): s_cci = 0.2
    else:                                                           s_cci = 0.0

    # 4h RSI — si 4h también en extremo, señal mucho más fiable
    r4h   = calc_raw(df_4h) if df_4h is not None and len(df_4h) >= 50 else None
    rsi4h = None
    s_4h  = 0.0
    if r4h is not None:
        rsi4h = r4h["rsi"]
        if   (is_bull and rsi4h < 35) or (not is_bull and rsi4h > 65): s_4h = 0.5
        elif (is_bull and rsi4h < 42) or (not is_bull and rsi4h > 58): s_4h = 0.2

    s_fib = fib_score(df_1h, price, direction)
    bb_w  = bb_range / (r["bbSma"] + 1e-10) if r["bbSma"] > 0 else 0
    s_sq  = 0.2 if bb_w < 0.025 else 0.0

    s_extra = _clamp(s_vol + s_stoch + s_cci + s_4h + s_fib + s_sq, 0, 1.8)

    score = _clamp(_r2(s_rsi + s_bb + s_div + s_extra), 0, 10)
    if score < BTC_MR_THRESHOLD:
        return None

    # ── Trade params: Fibonacci (fallback ATR) ────────────────────────────────
    atr_v   = r["atr"]
    atr_pct = (atr_v / price) * 100 if price > 0 else 2.0
    # SL mínimo 3% (grid search óptimo: WR 58.3%, PF 2.10)
    # TP = SL × 1.5 siempre (RR 1.5 óptimo)
    atr_dist = max(atr_v * 2.0, price * 0.030)   # mínimo 3%
    atr_sl   = price - atr_dist if is_bull else price + atr_dist
    atr_tp1  = price + atr_dist * 1.5 if is_bull else price - atr_dist * 1.5

    fib_sl, fib_tp = calc_fib_tp_sl(df_1h, price, direction, "mean_reversion")

    # Fibonacci válido → usarlo con floor mínimo; si no, ATR como fallback
    if fib_sl is not None and fib_tp is not None:
        raw_sl_dist = abs(price - fib_sl)
        sl_dist_mr = max(raw_sl_dist, price * 0.030)
        sl  = price - sl_dist_mr if is_bull else price + sl_dist_mr
        tp1 = price + sl_dist_mr * 1.5 if is_bull else price - sl_dist_mr * 1.5  # RR 1.5
        rr_val = 1.5
    else:
        sl  = atr_sl
        tp1 = atr_tp1
        rr_val = 1.5

    lev = 5

    reasons_out = [
        f"RSI 1h en extremo {'sobrevendido' if is_bull else 'sobrecomprado'} ({rsi:.1f})",
        f"Precio en {'banda inferior' if is_bull else 'banda superior'} de BB (pos {bb_pos:.2f})",
        f"Stochastic K confirma extremo ({r['stochK']:.0f})",
        f"Volumen de {'pánico' if is_bull else 'euforia'} ({vr:.1f}× media)",
    ]
    if s_div > 0:
        reasons_out.append(f"Divergencia RSI detectada (+{s_div:.1f}pts)")
    if s_4h > 0 and rsi4h is not None:
        reasons_out.append(f"4h RSI alineado en extremo ({rsi4h:.1f})")
    if s_cci > 0:
        reasons_out.append(f"CCI confirma agotamiento ({cci:.0f})")

    sl_dist_mr = abs(price - sl)
    be_level   = round(price + sl_dist_mr * 0.8, 6) if is_bull else round(price - sl_dist_mr * 0.8, 6)

    return {
        "symbol":    symbol,
        "direction": direction,
        "strategy":  "mean_reversion",
        "entry":     round(price, 6),
        "sl":        round(sl, 6),
        "tp1":       round(tp1, 6),
        "breakeven_level": be_level,
        "leverage":  lev,
        "score":     score,
        "rr":        rr_val,
        "risk_pct":  1.0,
        "breakdown": {
            "rsiExtreme":  s_rsi,
            "bbPosition":  s_bb,
            "divergencia": s_div,
            "volumen":     s_vol,
            "stoch":       s_stoch,
            "cci":         s_cci,
            "mtf_4h":      s_4h,
        },
        "reasons": reasons_out,
    }


# Alias para compatibilidad con código existente
score_btc_mean_reversion = score_mean_reversion


# ── RSI Pullback en Tendencia ─────────────────────────────────────────────────

def score_rsi_pullback(
    symbol: str,
    df_1h: pd.DataFrame,
    df_4h: Optional[pd.DataFrame] = None,
) -> Optional[Dict]:
    """
    RSI Pullback en Tendencia — Connors-style adaptado a crypto.

    Lógica: el mercado está en tendencia 4h confirmada → RSI 1h cae a zona
    35-50 (corrección temporal) → precio toca EMA21 (soporte dinámico) →
    los compradores institucionales absorben y el precio retoma la tendencia.

    WR histórico documentado: 55-65% (Connors Research 2009, Caginalp 2015).

    GATES obligatorios:
      1. 4h: EMA9 > EMA21 > EMA50 (o inverso para short) — tendencia establecida
      2. 4h: ADX > 20 — hay fuerza real, no ruido lateral
      3. 1h: RSI entre 35-50 (long) / 50-65 (short) — corrección, no reversión
      4. 1h: precio a max 1.5% de EMA21 — toca soporte dinámico
      5. 1h: volRatio > 1.2 — compradores llegando

    TP = SL × 1.5  (RR 1.5:1)
    Objetivo WR: 55-62%
    """
    r1h = calc_raw(df_1h)
    if r1h is None:
        return None

    r4h = calc_raw(df_4h) if df_4h is not None and len(df_4h) >= 50 else None
    if r4h is None:
        return None  # Sin confirmación 4h no hay señal

    price   = r1h["price"]
    rsi_1h  = r1h["rsi"]
    adx_4h  = _calc_adx(df_4h) if df_4h is not None and len(df_4h) >= 30 else 0.0

    # ── Gate 1: Tendencia 4h clara ───────────────────────────────────────────
    bull_4h = r4h["ema9"] > r4h["ema21"] and r4h["ema21"] > r4h["ema50"]
    bear_4h = r4h["ema9"] < r4h["ema21"] and r4h["ema21"] < r4h["ema50"]
    if not bull_4h and not bear_4h:
        return None

    direction = "long" if bull_4h else "short"
    is_bull   = direction == "long"

    # ── Gate 2: ADX 4h > 20 ─────────────────────────────────────────────────
    if adx_4h < 20:
        return None

    # ── Gate 3: RSI 1h en zona de pullback ──────────────────────────────────
    if is_bull and not (35 <= rsi_1h <= 50):
        return None
    if not is_bull and not (50 <= rsi_1h <= 65):
        return None

    # ── Gate 4: Precio cerca de EMA21 1h ────────────────────────────────────
    ema21_1h  = r1h["ema21"]
    proximity = abs(price - ema21_1h) / (price + 1e-10)
    if proximity > 0.015:
        return None

    # ── Gate 5: Volumen confirmando entrada ──────────────────────────────────
    if r1h["volRatio"] < 1.2:
        return None

    # ── Filtro direccional ───────────────────────────────────────────────────
    allowed = PAIR_DIR_FILTER.get(symbol)
    if allowed and direction != allowed:
        return None

    # ── Scoring (0-10) ───────────────────────────────────────────────────────
    s = 0.0

    # Fuerza de tendencia 4h (0-3.0)
    s += 2.0  # ya pasó gate EMA stack
    if   adx_4h >= 35: s += 1.0
    elif adx_4h >= 28: s += 0.7
    elif adx_4h >= 22: s += 0.4
    else:              s += 0.2

    # Profundidad del pullback RSI (0-2.5) — más oversold = mejor entrada
    if is_bull:
        if   rsi_1h <= 38: s += 2.5
        elif rsi_1h <= 42: s += 2.0
        elif rsi_1h <= 46: s += 1.5
        else:              s += 1.0
    else:
        if   rsi_1h >= 62: s += 2.5
        elif rsi_1h >= 58: s += 2.0
        elif rsi_1h >= 54: s += 1.5
        else:              s += 1.0

    # Proximidad EMA21 (0-1.5) — más cerca = soporte más preciso
    if   proximity < 0.003: s += 1.5
    elif proximity < 0.007: s += 1.0
    elif proximity < 0.012: s += 0.6
    else:                   s += 0.3

    # Volumen (0-1.0)
    vr = r1h["volRatio"]
    if   vr >= 2.5: s += 1.0
    elif vr >= 1.8: s += 0.7
    elif vr >= 1.4: s += 0.5
    else:           s += 0.3

    # MACD 1h acelerando en dirección de la tendencia (0-1.0)
    macd_acc = (is_bull  and r1h["macdHist"] > 0 and r1h["macdHist"] > r1h["macdHistPrev"]) or \
               (not is_bull and r1h["macdHist"] < 0 and r1h["macdHist"] < r1h["macdHistPrev"])
    if macd_acc:        s += 1.0
    elif (is_bull and r1h["macdHist"] > r1h["macdHistPrev"]) or \
         (not is_bull  and r1h["macdHist"] < r1h["macdHistPrev"]):
        s += 0.5

    # 4h RSI en zona saludable de tendencia (0-1.0)
    rsi_4h = r4h["rsi"]
    if is_bull  and 45 <= rsi_4h <= 68: s += 1.0
    elif not is_bull and 32 <= rsi_4h <= 55: s += 1.0
    elif is_bull  and 40 <= rsi_4h <= 72: s += 0.5
    elif not is_bull and 28 <= rsi_4h <= 60: s += 0.5

    score = _clamp(_r2(s), 0, 10)

    PULLBACK_THRESHOLD = 6.5
    if score < PULLBACK_THRESHOLD:
        return None

    # ── Trade params ──────────────────────────────────────────────────────────
    atr_v   = r1h["atr"]
    atr_pct = (atr_v / price) * 100 if price > 0 else 2.0

    # SL: por debajo/encima del mínimo/máximo del pullback (últimas 5 velas)
    if is_bull:
        swing_sl  = float(df_1h.tail(5)["low"].min()) - atr_v * 0.3
        atr_sl    = price - max(atr_v * 1.5, price * 0.012)
        sl        = max(swing_sl, atr_sl)   # el más cercano al precio
    else:
        swing_sl  = float(df_1h.tail(5)["high"].max()) + atr_v * 0.3
        atr_sl    = price + max(atr_v * 1.5, price * 0.012)
        sl        = min(swing_sl, atr_sl)

    sl_dist = max(abs(price - sl), price * 0.030)    # mínimo 3% — grid search óptimo
    sl  = price - sl_dist if is_bull else price + sl_dist
    tp1 = price + sl_dist * 1.5 if is_bull else price - sl_dist * 1.5  # RR 1.5
    rr_val = 1.5

    lev = 4 if atr_pct > 3 else (5 if atr_pct > 1.5 else 6)

    reasons = [
        f"Tendencia 4h {'alcista' if is_bull else 'bajista'} confirmada "
        f"(EMA9>EMA21>EMA50, ADX {adx_4h:.0f})",
        f"RSI 1h en zona de pullback ({rsi_1h:.1f}) — corrección temporal",
        f"Precio en EMA21 1h (soporte dinámico, {proximity*100:.2f}% distancia)",
        f"Volumen de {'compra' if is_bull else 'venta'} confirmado ({vr:.1f}x media)",
    ]
    if macd_acc:
        reasons.append(f"MACD 1h acelerando en dirección de la tendencia")
    if s >= 8.0:
        reasons.append(f"Alta confluencia de factores (score {score:.1f}/10)")

    be_level = round(price + sl_dist * 0.8, 6) if is_bull else round(price - sl_dist * 0.8, 6)

    return {
        "symbol":    symbol,
        "direction": direction,
        "strategy":  "pullback",
        "entry":     round(price, 6),
        "sl":        round(sl, 6),
        "tp1":       round(tp1, 6),
        "breakeven_level": be_level,
        "leverage":  lev,
        "score":     score,
        "rr":        rr_val,
        "risk_pct":  1.0,
        "breakdown": {
            "adx_4h":    round(adx_4h, 1),
            "rsi_1h":    round(rsi_1h, 1),
            "rsi_4h":    round(rsi_4h, 1),
            "ema_prox":  round(proximity * 100, 2),
            "vol_ratio": round(vr, 2),
        },
        "reasons": reasons,
    }


# ── Public entry point ────────────────────────────────────────────────────────

def score_signal(
    symbol: str,
    df_1h: pd.DataFrame,
    df_4h: Optional[pd.DataFrame] = None,
    regime: str = "Lateral",
    higher_regime: Optional[str] = None,
) -> Optional[Dict]:
    """
    Main entry point. Routes to correct strategy based on PAIR_STRATEGIES.
    Returns a complete signal dict or None.
    """
    strategy = PAIR_STRATEGIES.get(symbol, "trend")
    if strategy == "mean_reversion":
        return score_mean_reversion(symbol, df_1h, df_4h)
    if strategy == "pullback":
        return score_rsi_pullback(symbol, df_1h, df_4h)
    return score_trend_following(symbol, df_1h, df_4h, regime, higher_regime)


# ── Reason builder ────────────────────────────────────────────────────────────

def _build_reasons(
    r: Dict, direction: str,
    s1: float, s2: float, s3: float, s4: float, s5: float, s6: float,
    regime: str = "", higher_regime: Optional[str] = None,
) -> List[str]:
    is_bull = direction == "long"
    reasons = []
    if s1 >= 1.5:
        reasons.append(f"Tendencia fuerte {'alcista' if is_bull else 'bajista'} (EMAs+ADX {s1:.1f})")
    if s2 >= 1.5:
        reasons.append(f"Momentum {'alcista' if is_bull else 'bajista'} confirmado (RSI+MACD {s2:.1f})")
    if s3 >= 0.7:
        reasons.append(f"Volumen institucional (ratio {r['volRatio']:.1f}x, {s3:.1f}pts)")
    if s4 >= 1.0:
        reasons.append(f"4h alineado {'alcista' if is_bull else 'bajista'} ({s4:.1f}pts)")
    if s5 >= 0.5:
        reasons.append(f"Price action: BB+VWAP+Fib ({s5:.1f}pts)")
    if s6 >= 0.8:
        reasons.append(f"Patrón de velas confirmado ({s6:.1f}pts)")
    if regime and regime not in ("Lateral", "Sideways"):
        reasons.append(f"Régimen HMM 1h: {regime}")
    if higher_regime and higher_regime not in ("Lateral", "Sideways"):
        reasons.append(f"Régimen HMM 4h: {higher_regime}")
    return reasons
