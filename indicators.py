"""
indicators.py
=============
30+ indicadores técnicos equivalentes a TradingView / Binance.
"""

import numpy as np
import pandas as pd


def sma(close, period):
    return close.rolling(period).mean()

def ema(close, period):
    return close.ewm(span=period, adjust=False).mean()

def wma(close, period):
    w = np.arange(1, period + 1)
    return close.rolling(period).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)

def hull_ma(close, period):
    h = int(period / 2); s = int(np.sqrt(period))
    return wma(2 * wma(close, h) - wma(close, period), s)

def vwap(df):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return (tp * df["volume"]).cumsum() / df["volume"].cumsum()

def vwma(df, period=20):
    return (df["close"] * df["volume"]).rolling(period).sum() / df["volume"].rolling(period).sum()

def rsi(close, period=14):
    d = close.diff()
    g = d.clip(lower=0).rolling(period).mean()
    l = (-d.clip(upper=0)).rolling(period).mean()
    return 100 - (100 / (1 + g / (l + 1e-10)))

def macd(close, fast=12, slow=26, signal=9):
    macd_line = ema(close, fast) - ema(close, slow)
    sig_line  = ema(macd_line, signal)
    return macd_line, sig_line, macd_line - sig_line

def stochastic(df, k=14, d=3):
    lo = df["low"].rolling(k).min()
    hi = df["high"].rolling(k).max()
    pk = 100 * (df["close"] - lo) / (hi - lo + 1e-10)
    return pk, pk.rolling(d).mean()

def cci(df, period=20):
    tp  = (df["high"] + df["low"] + df["close"]) / 3
    ma  = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - ma) / (0.015 * mad + 1e-10)

def williams_r(df, period=14):
    hh = df["high"].rolling(period).max()
    ll = df["low"].rolling(period).min()
    return -100 * (hh - df["close"]) / (hh - ll + 1e-10)

def roc(close, period=12):
    return ((close - close.shift(period)) / (close.shift(period) + 1e-10)) * 100

def mfi(df, period=14):
    tp  = (df["high"] + df["low"] + df["close"]) / 3
    rmf = tp * df["volume"]
    pos = rmf.where(tp > tp.shift(1), 0).rolling(period).sum()
    neg = rmf.where(tp < tp.shift(1), 0).rolling(period).sum()
    return 100 - (100 / (1 + pos / (neg + 1e-10)))

def atr(df, period=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"]  - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def bollinger_bands(close, period=20, std_dev=2.0):
    mid   = sma(close, period)
    std   = close.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower, (upper - lower) / (mid + 1e-10) * 100, (close - lower) / (upper - lower + 1e-10)

def keltner_channel(df, ema_p=20, atr_p=10, mult=2.0):
    mid = ema(df["close"], ema_p)
    a   = atr(df, atr_p)
    return mid + mult * a, mid, mid - mult * a

def donchian_channel(df, period=20):
    hi = df["high"].rolling(period).max()
    lo = df["low"].rolling(period).min()
    return hi, (hi + lo) / 2, lo

def historical_volatility(close, period=20):
    return np.log(close / close.shift(1)).rolling(period).std() * np.sqrt(252) * 100

def obv(df):
    return (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()

def cmf(df, period=20):
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"] + 1e-10)
    return (mfm * df["volume"]).rolling(period).sum() / df["volume"].rolling(period).sum()

def adl(df):
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"] + 1e-10)
    return (mfm * df["volume"]).cumsum()

def force_index(df, period=13):
    return (df["close"].diff() * df["volume"]).ewm(span=period, adjust=False).mean()


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c   = df["close"]
    for p in [9, 20, 50, 100, 200]:
        out[f"sma_{p}"] = sma(c, p)
        out[f"ema_{p}"] = ema(c, p)
    out["wma_20"]  = wma(c, 20)
    out["hma_20"]  = hull_ma(c, 20)
    out["vwap"]    = vwap(df)
    out["vwma_20"] = vwma(df, 20)
    out["macd"], out["macd_signal"], out["macd_hist"] = macd(c)
    out["rsi_14"]  = rsi(c, 14)
    out["rsi_7"]   = rsi(c, 7)
    out["stoch_k"], out["stoch_d"] = stochastic(df)
    out["bb_upper"], out["bb_mid"], out["bb_lower"], out["bb_bw"], out["bb_pctb"] = bollinger_bands(c)
    out["kc_upper"], out["kc_mid"], out["kc_lower"] = keltner_channel(df)
    out["dc_upper"], out["dc_mid"], out["dc_lower"] = donchian_channel(df)
    out["atr_14"]    = atr(df, 14)
    out["cci_20"]    = cci(df, 20)
    out["williams_r"]= williams_r(df)
    out["roc_12"]    = roc(c, 12)
    out["mfi_14"]    = mfi(df, 14)
    out["obv"]       = obv(df)
    out["cmf_20"]    = cmf(df, 20)
    out["adl"]       = adl(df)
    out["force_idx"] = force_index(df)
    out["hv_20"]     = historical_volatility(c, 20)
    return out


def indicator_signals(df: pd.DataFrame) -> dict:
    ind  = compute_all_indicators(df)
    last = ind.iloc[-1]
    prev = ind.iloc[-2] if len(ind) > 1 else last
    c    = last["close"]
    sigs = {}

    def s(buy, sell, name, val, fmt=".2f"):
        sigs[name] = {"signal": "BUY" if buy else "SELL" if sell else "NEUTRAL",
                      "value": val, "fmt": fmt}

    s(c > last["sma_20"],  c < last["sma_20"],  "SMA 20",  last["sma_20"])
    s(c > last["sma_50"],  c < last["sma_50"],  "SMA 50",  last["sma_50"])
    s(c > last["sma_200"], c < last["sma_200"], "SMA 200", last["sma_200"])
    s(c > last["ema_20"],  c < last["ema_20"],  "EMA 20",  last["ema_20"])
    s(c > last["ema_50"],  c < last["ema_50"],  "EMA 50",  last["ema_50"])
    s(c > last["ema_200"], c < last["ema_200"], "EMA 200", last["ema_200"])
    s(c > last["vwap"],    c < last["vwap"],    "VWAP",    last["vwap"])
    s(last["macd"] > last["macd_signal"] and last["macd_hist"] > 0,
      last["macd"] < last["macd_signal"] and last["macd_hist"] < 0,
      "MACD", last["macd_hist"])
    s(last["rsi_14"] < 40,  last["rsi_14"] > 60,  "RSI 14",     last["rsi_14"], ".1f")
    s(last["stoch_k"] < 25 and last["stoch_k"] > prev["stoch_k"],
      last["stoch_k"] > 75 and last["stoch_k"] < prev["stoch_k"],
      "Stoch %K", last["stoch_k"], ".1f")
    s(c <= last["bb_lower"], c >= last["bb_upper"], "Bollinger %B", last["bb_pctb"], ".3f")
    s(last["cci_20"] < -100, last["cci_20"] > 100,  "CCI 20",      last["cci_20"], ".1f")
    s(last["williams_r"] < -80, last["williams_r"] > -20, "Williams %R", last["williams_r"], ".1f")
    s(last["mfi_14"] < 25, last["mfi_14"] > 75,  "MFI 14",  last["mfi_14"], ".1f")
    s(last["roc_12"] > 0,  last["roc_12"] < 0,   "ROC 12",  last["roc_12"], ".2f")
    s(last["obv"] > prev["obv"], last["obv"] < prev["obv"], "OBV", last["obv"], ".0f")
    s(last["cmf_20"] > 0.05, last["cmf_20"] < -0.05, "CMF 20", last["cmf_20"], ".3f")
    sigs["ATR 14"] = {"signal": "NEUTRAL", "value": last["atr_14"], "fmt": ".2f"}
    return sigs


def summary_signal(signals: dict) -> dict:
    counts = {"BUY": 0, "SELL": 0, "NEUTRAL": 0}
    for v in signals.values():
        counts[v["signal"]] += 1
    total    = sum(counts.values())
    pct_buy  = counts["BUY"]  / total * 100
    pct_sell = counts["SELL"] / total * 100
    if   pct_buy  >= 60: overall = "COMPRAR"
    elif pct_buy  >= 45: overall = "COMPRAR (DÉBIL)"
    elif pct_sell >= 60: overall = "VENDER"
    elif pct_sell >= 45: overall = "VENDER (DÉBIL)"
    else:                overall = "NEUTRAL"
    return {"overall": overall,
            "score":   round((counts["BUY"] - counts["SELL"]) / total, 3),
            "buy": counts["BUY"], "sell": counts["SELL"], "neutral": counts["NEUTRAL"],
            "pct_buy": round(pct_buy, 1), "pct_sell": round(pct_sell, 1), "total": total}
