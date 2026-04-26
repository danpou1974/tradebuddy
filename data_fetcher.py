"""
data_fetcher.py
===============
Crypto  → KuCoin / OKX / Bybit via ccxt (sin restricciones en cloud)
         (Binance bloquea IPs de Render/AWS/cloud providers)
Forex / Gold / Acciones → yfinance (~15 min delay)
"""

import time
import numpy as np
import pandas as pd
import ccxt
import yfinance as yf
from typing import Optional

CRYPTO_SYMBOLS = {
    # Pares activos v3.2 (backtest confirmado)
    "BTC/USDT","ETH/USDT","BNB/USDT","ADA/USDT",
    "DOT/USDT","ATOM/USDT","NEAR/USDT","ARB/USDT","PAXG/USDT",
    # Pares legacy (mantenidos para compatibilidad con otras partes del app)
    "SOL/USDT","XRP/USDT","AVAX/USDT","DOGE/USDT","LINK/USDT","MATIC/USDT",
}

YAHOO_TICKERS = {
    "EUR/USD":"EURUSD=X","GBP/USD":"GBPUSD=X","USD/JPY":"JPY=X",
    "AUD/USD":"AUDUSD=X","USD/CAD":"CAD=X","USD/CHF":"CHF=X",
    "XAU/USD":"GC=F","XAG/USD":"SI=F","WTI/USD":"CL=F","NG/USD":"NG=F",
    "AAPL":"AAPL","TSLA":"TSLA","NVDA":"NVDA","MSFT":"MSFT",
    "AMZN":"AMZN","GOOGL":"GOOGL","SPY":"SPY","QQQ":"QQQ",
}

YAHOO_TF = {
    "5m": {"period":"5d",  "interval":"5m"},
    "15m":{"period":"5d",  "interval":"15m"},
    "1h": {"period":"1mo", "interval":"1h"},
    "4h": {"period":"3mo", "interval":"1h"},   # Yahoo no tiene 4h, usamos 1h
    "1d": {"period":"2y",  "interval":"1d"},
}

BINANCE_TF    = {"5m":"5m","15m":"15m","1h":"1h","4h":"4h","1d":"1d"}
BINANCE_LIMIT = {"5m":1000,"15m":1000,"1h":1000,"4h":500,"1d":500}


class CryptoFetcher:
    """
    Fetcher multi-exchange para OHLCV de cripto.
    Prueba en orden: KuCoin → OKX → Bybit
    Binance bloquea IPs de Render y otros cloud providers.
    """
    _EXCHANGES = [
        lambda: ccxt.kucoin({"enableRateLimit": True, "timeout": 12000}),
        lambda: ccxt.okx({"enableRateLimit": True, "timeout": 12000}),
        lambda: ccxt.bybit({"enableRateLimit": True, "timeout": 12000, "options": {"defaultType": "spot"}}),
    ]

    def __init__(self, symbol="ETH/USDT"):
        self.symbol = symbol

    def _fetch_raw(self, timeframe="1h", limit=500):
        interval  = BINANCE_TF.get(timeframe, "1h")
        last_err  = None
        for factory in self._EXCHANGES:
            try:
                ex  = factory()
                raw = ex.fetch_ohlcv(self.symbol, timeframe=interval, limit=limit)
                if raw and len(raw) >= 60:
                    print(f"  [{ex.id.upper()}] {self.symbol} {timeframe} OK ({len(raw)} velas)")
                    return raw
            except Exception as e:
                last_err = e
        raise Exception(f"Todos los exchanges fallaron para {self.symbol}: {last_err}")

    def fetch_ticker(self):
        for factory in self._EXCHANGES:
            try:
                ex = factory()
                t  = ex.fetch_ticker(self.symbol)
                return {
                    "price":      t["last"],
                    "change_pct": t.get("percentage", 0) or 0,
                    "change_abs": t.get("change", 0) or 0,
                    "volume_24h": t.get("quoteVolume", 0) or 0,
                    "high_24h":   t.get("high", 0) or 0,
                    "low_24h":    t.get("low", 0) or 0,
                    "source":     ex.id,
                    "realtime":   True,
                }
            except Exception:
                continue
        return {"price": None, "change_pct": 0, "source": "unavailable", "realtime": False}

    def fetch_ohlcv(self, timeframe="1h", limit=None):
        limit = limit or BINANCE_LIMIT.get(timeframe, 500)
        raw   = self._fetch_raw(timeframe, limit)
        df    = pd.DataFrame(raw, columns=["ts","open","high","low","close","volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df.set_index("ts", inplace=True)
        return df.astype(float)

    def fetch_all_timeframes(self, timeframes=None):
        tfs  = timeframes or list(BINANCE_TF.keys())
        data = {}
        for tf in tfs:
            try:
                data[tf] = self.fetch_ohlcv(tf)
                time.sleep(0.4)
            except Exception as e:
                print(f"  [ERROR] {self.symbol} {tf}: {e}")
        return data


# Alias para compatibilidad con imports existentes
BinanceFetcher = CryptoFetcher


class YahooFetcher:
    def __init__(self, symbol):
        self.symbol = symbol
        self.ticker = YAHOO_TICKERS.get(symbol, symbol)
        self._yf    = yf.Ticker(self.ticker)

    def fetch_ticker(self):
        try:
            info  = self._yf.fast_info
            price = info.last_price
            prev  = getattr(info, "previous_close", None) or price
            chg   = ((price - prev) / prev * 100) if prev else 0
            return {"price": price, "change_pct": round(chg, 4),
                    "change_abs": round(price - prev, 6),
                    "volume_24h": getattr(info, "three_month_average_volume", 0),
                    "source": "Yahoo Finance", "realtime": False}
        except Exception as e:
            print(f"  [YAHOO TICKER] {self.symbol}: {e}")
            return {"price": None, "change_pct": 0, "source": "Yahoo Finance", "realtime": False}

    def fetch_ohlcv(self, timeframe="1h"):
        cfg = YAHOO_TF.get(timeframe, {"period":"1mo","interval":"1h"})
        try:
            df = self._yf.history(period=cfg["period"], interval=cfg["interval"])
            if df.empty:
                return pd.DataFrame()
            df = df[["Open","High","Low","Close","Volume"]].copy()
            df.columns = ["open","high","low","close","volume"]
            df.index.name = "ts"
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df.dropna()
        except Exception as e:
            print(f"  [YAHOO OHLCV] {self.symbol} {timeframe}: {e}")
            return pd.DataFrame()

    def fetch_all_timeframes(self, timeframes=None):
        tfs  = timeframes or list(YAHOO_TF.keys())
        data = {}
        for tf in tfs:
            print(f"  [YAHOO] {self.symbol} {tf}...")
            df = self.fetch_ohlcv(tf)
            if not df.empty:
                data[tf] = df
            time.sleep(0.2)
        return data


def get_fetcher(symbol):
    return CryptoFetcher(symbol) if symbol in CRYPTO_SYMBOLS else YahooFetcher(symbol)

def fetch_ticker_universal(symbol):
    return get_fetcher(symbol).fetch_ticker()

def fetch_all_timeframes_universal(symbol, timeframes=None):
    return get_fetcher(symbol).fetch_all_timeframes(timeframes)


def make_synthetic_ohlcv(n=1500, seed=42, freq="1h", start_price=2000.0):
    rng = np.random.default_rng(seed)
    p   = n // 3
    r1  = rng.normal( 0.0012, 0.010, p)
    r2  = rng.normal(-0.0010, 0.022, p)
    r3  = rng.normal( 0.0001, 0.005, n - 2*p)
    rets  = np.concatenate([r1, r2, r3])
    close = start_price * np.exp(np.cumsum(rets))
    noise = np.abs(rng.normal(0, 0.004, n))
    high  = close * (1 + noise)
    low   = close * (1 - noise)
    open_ = np.concatenate([[close[0]], close[:-1]])
    vol   = np.abs(rng.normal(800_000, 150_000, n))
    idx   = pd.date_range("2023-01-01", periods=n, freq=freq)
    return pd.DataFrame({"open":open_,"high":high,"low":low,"close":close,"volume":vol}, index=idx)

def make_synthetic_all_timeframes(start_price=2000.0):
    configs = {
        "5m":  (1500, "5min"),
        "15m": (1500, "15min"),
        "1h":  (500,  "1h"),
        "4h":  (250,  "4h"),
        "1d":  (500,  "D"),
    }
    return {tf: make_synthetic_ohlcv(n=c[0], freq=c[1], start_price=start_price, seed=i)
            for i,(tf,c) in enumerate(configs.items())}
