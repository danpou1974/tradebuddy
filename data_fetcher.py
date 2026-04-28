"""
data_fetcher.py
===============
Crypto  → KuCoin REST + OKX REST (sin ccxt, sin load_markets, ~1-2s/fetch)
         Binance bloquea IPs de Render/AWS. KuCoin y OKX no.
Forex / Gold / Acciones → yfinance (~15 min delay)
"""

import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Optional

CRYPTO_SYMBOLS = {
    # Pares activos v3.2 (backtest confirmado)
    "BTC/USDT","ETH/USDT","BNB/USDT","ADA/USDT",
    "DOT/USDT","ATOM/USDT","NEAR/USDT","ARB/USDT","PAXG/USDT",
    # Pares legacy (mantenidos para compatibilidad)
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
BINANCE_LIMIT = {"5m":300,"15m":300,"1h":300,"4h":200,"1d":300}

# Mapeos de timeframe para cada exchange
_KUCOIN_TF = {"5m":"5min","15m":"15min","1h":"1hour","4h":"4hour","1d":"1day"}
_OKX_TF    = {"5m":"5m",  "15m":"15m",  "1h":"1H",   "4h":"4H",  "1d":"1D"}


class CryptoFetcher:
    """
    Fetcher directo via REST API de KuCoin y OKX.
    - Sin ccxt, sin load_markets() (era la causa de los 20s de overhead)
    - Thread-safe: cada llamada crea su propia request HTTP
    - ~1-2s por fetch en condiciones normales
    - Binance bloquea IPs de Render — KuCoin y OKX no tienen restricciones
    """

    def __init__(self, symbol="ETH/USDT"):
        self.symbol      = symbol
        self._kucoin_sym = symbol.replace("/", "-")   # "ETH/USDT" → "ETH-USDT"
        self._okx_sym    = symbol.replace("/", "-")   # igual para OKX

    # ── KuCoin ────────────────────────────────────────────────────────────────

    def _fetch_kucoin(self, timeframe: str, limit: int):
        tf   = _KUCOIN_TF.get(timeframe, "1hour")
        cap  = min(limit, 1500)   # KuCoin max 1500 candles
        url  = (f"https://api.kucoin.com/api/v1/market/candles"
                f"?symbol={self._kucoin_sym}&type={tf}&pageSize={cap}")
        resp = requests.get(url, timeout=12)
        resp.raise_for_status()
        data = resp.json()
        if str(data.get("code")) != "200000":
            raise Exception(f"KuCoin API error: {data.get('msg', data)}")
        candles = data.get("data", [])
        # Formato: [ts_sec, open, close, high, low, volume, turnover] — NEWEST FIRST
        result = []
        for row in reversed(candles):
            ts_ms = int(row[0]) * 1000
            o  = float(row[1])
            c  = float(row[2])
            h  = float(row[3])
            l  = float(row[4])
            v  = float(row[5])
            result.append([ts_ms, o, h, l, c, v])
        return result[-limit:] if len(result) > limit else result

    # ── OKX ───────────────────────────────────────────────────────────────────

    def _fetch_okx(self, timeframe: str, limit: int):
        tf  = _OKX_TF.get(timeframe, "1H")
        cap = min(limit, 300)   # OKX max 300 por request
        url = (f"https://www.okx.com/api/v5/market/candles"
               f"?instId={self._okx_sym}&bar={tf}&limit={cap}")
        resp = requests.get(url, timeout=12)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "0":
            raise Exception(f"OKX API error: {data.get('msg', data)}")
        candles = data.get("data", [])
        # Formato: [ts_ms, open, high, low, close, vol, ...] — NEWEST FIRST
        result = []
        for row in reversed(candles):
            ts_ms = int(row[0])
            o = float(row[1])
            h = float(row[2])
            l = float(row[3])
            c = float(row[4])
            v = float(row[5])
            result.append([ts_ms, o, h, l, c, v])
        return result

    # ── Fetch raw con fallback ─────────────────────────────────────────────────

    def _fetch_raw(self, timeframe="1h", limit=300):
        errors = []

        try:
            raw = self._fetch_kucoin(timeframe, limit)
            if raw and len(raw) >= 60:
                print(f"  [KUCOIN] {self.symbol} {timeframe} OK ({len(raw)} velas)")
                return raw
            errors.append(f"KuCoin: solo {len(raw)} velas")
        except Exception as e:
            errors.append(f"KuCoin: {e}")

        try:
            raw = self._fetch_okx(timeframe, min(limit, 300))
            if raw and len(raw) >= 60:
                print(f"  [OKX] {self.symbol} {timeframe} OK ({len(raw)} velas)")
                return raw
            errors.append(f"OKX: solo {len(raw)} velas")
        except Exception as e:
            errors.append(f"OKX: {e}")

        raise Exception(f"Todos los exchanges fallaron para {self.symbol} {timeframe}: {'; '.join(errors)}")

    # ── Ticker ────────────────────────────────────────────────────────────────

    def fetch_ticker(self):
        # KuCoin
        try:
            url  = f"https://api.kucoin.com/api/v1/market/stats?symbol={self._kucoin_sym}"
            resp = requests.get(url, timeout=8)
            data = resp.json()
            if str(data.get("code")) == "200000":
                d     = data["data"]
                price = float(d.get("last", 0) or 0)
                open_ = float(d.get("open", price) or price)
                chg   = ((price - open_) / open_ * 100) if open_ else 0
                return {
                    "price":      price,
                    "change_pct": round(chg, 4),
                    "change_abs": round(price - open_, 6),
                    "volume_24h": float(d.get("volValue", 0) or 0),
                    "high_24h":   float(d.get("high", 0) or 0),
                    "low_24h":    float(d.get("low", 0) or 0),
                    "source": "KuCoin", "realtime": True,
                }
        except Exception:
            pass

        # OKX fallback
        try:
            url  = f"https://www.okx.com/api/v5/market/ticker?instId={self._okx_sym}"
            resp = requests.get(url, timeout=8)
            data = resp.json()
            if data.get("code") == "0" and data.get("data"):
                d     = data["data"][0]
                price = float(d.get("last", 0) or 0)
                open_ = float(d.get("open24h", price) or price)
                chg   = ((price - open_) / open_ * 100) if open_ else 0
                return {
                    "price":      price,
                    "change_pct": round(chg, 4),
                    "change_abs": round(price - open_, 6),
                    "volume_24h": float(d.get("volCcy24h", 0) or 0),
                    "high_24h":   float(d.get("high24h", 0) or 0),
                    "low_24h":    float(d.get("low24h", 0) or 0),
                    "source": "OKX", "realtime": True,
                }
        except Exception:
            pass

        return {"price": None, "change_pct": 0, "source": "unavailable", "realtime": False}

    # ── OHLCV DataFrame ───────────────────────────────────────────────────────

    def fetch_ohlcv(self, timeframe="1h", limit=None):
        limit = limit or BINANCE_LIMIT.get(timeframe, 300)
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
                time.sleep(0.2)
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
