"""
backtest_v3.py — walk-forward rapido (3 meses, timeout por par)
"""
import time, requests, numpy as np, pandas as pd, scorer_v3
from typing import List, Dict

MONTHS_BACK  = 3
COOLDOWN_H   = 4
MIN_CANDLES  = 100
LEVERAGE_CAP = 10
MAX_BARS     = 144  # max horas hasta resolver trade (6 dias — grid search óptimo)

PAIRS_ALL = [
    # Mean Reversion
    "BTC/USDT", "ETH/USDT",
    # RSI Pullback (nuevos)
    "SOL/USDT", "BNB/USDT", "LINK/USDT", "AVAX/USDT", "MATIC/USDT",
    # Trend Following
    "ADA/USDT", "DOT/USDT", "ATOM/USDT", "NEAR/USDT", "ARB/USDT", "PAXG/USDT",
]

# ── Fetch 1 chunk KuCoin (max 1500 velas) con timeout ────────────────────────
def fetch_pair(symbol: str) -> pd.DataFrame:
    sym = symbol.replace("/", "-")
    end = int(time.time())
    start = end - MONTHS_BACK * 30 * 24 * 3600
    rows = []
    current_end = end
    for _ in range(4):   # max 4 chunks = 6000 velas ~ 8 meses
        url = (f"https://api.kucoin.com/api/v1/market/candles"
               f"?symbol={sym}&type=1hour&startAt={max(start, current_end-1500*3600)}&endAt={current_end}")
        try:
            r = requests.get(url, timeout=12)
            data = r.json()
            if str(data.get("code")) != "200000": break
            candles = data.get("data", [])
            if not candles: break
            for c in candles:
                ts = int(c[0])
                if ts < start: continue
                rows.append({"ts": pd.Timestamp(ts, unit="s"),
                              "open": float(c[1]), "close": float(c[2]),
                              "high": float(c[3]), "low":  float(c[4]),
                              "volume": float(c[5])})
            oldest = min(int(c[0]) for c in candles)
            if oldest <= start: break
            current_end = oldest - 1
            time.sleep(0.25)
        except Exception as e:
            print(f"    ERROR fetch {symbol}: {e}"); break
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates("ts").sort_values("ts").set_index("ts")
    return df.astype(float)

# ── Resolver trade ────────────────────────────────────────────────────────────
def resolve(df, idx, entry, sl, tp, direction):
    is_bull = direction == "long"
    for i in range(idx+1, min(idx+MAX_BARS+1, len(df))):
        hi, lo = df.iloc[i]["high"], df.iloc[i]["low"]
        tp_hit = hi >= tp if is_bull else lo <= tp
        sl_hit = lo <= sl if is_bull else hi >= sl
        if sl_hit and tp_hit: outcome = "sl"
        elif tp_hit:           outcome = "tp"
        elif sl_hit:           outcome = "sl"
        else:                  continue
        exit_p = tp if outcome == "tp" else sl
        raw = (exit_p-entry)/entry*100 if is_bull else (entry-exit_p)/entry*100
        return outcome, round(raw, 4)
    # expirado: usar precio de cierre
    exit_p = df.iloc[min(idx+MAX_BARS, len(df)-1)]["close"]
    raw = (exit_p-entry)/entry*100 if is_bull else (entry-exit_p)/entry*100
    return "expired", round(raw, 4)

# ── Walk-forward por par ──────────────────────────────────────────────────────
def backtest_pair(symbol, df):
    trades = []
    last_bar = -COOLDOWN_H * 2
    for i in range(MIN_CANDLES, len(df)-5):
        if i - last_bar < COOLDOWN_H: continue
        w1h = df.iloc[:i+1]
        df4h = w1h.resample("4h").agg({"open":"first","high":"max",
                                        "low":"min","close":"last","volume":"sum"}).dropna()
        try:
            sig = scorer_v3.score_signal(symbol, w1h, df4h)
        except: continue
        if sig is None: continue

        entry = df.iloc[i]["close"]
        sl, tp1 = sig.get("sl",0), sig.get("tp1",0)
        lev = min(sig.get("leverage",1), LEVERAGE_CAP)
        direction = sig.get("direction","long")

        if sl<=0 or tp1<=0: continue
        if direction=="long"  and (sl>=entry or tp1<=entry): continue
        if direction=="short" and (sl<=entry or tp1>=entry): continue

        outcome, raw = resolve(df, i, entry, sl, tp1, direction)
        sl_pct = abs(entry-sl)/entry*100
        tp_pct = abs(tp1-entry)/entry*100

        trades.append({
            "symbol": symbol,
            "strategy": scorer_v3.PAIR_STRATEGIES.get(symbol,"trend"),
            "direction": direction,
            "date": df.index[i],
            "score": sig.get("score",0),
            "sl_pct": round(sl_pct,3),
            "tp_pct": round(tp_pct,3),
            "rr": round(tp_pct/sl_pct,2) if sl_pct else 0,
            "leverage": lev,
            "outcome": outcome,
            "raw_pct": raw,
            "pnl_lev": round(raw*lev, 3),
        })
        last_bar = i
    return trades

# ── Métricas ──────────────────────────────────────────────────────────────────
def metrics(trades, label=""):
    if not trades:
        return {"label":label,"n":0,"wr":0,"pf":0,"avg_rr":0,
                "pnl_tp_sl":0,"per_month":0,"wins":0,"losses":0,"expired":0}
    wins    = [t for t in trades if t["outcome"]=="tp"]
    losses  = [t for t in trades if t["outcome"]=="sl"]
    expired = [t for t in trades if t["outcome"]=="expired"]
    n = len(trades)
    wr = len(wins)/n*100
    gw = sum(t["raw_pct"] for t in wins) if wins else 0
    gl = sum(abs(t["raw_pct"]) for t in losses) if losses else 0
    pf = gw/gl if gl>0 else (99 if gw>0 else 0)
    avg_rr = sum(t["rr"] for t in trades)/n
    # PnL solo sobre TP/SL (sin expired — más honesto)
    resolved = wins + losses
    pnl_ts = sum(t["pnl_lev"] for t in resolved) if resolved else 0
    days = (trades[-1]["date"]-trades[0]["date"]).days or 1
    per_month = n/(days/30)
    return {"label":label,"n":n,"wins":len(wins),"losses":len(losses),
            "expired":len(expired),"wr":round(wr,1),"pf":round(pf,2),
            "avg_rr":round(avg_rr,2),"pnl_tp_sl":round(pnl_ts,1),
            "per_month":round(per_month,1)}

# ── Main ─────────────────────────────────────────────────────────────────────
def run():
    print("="*65)
    print(f"BACKTEST v3 | {MONTHS_BACK} meses | trend>={scorer_v3.SIGNAL_THRESHOLD} | MR>={scorer_v3.BTC_MR_THRESHOLD}")
    print("="*65)
    print(f"{'Par':<14} {'N':>4} {'W':>4} {'L':>4} {'Exp':>4} "
          f"{'WR':>6} {'PF':>6} {'RR':>5} {'PnL(lev)':>9} {'Sig/m':>6}")
    print("-"*65)

    all_trades, results = [], []

    for symbol in PAIRS_ALL:
        print(f"  [{symbol}] fetch...", end="", flush=True)
        df = fetch_pair(symbol)
        if df.empty or len(df) < MIN_CANDLES+20:
            print(f" sin datos"); continue
        print(f" {len(df)}v | scoring...", end="", flush=True)
        trades = backtest_pair(symbol, df)
        m = metrics(trades, symbol)
        results.append(m)
        all_trades.extend(trades)
        print(f" n={m['n']:>3} WR={m['wr']:>5.1f}% PF={m['pf']:>5.2f} "
              f"RR={m['avg_rr']:.2f} PnL={m['pnl_tp_sl']:>+7.1f}% {m['per_month']:.1f}/m")

    # ── Ranking ───────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("RANKING POR PROFIT FACTOR")
    print("="*65)
    print(f"{'Par':<14} {'N':>4} {'WR':>6} {'PF':>6} {'RR':>5} {'PnL':>8}  DECISION")
    print("-"*65)

    ranked = sorted([r for r in results if r["n"]>=3], key=lambda x: x["pf"], reverse=True)
    keep, drop = [], []

    for r in ranked:
        if r["pf"] >= 1.2 and r["wr"] >= 38:
            keep.append(r["label"]); tag = "KEEP [OK]"
        elif r["pf"] >= 1.0 and r["n"] < 6:
            keep.append(r["label"]); tag = "POCOS DATOS"
        else:
            drop.append(r["label"]); tag = "ELIMINAR [X]"
        print(f"{r['label']:<14} {r['n']:>4} {r['wr']:>5.1f}% {r['pf']:>6.2f} "
              f"{r['avg_rr']:>5.2f} {r['pnl_tp_sl']:>+7.1f}%  {tag}")

    # ── Por estrategia ────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("POR ESTRATEGIA")
    print("="*65)
    for strat in ["trend","mean_reversion"]:
        t = [x for x in all_trades if x["strategy"]==strat]
        m = metrics(t, strat)
        print(f"{strat:<20} n={m['n']:>3} WR={m['wr']}% PF={m['pf']} "
              f"RR={m['avg_rr']} PnL={m['pnl_tp_sl']:+.1f}% {m['per_month']:.1f}/mes")

    # ── Conclusión ────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("CONCLUSION FINAL")
    print("="*65)
    print(f"MANTENER: {', '.join(keep)}")
    print(f"ELIMINAR: {', '.join(drop)}")
    no_data = [r["label"] for r in results if r["n"]<3]
    if no_data: print(f"SIN SUFICIENTES DATOS: {', '.join(no_data)}")
    gm = metrics(all_trades,"GLOBAL")
    print(f"\nGLOBAL: {gm['n']} trades | WR {gm['wr']}% | PF {gm['pf']} | {gm['per_month']:.1f}/mes")
    print("Fin.")

if __name__=="__main__":
    run()
