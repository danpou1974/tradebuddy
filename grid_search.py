"""
grid_search.py — Busqueda de la mejor configuracion SL/TP para crypto
======================================================================
Estrategia: genera senales con scorer_v3 (una vez por par),
luego aplica TODAS las combinaciones de parametros sobre esas mismas senales.

Grid de parametros:
  SL%      : 2, 3, 4, 5, 6, 7, 8%
  RR (TP/SL): 1.5, 2.0, 2.5, 3.0
  MAX_BARS : 72h, 96h, 120h, 144h

Pares activos: BTC/USDT (MR), ETH/USDT (MR), LINK/USDT (Pullback), BNB/USDT (Pullback)
"""
import time, requests, numpy as np, pandas as pd, scorer_v3
from itertools import product

# ── Config ────────────────────────────────────────────────────────────────────
MONTHS_BACK   = 3
MIN_CANDLES   = 100
COOLDOWN_H    = 4
PAIRS         = ["BTC/USDT", "ETH/USDT", "LINK/USDT", "BNB/USDT"]

SL_PCTS       = [2, 3, 4, 5, 6, 7, 8]
RR_VALS       = [1.5, 2.0, 2.5, 3.0]
MAX_BARS_LIST = [72, 96, 120, 144]

MIN_N_TRADES  = 15   # minimo de trades para ser estadisticamente valido
MIN_WR        = 38.0 # minimo WR para considerar la config

# ── Fetch ──────────────────────────────────────────────────────────────────────
def fetch_pair(symbol: str) -> pd.DataFrame:
    sym = symbol.replace("/", "-")
    end = int(time.time())
    start = end - MONTHS_BACK * 30 * 24 * 3600
    rows = []
    current_end = end
    for _ in range(4):
        url = (f"https://api.kucoin.com/api/v1/market/candles"
               f"?symbol={sym}&type=1hour"
               f"&startAt={max(start, current_end-1500*3600)}&endAt={current_end}")
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

# ── Paso 1: Generar senales (una vez por par) ─────────────────────────────────
def extract_signals(symbol: str, df: pd.DataFrame) -> list:
    """
    Corre el scorer sobre el historico y devuelve una lista de senales:
    [{idx, entry, direction, df_snapshot}, ...]
    Sin aplicar SL/TP — eso lo hacemos en el grid search.
    """
    signals = []
    last_bar = -COOLDOWN_H * 2

    for i in range(MIN_CANDLES, len(df) - 5):
        if i - last_bar < COOLDOWN_H:
            continue
        w1h = df.iloc[:i+1]
        df4h = w1h.resample("4h").agg({
            "open": "first", "high": "max",
            "low": "min", "close": "last", "volume": "sum"
        }).dropna()
        try:
            sig = scorer_v3.score_signal(symbol, w1h, df4h)
        except:
            continue
        if sig is None:
            continue

        entry     = df.iloc[i]["close"]
        direction = sig.get("direction", "long")
        signals.append({
            "idx":       i,
            "entry":     entry,
            "direction": direction,
            "strategy":  sig.get("strategy", "trend"),
            "score":     sig.get("score", 0),
        })
        last_bar = i

    return signals

# ── Paso 2: Resolver trade con SL/TP fijos en % ───────────────────────────────
def resolve_fixed(df: pd.DataFrame, idx: int, entry: float,
                  sl_pct: float, rr: float, direction: str, max_bars: int):
    is_bull = direction == "long"
    sl_dist = entry * sl_pct / 100
    tp_dist = sl_dist * rr
    sl = entry - sl_dist if is_bull else entry + sl_dist
    tp = entry + tp_dist if is_bull else entry - tp_dist

    for i in range(idx + 1, min(idx + max_bars + 1, len(df))):
        hi, lo = df.iloc[i]["high"], df.iloc[i]["low"]
        tp_hit = hi >= tp if is_bull else lo <= tp
        sl_hit = lo <= sl if is_bull else hi >= sl
        if sl_hit and tp_hit:  outcome = "sl"
        elif tp_hit:            outcome = "tp"
        elif sl_hit:            outcome = "sl"
        else:                   continue
        raw = sl_pct * rr if outcome == "tp" else -sl_pct
        return outcome, raw

    # Expirado
    exit_p = df.iloc[min(idx + max_bars, len(df) - 1)]["close"]
    raw = (exit_p - entry) / entry * 100 * (1 if is_bull else -1)
    return "expired", round(raw, 3)

# ── Metricas ──────────────────────────────────────────────────────────────────
def metrics(trades: list) -> dict:
    if not trades:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "exp_pct": 0.0}
    wins    = [t for t in trades if t["outcome"] == "tp"]
    losses  = [t for t in trades if t["outcome"] == "sl"]
    expired = [t for t in trades if t["outcome"] == "expired"]
    n  = len(trades)
    nr = len(wins) + len(losses)   # resueltos (excluye expired)
    wr = len(wins) / nr * 100 if nr > 0 else 0
    gw = sum(abs(t["raw"]) for t in wins)
    gl = sum(abs(t["raw"]) for t in losses)
    pf = gw / gl if gl > 0 else (99.0 if gw > 0 else 0.0)
    return {
        "n":       n,
        "nr":      nr,
        "wins":    len(wins),
        "losses":  len(losses),
        "expired": len(expired),
        "wr":      round(wr, 1),
        "pf":      round(min(pf, 99.0), 2),
        "exp_pct": round(len(expired) / n * 100, 1),
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    print("=" * 72)
    print("GRID SEARCH — Mejor configuracion SL/TP/MaxBars para crypto")
    print(f"Pares: {', '.join(PAIRS)}")
    print(f"Grid: {len(SL_PCTS)} SL x {len(RR_VALS)} RR x {len(MAX_BARS_LIST)} MaxBars"
          f" = {len(SL_PCTS)*len(RR_VALS)*len(MAX_BARS_LIST)} combinaciones")
    print("=" * 72)

    # ── Descargar datos ───────────────────────────────────────────────────────
    print("\n[1/3] Descargando datos historicos...")
    dfs = {}
    for symbol in PAIRS:
        print(f"  {symbol}...", end="", flush=True)
        df = fetch_pair(symbol)
        if not df.empty and len(df) >= MIN_CANDLES + 20:
            dfs[symbol] = df
            print(f" {len(df)} velas OK")
        else:
            print(" sin datos, saltando")

    # ── Generar senales (una vez) ─────────────────────────────────────────────
    print("\n[2/3] Generando senales (una vez por par)...")
    all_signals = {}
    for symbol, df in dfs.items():
        print(f"  {symbol}...", end="", flush=True)
        sigs = extract_signals(symbol, df)
        all_signals[symbol] = sigs
        print(f" {len(sigs)} senales")

    # ── Grid search ───────────────────────────────────────────────────────────
    print("\n[3/3] Aplicando grid de parametros...\n")
    combos = list(product(SL_PCTS, RR_VALS, MAX_BARS_LIST))
    results = []

    for sl_pct, rr, max_bars in combos:
        tp_pct = round(sl_pct * rr, 1)
        trades_all = []

        for symbol, sigs in all_signals.items():
            df = dfs[symbol]
            for s in sigs:
                outcome, raw = resolve_fixed(
                    df, s["idx"], s["entry"], sl_pct, rr, s["direction"], max_bars
                )
                trades_all.append({
                    "symbol":    symbol,
                    "strategy":  s["strategy"],
                    "outcome":   outcome,
                    "raw":       raw,
                })

        m = metrics(trades_all)
        results.append({
            "sl_pct":   sl_pct,
            "tp_pct":   tp_pct,
            "rr":       rr,
            "max_bars": max_bars,
            **m,
        })

    # ── Resultados ────────────────────────────────────────────────────────────
    # Filtrar por minimos y ordenar por PF
    valid = [r for r in results if r["wr"] >= MIN_WR and r["nr"] >= MIN_N_TRADES]
    valid.sort(key=lambda x: x["pf"], reverse=True)

    print("=" * 72)
    print(f"TOP 15 CONFIGS  (WR>={MIN_WR}%, trades resueltos>={MIN_N_TRADES})")
    print("Ordenado por Profit Factor")
    print("=" * 72)
    print(f"{'SL%':>5} {'TP%':>6} {'RR':>5} {'MaxH':>5} {'N':>5} {'WR':>7} "
          f"{'PF':>6} {'Exp%':>6}")
    print("-" * 72)
    for r in valid[:15]:
        print(f"{r['sl_pct']:>5}% {r['tp_pct']:>5}% {r['rr']:>5} "
              f"{r['max_bars']:>5}h {r['n']:>5} {r['wr']:>6.1f}% "
              f"{r['pf']:>6.2f} {r['exp_pct']:>5.1f}%")

    # Top con WR >= 50%
    best_50 = [r for r in results if r["wr"] >= 50 and r["nr"] >= MIN_N_TRADES]
    best_50.sort(key=lambda x: x["pf"], reverse=True)
    if best_50:
        print()
        print("=" * 72)
        print("TOP 10 CONFIGS  (WR>=50%)")
        print("=" * 72)
        print(f"{'SL%':>5} {'TP%':>6} {'RR':>5} {'MaxH':>5} {'N':>5} {'WR':>7} "
              f"{'PF':>6} {'Exp%':>6}")
        print("-" * 72)
        for r in best_50[:10]:
            print(f"{r['sl_pct']:>5}% {r['tp_pct']:>5}% {r['rr']:>5} "
                  f"{r['max_bars']:>5}h {r['n']:>5} {r['wr']:>6.1f}% "
                  f"{r['pf']:>6.2f} {r['exp_pct']:>5.1f}%")

    # CONCLUSION
    print()
    print("=" * 72)
    print("CONCLUSION")
    print("=" * 72)
    if valid:
        best = valid[0]
        print(f"MEJOR PF   : SL={best['sl_pct']}%  TP={best['tp_pct']}%  "
              f"RR={best['rr']}  MaxBars={best['max_bars']}h")
        print(f"             WR={best['wr']}%  PF={best['pf']}  "
              f"N={best['n']} trades  Expirados={best['exp_pct']}%")
    if best_50:
        b = best_50[0]
        print(f"MEJOR WR50+: SL={b['sl_pct']}%  TP={b['tp_pct']}%  "
              f"RR={b['rr']}  MaxBars={b['max_bars']}h")
        print(f"             WR={b['wr']}%  PF={b['pf']}  "
              f"N={b['n']} trades  Expirados={b['exp_pct']}%")
    print()
    print("Fin.")

if __name__ == "__main__":
    run()
