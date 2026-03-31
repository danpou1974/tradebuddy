"""
ai_signal.py
============
Motor de señales de trading con IA.
Combina HMM regime (7 regímenes) + indicadores técnicos + contexto de mercado.
"""

import numpy as np
import pandas as pd
from indicators import compute_all_indicators, indicator_signals, summary_signal


def find_support_resistance(df, lookback=50, n=3):
    r = df.tail(lookback)
    res = sorted(r["high"].nlargest(n * 2).values[:n], reverse=True)
    sup = sorted(r["low"].nsmallest(n * 2).values[:n])
    return sup, res


def market_context(df, ind):
    last = ind.iloc[-1]
    prev = ind.iloc[-2] if len(ind) > 1 else last

    price_trend = ("alcista"  if last["ema_20"] > last["ema_50"] > last["ema_200"] else
                   "bajista"  if last["ema_20"] < last["ema_50"] < last["ema_200"] else "mixta")

    rv = last["rsi_14"]
    rsi_state = ("sobrecomprado" if rv > 70 else "sobrevendido" if rv < 30 else
                 "alcista" if rv > 55 else "bajista" if rv < 45 else "neutral")

    atr_pct = last["atr_14"] / last["close"] * 100
    vol_ctx = "alta" if atr_pct > 3 else "media" if atr_pct > 1.5 else "baja"

    vol_avg  = df["volume"].tail(20).mean()
    vol_last = df["volume"].iloc[-1]
    vol_rel  = vol_last / (vol_avg + 1e-10)
    vol_desc = "alto" if vol_rel > 1.5 else "bajo" if vol_rel < 0.7 else "normal"

    macd_mom = ("positivo"    if last["macd_hist"] > 0 and last["macd_hist"] > prev["macd_hist"] else
                "negativo"    if last["macd_hist"] < 0 and last["macd_hist"] < prev["macd_hist"] else
                "convergiendo")

    bb_pos = ("cerca_upper" if last["bb_pctb"] > 0.85 else
              "cerca_lower"  if last["bb_pctb"] < 0.15 else
              "medio_upper"  if last["bb_pctb"] > 0.5  else "medio_lower")

    return {"price_trend": price_trend, "rsi_state": rsi_state, "rsi_value": round(rv, 1),
            "volatility": vol_ctx, "atr_pct": round(atr_pct, 2),
            "volume": vol_desc, "vol_ratio": round(vol_rel, 2),
            "macd_momentum": macd_mom, "bb_position": bb_pos, "bb_pctb": round(last["bb_pctb"], 3)}


# Scores de régimen: positivo = bullish, negativo = bearish
REGIME_SCORES = {
    "Alcista Fuerte":  +1.0,
    "Alcista":         +0.7,
    "Acumulación":     +0.3,
    "Lateral":          0.0,
    "Distribución":    -0.3,
    "Bajista":         -0.7,
    "Bajista Fuerte":  -1.0,
    "Desconocido":      0.0,
}


class AITradingSignal:

    def analyze(self, df: pd.DataFrame, regime_info: dict, symbol: str = "ETH/USDT") -> dict:
        if len(df) < 30:
            return {"action": "SIN DATOS", "action_color": "#9E9E9E",
                    "confidence": 0, "reasons": ["Datos insuficientes"], "warnings": [], "signals": {}}

        ind     = compute_all_indicators(df)
        signals = indicator_signals(df)
        summary = summary_signal(signals)
        ctx     = market_context(df, ind)
        supports, resistances = find_support_resistance(df)

        last  = ind.iloc[-1]
        price = float(last["close"])
        atr_v = float(last["atr_14"])

        regime      = regime_info.get("regime", "Desconocido")
        regime_conf = regime_info.get("confidence", 0.5)
        regime_score = REGIME_SCORES.get(regime, 0.0)

        ind_score = summary["score"]

        ctx_score = 0.0
        if ctx["price_trend"] == "alcista":     ctx_score += 0.5
        if ctx["price_trend"] == "bajista":     ctx_score -= 0.5
        if ctx["rsi_state"] == "sobrevendido":  ctx_score += 0.4
        if ctx["rsi_state"] == "sobrecomprado": ctx_score -= 0.4
        if ctx["macd_momentum"] == "positivo":  ctx_score += 0.3
        if ctx["macd_momentum"] == "negativo":  ctx_score -= 0.3
        if ctx["volume"] == "alto":             ctx_score += 0.2 * np.sign(ind_score)
        ctx_score = max(-1, min(1, ctx_score))

        final_score = (0.40 * regime_score * regime_conf +
                       0.40 * ind_score +
                       0.20 * ctx_score)

        if   final_score >= 0.55:  action, action_en, ac, strength = "COMPRAR",        "LONG",  "#00C853", "FUERTE"
        elif final_score >= 0.35:  action, action_en, ac, strength = "COMPRAR",        "LONG",  "#00C853", "MODERADA"
        elif final_score <= -0.55: action, action_en, ac, strength = "VENDER / SHORT", "SHORT", "#D50000", "FUERTE"
        elif final_score <= -0.35: action, action_en, ac, strength = "VENDER / SHORT", "SHORT", "#D50000", "MODERADA"
        elif -0.15 <= final_score <= 0.15:
                                   action, action_en, ac, strength = "ESPERAR",        "WAIT",  "#9E9E9E", ""
        elif final_score > 0.15:   action, action_en, ac, strength = "POSIBLE COMPRA", "WATCH LONG",  "#FFD600", "DÉBIL"
        else:                      action, action_en, ac, strength = "POSIBLE VENTA",  "WATCH SHORT", "#FF6D00", "DÉBIL"

        if action_en in ("LONG", "WATCH LONG"):
            entry=price; sl=round(price-1.5*atr_v,4)
            tp1=round(price+1.5*atr_v,4); tp2=round(price+3.0*atr_v,4); tp3=round(price+5.0*atr_v,4)
        elif action_en in ("SHORT", "WATCH SHORT"):
            entry=price; sl=round(price+1.5*atr_v,4)
            tp1=round(price-1.5*atr_v,4); tp2=round(price-3.0*atr_v,4); tp3=round(price-5.0*atr_v,4)
        else:
            entry=sl=tp1=tp2=tp3=price

        rr = round(abs(tp2 - entry) / abs(sl - entry), 2) if sl != entry else 0
        confidence = min(95, max(30, int(abs(final_score) * 100 + regime_conf * 20 + 30)))

        reasons = []
        if regime in ("Alcista Fuerte", "Alcista", "Bajista", "Bajista Fuerte"):
            reasons.append(f"Régimen HMM {regime} con {regime_conf*100:.0f}% de confianza")
        if regime == "Acumulación":
            reasons.append(f"Régimen de Acumulación — posible base antes de subida")
        if regime == "Distribución":
            reasons.append(f"Régimen de Distribución — posible techo antes de caída")
        if ctx["price_trend"] != "mixta":
            sym = ">" if ctx["price_trend"] == "alcista" else "<"
            reasons.append(f"Tendencia {ctx['price_trend']}: EMA20 {sym} EMA50 {sym} EMA200")
        if summary["pct_buy"] > 55:
            reasons.append(f"{summary['buy']}/{summary['total']} indicadores en señal de compra")
        elif summary["pct_sell"] > 55:
            reasons.append(f"{summary['sell']}/{summary['total']} indicadores en señal de venta")
        if ctx["rsi_state"] in ("sobrevendido", "sobrecomprado"):
            reasons.append(f"RSI {ctx['rsi_value']} — zona de {ctx['rsi_state']}")
        if ctx["volume"] == "alto":
            reasons.append(f"Volumen {ctx['vol_ratio']}x sobre promedio — confirma movimiento")
        if ctx["macd_momentum"] in ("positivo", "negativo"):
            reasons.append(f"MACD histograma {ctx['macd_momentum']} y acelerando")
        if ctx["bb_position"] in ("cerca_upper", "cerca_lower"):
            reasons.append(f"Precio en {'resistencia' if ctx['bb_position']=='cerca_upper' else 'soporte'} de Bollinger (BB%B={ctx['bb_pctb']})")

        warnings = []
        if ctx["volatility"] == "alta":
            warnings.append(f"Alta volatilidad (ATR={ctx['atr_pct']}%) — ajustar tamaño de posición")
        if ctx["rsi_state"] == "sobrecomprado" and action_en == "LONG":
            warnings.append("RSI en sobrecompra — posible corrección inminente")
        if ctx["rsi_state"] == "sobrevendido" and action_en == "SHORT":
            warnings.append("RSI en sobreventa — rebote posible")
        if abs(final_score) < 0.35:
            warnings.append("Señal débil — mejor esperar confirmación")
        if regime == "Lateral":
            warnings.append("Mercado lateral — operar con rangos, no tendencia")

        return {"symbol": symbol, "price": price, "action": action, "action_en": action_en,
                "action_color": ac, "strength": strength, "confidence": confidence,
                "final_score": round(final_score, 4), "regime": regime, "regime_conf": round(regime_conf*100,1),
                "entry": entry, "stop_loss": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3, "rr_ratio": rr,
                "atr": round(atr_v, 4), "supports": supports, "resistances": resistances,
                "context": ctx, "ind_summary": summary, "signals": signals,
                "reasons": reasons, "warnings": warnings}
