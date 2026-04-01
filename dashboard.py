"""
dashboard.py
============
Dashboard HMM + Indicadores Técnicos + Señal IA en tiempo real.
7 regímenes: Bajista Fuerte, Bajista, Distribución, Lateral,
             Acumulación, Alcista, Alcista Fuerte.
Selector de tema: Oscuro / Claro Celeste.
100% responsive para dispositivos móviles.

Fuentes:
  Crypto  → Binance API (tiempo real, sin key)
  Forex / Gold / Acciones → yfinance (~15 min delay)

Correr: streamlit run dashboard.py
"""

import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from hmm_engine   import MultiTimeframeHMM, RegimeHMM, REGIME_COLORS, TIMEFRAME_WEIGHTS
from data_fetcher import (fetch_ticker_universal, fetch_all_timeframes_universal,
                          make_synthetic_all_timeframes, CRYPTO_SYMBOLS)
from indicators   import compute_all_indicators, indicator_signals, summary_signal
from ai_signal    import AITradingSignal
from alerts       import RegimeAlertSystem
from themes       import THEMES, apply_css

# ─────────────────────────────────────────
# PAGE CONFIG — auto sidebar para mobile
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Trade Buddy — HMM AI Signal",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto",
)

# Viewport meta tag para correcta escala móvil
st.markdown(
    '<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0">',
    unsafe_allow_html=True
)

# ─────────────────────────────────────────
# CATÁLOGO DE ACTIVOS
# ─────────────────────────────────────────
ASSET_CATALOG = {
    "🪙 Crypto — Binance": {
        "BTC/USDT":  {"icon":"₿",  "color":"#F7931A"},
        "ETH/USDT":  {"icon":"Ξ",  "color":"#627EEA"},
        "SOL/USDT":  {"icon":"◎",  "color":"#9945FF"},
        "BNB/USDT":  {"icon":"B",  "color":"#F3BA2F"},
        "XRP/USDT":  {"icon":"✕",  "color":"#00AAE4"},
        "AVAX/USDT": {"icon":"▲",  "color":"#E84142"},
        "ADA/USDT":  {"icon":"₳",  "color":"#0033AD"},
        "DOGE/USDT": {"icon":"Ð",  "color":"#C3A634"},
        "LINK/USDT": {"icon":"⬡",  "color":"#2A5ADA"},
        "MATIC/USDT":{"icon":"⬟",  "color":"#8247E5"},
    },
    "💱 Forex — Yahoo Finance": {
        "EUR/USD":{"icon":"€","color":"#003399"},
        "GBP/USD":{"icon":"£","color":"#012169"},
        "USD/JPY":{"icon":"¥","color":"#BC002D"},
        "AUD/USD":{"icon":"A","color":"#00843D"},
        "USD/CAD":{"icon":"C","color":"#FF0000"},
        "USD/CHF":{"icon":"F","color":"#D52B1E"},
    },
    "🥇 Commodities — Yahoo Finance": {
        "XAU/USD":{"icon":"Au","color":"#FFD700"},
        "XAG/USD":{"icon":"Ag","color":"#C0C0C0"},
        "WTI/USD":{"icon":"🛢", "color":"#4a4a4a"},
        "NG/USD": {"icon":"🔥","color":"#FF8C00"},
    },
    "📈 Acciones / ETFs — Yahoo Finance": {
        "AAPL": {"icon":"","color":"#555555"},
        "TSLA": {"icon":"","color":"#CC0000"},
        "NVDA": {"icon":"","color":"#76B900"},
        "MSFT": {"icon":"","color":"#00A4EF"},
        "AMZN": {"icon":"","color":"#FF9900"},
        "GOOGL":{"icon":"","color":"#4285F4"},
        "SPY":  {"icon":"S","color":"#1a56db"},
        "QQQ":  {"icon":"Q","color":"#7e3af2"},
    },
}

def get_meta(symbol):
    for cat, assets in ASSET_CATALOG.items():
        if symbol in assets:
            return {**assets[symbol], "category": cat}
    return {"icon":"?","color":"#888","category":"Desconocido"}


# ─────────────────────────────────────────
# FORMATO PRECIOS
# ─────────────────────────────────────────
def fmt_price(price, symbol):
    if price is None: return "N/A"
    if symbol in CRYPTO_SYMBOLS:
        if price >= 1000: return f"${price:,.2f}"
        if price >= 1:    return f"${price:.4f}"
        return f"${price:.6f}"
    elif "JPY" in symbol: return f"{price:.3f}"
    elif symbol in ("XAU/USD","XAG/USD","WTI/USD","NG/USD"): return f"${price:,.2f}"
    elif symbol in ("AAPL","TSLA","NVDA","MSFT","AMZN","GOOGL","SPY","QQQ"): return f"${price:,.2f}"
    return f"{price:.5f}"

def fmt_vol(vol):
    if not vol or vol == 0: return "N/A"
    if vol >= 1e9: return f"${vol/1e9:.2f}B"
    if vol >= 1e6: return f"${vol/1e6:.2f}M"
    if vol >= 1e3: return f"${vol/1e3:.0f}K"
    return f"{vol:.0f}"


# ─────────────────────────────────────────
# CARGA DE DATOS (cache)
# ─────────────────────────────────────────
@st.cache_data(ttl=30, show_spinner=False)
def load_ticker(symbol):
    try:    return fetch_ticker_universal(symbol)
    except: return {"price":None,"change_pct":0,"source":"Error","realtime":False}

@st.cache_data(ttl=300, show_spinner=False)
def load_ohlcv(symbol, use_real):
    if use_real: return fetch_all_timeframes_universal(symbol)
    base = {"BTC/USDT":68000,"ETH/USDT":3500,"XAU/USD":2300}.get(symbol, 100)
    return make_synthetic_all_timeframes(start_price=base)

@st.cache_resource(show_spinner=False)
def train_system(symbol, n_states, auto_select, use_real, _key):
    data   = load_ohlcv(symbol, use_real)
    system = MultiTimeframeHMM(n_states=n_states, auto_select=auto_select)
    system.fit(data)
    return system, data


# ─────────────────────────────────────────
# HELPERS HTML
# ─────────────────────────────────────────
def card_open(T):
    return f'<div class="hmm-card" style="background:{T["card_bg"]};border-color:{T["card_border"]}">'

def card_close():
    return '</div>'

def card_title(text):
    return f'<div class="hmm-card-title">{text}</div>'

def metric_html(label, value, sub="", sub_color=None, T=None):
    sc = sub_color or T["text_secondary"]
    return f"""
    <div class="hmm-metric" style="background:{T['metric_bg']}">
        <div class="hmm-metric-label">{label}</div>
        <div class="hmm-metric-value" style="color:{T['text_primary']}">{value}</div>
        {f'<div class="hmm-metric-sub" style="color:{sc}">{sub}</div>' if sub else ''}
    </div>"""

def regime_badge(regime, confidence, T):
    color = REGIME_COLORS.get(regime, "#9E9E9E")
    return f"""
    <div class="ai-signal-box" style="background:{color}20;border:2px solid {color};">
        <div class="ai-action" style="color:{color}">{regime}</div>
        <div class="ai-conf" style="color:{color}">Confianza: {confidence*100:.1f}%</div>
    </div>"""

def source_pill(symbol, T):
    is_crypto = symbol in CRYPTO_SYMBOLS
    color  = T["primaryColor"]
    label  = "⚡ Binance · Tiempo real" if is_crypto else "⏱ Yahoo Finance · ~15 min"
    return f'<span style="font-size:10px;padding:3px 9px;border-radius:4px;background:{color}18;color:{color};border:1px solid {color}44">{label}</span>'


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:

    # TEMA
    st.markdown("### 🎨 Tema")
    theme_name = st.radio("", list(THEMES.keys()), index=0, horizontal=True,
                           label_visibility="collapsed")
    T = THEMES[theme_name]
    st.markdown(apply_css(T), unsafe_allow_html=True)

    # Aplicar CSS también al body principal (fuera del sidebar)
    _css_applied = True

    st.divider()
    st.markdown("### 📡 Datos")
    use_real = st.toggle("Datos reales", value=False,
                          help="ON = Binance/Yahoo Finance | OFF = demo offline")

    st.markdown("### 🎯 Activo")
    cat    = st.selectbox("Categoría", list(ASSET_CATALOG.keys()))
    symbol = st.selectbox("Par / Símbolo", list(ASSET_CATALOG[cat].keys()))
    meta   = get_meta(symbol)
    is_crypto = symbol in CRYPTO_SYMBOLS
    st.markdown(source_pill(symbol, T), unsafe_allow_html=True)

    st.markdown("### 🧠 Modelo HMM")
    auto_select = st.toggle("Auto-selección de estados (BIC)", value=False)
    n_states    = st.slider("Nº de estados", 3, 7, 7, disabled=auto_select,
                             help="7 = máximo detalle (7 regímenes)")

    st.markdown("### 📊 Gráfico")
    selected_tf = st.select_slider("Timeframe", ["5m","15m","1h","4h","1d"], value="1h")

    st.markdown("**Overlays**")
    show_emas = st.multiselect("EMAs", [9,20,50,200], default=[20,50,200])
    col_ov1, col_ov2 = st.columns(2)
    with col_ov1:
        show_bb   = st.checkbox("Bollinger",  value=True)
        show_vwap = st.checkbox("VWAP",       value=True)
    with col_ov2:
        show_st   = st.checkbox("Supertrend", value=False)
        show_ich  = st.checkbox("Ichimoku",   value=False)

    st.markdown("**Paneles inferiores**")
    panel_opts = st.multiselect(
        "Indicadores", ["MACD","RSI","Volumen","Stochastic","CCI","ATR"],
        default=["MACD","RSI","Volumen"]
    )

    train_btn = st.button("🚀 Entrenar / Actualizar", type="primary", use_container_width=True)

    if st.button("🔄 Actualizar precios", use_container_width=True):
        st.cache_data.clear(); st.rerun()

    st.divider()
    st.caption("Trade Buddy v3.0 · 7 Regímenes\nCrypto: Binance RT · Resto: Yahoo Finance\nPesos: 5m=1 · 15m=2 · 1h=3 · 4h=4 · 1d=5")


# ─── Aplicar CSS al body principal también ───
st.markdown(apply_css(T), unsafe_allow_html=True)

# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
cache_key = f"{symbol}_{n_states}_{auto_select}_{use_real}"

# Inicializar todas las claves necesarias
for k, v in [("last_key",""), ("alerts_sys",None), ("ai_engine",None),
             ("system",None), ("data",None)]:
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.alerts_sys is None: st.session_state.alerts_sys = RegimeAlertSystem()
if st.session_state.ai_engine  is None: st.session_state.ai_engine  = AITradingSignal()

# Auto-entrenar al cargar (datos demo = instantáneo, sin internet)
# o re-entrenar si cambió la configuración o se presionó el botón
needs_train = (st.session_state.system is None or
               st.session_state.last_key != cache_key or
               train_btn)

if needs_train:
    with st.spinner(f"⏳ Entrenando HMM para {symbol} ({n_states} estados)..."):
        try:
            system, data = train_system(symbol, n_states, auto_select, use_real, cache_key)
            st.session_state.system   = system
            st.session_state.data     = data
            st.session_state.last_key = cache_key
        except Exception as e:
            st.error(f"❌ Error de entrenamiento: {e}")
            st.info("Desactiva 'Datos reales' en el panel lateral para usar datos demo.")
            st.stop()

system     = st.session_state.system
data       = st.session_state.data
alerts_sys = st.session_state.alerts_sys
ai_engine  = st.session_state.ai_engine

ticker    = load_ticker(symbol) if use_real else {
    "price":68708.0,"change_pct":1.8,"volume_24h":31e9,"source":"Demo","realtime":False}
composite  = system.composite_regime(data)
new_alerts = alerts_sys.check_composite(symbol, composite)
for a in new_alerts:
    st.toast(f"🔔 [{a['timeframe']}] {a['from']} → {a['to']}", icon="🔔")

df_tf      = data.get(selected_tf, pd.DataFrame())
model_tf   = system.models.get(selected_tf)
regime_info = model_tf.current_regime(df_tf) if model_tf and len(df_tf) > 30 else \
              {"regime":"Desconocido","confidence":0.5,"stability":0,"posteriors":{}}
ind_df     = compute_all_indicators(df_tf) if len(df_tf) > 30 else pd.DataFrame()
signals    = indicator_signals(df_tf) if len(df_tf) > 30 else {}
ind_sum    = summary_signal(signals) if signals else {}
ai_result  = ai_engine.analyze(df_tf, regime_info, symbol) if len(df_tf) > 30 else {}

price = ticker.get("price")
chg   = ticker.get("change_pct", 0) or 0


# ════════════════════════════════════════
# ══════════  LAYOUT PRINCIPAL  ══════════
# ════════════════════════════════════════

# ─── HEADER ─────────────────────────────
col_h1, col_h2 = st.columns([3,1])
with col_h1:
    st.markdown(f"## {meta['icon']} {symbol}")
    st.caption(f"{meta['category']} · {ticker.get('source','—')} · Timeframe: {selected_tf}")
with col_h2:
    st.markdown(source_pill(symbol, T), unsafe_allow_html=True)
    st.caption(f"Actualizado: {time.strftime('%H:%M:%S')}")

# ─── MÉTRICAS TOP ───────────────────────
c1,c2,c3,c4,c5 = st.columns(5)
with c1:
    delta_color = "normal" if chg >= 0 else "inverse"
    st.metric("Precio actual", fmt_price(price, symbol), f"{chg:+.2f}% 24h", delta_color=delta_color)
with c2:
    st.metric("Volumen 24h", fmt_vol(ticker.get("volume_24h",0)))
with c3:
    rc = REGIME_COLORS.get(regime_info["regime"],"#888")
    st.metric("Régimen HMM", regime_info["regime"], f"{regime_info['confidence']*100:.0f}% conf")
with c4:
    rsi_val = float(ind_df["rsi_14"].iloc[-1]) if not ind_df.empty and "rsi_14" in ind_df.columns else 0
    rsi_state = "Sobrecompra" if rsi_val>70 else "Sobreventa" if rsi_val<30 else "Neutral"
    st.metric("RSI 14", f"{rsi_val:.1f}", rsi_state)
with c5:
    atr_val = float(ind_df["atr_14"].iloc[-1]) if not ind_df.empty and "atr_14" in ind_df.columns else 0
    atr_pct = atr_val/price*100 if price else 0
    st.metric("ATR 14", fmt_price(atr_val, symbol), f"{atr_pct:.1f}% del precio")

st.divider()


# ════════════════════════════════════════
# A — SEÑAL IA
# ════════════════════════════════════════
st.subheader("🤖 Señal IA en Tiempo Real")

if ai_result:
    col_a1, col_a2, col_a3, col_a4 = st.columns([1.2, 1.4, 1.2, 1.2])

    # Acción
    with col_a1:
        ac    = ai_result.get("action_color","#9E9E9E")
        score = ai_result.get("final_score", 0)
        sp    = int((score + 1) / 2 * 100)
        b = ai_result["ind_summary"].get("buy",0)
        s = ai_result["ind_summary"].get("sell",0)
        n = ai_result["ind_summary"].get("neutral",0)

        st.markdown(f"""
        <div class="ai-signal-box" style="background:{ac}18;border:2px solid {ac};margin-bottom:10px">
            <div class="ai-action" style="color:{ac}">{ai_result['action']}</div>
            <div class="ai-strength" style="color:{ac}">{ai_result.get('strength','')}</div>
            <div class="ai-conf" style="color:{ac}">Confianza IA: {ai_result['confidence']}%</div>
        </div>
        <div style="margin-bottom:10px">
            <div style="font-size:9px;color:{T['text_tertiary']};margin-bottom:5px">Score compuesto</div>
            <div class="score-track" style="background:{T['metric_bg']};border-color:{T['card_border']}">
                <div style="width:{sp}%;height:100%;background:{'#00C853' if score>0 else '#D50000'};border-radius:6px"></div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:9px;color:{T['text_tertiary']};margin-top:3px">
                <span>SELL</span><span>NEUTRAL</span><span>BUY</span>
            </div>
        </div>
        <div style="display:flex;gap:5px;justify-content:center;flex-wrap:wrap">
            <span style="font-size:9px;padding:3px 8px;border-radius:4px;background:#00C85318;color:#00C853;border:1px solid #00C85344">▲ {b} BUY</span>
            <span style="font-size:9px;padding:3px 8px;border-radius:4px;background:#D5000018;color:#D50000;border:1px solid #D5000044">▼ {s} SELL</span>
            <span style="font-size:9px;padding:3px 8px;border-radius:4px;background:{T['neutral_bg']};color:{T['text_secondary']};border:1px solid {T['card_border']}">— {n} NEU</span>
        </div>
        """, unsafe_allow_html=True)

    # Razones
    with col_a2:
        st.markdown(f"<div style='font-size:12px;font-weight:600;color:{T['text_primary']};margin-bottom:8px'>Análisis IA</div>", unsafe_allow_html=True)
        for reason in ai_result.get("reasons",[]):
            st.markdown(f'<div class="reason-item" style="background:#00C85312;border-left-color:#00C853;color:{T["text_secondary"]}">✓ {reason}</div>', unsafe_allow_html=True)
        for w in ai_result.get("warnings",[]):
            st.markdown(f'<div class="reason-item" style="background:#FFD60012;border-left-color:#FFD600;color:{T["text_secondary"]}">⚠ {w}</div>', unsafe_allow_html=True)

        ctx = ai_result.get("context",{})
        if ctx:
            st.markdown(f"<div style='font-size:11px;font-weight:600;color:{T['text_primary']};margin:10px 0 6px'>Contexto</div>", unsafe_allow_html=True)
            rows = [("Tendencia",ctx.get("price_trend","—")),
                    ("RSI",f"{ctx.get('rsi_value','—')} ({ctx.get('rsi_state','—')})"),
                    ("Volatilidad",ctx.get("volatility","—")),
                    ("Volumen",f"{ctx.get('volume','—')} ({ctx.get('vol_ratio','—')}x)"),
                    ("MACD",ctx.get("macd_momentum","—"))]
            for lbl,val in rows:
                st.markdown(f"""<div style="display:flex;justify-content:space-between;font-size:10px;
                    padding:3px 0;border-bottom:1px solid {T['divider']}">
                    <span style="color:{T['text_secondary']}">{lbl}</span>
                    <span style="color:{T['text_primary']}">{val}</span></div>""", unsafe_allow_html=True)

    # Targets
    with col_a3:
        st.markdown(f"<div style='font-size:12px;font-weight:600;color:{T['text_primary']};margin-bottom:8px'>Targets (ATR-based)</div>", unsafe_allow_html=True)
        tps = [("Entrada",   ai_result.get("entry",price),   "#60a5fa"),
               ("Stop Loss", ai_result.get("stop_loss",0),    "#D50000"),
               ("TP1 — 1.5×ATR", ai_result.get("tp1",0),     "#00C853"),
               ("TP2 — 3×ATR",   ai_result.get("tp2",0),     "#00C853"),
               ("TP3 — 5×ATR",   ai_result.get("tp3",0),     "#00C853")]
        for lbl,val,col in tps:
            st.markdown(f"""<div style="display:flex;justify-content:space-between;
                padding:5px 10px;border-radius:5px;background:{col}15;
                border-left:3px solid {col};margin-bottom:4px">
                <span style="font-size:10px;color:{T['text_secondary']}">{lbl}</span>
                <strong style="font-size:10px;color:{col}">{fmt_price(val,symbol)}</strong>
                </div>""", unsafe_allow_html=True)
        rr = ai_result.get("rr_ratio",0)
        st.markdown(f"""<div style="padding:6px 10px;border-radius:5px;
            background:{T['metric_bg']};text-align:center;margin-top:6px">
            <span style="font-size:10px;color:{T['text_secondary']}">Risk / Reward</span>
            <strong style="font-size:13px;color:{T['primaryColor']};margin-left:10px">1 : {rr}</strong>
            </div>""", unsafe_allow_html=True)

    # Niveles
    with col_a4:
        st.markdown(f"<div style='font-size:12px;font-weight:600;color:{T['text_primary']};margin-bottom:8px'>Soporte / Resistencia</div>", unsafe_allow_html=True)
        for r in ai_result.get("resistances",[])[:3]:
            st.markdown(f'<div style="font-size:10px;padding:4px 8px;border-radius:4px;background:#FF6D0015;border-left:2px solid #FF6D00;margin-bottom:3px;color:#FF6D00">R: {fmt_price(r,symbol)}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:10px;padding:4px 8px;border-radius:4px;background:{T["primaryColor"]}20;border-left:2px solid {T["primaryColor"]};margin-bottom:3px;color:{T["primaryColor"]}">Precio: {fmt_price(price,symbol)}</div>', unsafe_allow_html=True)
        for sv in ai_result.get("supports",[])[:3]:
            st.markdown(f'<div style="font-size:10px;padding:4px 8px;border-radius:4px;background:#00C85315;border-left:2px solid #00C853;margin-bottom:3px;color:#00C853">S: {fmt_price(sv,symbol)}</div>', unsafe_allow_html=True)

st.divider()


# ════════════════════════════════════════
# B — GRÁFICO PRINCIPAL
# ════════════════════════════════════════
st.subheader(f"📈 Precio con Indicadores — {symbol} · {selected_tf}")

if df_tf.empty or ind_df.empty:
    st.warning("Sin datos para este timeframe.")
else:
    n_panels = len(panel_opts)
    row_h    = [0.55] + [0.45 / max(n_panels, 1)] * n_panels if n_panels else [1.0]
    specs    = [[{}]] * (n_panels + 1)

    fig = make_subplots(rows=n_panels+1, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02, row_heights=row_h[:n_panels+1], specs=specs)

    # Overlay régimen HMM
    if model_tf:
        pred_tf = model_tf.predict(df_tf)
        for regime, color in REGIME_COLORS.items():
            mask = pred_tf["regime"] == regime
            if not mask.any(): continue
            blocks = (mask != mask.shift()).cumsum()
            for _, grp in pred_tf[mask].groupby(blocks[mask]):
                fig.add_vrect(x0=grp.index[0], x1=grp.index[-1],
                              fillcolor=color, opacity=0.08, layer="below",
                              line_width=0, row=1, col=1)

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df_tf.index, open=df_tf["open"], high=df_tf["high"],
        low=df_tf["low"], close=df_tf["close"], name="Precio",
        increasing_line_color=T["candle_up"],   increasing_fillcolor=T["candle_up"],
        decreasing_line_color=T["candle_down"], decreasing_fillcolor=T["candle_down"],
    ), row=1, col=1)

    # EMAs
    ema_colors = {9:"#f59e0b",20:"#60a5fa",50:"#a78bfa",200:"#fb923c"}
    for p in show_emas:
        col_name = f"ema_{p}"
        if col_name in ind_df.columns:
            fig.add_trace(go.Scatter(x=ind_df.index, y=ind_df[col_name],
                                      name=f"EMA {p}", line=dict(color=ema_colors.get(p,"#aaa"), width=1.2),
                                      opacity=0.85), row=1, col=1)

    # Bollinger Bands
    if show_bb and "bb_upper" in ind_df.columns:
        fig.add_trace(go.Scatter(x=ind_df.index, y=ind_df["bb_upper"], name="BB Upper",
                                  line=dict(color="#7c3aed", width=0.8, dash="dot"), opacity=0.7), row=1, col=1)
        fig.add_trace(go.Scatter(x=ind_df.index, y=ind_df["bb_lower"], name="BB Lower",
                                  line=dict(color="#7c3aed", width=0.8, dash="dot"), opacity=0.7,
                                  fill="tonexty", fillcolor="rgba(124,58,237,0.04)"), row=1, col=1)

    # VWAP
    if show_vwap and "vwap" in ind_df.columns:
        fig.add_trace(go.Scatter(x=ind_df.index, y=ind_df["vwap"], name="VWAP",
                                  line=dict(color="#f97316", width=1.5, dash="dash"), opacity=0.85), row=1, col=1)

    # Precio actual
    if price:
        fig.add_hline(y=price, line_color="#facc15", line_width=1.2, line_dash="dash",
                      annotation_text=f"  {fmt_price(price,symbol)}", annotation_font_color="#facc15",
                      row=1, col=1)

    # Paneles inferiores
    for pi, panel in enumerate(panel_opts, start=2):
        if panel == "MACD" and "macd" in ind_df.columns:
            hc = ["#26a69a" if v >= 0 else "#ef5350" for v in ind_df["macd_hist"]]
            fig.add_trace(go.Bar(x=ind_df.index, y=ind_df["macd_hist"], marker_color=hc,
                                  name="MACD Hist", opacity=0.7, showlegend=False), row=pi, col=1)
            fig.add_trace(go.Scatter(x=ind_df.index, y=ind_df["macd"], name="MACD",
                                      line=dict(color="#60a5fa", width=1.2)), row=pi, col=1)
            fig.add_trace(go.Scatter(x=ind_df.index, y=ind_df["macd_signal"], name="Signal",
                                      line=dict(color="#f59e0b", width=1.2)), row=pi, col=1)

        elif panel == "RSI" and "rsi_14" in ind_df.columns:
            fig.add_trace(go.Scatter(x=ind_df.index, y=ind_df["rsi_14"], name="RSI 14",
                                      line=dict(color="#a78bfa", width=1.5)), row=pi, col=1)
            fig.add_hline(y=70, line_color="#ef5350", line_width=1, line_dash="dot", row=pi, col=1)
            fig.add_hline(y=30, line_color="#26a69a", line_width=1, line_dash="dot", row=pi, col=1)
            fig.update_yaxes(range=[0,100], row=pi, col=1)

        elif panel == "Volumen":
            vc = ["#26a69a" if c >= o else "#ef5350" for c,o in zip(df_tf["close"], df_tf["open"])]
            fig.add_trace(go.Bar(x=df_tf.index, y=df_tf["volume"], marker_color=vc,
                                  name="Volumen", opacity=0.7), row=pi, col=1)
            fig.add_trace(go.Scatter(x=df_tf.index, y=df_tf["volume"].rolling(20).mean(),
                                      name="Vol MA20", line=dict(color="#f59e0b", width=1.2)), row=pi, col=1)

        elif panel == "Stochastic" and "stoch_k" in ind_df.columns:
            fig.add_trace(go.Scatter(x=ind_df.index, y=ind_df["stoch_k"], name="%K",
                                      line=dict(color="#60a5fa", width=1.2)), row=pi, col=1)
            fig.add_trace(go.Scatter(x=ind_df.index, y=ind_df["stoch_d"], name="%D",
                                      line=dict(color="#f59e0b", width=1.2)), row=pi, col=1)
            fig.add_hline(y=80, line_color="#ef5350", line_width=1, line_dash="dot", row=pi, col=1)
            fig.add_hline(y=20, line_color="#26a69a", line_width=1, line_dash="dot", row=pi, col=1)
            fig.update_yaxes(range=[0,100], row=pi, col=1)

        elif panel == "CCI" and "cci_20" in ind_df.columns:
            cc = ["#26a69a" if v>=0 else "#ef5350" for v in ind_df["cci_20"]]
            fig.add_trace(go.Bar(x=ind_df.index, y=ind_df["cci_20"], marker_color=cc,
                                  name="CCI 20", opacity=0.7), row=pi, col=1)
            fig.add_hline(y=100,  line_color="#ef5350", line_width=1, line_dash="dot", row=pi, col=1)
            fig.add_hline(y=-100, line_color="#26a69a", line_width=1, line_dash="dot", row=pi, col=1)

        elif panel == "ATR" and "atr_14" in ind_df.columns:
            fig.add_trace(go.Scatter(x=ind_df.index, y=ind_df["atr_14"], name="ATR 14",
                                      line=dict(color="#fb923c", width=1.5),
                                      fill="tozeroy", fillcolor="rgba(251,146,60,0.08)"), row=pi, col=1)

    fig.update_layout(
        height=520 + n_panels * 140,
        paper_bgcolor=T["paper_bg"],
        plot_bgcolor=T["plot_bg"],
        font=dict(color=T["tick_color"]),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font=dict(size=10)),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    for i in range(1, n_panels + 2):
        fig.update_xaxes(gridcolor=T["grid_color"], row=i, col=1)
        fig.update_yaxes(gridcolor=T["grid_color"], row=i, col=1)

    st.plotly_chart(fig, use_container_width=True)

st.divider()


# ════════════════════════════════════════
# C — TABLA DE SEÑALES + HMM
# ════════════════════════════════════════
st.subheader("📋 Señales por Indicador")

col_sigs, col_hmm = st.columns([2.5, 1])

with col_sigs:
    if signals:
        cats = {
            "📈 Medias móviles": ["SMA 20","SMA 50","SMA 200","EMA 20","EMA 50","EMA 200","VWAP"],
            "⚡ Momentum":       ["MACD","RSI 14","Stoch %K","CCI 20","Williams %R","MFI 14","ROC 12"],
            "📊 Volumen":        ["OBV","CMF 20"],
            "🎯 Volatilidad":    ["Bollinger %B","ATR 14"],
        }
        col_c1, col_c2 = st.columns(2)
        col_list = [col_c1, col_c2]
        for ci, (cat_name, ind_names) in enumerate(cats.items()):
            with col_list[ci % 2]:
                st.markdown(f"**{cat_name}**")
                for name in ind_names:
                    if name not in signals: continue
                    sig = signals[name]["signal"]
                    val = signals[name]["value"]
                    fmt = signals[name].get("fmt",".2f")
                    if sig == "BUY":     bg,tc,ic = "#00C85315","#00C853","▲"
                    elif sig == "SELL":  bg,tc,ic = "#D5000015","#D50000","▼"
                    else:                bg,tc,ic = T["neutral_bg"],T["text_secondary"],"—"
                    val_str = f"{val:{fmt}}" if isinstance(val,(int,float)) and not np.isnan(val) else "N/A"
                    st.markdown(f"""<div class="sig-row" style="background:{bg};border-left-color:{tc}">
                        <span style="color:{T['text_secondary']}">{name}</span>
                        <span style="color:{T['text_tertiary']};font-size:9px">{val_str}</span>
                        <span style="color:{tc};font-weight:600">{ic} {sig}</span>
                    </div>""", unsafe_allow_html=True)

with col_hmm:
    st.markdown(f"**Multi-Timeframe**")
    for tf, info in composite["breakdown"].items():
        color = REGIME_COLORS.get(info["regime"],"#888")
        w     = TIMEFRAME_WEIGHTS.get(tf, 1)
        st.markdown(f"""<div class="tf-pill" style="background:{color}18;border-color:{color}">
            <div style="font-size:10px;color:{T['text_tertiary']};margin-bottom:2px">{tf} <small style="opacity:.6">(w={w})</small></div>
            <div style="font-size:13px;font-weight:600;color:{color}">{info['regime']}</div>
            <div style="font-size:9px;color:{T['text_tertiary']};margin-top:2px">{info['confidence']*100:.0f}% · {info['stability']} velas</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"**Probabilidades**")
    for regime, prob in sorted(regime_info.get("posteriors",{}).items(), key=lambda x:-x[1]):
        color = REGIME_COLORS.get(regime,"#888")
        st.markdown(f"""<div class="prob-row" style="background:{color}18;border-left-color:{color}">
            <span style="color:{color}">{regime}</span>
            <strong style="color:{color}">{prob*100:.1f}%</strong>
        </div>""", unsafe_allow_html=True)

    st.metric("Estabilidad", f"{regime_info.get('stability',0)} velas")
    st.metric("Confianza",   f"{regime_info.get('confidence',0)*100:.1f}%")

st.divider()


# ════════════════════════════════════════
# D — RÉGIMEN COMPUESTO + SCORES
# ════════════════════════════════════════
st.subheader("🎯 Régimen Compuesto Multi-Timeframe")

comp_reg   = composite["composite_regime"]
comp_color = REGIME_COLORS.get(comp_reg, "#9E9E9E")
comp_conf  = composite["composite_confidence"]

col_cr1, col_cr2 = st.columns([1, 2])
with col_cr1:
    st.markdown(f"""
    <div class="ai-signal-box" style="background:{comp_color}20;border:2px solid {comp_color};padding:20px">
        <div class="ai-action" style="color:{comp_color}">{comp_reg}</div>
        <div class="ai-conf" style="color:{comp_color}">Confianza global: {comp_conf*100:.1f}%</div>
    </div>""", unsafe_allow_html=True)

with col_cr2:
    # Barra de scores por régimen
    scores = composite.get("composite_scores", {})
    # Filtrar regímenes con score > 0
    active = {k: v for k,v in scores.items() if v > 0.001 and k != "Desconocido"}
    if active:
        sorted_scores = sorted(active.items(), key=lambda x: -x[1])
        for reg, scr in sorted_scores[:7]:
            color = REGIME_COLORS.get(reg, "#9E9E9E")
            pct   = min(int(scr * 100), 100)
            st.markdown(f"""
            <div style="margin-bottom:6px">
                <div style="display:flex;justify-content:space-between;font-size:10px;margin-bottom:2px">
                    <span style="color:{color};font-weight:600">{reg}</span>
                    <span style="color:{T['text_tertiary']}">{scr*100:.1f}%</span>
                </div>
                <div style="background:{T['metric_bg']};border-radius:4px;height:8px;overflow:hidden">
                    <div style="width:{pct}%;height:100%;background:{color};border-radius:4px;opacity:0.85"></div>
                </div>
            </div>""", unsafe_allow_html=True)

st.divider()


# ════════════════════════════════════════
# E — MATRIZ DE TRANSICIÓN
# ════════════════════════════════════════
st.subheader("🔄 Matriz de Transición HMM")
if model_tf:
    tm = model_tf.transition_df()
    fig_hm = px.imshow(tm, text_auto=".2f", color_continuous_scale="RdYlGn",
                        zmin=0, zmax=1, aspect="auto",
                        title=f"Probabilidades de transición — {selected_tf}")
    fig_hm.update_layout(height=350, paper_bgcolor=T["paper_bg"],
                          font=dict(color=T["tick_color"]), margin=dict(l=0,r=0,t=35,b=0))
    st.plotly_chart(fig_hm, use_container_width=True)

# AIC/BIC
if auto_select and model_tf and model_tf.aic_bic_table is not None:
    st.subheader("📐 Selección de estados — AIC / BIC")
    tbl = model_tf.aic_bic_table
    fig_ic = go.Figure()
    fig_ic.add_trace(go.Scatter(x=tbl["n_states"],y=tbl["aic"],mode="lines+markers",name="AIC",line=dict(color="#FF9800")))
    fig_ic.add_trace(go.Scatter(x=tbl["n_states"],y=tbl["bic"],mode="lines+markers",name="BIC",line=dict(color="#9C27B0")))
    fig_ic.update_layout(height=260,paper_bgcolor=T["paper_bg"],plot_bgcolor=T["plot_bg"],
                          font=dict(color=T["tick_color"]),
                          xaxis=dict(title="Nº estados",dtick=1,gridcolor=T["grid_color"]),
                          yaxis=dict(gridcolor=T["grid_color"]),margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig_ic, use_container_width=True)
    st.dataframe(tbl, use_container_width=True)

st.divider()


# ════════════════════════════════════════
# F — ALERTAS
# ════════════════════════════════════════
st.subheader("🔔 Alertas de Cambio de Régimen")
alert_df = alerts_sys.get_history()
if alert_df.empty:
    st.info("Sin alertas en esta sesión. Se generan al detectar cambios de régimen.")
else:
    st.dataframe(alert_df.sort_values("timestamp",ascending=False),
                 use_container_width=True, hide_index=True)

st.divider()


# ════════════════════════════════════════
# G — GUARDAR / CARGAR
# ════════════════════════════════════════
st.subheader("💾 Modelos")
col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    if st.button("💾 Guardar modelos", use_container_width=True):
        system.save_all("models"); st.success("Guardado en 'models/'")
with col_s2:
    if st.button("📂 Cargar modelos", use_container_width=True):
        import os
        if os.path.exists("models"): system.load_all("models"); st.success("Cargado")
        else: st.warning("Carpeta 'models/' no encontrada")
with col_s3:
    if st.button("🗑 Limpiar cache", use_container_width=True):
        st.cache_data.clear(); st.cache_resource.clear(); st.rerun()

# Auto-refresh crypto
if use_real and is_crypto:
    st.markdown('<meta http-equiv="refresh" content="30">', unsafe_allow_html=True)

st.caption(f"Trade Buddy v3.0 · 7 Regímenes HMM · {symbol} · Tema: {theme_name} · {time.strftime('%H:%M:%S')}")
