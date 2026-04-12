"""
api_server.py
=============
FastAPI backend for TradeBuddy mobile app.
- Chat proxy (Groq AI, key server-side)
- HMM regime detection (real model, trained on-demand)
- Stripe payment integration
"""

import os
import re
import json
import httpx
import stripe
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict

from hmm_engine import RegimeHMM, build_features
from data_fetcher import BinanceFetcher, fetch_ticker_universal, fetch_all_timeframes_universal

app = FastAPI(title="TradeBuddy API", version="1.0.0")

# CORS — allow mobile app and web preview
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq config — key from environment variable (set in Render dashboard)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"

# Stripe config — keys from environment variables
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
stripe.api_key = STRIPE_SECRET_KEY

# Stripe price IDs for each plan (create these in Stripe Dashboard)
STRIPE_PRICES = {
    "trader_monthly": os.environ.get("STRIPE_PRICE_TRADER_MONTHLY", ""),
    "trader_annual": os.environ.get("STRIPE_PRICE_TRADER_ANNUAL", ""),
    "elite_monthly": os.environ.get("STRIPE_PRICE_ELITE_MONTHLY", ""),
    "elite_annual": os.environ.get("STRIPE_PRICE_ELITE_ANNUAL", ""),
}


# ===== Request/Response models =====

class MarketItem(BaseModel):
    symbol: str = ""
    name: str = ""
    price: float = 0
    change24h: float = 0
    regime: str = ""
    confidence: float = 0
    signal: Optional[str] = None
    support: Optional[float] = None
    resistance: Optional[float] = None
    volume: Optional[str] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    marketData: List[MarketItem] = []
    conversationHistory: List[ChatMessage] = []
    lang: str = "en"


class ChatResponse(BaseModel):
    text: str
    title: Optional[str] = None


# ===== System prompt builder =====

def build_system_prompt(market_data: List[MarketItem], user_lang: str) -> str:
    lang_instructions = {
        "es": "SIEMPRE responde en español.",
        "pt": "SEMPRE responda em português.",
        "fr": "Réponds TOUJOURS en français.",
        "de": "Antworte IMMER auf Deutsch.",
        "it": "Rispondi SEMPRE in italiano.",
        "ru": "ВСЕГДА отвечай на русском языке.",
        "zh": "始终用中文回复。",
        "ja": "常に日本語で返答してください。",
        "ko": "항상 한국어로 답변해주세요.",
        "ar": "أجب دائماً باللغة العربية.",
    }
    lang_instruction = lang_instructions.get(user_lang, "Respond in the same language as the user message.")

    market_context = ""
    if market_data:
        market_context = "\n\n## DATOS DE MERCADO EN TIEMPO REAL (actualizados ahora mismo):\n\n"
        for m in market_data[:20]:
            change_sign = "+" if m.change24h > 0 else ""
            confidence_pct = round(m.confidence * 100)
            market_context += (
                f"**{m.symbol}** ({m.name}): Precio ${m.price}, "
                f"Cambio 24h: {change_sign}{m.change24h}%, "
                f"Régimen HMM: {m.regime} (confianza: {confidence_pct}%), "
                f"Señal: {m.signal or 'N/A'}, "
                f"Soporte: ${m.support or 'N/A'}, Resistencia: ${m.resistance or 'N/A'}, "
                f"Volumen: {m.volume or 'N/A'}\n"
            )

    return f"""Eres TradeBuddy AI, el asistente inteligente de análisis de mercados de la app TradeBuddy. Eres experto en trading, criptomonedas, forex, commodities, acciones y análisis técnico.

## TU PERSONALIDAD:
- Eres profesional pero amigable y accesible
- Usas datos reales para fundamentar tus respuestas
- Eres conciso pero completo — respuestas de 2-4 párrafos máximo
- Usas emojis con moderación (📊 🟢 🔴 📈 📉)
- NUNCA das consejos financieros directos — todo es educativo e informativo
- Siempre recuerdas al usuario que haga su propia investigación (DYOR)

## SISTEMA DE ANÁLISIS HMM (Hidden Markov Model):
TradeBuddy usa un modelo HMM de 7 regímenes para detectar el estado del mercado:

1. **Alcista Fuerte** 🟢🟢 — Tendencia alcista con alto momentum. Alta probabilidad de continuación.
2. **Alcista** 🟢 — Tendencia positiva moderada. Momentum favorable.
3. **Acumulación** 🟡 — Fase lateral-alcista. Posible preparación para movimiento alcista. Volumen creciente.
4. **Lateral** ⚪ — Sin dirección clara. Consolidación. Esperar confirmación.
5. **Distribución** 🟠 — Fase lateral-bajista. Posible preparación para caída. Volumen decreciente.
6. **Bajista** 🔴 — Tendencia negativa moderada. Momentum desfavorable.
7. **Bajista Fuerte** 🔴🔴 — Tendencia bajista con alto momentum. Alta volatilidad.

## INDICADORES TÉCNICOS DISPONIBLES (30+):
- Tendencia: RSI, MACD, EMA (20/50/200), ADX
- Momentum: Stochastic K/D, CCI, Williams %R, MFI, ROC
- Volatilidad: Bollinger Bands, ATR, Std Dev, Keltner Channels
- Volumen: OBV, VWAP, Volume SMA, Chaikin MF, A/D Line

## REGLAS IMPORTANTES:
1. {lang_instruction}
2. SIEMPRE incluye disclaimer: "Esto es solo con fines educativos, no constituye consejo financiero."
3. Si te preguntan sobre un activo específico, usa los datos reales proporcionados abajo.
4. Si no tienes datos de un activo, dilo honestamente.
5. Puedes explicar conceptos de trading, análisis técnico, estrategias, gestión de riesgo, etc.
6. NO inventes datos — si no tienes info, dilo.
7. Mantén respuestas cortas y directas (máximo 300 palabras).
8. Si el usuario pregunta algo no relacionado con trading/finanzas, puedes responder brevemente pero redirige a tu expertise.
{market_context}"""


# ===== Endpoints =====

@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "TradeBuddy API"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")

    system_prompt = build_system_prompt(req.marketData, req.lang)

    messages = [
        {"role": "system", "content": system_prompt},
        *[{"role": m.role, "content": m.content} for m in req.conversationHistory[-6:]],
        {"role": "user", "content": req.message},
    ]

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 800,
                    "top_p": 0.9,
                },
            )

        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=f"Groq error: {resp.text}")

        data = resp.json()
        reply = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        if not reply:
            raise HTTPException(status_code=500, detail="Empty response from AI")

        # Try to detect asset title
        title = None
        match = re.search(r'\*?\*?(\w+/\w+|\w{2,6})\*?\*?\s*\(', reply)
        if match:
            title = match.group(1)

        return ChatResponse(text=reply.strip(), title=title)

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="AI response timeout")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===== HMM Regime Detection =====

class RegimeRequest(BaseModel):
    symbol: str  # e.g. "BTC/USDT", "ETH/USDT", "AAPL", "EUR/USD"
    timeframe: str = "1d"  # "5m", "15m", "1h", "4h", "1d"


class RegimeResponse(BaseModel):
    symbol: str
    regime: str
    confidence: float
    stability: int
    posteriors: Dict[str, float]
    features: Dict[str, float] = {}


# Cache trained models to avoid retraining on every request
_hmm_cache: Dict[str, tuple] = {}  # key: "symbol_tf" -> (model, timestamp)
import time as _time


@app.post("/api/regime", response_model=RegimeResponse)
async def detect_regime(req: RegimeRequest):
    """
    Real HMM regime detection using trained Gaussian HMM.
    Fetches market data, builds features, trains/caches model, returns regime.
    """
    cache_key = f"{req.symbol}_{req.timeframe}"
    now = _time.time()

    try:
        # Fetch OHLCV data
        df = None
        try:
            data = fetch_all_timeframes_universal(req.symbol)
            df = data.get(req.timeframe)
        except Exception as e:
            print(f"Data fetch error for {req.symbol}: {e}")

        if df is None or len(df) < 60:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data for {req.symbol} ({req.timeframe}). Need 60+ candles."
            )

        # Check cache (reuse model if < 15 min old)
        cached = _hmm_cache.get(cache_key)
        if cached and (now - cached[1]) < 900:
            model = cached[0]
        else:
            # Train new model
            model = RegimeHMM(n_states=7)
            model.fit(df, auto_select=False)
            _hmm_cache[cache_key] = (model, now)

        # Get current regime
        info = model.current_regime(df)

        # Get latest feature values for the app
        feat = build_features(df)
        latest_feat = {}
        if len(feat) > 0:
            last = feat.iloc[-1]
            latest_feat = {
                "log_return": round(float(last.get("log_return", 0)), 6),
                "volatility": round(float(last.get("volatility", 0)), 6),
                "rsi": round(float(last.get("rsi", 50)), 2),
                "macd_hist": round(float(last.get("macd_hist", 0)), 4),
            }

        return RegimeResponse(
            symbol=req.symbol,
            regime=info["regime"],
            confidence=round(info["confidence"], 4),
            stability=info["stability"],
            posteriors={k: round(v, 4) for k, v in info["posteriors"].items()},
            features=latest_feat,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"HMM error: {str(e)}")


# ===== Stripe Payment Endpoints =====

class CheckoutRequest(BaseModel):
    plan: str  # "trader" or "elite"
    billing: str = "monthly"  # "monthly" or "annual"
    user_id: str  # Firebase UID
    email: str


class CheckoutResponse(BaseModel):
    checkout_url: str
    session_id: str


@app.post("/api/checkout", response_model=CheckoutResponse)
async def create_checkout(req: CheckoutRequest):
    """Create a Stripe Checkout session for plan upgrade."""
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=500, detail="Stripe not configured")

    price_key = f"{req.plan}_{req.billing}"
    price_id = STRIPE_PRICES.get(price_key)

    if not price_id:
        raise HTTPException(status_code=400, detail=f"Invalid plan: {price_key}")

    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            mode="subscription",
            customer_email=req.email,
            line_items=[{"price": price_id, "quantity": 1}],
            success_url="https://tradebuddy.app/payment/success?session_id={CHECKOUT_SESSION_ID}",
            cancel_url="https://tradebuddy.app/payment/cancel",
            metadata={
                "user_id": req.user_id,
                "plan": req.plan,
                "billing": req.billing,
            },
        )
        return CheckoutResponse(checkout_url=session.url, session_id=session.id)
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/webhook/stripe")
async def stripe_webhook(request: Request):
    """
    Handle Stripe webhook events.
    Verifies signature, processes subscription events, updates user plan.
    """
    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="Webhook secret not configured")

    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    # Handle relevant events
    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        user_id = session.get("metadata", {}).get("user_id")
        plan = session.get("metadata", {}).get("plan", "trader")

        if user_id:
            # Update Firestore via Firebase Admin SDK (or store locally)
            print(f"[STRIPE] User {user_id} upgraded to {plan}")
            # TODO: Update Firestore user document with new plan
            # For now, the app handles plan update client-side after successful checkout

    elif event["type"] == "customer.subscription.deleted":
        # Subscription cancelled — downgrade to starter
        subscription = event["data"]["object"]
        print(f"[STRIPE] Subscription cancelled: {subscription.get('id')}")
        # TODO: Find user by subscription and downgrade to starter

    elif event["type"] == "invoice.payment_failed":
        invoice = event["data"]["object"]
        print(f"[STRIPE] Payment failed: {invoice.get('customer_email')}")
        # TODO: Notify user about failed payment

    return JSONResponse(content={"received": True})


@app.get("/api/subscription/{user_id}")
async def get_subscription(user_id: str):
    """Check user's subscription status."""
    if not STRIPE_SECRET_KEY:
        return {"active": False, "plan": "starter", "message": "Stripe not configured"}

    try:
        # Search for customer by metadata
        customers = stripe.Customer.search(
            query=f"metadata['user_id']:'{user_id}'"
        )

        if not customers.data:
            return {"active": False, "plan": "starter"}

        customer = customers.data[0]
        subscriptions = stripe.Subscription.list(customer=customer.id, status="active")

        if not subscriptions.data:
            return {"active": False, "plan": "starter"}

        sub = subscriptions.data[0]
        price_id = sub["items"]["data"][0]["price"]["id"]

        # Reverse lookup plan from price ID
        plan = "trader"
        for key, pid in STRIPE_PRICES.items():
            if pid == price_id:
                plan = key.split("_")[0]
                break

        return {
            "active": True,
            "plan": plan,
            "current_period_end": sub["current_period_end"],
            "cancel_at_period_end": sub.get("cancel_at_period_end", False),
        }
    except Exception as e:
        return {"active": False, "plan": "starter", "error": str(e)}
