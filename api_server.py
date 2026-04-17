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
from signal_engine import scan_and_emit, debug_scan, WATCHLIST
from expo_push import send_push_batch, build_signal_message

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


# ============================================================
# ===============  BUDDY VIP SIGNALS  ========================
# ============================================================
#
# Private educational signals for an invitation-only circle.
# Storage is in-memory (upgrade to Firestore Admin when needed).
# Push delivery via Expo Push Service (no API key required).
# ------------------------------------------------------------

ADMIN_EMAIL = "danpou1974@gmail.com"
SCAN_SECRET = os.environ.get("SCAN_SECRET", "buddy-scan-secret-change-me")

# user_id -> {token, lang, vip, email}
_push_registry: Dict[str, Dict] = {}
# list of signals (newest first), capped
_signals_history: List[Dict] = []
_SIGNALS_MAX = 200
# user_id -> signal_id -> {"action": "took"|"passed", "ts": ...}
_signal_tracking: Dict[str, Dict[str, Dict]] = {}


class PushTokenRequest(BaseModel):
    user_id: str
    email: Optional[str] = ""
    token: str
    lang: str = "en"
    vip: bool = False


@app.post("/api/push-token")
async def register_push_token(req: PushTokenRequest):
    """Register / update an Expo push token for a user."""
    if not req.token.startswith("ExponentPushToken"):
        raise HTTPException(status_code=400, detail="Invalid Expo push token")
    is_admin = (req.email or "").strip().lower() == ADMIN_EMAIL
    _push_registry[req.user_id] = {
        "token": req.token,
        "lang": req.lang,
        "vip": bool(req.vip) or is_admin,
        "email": req.email or "",
        "is_admin": is_admin,
    }
    return {"ok": True, "registered_users": len(_push_registry)}


class VipToggleRequest(BaseModel):
    admin_email: str
    user_id: str
    vip: bool


@app.post("/api/admin/toggle-vip")
async def admin_toggle_vip(req: VipToggleRequest):
    if (req.admin_email or "").strip().lower() != ADMIN_EMAIL:
        raise HTTPException(status_code=403, detail="Not authorized")
    rec = _push_registry.get(req.user_id)
    if not rec:
        raise HTTPException(status_code=404, detail="User not found in registry")
    rec["vip"] = bool(req.vip)
    return {"ok": True, "user": {req.user_id: rec}}


@app.get("/api/admin/users")
async def admin_users(admin_email: str):
    if (admin_email or "").strip().lower() != ADMIN_EMAIL:
        raise HTTPException(status_code=403, detail="Not authorized")
    return {
        "users": [{"user_id": uid, **info} for uid, info in _push_registry.items()],
        "count": len(_push_registry),
    }


@app.get("/api/admin/stats")
async def admin_stats(admin_email: str):
    if (admin_email or "").strip().lower() != ADMIN_EMAIL:
        raise HTTPException(status_code=403, detail="Not authorized")
    vip_users = [u for u in _push_registry.values() if u.get("vip")]
    took = 0; passed = 0
    for uid, sigs in _signal_tracking.items():
        for sid, d in sigs.items():
            if d.get("action") == "took":
                took += 1
            elif d.get("action") == "passed":
                passed += 1
    return {
        "total_users": len(_push_registry),
        "vip_users": len(vip_users),
        "signals_sent": len(_signals_history),
        "took_count": took,
        "passed_count": passed,
        "watchlist": WATCHLIST,
    }


@app.get("/api/signals/list")
async def signals_list(user_id: Optional[str] = None, limit: int = 50):
    """Return recent signals. Only VIP users get the full feed."""
    rec = _push_registry.get(user_id or "")
    is_vip = bool(rec and (rec.get("vip") or rec.get("is_admin")))
    if not is_vip:
        # Return only a count so non-VIP can display teaser
        return {"vip": False, "count": len(_signals_history), "signals": []}
    return {"vip": True, "count": len(_signals_history), "signals": _signals_history[:limit]}


class TrackRequest(BaseModel):
    user_id: str
    signal_id: str
    action: str  # "took" or "passed"


@app.post("/api/signal/track")
async def track_signal(req: TrackRequest):
    if req.action not in ("took", "passed"):
        raise HTTPException(status_code=400, detail="action must be 'took' or 'passed'")
    _signal_tracking.setdefault(req.user_id, {})[req.signal_id] = {
        "action": req.action, "ts": int(_time.time()),
    }
    return {"ok": True}


@app.post("/api/scan-signals")
async def scan_signals(x_scan_secret: Optional[str] = Header(default=None)):
    """
    Triggered by cron-job.org every 15 min.
    Returns the new signals emitted in this run.
    """
    if x_scan_secret != SCAN_SECRET:
        raise HTTPException(status_code=403, detail="Bad secret")

    try:
        new_signals = scan_and_emit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"scan error: {e}")

    delivered = []
    for sig in new_signals:
        sig["id"] = f"{sig['symbol'].replace('/', '')}_{sig['generated_at']}"
        _signals_history.insert(0, sig)

        # Dispatch per-language (group tokens by lang to preserve localization)
        vip_records = [r for r in _push_registry.values() if r.get("vip") or r.get("is_admin")]
        by_lang: Dict[str, List[str]] = {}
        for r in vip_records:
            by_lang.setdefault(r.get("lang", "en"), []).append(r["token"])

        push_results = []
        for lang, tokens in by_lang.items():
            msg = build_signal_message(sig, lang=lang)
            res = await send_push_batch(
                tokens=tokens,
                title=msg["title"],
                body=msg["body"],
                data={"type": "vip_signal", "signal_id": sig["id"], "symbol": sig["symbol"]},
                channel_id="vip-signals",
            )
            push_results.append({"lang": lang, **res})
        delivered.append({"signal": sig, "push": push_results})

    # Trim history
    if len(_signals_history) > _SIGNALS_MAX:
        del _signals_history[_SIGNALS_MAX:]

    return {
        "ok": True,
        "new_signals": len(new_signals),
        "vip_recipients": sum(1 for r in _push_registry.values() if r.get("vip") or r.get("is_admin")),
        "delivered": delivered,
    }


@app.get("/api/scan-signals/debug")
async def scan_signals_debug(admin_email: str):
    """Run a dry scan (below threshold too) for admin visibility."""
    if (admin_email or "").strip().lower() != ADMIN_EMAIL:
        raise HTTPException(status_code=403, detail="Not authorized")
    return {"scan": debug_scan()}


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


# ============================================================
# ===============  SISTEMA DE REFERIDOS  =====================
# ============================================================
#
# Códigos de afiliado para creadores de contenido.
# Cada código da 10% de descuento al usuario y genera
# 20% de comisión para el creador (sobre el 1er pago).
# Storage in-memory (migrar a Firestore cuando escale).
# ------------------------------------------------------------

REFERRAL_DISCOUNT_PCT = 10   # % descuento para el usuario
REFERRAL_COMMISSION_PCT = 20  # % comisión para el creador (1er pago)

# code -> { code, creatorEmail, creatorName, discount, commission,
#           uses, active, conversions: [{userId, plan, amount, ts}] }
_referral_codes: Dict[str, Dict] = {}


class ReferralCreateRequest(BaseModel):
    admin_email: str
    code: str            # e.g. "CRYPTODAN" — uppercase recommended
    creator_email: str
    creator_name: str


class ReferralValidateRequest(BaseModel):
    code: str


class ReferralConvertRequest(BaseModel):
    code: str
    user_id: str
    user_email: str
    plan: str
    billing: str       # "monthly" | "annual"
    amount: float      # final charged amount (after discount)


def _normalize_code(code: str) -> str:
    return code.strip().upper()


@app.post("/api/referral/create")
async def referral_create(req: ReferralCreateRequest):
    """Admin creates a referral code for a content creator."""
    if (req.admin_email or "").strip().lower() != ADMIN_EMAIL:
        raise HTTPException(status_code=403, detail="Not authorized")
    code = _normalize_code(req.code)
    if not code or len(code) < 3:
        raise HTTPException(status_code=400, detail="Code too short (min 3 chars)")
    if code in _referral_codes:
        raise HTTPException(status_code=409, detail="Code already exists")
    _referral_codes[code] = {
        "code": code,
        "creator_email": req.creator_email.lower().strip(),
        "creator_name": req.creator_name.strip(),
        "discount_pct": REFERRAL_DISCOUNT_PCT,
        "commission_pct": REFERRAL_COMMISSION_PCT,
        "uses": 0,
        "active": True,
        "created_at": int(_time.time()),
        "conversions": [],
    }
    return {"ok": True, "code": code}


@app.post("/api/referral/validate")
async def referral_validate(req: ReferralValidateRequest):
    """Check if a referral code is valid. Called before checkout."""
    code = _normalize_code(req.code)
    rec = _referral_codes.get(code)
    if not rec or not rec.get("active"):
        raise HTTPException(status_code=404, detail="Código no válido o inactivo")
    return {
        "valid": True,
        "code": code,
        "creator_name": rec["creator_name"],
        "discount_pct": rec["discount_pct"],
        "uses": rec.get("uses", 0),
        "message": f"¡Código válido! {rec['discount_pct']}% de descuento aplicado",
    }


@app.post("/api/referral/convert")
async def referral_convert(req: ReferralConvertRequest):
    """Record a successful conversion. Call after Stripe checkout completes."""
    code = _normalize_code(req.code)
    rec = _referral_codes.get(code)
    if not rec:
        return {"ok": False, "error": "Code not found"}
    commission = round(req.amount * rec["commission_pct"] / 100, 2)
    rec["uses"] += 1
    rec["conversions"].append({
        "user_id": req.user_id,
        "user_email": req.user_email,
        "plan": req.plan,
        "billing": req.billing,
        "amount": req.amount,
        "commission": commission,
        "ts": int(_time.time()),
    })
    return {
        "ok": True,
        "commission_earned": commission,
        "total_uses": rec["uses"],
    }


@app.get("/api/referral/by-email")
async def referral_by_email(creator_email: str):
    """Find a referral code by creator email. Used by the app to load creator dashboard."""
    email = (creator_email or "").strip().lower()
    if not email:
        raise HTTPException(status_code=400, detail="creator_email required")
    for code, rec in _referral_codes.items():
        if rec.get("creator_email") == email:
            total_commission = sum(cv.get("commission", 0) for cv in rec["conversions"])
            return {
                "code": code,
                "creator_name": rec["creator_name"],
                "uses": rec["uses"],
                "active": rec["active"],
                "discount_pct": rec["discount_pct"],
                "commission_pct": rec["commission_pct"],
                "total_commission_usd": round(total_commission, 2),
                "conversions": rec["conversions"],
            }
    raise HTTPException(status_code=404, detail="No code found for this email")


@app.get("/api/referral/stats/{code}")
async def referral_stats(code: str, creator_email: str):
    """Creator checks their own code stats."""
    c = _normalize_code(code)
    rec = _referral_codes.get(c)
    if not rec:
        raise HTTPException(status_code=404, detail="Code not found")
    if rec["creator_email"] != creator_email.lower().strip() and \
       creator_email.lower().strip() != ADMIN_EMAIL:
        raise HTTPException(status_code=403, detail="Not authorized")
    total_commission = sum(cv.get("commission", 0) for cv in rec["conversions"])
    return {
        "code": c,
        "creator_name": rec["creator_name"],
        "uses": rec["uses"],
        "active": rec["active"],
        "discount_pct": rec["discount_pct"],
        "commission_pct": rec["commission_pct"],
        "total_commission_usd": round(total_commission, 2),
        "conversions": rec["conversions"],
    }


@app.get("/api/admin/referrals")
async def admin_referrals(admin_email: str):
    """Admin sees all referral codes and totals."""
    if (admin_email or "").strip().lower() != ADMIN_EMAIL:
        raise HTTPException(status_code=403, detail="Not authorized")
    result = []
    for code, rec in _referral_codes.items():
        total = sum(cv.get("commission", 0) for cv in rec["conversions"])
        result.append({
            "code": code,
            "creator_name": rec["creator_name"],
            "creator_email": rec["creator_email"],
            "uses": rec["uses"],
            "active": rec["active"],
            "total_commission_usd": round(total, 2),
        })
    result.sort(key=lambda x: x["uses"], reverse=True)
    return {"referrals": result, "count": len(result)}


@app.post("/api/admin/referral/toggle")
async def admin_referral_toggle(admin_email: str, code: str, active: bool):
    """Admin activates/deactivates a referral code."""
    if (admin_email or "").strip().lower() != ADMIN_EMAIL:
        raise HTTPException(status_code=403, detail="Not authorized")
    c = _normalize_code(code)
    rec = _referral_codes.get(c)
    if not rec:
        raise HTTPException(status_code=404, detail="Code not found")
    rec["active"] = active
    return {"ok": True, "code": c, "active": active}
