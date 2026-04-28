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

import asyncio
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(application):
    """Ejecuta un scan al iniciar para poblar _signals_history desde el arranque."""
    async def _startup_scan():
        await asyncio.sleep(10)   # espera a que el servidor esté listo
        try:
            loop = asyncio.get_event_loop()
            new_sigs = await loop.run_in_executor(None, scan_and_emit)
            for s in new_sigs:
                s["id"] = f"{s['symbol'].replace('/', '')}_{s['generated_at']}"
                _signals_history.insert(0, s)
            if len(_signals_history) > _SIGNALS_MAX:
                del _signals_history[_SIGNALS_MAX:]
            if new_sigs:
                _save_signals_cache(_signals_history)
            cached = _load_signals_cache()
            if not new_sigs and cached:
                print(f"[startup] sin señales nuevas — {len(cached)} señal(es) cargadas del cache")
            else:
                print(f"[startup] scan OK — {len(new_sigs)} señal(es) nueva(s)")
        except Exception as e:
            print(f"[startup] scan error: {e}")
    asyncio.create_task(_startup_scan())
    yield

app = FastAPI(title="TradeBuddy API", version="1.0.0", lifespan=lifespan)

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
    userPlan: str = "starter"  # "starter" | "trader" | "elite"


class ChatResponse(BaseModel):
    text: str
    title: Optional[str] = None


# ===== System prompt builder =====

def build_system_prompt(market_data: List[MarketItem], user_lang: str, user_plan: str = "starter") -> str:
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

    # Límites de respuesta según plan
    is_premium = user_plan in ("trader", "elite")
    max_words = 250 if is_premium else 100
    detail_level = (
        "Puedes dar análisis detallados de hasta 250 palabras cuando sea relevante."
        if is_premium else
        "Responde en máximo 2-3 oraciones directas (≤100 palabras). Si el usuario necesita más profundidad, sugiérele que actualice a plan Trader o Elite."
    )

    market_context = ""
    if market_data:
        market_context = "\n\nDATOS DE MERCADO:\n"
        for m in market_data[:15]:
            change_sign = "+" if m.change24h > 0 else ""
            market_context += (
                f"{m.symbol}: ${m.price} ({change_sign}{m.change24h}%) "
                f"— {m.regime}, Soporte ${m.support or 'N/A'}, Resistencia ${m.resistance or 'N/A'}\n"
            )

    return f"""Eres TradeBuddy AI, asistente de análisis de mercados. Experto en crypto, forex, acciones y trading.

REGLAS:
1. {lang_instruction}
2. {detail_level}
3. Usa datos reales del mercado cuando estén disponibles.
4. NO inventes datos — si no tienes info, dilo en una línea.
5. Incluye siempre al final: "⚠️ Solo educativo, no es consejo financiero."
6. Sé directo, sin relleno, sin explicaciones largas de metodología interna.
7. No menciones tecnología interna del sistema (modelos, algoritmos, etc.).
8. Usa máximo 1-2 emojis por respuesta.
{market_context}"""


# ===== Endpoints =====

@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "TradeBuddy API"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")

    system_prompt = build_system_prompt(req.marketData, req.lang, req.userPlan)

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
                    "temperature": 0.65,
                    "max_tokens": 400 if req.userPlan in ("trader", "elite") else 180,
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

# ── Email config (Gmail SMTP) ─────────────────────────────────────────────────
GMAIL_USER     = os.environ.get("GMAIL_USER", "")          # ej: tradebuddy.signals@gmail.com
GMAIL_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")  # App Password de Google


def _send_signal_emails(sig: Dict) -> None:
    """Envía email con todos los detalles de la señal a los VIP emails."""
    import smtplib, ssl
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    if not GMAIL_USER or not GMAIL_PASSWORD:
        print("[email] GMAIL_USER / GMAIL_APP_PASSWORD no configurados — email omitido")
        return

    symbol    = sig.get("symbol", "")
    direction = (sig.get("direction") or "").upper()
    entry     = sig.get("entry", 0)
    tp1       = sig.get("tp1", 0)
    tp2       = sig.get("tp2", tp1)
    sl        = sig.get("sl", 0)
    leverage  = sig.get("leverage", 1)
    score     = sig.get("score", 0)
    regime    = sig.get("regime", "")
    strategy  = sig.get("strategy", "trend")
    reasons   = sig.get("reasons", [])

    dir_emoji = "📈" if direction == "LONG" else "📉"
    dir_color = "#00C896" if direction == "LONG" else "#ef4444"
    rr        = round(abs(float(tp1) - float(entry)) / (abs(float(entry) - float(sl)) + 1e-10), 2) if entry and sl else 0

    reasons_html = "".join(
        f'<li style="margin-bottom:6px;color:#ccc;">{r}</li>'
        for r in (reasons if isinstance(reasons, list) else [str(reasons)])
    )

    html = f"""
<!DOCTYPE html><html><body style="background:#0d1117;color:#e6edf3;font-family:Arial,sans-serif;margin:0;padding:24px;">
<div style="max-width:520px;margin:0 auto;background:#161b22;border-radius:12px;overflow:hidden;border:1px solid #30363d;">

  <!-- Header -->
  <div style="background:#1c2128;padding:20px 24px;border-bottom:1px solid #30363d;">
    <div style="font-size:11px;color:#8b949e;letter-spacing:2px;margin-bottom:4px;">⚡ BUDDY SIGNALS VIP</div>
    <div style="font-size:22px;font-weight:800;color:#e6edf3;">
      {dir_emoji} {symbol} &nbsp;
      <span style="color:{dir_color};font-size:18px;">{direction}</span>
    </div>
    <div style="margin-top:8px;">
      <span style="background:{dir_color}22;color:{dir_color};border:1px solid {dir_color}44;
                   padding:3px 10px;border-radius:20px;font-size:12px;font-weight:700;">
        Score {score}/10
      </span>
      &nbsp;
      <span style="background:#30363d;color:#8b949e;padding:3px 10px;border-radius:20px;font-size:12px;">
        {strategy.upper()} · {regime}
      </span>
    </div>
  </div>

  <!-- Precios -->
  <div style="padding:20px 24px;">
    <table width="100%" cellpadding="0" cellspacing="0">
      <tr>
        <td style="text-align:center;padding:12px;background:#0d1117;border-radius:8px;">
          <div style="font-size:10px;color:#8b949e;margin-bottom:4px;">ENTRADA</div>
          <div style="font-size:20px;font-weight:800;color:#e6edf3;">${entry}</div>
        </td>
        <td width="8"></td>
        <td style="text-align:center;padding:12px;background:#0d1117;border-radius:8px;">
          <div style="font-size:10px;color:#8b949e;margin-bottom:4px;">STOP LOSS</div>
          <div style="font-size:20px;font-weight:800;color:#ef4444;">${sl}</div>
        </td>
      </tr>
      <tr><td height="8" colspan="3"></td></tr>
      <tr>
        <td style="text-align:center;padding:12px;background:#0d1117;border-radius:8px;">
          <div style="font-size:10px;color:#8b949e;margin-bottom:4px;">TP1</div>
          <div style="font-size:20px;font-weight:800;color:#00C896;">${tp1}</div>
        </td>
        <td width="8"></td>
        <td style="text-align:center;padding:12px;background:#0d1117;border-radius:8px;">
          <div style="font-size:10px;color:#8b949e;margin-bottom:4px;">TP2</div>
          <div style="font-size:20px;font-weight:800;color:#00C896;">${tp2}</div>
        </td>
      </tr>
    </table>

    <!-- Leverage + RR -->
    <div style="display:flex;gap:8px;margin-top:12px;">
      <div style="flex:1;text-align:center;padding:10px;background:#0d1117;border-radius:8px;">
        <div style="font-size:10px;color:#8b949e;margin-bottom:2px;">LEVERAGE</div>
        <div style="font-size:18px;font-weight:800;color:#FFD600;">{leverage}x</div>
      </div>
      <div style="flex:1;text-align:center;padding:10px;background:#0d1117;border-radius:8px;">
        <div style="font-size:10px;color:#8b949e;margin-bottom:2px;">RATIO R:R</div>
        <div style="font-size:18px;font-weight:800;color:#e6edf3;">1:{rr}</div>
      </div>
    </div>
  </div>

  <!-- Razones -->
  <div style="padding:0 24px 20px;">
    <div style="font-size:12px;color:#8b949e;font-weight:700;letter-spacing:1px;margin-bottom:10px;">
      💡 POR QUÉ ENTRAR
    </div>
    <ul style="margin:0;padding-left:18px;line-height:1.8;">
      {reasons_html}
    </ul>
  </div>

  <!-- Footer -->
  <div style="background:#0d1117;padding:14px 24px;border-top:1px solid #30363d;
               font-size:11px;color:#484f58;text-align:center;">
    Señal educativa — ejecutá a tu propio criterio y riesgo · TradeBuddy VIP
  </div>
</div>
</body></html>
"""

    subject = f"{dir_emoji} Señal VIP: {symbol} {direction} — Entrada ${entry} · {leverage}x"

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(GMAIL_USER, GMAIL_PASSWORD)
            for recipient in VIP_EMAILS:
                msg = MIMEMultipart("alternative")
                msg["Subject"] = subject
                msg["From"]    = f"TradeBuddy Signals <{GMAIL_USER}>"
                msg["To"]      = recipient
                msg.attach(MIMEText(html, "html"))
                server.sendmail(GMAIL_USER, recipient, msg.as_string())
                print(f"[email] enviado a {recipient}")
    except Exception as e:
        print(f"[email] error SMTP: {e}")

# Whitelist VIP — acceso manual (no pago). Espejo del frontend vipWhitelist.js.
# Persiste en código: sobrevive reinicios de Render sin perder acceso VIP.
VIP_EMAILS: set = {
    "danpou1974@gmail.com",
    "scimelorena@hotmail.com",
}

# user_id -> {token, lang, vip, email}
_push_registry: Dict[str, Dict] = {}

# ── Señales — persistencia en archivo (sobrevive sleep/wake de Render) ─────────
_SIGNALS_CACHE_FILE = "/tmp/tradebuddy_signals.json"
_SIGNALS_MAX = 10   # solo las últimas 10

def _load_signals_cache() -> List[Dict]:
    """Carga las últimas señales guardadas en disco."""
    try:
        with open(_SIGNALS_CACHE_FILE, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data[:_SIGNALS_MAX]
    except Exception:
        pass
    return []

def _save_signals_cache(signals: List[Dict]) -> None:
    """Persiste las últimas señales en disco."""
    try:
        with open(_SIGNALS_CACHE_FILE, "w") as f:
            json.dump(signals[:_SIGNALS_MAX], f)
    except Exception as e:
        print(f"[cache] error guardando señales: {e}")

_signals_history: List[Dict] = _load_signals_cache()   # carga al iniciar
# user_id -> signal_id -> {"action": "took"|"passed", "ts": ...}
_signal_tracking: Dict[str, Dict[str, Dict]] = {}
# user_id -> list of price alert dicts
_alerts_registry: Dict[str, List[Dict]] = {}
# Error log (últimos 200 errores)
_error_log: List[Dict] = []
_ERROR_LOG_MAX = 200

# ── Último resultado de debug scan (en memoria) ──────────────────────────────
import time as _time_module
_last_scan_status: Dict = {
    "status": "never_run",   # "never_run" | "running" | "done" | "error"
    "started_at": None,
    "finished_at": None,
    "duration_s": None,
    "results": [],
    "error": None,
}
_scan_lock = False   # evita scans concurrentes


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
    clean_email = (req.email or "").strip().lower()
    is_admin    = clean_email == ADMIN_EMAIL
    is_vip_email = clean_email in VIP_EMAILS   # server-side whitelist (persiste en código)
    _push_registry[req.user_id] = {
        "token": req.token,
        "lang": req.lang,
        "vip": bool(req.vip) or is_admin or is_vip_email,
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
async def signals_list(user_id: Optional[str] = None, email: Optional[str] = None, limit: int = 50):
    """Return recent signals. Only VIP users get the full feed."""
    rec = _push_registry.get(user_id or "")
    rec_email     = (rec.get("email", "") if rec else "").strip().lower()
    direct_email  = (email or "").strip().lower()   # email enviado directo desde el frontend
    is_vip = bool(
        (rec and (rec.get("vip") or rec.get("is_admin")))
        or rec_email    in VIP_EMAILS   # email en registry
        or direct_email in VIP_EMAILS   # email enviado directo (sobrevive restart sin push token)
    )
    if not is_vip:
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


# ===== Error Reporting =======================================================

import time as _time_module

class ErrorReport(BaseModel):
    message: str
    stack: Optional[str] = None
    context: Optional[str] = None      # pantalla / acción
    user_id: Optional[str] = None
    app_version: Optional[str] = None
    platform: Optional[str] = None     # "ios" | "android"


@app.post("/api/errors/report")
async def report_error(req: ErrorReport):
    """Recibe errores del cliente y los guarda para revisión del admin."""
    entry = {
        "ts": int(_time_module.time()),
        "message": req.message[:500],        # truncar para evitar abuse
        "stack": (req.stack or "")[:2000],
        "context": req.context,
        "user_id": req.user_id,
        "app_version": req.app_version,
        "platform": req.platform,
    }
    _error_log.insert(0, entry)
    if len(_error_log) > _ERROR_LOG_MAX:
        del _error_log[_ERROR_LOG_MAX:]
    # No levantamos excepción — el cliente no debe interrumpirse por un error de reporte
    return {"ok": True}


@app.get("/api/errors")
async def get_errors(admin_email: str, limit: int = 50):
    """Solo el admin puede ver los errores registrados."""
    if (admin_email or "").strip().lower() != ADMIN_EMAIL:
        raise HTTPException(status_code=403, detail="Not authorized")
    return {
        "total": len(_error_log),
        "errors": _error_log[:limit],
    }


# ===== Price Alerts ==========================================================

class AlertItem(BaseModel):
    id: str
    symbol: str          # "BTC/USDT" o "BTCUSDT"
    targetPrice: float
    condition: str       # "above" | "below"
    category: str = "crypto"
    active: bool = True

class AlertsSyncRequest(BaseModel):
    user_id: str
    alerts: List[AlertItem]


@app.post("/api/alerts/sync")
async def sync_alerts(req: AlertsSyncRequest):
    """El frontend sincroniza su lista completa de alertas al backend."""
    _alerts_registry[req.user_id] = [a.dict() for a in req.alerts if a.active]
    return {"ok": True, "stored": len(_alerts_registry[req.user_id])}


@app.get("/api/alerts/{user_id}")
async def get_user_alerts(user_id: str):
    """Retorna las alertas activas de un usuario (útil para sincronía entre dispositivos)."""
    return {"alerts": _alerts_registry.get(user_id, [])}


async def fetch_binance_prices() -> Dict[str, float]:
    """Obtiene todos los precios USDT de Binance en un solo request."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get("https://api.binance.com/api/v3/ticker/price")
            if resp.status_code != 200:
                return {}
            prices: Dict[str, float] = {}
            for item in resp.json():
                sym = item["symbol"]
                price = float(item["price"])
                if sym.endswith("USDT"):
                    base = sym[:-4]
                    prices[f"{base}/USDT"] = price   # formato app: "BTC/USDT"
                    prices[sym] = price              # formato Binance: "BTCUSDT"
            return prices
    except Exception as e:
        print(f"[alerts] Binance price fetch error: {e}")
        return {}


async def check_and_send_price_alerts(prices: Dict[str, float]) -> Dict:
    """
    Compara alertas registradas contra precios actuales.
    Envía push a los usuarios cuyas alertas se dispararon y las elimina del registro.
    """
    if not prices:
        return {"checked": 0, "triggered": 0}

    total_checked = 0
    total_triggered = 0

    for user_id, alerts in list(_alerts_registry.items()):
        if not alerts:
            continue

        remaining = []
        user_rec = _push_registry.get(user_id)
        total_checked += len(alerts)

        for alert in alerts:
            symbol = alert.get("symbol", "")
            current_price = prices.get(symbol)

            if current_price is None:
                remaining.append(alert)
                continue

            target = float(alert.get("targetPrice", 0))
            condition = alert.get("condition", "above")
            hit = (condition == "above" and current_price >= target) or \
                  (condition == "below" and current_price <= target)

            if hit:
                total_triggered += 1
                # Solo enviar push si el usuario tiene token registrado
                if user_rec and user_rec.get("token"):
                    arrow = "📈" if condition == "above" else "📉"
                    verb  = "superó" if condition == "above" else "cayó bajo"
                    try:
                        await send_push_batch(
                            tokens=[user_rec["token"]],
                            title=f"{arrow} Alerta: {symbol}",
                            body=f"{symbol} {verb} ${target:,.4g}. Precio actual: ${current_price:,.4g}",
                            data={
                                "type": "price-alert",
                                "symbol": symbol,
                                "targetPrice": target,
                                "currentPrice": current_price,
                            },
                            channel_id="price-alerts",
                        )
                    except Exception as e:
                        print(f"[alerts] push error for {user_id}: {e}")
            else:
                remaining.append(alert)

        # Actualizar el registro sin las alertas disparadas
        _alerts_registry[user_id] = remaining

    return {"checked": total_checked, "triggered": total_triggered}


@app.post("/api/alerts/check")
async def manual_alert_check(x_scan_secret: Optional[str] = Header(default=None)):
    """
    Endpoint manual para revisar alertas (se puede llamar desde cron-job.org).
    Puede ejecutarse con más frecuencia que scan-signals (cada 5 min).
    """
    if x_scan_secret != SCAN_SECRET:
        raise HTTPException(status_code=403, detail="Bad secret")
    prices = await fetch_binance_prices()
    result = await check_and_send_price_alerts(prices)
    return {"ok": True, **result, "users_with_alerts": len(_alerts_registry)}


async def _fetch_prices_for_signals(symbols: list) -> Dict[str, float]:
    """Obtiene precios via KuCoin REST (sin restricciones de Render IP)."""
    prices: Dict[str, float] = {}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            for sym in symbols:
                kucoin_sym = sym.replace("/", "-")
                try:
                    r = await client.get(
                        f"https://api.kucoin.com/api/v1/market/stats?symbol={kucoin_sym}")
                    data = r.json()
                    if str(data.get("code")) == "200000":
                        prices[sym] = float(data["data"].get("last", 0) or 0)
                except Exception:
                    pass
    except Exception as e:
        print(f"[outcomes] price fetch error: {e}")
    return prices


def _check_signal_outcomes(prices: Dict[str, float]) -> int:
    """
    Revisa señales activas y marca automáticamente las que alcanzaron TP1 o SL.
    Devuelve el número de señales cerradas.
    """
    import time as _t
    closed = 0
    changed = False
    for sig in _signals_history:
        if sig.get("outcome"):        # ya tiene resultado
            continue
        sym    = sig.get("symbol", "")
        price  = prices.get(sym)
        if not price:
            continue

        entry     = float(sig.get("entry", 0) or 0)
        sl        = float(sig.get("sl",    0) or 0)
        tp1       = float(sig.get("tp1",   0) or 0)
        direction = sig.get("direction", "long")
        leverage  = sig.get("leverage",  1)

        if not entry:
            continue

        is_long = direction == "long"
        hit_tp  = (is_long  and price >= tp1) or (not is_long and price <= tp1)
        hit_sl  = (is_long  and price <= sl)  or (not is_long and price >= sl)

        if hit_tp or hit_sl:
            outcome    = "tp1" if hit_tp else "sl"
            exit_price = tp1 if hit_tp else sl
            raw_pct    = ((exit_price - entry) / entry * 100) if is_long \
                         else ((entry - exit_price) / entry * 100)
            pnl_pct    = round(raw_pct * leverage, 2)
            sig["outcome"]     = outcome
            sig["exit_price"]  = exit_price
            sig["exit_at"]     = int(_t.time())
            sig["pnl_pct"]     = pnl_pct
            closed  += 1
            changed  = True
            print(f"[outcomes] {sym} → {outcome.upper()} | PnL: {pnl_pct:+.1f}%")

    if changed:
        _save_signals_cache(_signals_history)
    return closed


@app.post("/api/scan-signals")
async def scan_signals(x_scan_secret: Optional[str] = Header(default=None)):
    """
    Triggered by cron-job.org every 15 min.
    Devuelve 200 OK inmediatamente y corre el scan en background.
    Así nunca hace timeout en cron-job.org (límite 30s).
    """
    if x_scan_secret != SCAN_SECRET:
        raise HTTPException(status_code=403, detail="Bad secret")

    async def _run_scan():
        try:
            loop = asyncio.get_event_loop()
            new_signals = await loop.run_in_executor(None, scan_and_emit)

            for sig in new_signals:
                sig["id"] = f"{sig['symbol'].replace('/', '')}_{sig['generated_at']}"
                _signals_history.insert(0, sig)

                # Push notifications por idioma
                vip_records = [r for r in _push_registry.values() if r.get("vip") or r.get("is_admin")]
                by_lang: Dict[str, List[str]] = {}
                for r in vip_records:
                    by_lang.setdefault(r.get("lang", "en"), []).append(r["token"])
                for lang, tokens in by_lang.items():
                    msg = build_signal_message(sig, lang=lang)
                    await send_push_batch(
                        tokens=tokens,
                        title=msg["title"],
                        body=msg["body"],
                        data={"type": "vip_signal", "signal_id": sig["id"], "symbol": sig["symbol"]},
                        channel_id="vip-signals",
                    )

            # Trim + persistir
            if len(_signals_history) > _SIGNALS_MAX:
                del _signals_history[_SIGNALS_MAX:]
            if new_signals:
                _save_signals_cache(_signals_history)
                print(f"[cron] {len(new_signals)} señal(es) nueva(s) guardadas")
                # Enviar email a todos los VIP
                for sig in new_signals:
                    try:
                        await asyncio.get_event_loop().run_in_executor(
                            None, _send_signal_emails, sig)
                    except Exception as e:
                        print(f"[email] error: {e}")

            # Cierre automático TP/SL
            active_syms = [s["symbol"] for s in _signals_history if not s.get("outcome") and s.get("symbol")]
            if active_syms:
                prices = await _fetch_prices_for_signals(list(set(active_syms)))
                closed = _check_signal_outcomes(prices)
                if closed:
                    print(f"[cron] {closed} señal(es) cerrada(s) automaticamente")
                # Alertas de precio
                if _alerts_registry:
                    await check_and_send_price_alerts(prices)

        except Exception as e:
            print(f"[cron] scan error: {e}")

    asyncio.create_task(_run_scan())
    return {"ok": True, "status": "scan iniciado en background"}


@app.get("/api/scan-signals/debug")
async def scan_signals_debug(admin_email: str):
    """
    Lanza un debug scan en background y devuelve inmediatamente.
    Consultá /api/scan-status para ver los resultados cuando terminen.
    """
    global _scan_lock
    if (admin_email or "").strip().lower() != ADMIN_EMAIL:
        raise HTTPException(status_code=403, detail="Not authorized")

    if _scan_lock:
        return {"status": "already_running", "message": "Scan en progreso, consultá /api/scan-status"}

    async def _run_debug():
        global _scan_lock, _last_scan_status
        _scan_lock = True
        t0 = _time_module.time()
        _last_scan_status.update({"status": "running", "started_at": t0,
                                   "finished_at": None, "results": [], "error": None})
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, debug_scan)
            t1 = _time_module.time()
            _last_scan_status.update({
                "status": "done",
                "finished_at": t1,
                "duration_s": round(t1 - t0, 1),
                "results": results,
            })
        except Exception as e:
            _last_scan_status.update({"status": "error", "error": str(e),
                                       "finished_at": _time_module.time()})
        finally:
            _scan_lock = False

    asyncio.create_task(_run_debug())
    return {"status": "started", "message": "Scan iniciado en background. Consultá /api/scan-status en ~2 minutos."}


@app.get("/api/scan-status")
async def scan_status(admin_email: str):
    """Devuelve el resultado del último debug scan (instantáneo)."""
    if (admin_email or "").strip().lower() != ADMIN_EMAIL:
        raise HTTPException(status_code=403, detail="Not authorized")
    return _last_scan_status


@app.post("/api/signals/inject")
async def inject_signal(request: Request):
    """
    Admin endpoint: inyecta una señal manual al historial VIP.
    Útil para agregar señales reales cuando el mercado no genera setups automáticos.
    Body: { "admin_email": "...", "signal": { symbol, direction, entry, sl, tp1, leverage, reasons: [...] } }
    """
    body = await request.json()
    if (body.get("admin_email") or "").strip().lower() != ADMIN_EMAIL:
        raise HTTPException(status_code=403, detail="Not authorized")

    sig = body.get("signal", {})
    if not sig.get("symbol") or not sig.get("direction") or not sig.get("entry"):
        raise HTTPException(status_code=400, detail="signal.symbol, direction y entry son obligatorios")

    import time as _t
    sig.setdefault("generated_at", int(_t.time()))
    sig.setdefault("strategy",     "manual")
    sig.setdefault("score",        10.0)
    sig.setdefault("leverage",     sig.get("leverage", 5))
    sig.setdefault("reasons",      sig.get("reasons", ["Señal manual ingresada por el admin."]))
    sig["id"] = f"{sig['symbol'].replace('/', '')}_{sig['generated_at']}"

    _signals_history.insert(0, sig)
    if len(_signals_history) > _SIGNALS_MAX:
        del _signals_history[_SIGNALS_MAX:]
    _save_signals_cache(_signals_history)

    return {"ok": True, "signal_id": sig["id"], "total_signals": len(_signals_history)}


@app.get("/api/test-exchange")
async def test_exchange(admin_email: str):
    """Quick connectivity test — fetches 10 candles of ETH/USDT via REST directo."""
    if (admin_email or "").strip().lower() != ADMIN_EMAIL:
        raise HTTPException(status_code=403, detail="Not authorized")
    from data_fetcher import CryptoFetcher
    import time as _time

    def _test():
        fetcher = CryptoFetcher("ETH/USDT")
        results = []
        for name, fn in [("kucoin", fetcher._fetch_kucoin), ("okx", fetcher._fetch_okx)]:
            t0 = _time.time()
            try:
                raw = fn("1h", 10)
                elapsed = round(_time.time() - t0, 2)
                results.append({"exchange": name, "ok": True,
                                 "candles": len(raw), "elapsed_s": elapsed})
            except Exception as e:
                elapsed = round(_time.time() - t0, 2)
                results.append({"exchange": name, "ok": False,
                                 "error": str(e)[:120], "elapsed_s": elapsed})
        return results

    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, _test)
    return {"results": results}


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
