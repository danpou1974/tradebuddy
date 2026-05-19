"""
expo_push.py
============
Thin wrapper around Expo's Push Notification API (free, no key).
https://docs.expo.dev/push-notifications/sending-notifications/
"""

from __future__ import annotations
from typing import List, Dict, Optional
import httpx

EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"


async def send_push_batch(
    tokens: List[str],
    title: str,
    body: str,
    data: Optional[Dict] = None,
    sound: str = "default",
    priority: str = "high",
    channel_id: str = "vip-signals",
) -> Dict:
    """
    Send the same notification to many Expo tokens at once.
    Tokens must look like "ExponentPushToken[xxxxxxxxxxx]".
    """
    # Filter valid-looking tokens
    valid = [t for t in tokens if isinstance(t, str) and t.startswith("ExponentPushToken")]
    if not valid:
        return {"ok": False, "error": "no valid tokens", "sent": 0}

    messages = [
        {
            "to": t,
            "title": title,
            "body": body,
            "data": data or {},
            "sound": sound,
            "priority": priority,
            "channelId": channel_id,
        }
        for t in valid
    ]

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                EXPO_PUSH_URL,
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip, deflate",
                    "Content-Type": "application/json",
                },
                json=messages,
            )
        return {
            "ok": resp.status_code == 200,
            "status": resp.status_code,
            "sent": len(valid),
            "response": resp.json() if resp.status_code == 200 else resp.text,
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "sent": 0}


def build_signal_message(signal: Dict, lang: str = "en") -> Dict:
    """
    Build localized title + body for a VIP signal push.
    signal expected keys: symbol, direction, entry, sl, tp1, score, leverage, strategy
    """
    direction = signal.get("direction", "long").upper()
    symbol    = signal.get("symbol", "")
    entry     = signal.get("entry")
    sl        = signal.get("sl")
    tp1       = signal.get("tp1")
    leverage  = signal.get("leverage", 3)
    score     = signal.get("score", 0)
    strategy  = signal.get("strategy", "trend")

    emoji = "🟢" if direction == "LONG" else "🔴"

    # Etiqueta corta de estrategia para el push (sin ocupar demasiado espacio)
    _strat_labels = {
        "mean_reversion": {"es": "🔄 MR · WR 48-50%", "en": "🔄 MR · WR 48-50%",
                           "pt": "🔄 MR · WR 48-50%", "fr": "🔄 MR · WR 48-50%",
                           "de": "🔄 MR · WR 48-50%", "it": "🔄 MR · WR 48-50%"},
        "rsi_pullback":   {"es": "📐 Pullback · WR 44-62%", "en": "📐 Pullback · WR 44-62%",
                           "pt": "📐 Pullback · WR 44-62%", "fr": "📐 Pullback · WR 44-62%",
                           "de": "📐 Pullback · WR 44-62%", "it": "📐 Pullback · WR 44-62%"},
        "trend":          {"es": "📊 Trend · WR ~45%",      "en": "📊 Trend · WR ~45%",
                           "pt": "📊 Trend · WR ~45%",      "fr": "📊 Trend · WR ~45%",
                           "de": "📊 Trend · WR ~45%",      "it": "📊 Trend · WR ~45%"},
    }
    strat_tag = _strat_labels.get(strategy, {}).get(lang, strategy.upper())

    templates = {
        "es": {
            "title": f"{emoji} Señal VIP: {direction} {symbol}",
            "body":  f"Entrada ${entry} · SL ${sl} · TP ${tp1} · x{leverage} · Score {score}/10\n{strat_tag}",
        },
        "en": {
            "title": f"{emoji} VIP Signal: {direction} {symbol}",
            "body":  f"Entry ${entry} · SL ${sl} · TP ${tp1} · x{leverage} · Score {score}/10\n{strat_tag}",
        },
        "pt": {
            "title": f"{emoji} Sinal VIP: {direction} {symbol}",
            "body":  f"Entrada ${entry} · SL ${sl} · TP ${tp1} · x{leverage} · Score {score}/10\n{strat_tag}",
        },
        "fr": {
            "title": f"{emoji} Signal VIP : {direction} {symbol}",
            "body":  f"Entrée ${entry} · SL ${sl} · TP ${tp1} · x{leverage} · Score {score}/10\n{strat_tag}",
        },
        "de": {
            "title": f"{emoji} VIP-Signal: {direction} {symbol}",
            "body":  f"Einstieg ${entry} · SL ${sl} · TP ${tp1} · x{leverage} · Score {score}/10\n{strat_tag}",
        },
        "it": {
            "title": f"{emoji} Segnale VIP: {direction} {symbol}",
            "body":  f"Entrata ${entry} · SL ${sl} · TP ${tp1} · x{leverage} · Score {score}/10\n{strat_tag}",
        },
    }
    return templates.get(lang, templates["en"])
