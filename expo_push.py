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
    signal expected keys: symbol, direction, entry, sl, tp1, tp2, score, leverage, risk_pct
    """
    direction = signal.get("direction", "long").upper()
    symbol = signal.get("symbol", "")
    entry = signal.get("entry")
    sl = signal.get("sl")
    tp1 = signal.get("tp1")
    leverage = signal.get("leverage", 3)
    score = signal.get("score", 0)

    emoji = "🟢" if direction == "LONG" else "🔴"

    templates = {
        "es": {
            "title": f"{emoji} Señal VIP: {direction} {symbol}",
            "body": f"Entrada ${entry} · SL ${sl} · TP ${tp1} · x{leverage} · Score {score}/10",
        },
        "en": {
            "title": f"{emoji} VIP Signal: {direction} {symbol}",
            "body": f"Entry ${entry} · SL ${sl} · TP ${tp1} · x{leverage} · Score {score}/10",
        },
        "pt": {
            "title": f"{emoji} Sinal VIP: {direction} {symbol}",
            "body": f"Entrada ${entry} · SL ${sl} · TP ${tp1} · x{leverage} · Score {score}/10",
        },
        "fr": {
            "title": f"{emoji} Signal VIP : {direction} {symbol}",
            "body": f"Entrée ${entry} · SL ${sl} · TP ${tp1} · x{leverage} · Score {score}/10",
        },
        "de": {
            "title": f"{emoji} VIP-Signal: {direction} {symbol}",
            "body": f"Einstieg ${entry} · SL ${sl} · TP ${tp1} · x{leverage} · Score {score}/10",
        },
        "it": {
            "title": f"{emoji} Segnale VIP: {direction} {symbol}",
            "body": f"Entrata ${entry} · SL ${sl} · TP ${tp1} · x{leverage} · Score {score}/10",
        },
    }
    return templates.get(lang, templates["en"])
