"""alerts.py — Sistema de alertas de cambio de régimen."""
import os
import pandas as pd
from datetime import datetime

class RegimeAlertSystem:
    def __init__(self, log_path="logs/alerts.csv"):
        self.log_path = log_path
        self.last_regime: dict = {}
        self.alert_history: list = []

    def check(self, timeframe, symbol, current_info):
        new = current_info["regime"]
        old = self.last_regime.get(f"{symbol}_{timeframe}")
        alert = None
        if old and old != new:
            alert = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                     "symbol": symbol, "timeframe": timeframe,
                     "from": old, "to": new,
                     "confidence": round(current_info["confidence"] * 100, 1)}
            self.alert_history.append(alert)
            self._save(alert)
        self.last_regime[f"{symbol}_{timeframe}"] = new
        return alert

    def check_composite(self, symbol, composite):
        alerts = []
        for tf, info in composite["breakdown"].items():
            a = self.check(tf, symbol, info)
            if a: alerts.append(a)
        a = self.check("GLOBAL", symbol, {
            "regime": composite["composite_regime"],
            "confidence": composite["composite_confidence"]})
        if a: alerts.append(a)
        return alerts

    def _save(self, alert):
        try:
            log_dir = os.path.dirname(self.log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            row  = pd.DataFrame([alert])
            mode = "a" if os.path.exists(self.log_path) else "w"
            row.to_csv(self.log_path, mode=mode,
                       header=not os.path.exists(self.log_path), index=False)
        except Exception:
            pass

    def get_history(self):
        if not self.alert_history:
            return pd.DataFrame(columns=["timestamp","symbol","timeframe","from","to","confidence"])
        return pd.DataFrame(self.alert_history)
