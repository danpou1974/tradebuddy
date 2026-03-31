"""
hmm_engine.py
=============
Motor HMM de detección de regímenes de mercado.
7 regímenes: Bajista Fuerte, Bajista, Distribución, Lateral,
             Acumulación, Alcista, Alcista Fuerte.
Soporta múltiples timeframes con votación ponderada.
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
from typing import Optional
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

TIMEFRAME_WEIGHTS = {"5m": 1, "15m": 2, "1h": 3, "4h": 4, "1d": 5}

# 7 regímenes ordenados de más bajista a más alcista
ALL_REGIMES_ORDERED = [
    "Bajista Fuerte",
    "Bajista",
    "Distribución",
    "Lateral",
    "Acumulación",
    "Alcista",
    "Alcista Fuerte",
]

# Mapa de regímenes por número de estados
N_STATE_REGIMES = {
    3: ["Bajista", "Lateral", "Alcista"],
    4: ["Bajista Fuerte", "Bajista", "Alcista", "Alcista Fuerte"],
    5: ["Bajista Fuerte", "Bajista", "Lateral", "Alcista", "Alcista Fuerte"],
    6: ["Bajista Fuerte", "Bajista", "Distribución", "Acumulación", "Alcista", "Alcista Fuerte"],
    7: ["Bajista Fuerte", "Bajista", "Distribución", "Lateral", "Acumulación", "Alcista", "Alcista Fuerte"],
}

REGIME_COLORS = {
    "Bajista Fuerte":  "#B71C1C",
    "Bajista":         "#D50000",
    "Distribución":    "#FF6D00",
    "Lateral":         "#FFD600",
    "Acumulación":     "#64DD17",
    "Alcista":         "#00C853",
    "Alcista Fuerte":  "#00E676",
    "Desconocido":     "#9E9E9E",
}


def _rsi(close: pd.Series, period=14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-10)))


def _macd_hist(close: pd.Series, fast=12, slow=26, sig=9) -> pd.Series:
    macd   = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    signal = macd.ewm(span=sig, adjust=False).mean()
    return macd - signal


def build_features(df: pd.DataFrame, vol_window=20) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    f["log_return"] = np.log(df["close"] / df["close"].shift(1))
    f["volatility"]  = f["log_return"].rolling(vol_window).std()
    f["rsi"]         = _rsi(df["close"])
    f["macd_hist"]   = _macd_hist(df["close"])
    f["vol_change"]  = np.log(df["volume"] / df["volume"].shift(1) + 1e-10)
    return f.dropna()


def _train_hmm(X, n_states, n_iter=300, tol=1e-5):
    km = KMeans(n_clusters=n_states, random_state=42, n_init=10)
    km.fit(X)
    model = GaussianHMM(
        n_components=n_states, covariance_type="full",
        n_iter=n_iter, tol=tol, random_state=42,
        init_params="stc", params="stmc",
    )
    model.means_init = km.cluster_centers_
    model.fit(X)
    return model


def evaluate_states(X, min_s=3, max_s=7):
    rows = []
    for n in range(min_s, max_s + 1):
        try:
            m  = _train_hmm(X, n)
            ll = m.score(X)
            T, d = X.shape
            k = n*d + n*d*d + n*(n-1) + (n-1)
            rows.append({"n_states": n, "log_lik": round(ll, 4),
                         "aic": round(-2*ll*T + 2*k, 2),
                         "bic": round(-2*ll*T + k*np.log(T), 2)})
        except Exception as e:
            print(f"  [WARN] n={n}: {e}")
    return pd.DataFrame(rows)


def assign_labels(model, feature_names):
    """
    Asigna nombres de régimen a cada estado del HMM.
    Ordena los estados por score de tendencia (retorno/volatilidad)
    de más bajista a más alcista y asigna los nombres correspondientes.
    """
    ri = feature_names.index("log_return")
    vi = feature_names.index("volatility")
    rets = model.means_[:, ri]
    vols = model.means_[:, vi]
    n    = model.n_components

    # Score de tendencia: positivo = alcista, negativo = bajista
    scores = rets / (np.abs(vols) + 1e-10)
    order  = np.argsort(scores)  # ascendente: más bajista primero

    # Seleccionar lista de regímenes según n_states
    regime_list = N_STATE_REGIMES.get(n, ALL_REGIMES_ORDERED[:n])

    labels = {}
    for rank, state_idx in enumerate(order):
        labels[int(state_idx)] = regime_list[rank]

    print(f"    Regimenes: {labels}")
    return labels


class RegimeHMM:
    def __init__(self, n_states=7, n_iter=300, tol=1e-5):
        self.n_states = n_states
        self.n_iter   = n_iter
        self.tol      = tol
        self.model: Optional[GaussianHMM] = None
        self.scaler   = StandardScaler()
        self.labels: dict = {}
        self.feature_names: list = []
        self.aic_bic_table = None

    def fit(self, df, auto_select=False, min_s=3, max_s=7):
        feat = build_features(df)
        if len(feat) < 50:
            raise ValueError("Datos insuficientes (mínimo 50 filas).")
        self.feature_names = list(feat.columns)
        X = self.scaler.fit_transform(feat.values)
        if auto_select:
            self.aic_bic_table = evaluate_states(X, min_s, max_s)
            self.n_states = int(self.aic_bic_table.loc[self.aic_bic_table["bic"].idxmin(), "n_states"])
            print(f"    Auto: {self.n_states} estados")
        self.model  = _train_hmm(X, self.n_states, self.n_iter, self.tol)
        self.labels = assign_labels(self.model, self.feature_names)
        return self

    def predict(self, df):
        feat = build_features(df)
        X    = self.scaler.transform(feat.values)
        states     = self.model.predict(X)
        posteriors = self.model.predict_proba(X)
        out = feat.copy()
        out["state"]      = states
        out["regime"]     = [self.labels.get(s, "Desconocido") for s in states]
        out["confidence"] = posteriors.max(axis=1)
        for i in range(self.model.n_components):
            out[f"prob_{i}"] = posteriors[:, i]
        out["close"] = df["close"].reindex(out.index)
        return out

    def current_regime(self, df):
        pred  = self.predict(df)
        last  = pred.iloc[-1]
        state = int(last["state"])
        arr   = pred["state"].values
        stab  = 1
        for s in reversed(arr[:-1]):
            if s == state: stab += 1
            else: break
        return {
            "regime":     self.labels.get(state, "Desconocido"),
            "state":      state,
            "confidence": float(last["confidence"]),
            "stability":  stab,
            "posteriors": {
                self.labels.get(i, f"S{i}"): float(last[f"prob_{i}"])
                for i in range(self.model.n_components)
            },
        }

    def transition_df(self):
        idx = [self.labels.get(i, f"S{i}") for i in range(self.n_states)]
        return pd.DataFrame(self.model.transmat_, index=idx, columns=idx)

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler,
                     "labels": self.labels, "feature_names": self.feature_names,
                     "n_states": self.n_states}, path)

    def load(self, path):
        d = joblib.load(path)
        self.model = d["model"]; self.scaler = d["scaler"]
        self.labels = d["labels"]; self.feature_names = d["feature_names"]
        self.n_states = d["n_states"]
        return self


class MultiTimeframeHMM:
    def __init__(self, timeframes=None, n_states=7, auto_select=False):
        self.timeframes  = timeframes or list(TIMEFRAME_WEIGHTS.keys())
        self.n_states    = n_states
        self.auto_select = auto_select
        self.models: dict = {}

    def fit(self, data):
        for tf in self.timeframes:
            df = data.get(tf)
            if df is None or len(df) < 100:
                print(f"  [SKIP] {tf}: datos insuficientes"); continue
            print(f"\n  [TRAIN] {tf} — {len(df)} velas")
            m = RegimeHMM(n_states=self.n_states)
            m.fit(df, auto_select=self.auto_select)
            self.models[tf] = m
        return self

    def composite_regime(self, data):
        # Construir dict de votos dinámicamente para soportar cualquier nº de regímenes
        vote = {r: 0.0 for r in ALL_REGIMES_ORDERED}
        vote["Desconocido"] = 0.0
        total_w   = 0.0
        breakdown = {}

        for tf, model in self.models.items():
            df = data.get(tf)
            if df is None or len(df) < 50:
                continue
            info = model.current_regime(df)
            w    = TIMEFRAME_WEIGHTS.get(tf, 1)
            for regime, prob in info["posteriors"].items():
                if regime not in vote:
                    vote[regime] = 0.0
                vote[regime] += prob * w * info["confidence"]
            total_w += w
            breakdown[tf] = info

        if total_w > 0:
            for k in vote:
                vote[k] /= total_w

        # Filtrar solo regímenes con votos > 0 para composite
        active_vote = {k: v for k, v in vote.items() if v > 0}
        if active_vote:
            composite = max(active_vote, key=active_vote.get)
        else:
            composite = "Desconocido"

        return {
            "composite_regime":     composite,
            "composite_scores":     vote,
            "composite_confidence": vote.get(composite, 0),
            "breakdown":            breakdown,
        }

    def save_all(self, directory="models"):
        os.makedirs(directory, exist_ok=True)
        for tf, m in self.models.items():
            m.save(os.path.join(directory, f"hmm_{tf}.pkl"))
        print(f"  [SAVE] Modelos en '{directory}/'")

    def load_all(self, directory="models"):
        for tf in self.timeframes:
            path = os.path.join(directory, f"hmm_{tf}.pkl")
            if os.path.exists(path):
                m = RegimeHMM(); m.load(path)
                self.models[tf] = m
                print(f"  [LOAD] {tf}")
