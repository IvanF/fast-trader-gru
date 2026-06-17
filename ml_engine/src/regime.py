"""Lightweight regime detection via EMA volatility and GMM clustering."""

from __future__ import annotations

from enum import Enum
from typing import Dict

import numpy as np
from sklearn.mixture import GaussianMixture


class Regime(str, Enum):
    TRENDING = "Trending"
    CHOPPY = "Choppy"
    BREAKOUT = "Breakout"


class RegimeDetector:
    def __init__(self, ema_alpha: float = 0.1) -> None:
        self.ema_alpha = ema_alpha
        self._ema_std: Dict[str, float] = {}
        self._gmm = GaussianMixture(n_components=3, random_state=42, max_iter=50)
        self._fitted = False
        self._regimes: Dict[str, Regime] = {}

    def update(self, symbol: str, returns: np.ndarray) -> Regime:
        if len(returns) < 5:
            return Regime.CHOPPY

        std = float(np.std(returns))
        prev = self._ema_std.get(symbol, std)
        ema_std = self.ema_alpha * std + (1 - self.ema_alpha) * prev
        self._ema_std[symbol] = ema_std

        features = np.column_stack([
            returns[-min(60, len(returns)):],
            np.abs(returns[-min(60, len(returns)):]),
        ])
        if len(features) >= 10:
            try:
                self._gmm.fit(features)
                self._fitted = True
                label = int(self._gmm.predict(features[-1:])[0])
            except Exception:
                label = 1
        else:
            label = 1

        if ema_std < 0.0005:
            regime = Regime.CHOPPY
        elif ema_std > 0.002:
            regime = Regime.BREAKOUT
        elif label == 0:
            regime = Regime.TRENDING
        elif label == 2:
            regime = Regime.BREAKOUT
        else:
            regime = Regime.CHOPPY

        self._regimes[symbol] = regime
        return regime

    def get(self, symbol: str) -> Regime:
        return self._regimes.get(symbol, Regime.CHOPPY)
