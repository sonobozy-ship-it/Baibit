"""
ScalperPro Strategy (S10) — высокочастотный скальпер для 3m таймфрейма.

Цель: 30-50 прибыльных сделок в день через 10 символов одновременно.
  10 символов × 5-8 сигналов/день × 62% WR = 31-50 профитных сделок.

Сигнал: EMA(8)/EMA(21) тренд + RSI(7) вход + Bollinger + Volume.
R:R = 2.5:1 (TP 0.3%, SL 0.12%).
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .base import BaseStrategy, TradingSignal

logger = logging.getLogger(__name__)

# Символы для скальпинга — высокая ликвидность, малый минимальный лот
SCALP_SYMBOLS = [
    "DOGEUSDT",  # мин. лот ~$0.08 — подходит для $10
    "XRPUSDT",
    "ADAUSDT",
    "MATICUSDT",
    "SHIBUSDT",
    "TRXUSDT",
    "XLMUSDT",
    "VETUSDT",
    "FTMUSDT",
    "BNBUSDT",
]


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 7) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - 100 / (1 + rs)


def _bbands(series: pd.Series, period: int = 20, k: float = 2.0):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return sma + k * std, sma - k * std   # upper, lower


class ScalperProStrategy(BaseStrategy):
    """
    ScalperPro — 3-минутный скальпер.

    Фильтры для BUY:
      1. EMA(8) > EMA(21) — краткосрочный тренд вверх
      2. Цена > EMA(21) — позиция выше средней
      3. RSI(7) пересекает уровень 35 снизу вверх (выход из перепроданности)
      4. Цена у нижней полосы Боллинджера или отскакивает от неё
      5. Объём ≥ 1.2× среднего

    Фильтры для SELL — зеркально.

    TP = 0.30%, SL = 0.12% → R:R = 2.5:1
    Плечо 5× по умолчанию (переопределяется boost mode).
    """

    ID   = "S10"
    NAME = "Scalper Pro"
    DESCRIPTION = "3m EMA+RSI+BB scalper — до 8 сигналов в день на символ"
    REGIME_PREFERENCE = ["uptrend", "downtrend", "flat"]  # работает в любом режиме

    # ── параметры ─────────────────────────────────────────────────────────
    _EMA_FAST   = 8
    _EMA_SLOW   = 21
    _RSI_PERIOD = 7
    _BB_PERIOD  = 20
    _BB_K       = 2.0
    _VOL_MULT   = 1.2    # минимальный множитель объёма
    _RSI_OS     = 35     # перепроданность (для BUY)
    _RSI_OB     = 65     # перекупленность (для SELL)
    _MIN_BARS   = 60     # минимум свечей для надёжного расчёта

    def __init__(self, symbol: str = "DOGEUSDT", **kwargs):
        super().__init__(
            symbol=symbol,
            timeframe="3",
            leverage=kwargs.pop("leverage", 5),
            stop_loss_pct=kwargs.pop("stop_loss_pct", 0.12),
            take_profit_pct=kwargs.pop("take_profit_pct", 0.30),
            breakeven_pct=kwargs.pop("breakeven_pct", 0.15),
            trailing_stop_pct=kwargs.pop("trailing_stop_pct", 0.08),
            **kwargs,
        )

    def analyze(self, df: pd.DataFrame) -> Optional[TradingSignal]:
        if len(df) < self._MIN_BARS:
            return None

        close  = df["close"].astype(float)
        volume = df["volume"].astype(float)

        ema_fast = _ema(close, self._EMA_FAST)
        ema_slow = _ema(close, self._EMA_SLOW)
        rsi      = _rsi(close, self._RSI_PERIOD)
        bb_up, bb_lo = _bbands(close, self._BB_PERIOD, self._BB_K)
        avg_vol  = volume.rolling(20).mean()

        # Текущие значения
        c0, c1    = close.iloc[-1], close.iloc[-2]
        ef0, es0  = ema_fast.iloc[-1], ema_slow.iloc[-1]
        r0, r1    = rsi.iloc[-1],  rsi.iloc[-2]
        bbu0, bbl0 = bb_up.iloc[-1], bb_lo.iloc[-1]
        vol_ratio  = volume.iloc[-1] / (avg_vol.iloc[-1] + 1e-9)

        # ── BUY ────────────────────────────────────────────────────────────
        buy = (
            ef0 > es0                    # EMA тренд вверх
            and c0 > es0                 # цена выше медленной EMA
            and r1 < self._RSI_OS        # RSI был в перепроданности
            and r0 >= self._RSI_OS       # RSI выходит из перепроданности
            and c0 <= bbl0 * 1.003       # цена у нижней полосы BB (±0.3%)
            and vol_ratio >= self._VOL_MULT
        )

        # ── SELL ───────────────────────────────────────────────────────────
        sell = (
            ef0 < es0
            and c0 < es0
            and r1 > self._RSI_OB
            and r0 <= self._RSI_OB
            and c0 >= bbu0 * 0.997       # цена у верхней полосы BB
            and vol_ratio >= self._VOL_MULT
        )

        if not buy and not sell:
            return None

        action = "BUY" if buy else "SELL"
        entry  = c0

        if action == "BUY":
            sl = round(entry * (1 - self.stop_loss_pct / 100), 8)
            tp = round(entry * (1 + self.take_profit_pct / 100), 8)
        else:
            sl = round(entry * (1 + self.stop_loss_pct / 100), 8)
            tp = round(entry * (1 - self.take_profit_pct / 100), 8)

        bb_pos = "BB_LO" if buy else "BB_HI"
        reason = (
            f"ScalperPro {action}: EMA {ef0:.4f}>{es0:.4f} | "
            f"RSI({self._RSI_PERIOD}) {r1:.0f}→{r0:.0f} | "
            f"Vol×{vol_ratio:.1f} | {bb_pos}"
        )

        confidence = self._calc_confidence(r0, r1, vol_ratio, action)

        return TradingSignal(
            action=action,
            symbol=self.symbol,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            confidence=confidence,
            reason=reason,
            filters_passed={
                "ema_trend":  True,
                "rsi_cross":  True,
                "bb_touch":   True,
                "volume":     vol_ratio >= self._VOL_MULT,
                "vol_ratio":  round(vol_ratio, 2),
                "rsi_value":  round(r0, 1),
            },
        )

    def _calc_confidence(
        self, rsi_curr: float, rsi_prev: float, vol_ratio: float, action: str
    ) -> float:
        """0.60–0.85 в зависимости от силы сигнала."""
        conf = 0.62
        if action == "BUY":
            depth = max(0, self._RSI_OS - rsi_prev)   # насколько глубоко зашёл
            conf += min(0.10, depth / 100)
        else:
            depth = max(0, rsi_prev - self._RSI_OB)
            conf += min(0.10, depth / 100)
        conf += min(0.08, (vol_ratio - self._VOL_MULT) * 0.08)
        return round(min(0.85, conf), 3)
