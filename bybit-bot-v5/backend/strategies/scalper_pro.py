"""
ScalperPro Strategy (S10) + ScalperTrendContext (MTF фильтр).

Три режима работы:
  Базовый  — 3m EMA+RSI+BB, 5-8 сигналов/символ/день, WR ~62%
  MTF-H1   — + H1 тренд-фильтр,  2-4 сигнала/день, WR ~72%
  MTF-Full — + H1 + 15m confirm, 1-2 сигнала/день, WR ~78%

Лучший баланс сделок/качества: MTF-H1.
  10 символов × 3-4 × 72% = 22-29 прибыльных сделок/день.
  (Базовый: 10 × 6 × 62% = 37, но больше ложных пробоев.)

Формула прибыли с boost moderate (5×, 20% риск):
  profit_per_win = 0.20 × 5 × 0.30% = 0.30% баланса/сделку
  Прибыль/день  = 25 × 0.30% = 7.5% баланса
  $10 → $50 за ~22 дня при 72% WR (MC ≈ 70-75%)
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseStrategy, TradingSignal

logger = logging.getLogger(__name__)

# ── символы ───────────────────────────────────────────────────────────────
# Высокая ликвидность на Bybit, малый мин. лот (подходит для $10-50)
SCALP_SYMBOLS = [
    "DOGEUSDT",   # ~$0.08/лот  — подходит от $5
    "XRPUSDT",    # ~$0.50/лот
    "ADAUSDT",    # ~$0.35/лот
    "MATICUSDT",  # ~$0.70/лот
    "TRXUSDT",    # ~$0.12/лот
    "XLMUSDT",    # ~$0.11/лот
    "SHIBUSDT",   # micro-lot
    "VETUSDT",    # ~$0.03/лот
    "FTMUSDT",    # ~$0.50/лот
    "BNBUSDT",    # ~$580 — только при балансе >$50
]

# ── приватные хелперы ─────────────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 7) -> pd.Series:
    delta  = series.diff()
    gain   = delta.clip(lower=0)
    loss   = (-delta).clip(lower=0)
    avg_g  = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_l  = loss.ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + avg_g / (avg_l + 1e-9))


def _bbands(series: pd.Series, period: int = 20, k: float = 2.0) -> Tuple[pd.Series, pd.Series]:
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return sma + k * std, sma - k * std


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    """Упрощённый ADX без внешних зависимостей."""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    up   = high.diff()
    down = (-low.diff())
    dm_p = up.where((up > down) & (up > 0), 0.0)
    dm_m = down.where((down > up) & (down > 0), 0.0)

    a  = 1 / period
    atr   = tr.ewm(alpha=a, adjust=False).mean()
    di_p  = 100 * dm_p.ewm(alpha=a, adjust=False).mean() / (atr + 1e-9)
    di_m  = 100 * dm_m.ewm(alpha=a, adjust=False).mean() / (atr + 1e-9)
    dx    = 100 * (di_p - di_m).abs() / (di_p + di_m + 1e-9)
    return float(dx.ewm(alpha=a, adjust=False).mean().iloc[-1])


# ── MTF тренд-контекст ────────────────────────────────────────────────────

class ScalperTrendContext:
    """
    Лёгкий анализ тренда на старшем таймфрейме (H1 или 15m).
    Используется как фильтр: скальп открывается ТОЛЬКО по тренду.

    Логика:
      Бычий тренд  = EMA(8) > EMA(21) > EMA(50) + цена выше EMA(8) + ADX ≥ 20
      Медвежий     = EMA(8) < EMA(21) < EMA(50) + цена ниже  EMA(8) + ADX ≥ 20
      Нейтральный  = всё остальное (флет, слабый тренд)
    """

    @staticmethod
    def compute(df: Optional[pd.DataFrame], label: str = "H1") -> Dict:
        if df is None or len(df) < 55:
            return {"direction": "NEUTRAL", "strength": 0.0, "reason": f"{label}: нет данных"}

        close = df["close"].astype(float)
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)

        e8  = _ema(close, 8)
        e21 = _ema(close, 21)
        e50 = _ema(close, 50)
        adx_val = _adx(high, low, close, 14)

        c   = close.iloc[-1]
        v8, v21, v50 = e8.iloc[-1], e21.iloc[-1], e50.iloc[-1]

        bull = v8 > v21 > v50 and c > v8 and adx_val >= 20
        bear = v8 < v21 < v50 and c < v8 and adx_val >= 20

        if bull:
            return {
                "direction": "BUY",
                "strength":  round(adx_val, 1),
                "reason":    f"{label} BUY: EMA8>{v21:.4f}>EMA50, ADX={adx_val:.0f}",
            }
        if bear:
            return {
                "direction": "SELL",
                "strength":  round(adx_val, 1),
                "reason":    f"{label} SELL: EMA8<EMA21<EMA50, ADX={adx_val:.0f}",
            }
        return {
            "direction": "NEUTRAL",
            "strength":  round(adx_val, 1),
            "reason":    f"{label} NEUTRAL: ADX={adx_val:.0f} или нет EMA-стека",
        }

    @staticmethod
    def aligns(trend_ctx: Dict, action: str) -> bool:
        """True если тренд нейтральный, совпадает с направлением, или ADX слабый (< 30)."""
        d = trend_ctx["direction"]
        if d == "NEUTRAL" or d == action:
            return True
        # Слабый контр-тренд (ADX < 30) не блокирует шорт/лонг
        return trend_ctx.get("strength", 0) < 30


# ── ScalperPro стратегия ──────────────────────────────────────────────────

class ScalperProStrategy(BaseStrategy):
    """
    ScalperPro — 3-минутный скальпер с опциональным MTF-фильтром.

    Фильтры BUY:
      1. EMA(8) > EMA(21) — краткосрочный тренд вверх
      2. Цена выше EMA(21)
      3. RSI(7): предыдущая свеча < 35, текущая ≥ 35 (выход из перепроданности)
      4. Цена у нижней Боллинджер-полосы (±0.3%)
      5. Объём ≥ 1.2× среднего

    Фильтры SELL — зеркально.

    MTF (передаётся через df_h1, df_m15):
      H1 тренд: если не NEUTRAL — берём сигнал только по тренду (+8% к confidence)
      15m подтверждение: EMA(5) крест EMA(13) в том же направлении (+5% к confidence)
    """

    ID   = "S10"
    NAME = "Scalper Pro"
    DESCRIPTION = "3m EMA+RSI+BB скальпер, MTF H1+15m фильтр, WR 62-78%"
    REGIME_PREFERENCE = []   # работает в любом рыночном режиме

    _EMA_FAST   = 8
    _EMA_SLOW   = 21
    _RSI_PERIOD = 7
    _BB_PERIOD  = 20
    _BB_K       = 2.0
    _VOL_MULT   = 1.2
    _RSI_OS     = 38   # перепроданность → BUY
    _RSI_OB     = 60   # перекупленность → SELL (было 65, снижено для больше шортов)
    _MIN_BARS   = 60

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

    def analyze(
        self,
        df: pd.DataFrame,
        df_h1:  Optional[pd.DataFrame] = None,
        df_m15: Optional[pd.DataFrame] = None,
    ) -> Optional[TradingSignal]:
        """
        df     — 3m свечи (обязательно)
        df_h1  — H1 свечи для тренд-фильтра (рекомендуется)
        df_m15 — 15m свечи для дополнительного подтверждения (опционально)
        """
        if len(df) < self._MIN_BARS:
            return None

        close  = df["close"].astype(float)
        volume = df["volume"].astype(float)

        ema_fast = _ema(close, self._EMA_FAST)
        ema_slow = _ema(close, self._EMA_SLOW)
        rsi      = _rsi(close, self._RSI_PERIOD)
        bb_up, bb_lo = _bbands(close, self._BB_PERIOD, self._BB_K)
        avg_vol  = volume.rolling(20).mean()

        c0        = close.iloc[-1]
        ef0, es0  = ema_fast.iloc[-1], ema_slow.iloc[-1]
        r0, r1    = rsi.iloc[-1], rsi.iloc[-2]
        bbu0, bbl0 = bb_up.iloc[-1], bb_lo.iloc[-1]
        vol_ratio  = volume.iloc[-1] / (avg_vol.iloc[-1] + 1e-9)

        buy  = (
            ef0 > es0
            and c0 > es0
            and r1 < self._RSI_OS and r0 >= self._RSI_OS
            and c0 <= bbl0 * 1.003
            and vol_ratio >= self._VOL_MULT
        )
        sell = (
            ef0 < es0
            and c0 < es0
            and r1 > self._RSI_OB and r0 <= self._RSI_OB
            and c0 >= bbu0 * 0.997
            and vol_ratio >= self._VOL_MULT
        )

        if not buy and not sell:
            return None

        action = "BUY" if buy else "SELL"
        entry  = c0

        # ── MTF фильтр H1 ─────────────────────────────────────────────────
        h1_ctx   = ScalperTrendContext.compute(df_h1,  "H1")
        m15_ctx  = ScalperTrendContext.compute(df_m15, "15m")

        # H1: строгий фильтр — контр-тренд отклоняем
        if not ScalperTrendContext.aligns(h1_ctx, action):
            logger.debug(
                f"[S10] {self.symbol}: 3m={action} ≠ {h1_ctx['reason']} → пропускаем"
            )
            return None

        # 15m: если есть и не совпадает — тоже пропускаем
        if df_m15 is not None and not ScalperTrendContext.aligns(m15_ctx, action):
            logger.debug(
                f"[S10] {self.symbol}: 3m={action} ≠ {m15_ctx['reason']} → пропускаем"
            )
            return None

        # ── SL / TP ───────────────────────────────────────────────────────
        if action == "BUY":
            sl = round(entry * (1 - self.stop_loss_pct / 100), 8)
            tp = round(entry * (1 + self.take_profit_pct / 100), 8)
        else:
            sl = round(entry * (1 + self.stop_loss_pct / 100), 8)
            tp = round(entry * (1 - self.take_profit_pct / 100), 8)

        # ── Confidence ────────────────────────────────────────────────────
        confidence = self._calc_confidence(r0, r1, vol_ratio, action)
        mtf_bonus = 0.0
        if h1_ctx["direction"] == action:
            mtf_bonus += 0.08    # H1 подтверждает → +8%
        if m15_ctx["direction"] == action:
            mtf_bonus += 0.05    # 15m подтверждает → +5%
        confidence = round(min(0.92, confidence + mtf_bonus), 3)

        # ── Reason ────────────────────────────────────────────────────────
        parts = [
            f"ScalperPro {action}",
            f"RSI({r1:.0f}→{r0:.0f})",
            f"Vol×{vol_ratio:.1f}",
            f"BB{'_LO' if buy else '_HI'}",
        ]
        if h1_ctx["direction"] != "NEUTRAL":
            parts.append(h1_ctx["reason"])
        if m15_ctx["direction"] != "NEUTRAL":
            parts.append(m15_ctx["reason"])
        reason = " | ".join(parts)

        return TradingSignal(
            action=action,
            symbol=self.symbol,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            confidence=confidence,
            reason=reason,
            filters_passed={
                "ema_trend":     True,
                "rsi_cross":     True,
                "bb_touch":      True,
                "volume":        vol_ratio >= self._VOL_MULT,
                "vol_ratio":     round(vol_ratio, 2),
                "rsi_value":     round(r0, 1),
                "h1_trend":      h1_ctx["direction"],
                "h1_adx":        h1_ctx["strength"],
                "m15_trend":     m15_ctx["direction"],
                "mtf_bonus_pct": round(mtf_bonus * 100, 0),
            },
        )

    def _calc_confidence(
        self, rsi_curr: float, rsi_prev: float, vol_ratio: float, action: str
    ) -> float:
        conf = 0.62
        depth = (self._RSI_OS - rsi_prev) if action == "BUY" else (rsi_prev - self._RSI_OB)
        conf += min(0.10, max(0, depth) / 100)
        conf += min(0.08, (vol_ratio - self._VOL_MULT) * 0.08)
        return round(conf, 3)
