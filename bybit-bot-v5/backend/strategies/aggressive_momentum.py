"""
AggressiveMomentumStrategy (S15) — высокочастотный моментум-скальпер.

Таймфреймы:
  df      — 1m (вход): EMA9/21/50, RSI7, ATR, VWAP, Volume, ADX
  df_m15  — 15m (тренд): подтверждение направления + ADX-фильтр

LONG: EMA9>EMA21>EMA50 + RSI 55-78 + price>VWAP + vol×1.8 + ATR expansion
SHORT: зеркально

Частичный TP (metadata в filters_passed):
  TP1=0.7% (40%), TP2=1.5% (30%), TP3=2.5% (30%)

Оценка сигнала:
  <75  → пропуск
  75-85 → half size (size_multiplier=0.5)
  85+  → full size

Рыночный режим (market_regime):
  trending — ADX≥25 + EMA-стек выровнен
  volatile — ATR расширяется + объём × 1.5+
  flat     — ADX<20 → не торгуем
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseStrategy, TradingSignal

logger = logging.getLogger(__name__)


# ── приватные хелперы ─────────────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 7) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    ag    = gain.ewm(alpha=1 / period, adjust=False).mean()
    al    = loss.ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + ag / (al + 1e-9))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    up   = high.diff()
    down = (-low.diff())
    dm_p = up.where((up > down) & (up > 0), 0.0)
    dm_m = down.where((down > up) & (down > 0), 0.0)
    a    = 1 / period
    atr_ = tr.ewm(alpha=a, adjust=False).mean()
    di_p = 100 * dm_p.ewm(alpha=a, adjust=False).mean() / (atr_ + 1e-9)
    di_m = 100 * dm_m.ewm(alpha=a, adjust=False).mean() / (atr_ + 1e-9)
    dx   = 100 * (di_p - di_m).abs() / (di_p + di_m + 1e-9)
    return float(dx.ewm(alpha=a, adjust=False).mean().iloc[-1])


def _vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Сессионный VWAP (сбрасывается каждые 200 свечей — имитация торговой сессии)."""
    typical = (high + low + close) / 3
    cum_vol = volume.rolling(200, min_periods=1).sum()
    cum_pv  = (typical * volume).rolling(200, min_periods=1).sum()
    return cum_pv / (cum_vol + 1e-9)


def _detect_regime(
    adx_val: float,
    atr_ratio: float,
    vol_ratio: float,
    ema9: float, ema21: float, ema50: float,
) -> str:
    """Определяет рыночный режим: trending / volatile / flat."""
    ema_stacked = (ema9 > ema21 > ema50) or (ema9 < ema21 < ema50)
    if adx_val >= 25 and ema_stacked:
        return "trending"
    if atr_ratio >= 1.15 and vol_ratio >= 1.5:
        return "volatile"
    return "flat"


def _m15_trend(df_m15: Optional[pd.DataFrame]) -> Dict:
    """Анализ тренда на M15: EMA-стек + ADX."""
    if df_m15 is None or len(df_m15) < 55:
        return {"direction": "NEUTRAL", "adx": 0.0, "reason": "M15: нет данных"}

    close = df_m15["close"].astype(float)
    high  = df_m15["high"].astype(float)
    low   = df_m15["low"].astype(float)

    e9  = _ema(close, 9).iloc[-1]
    e21 = _ema(close, 21).iloc[-1]
    e50 = _ema(close, 50).iloc[-1]
    adx = _adx(high, low, close, 14)
    c   = close.iloc[-1]

    bull = e9 > e21 > e50 and c > e9 and adx >= 18
    bear = e9 < e21 < e50 and c < e9 and adx >= 18

    if bull:
        return {"direction": "BUY",  "adx": round(adx, 1),
                "reason": f"M15 BUY: EMA9>{e21:.4f}>EMA50 ADX={adx:.0f}"}
    if bear:
        return {"direction": "SELL", "adx": round(adx, 1),
                "reason": f"M15 SELL: EMA9<EMA21<EMA50 ADX={adx:.0f}"}
    return {"direction": "NEUTRAL", "adx": round(adx, 1),
            "reason": f"M15 NEUTRAL ADX={adx:.0f}"}


# ── Стратегия ─────────────────────────────────────────────────────────────

class AggressiveMomentumStrategy(BaseStrategy):
    """
    S15 — Aggressive Momentum AI.
    1m вход с подтверждением M15 тренда.
    Цель: максимизация прибыли в трендовых и волатильных режимах.
    """

    ID   = "S15"
    NAME = "Aggressive Momentum AI"
    DESCRIPTION = "1m EMA9/21/50+RSI7+VWAP+ATR+ADX, M15 trend filter, TP 0.7/1.5/2.5%"
    REGIME_PREFERENCE = ["uptrend", "downtrend", "volatile"]

    # Параметры индикаторов
    _RSI_LO  = 55    # RSI нижняя граница для BUY
    _RSI_HI  = 78    # RSI верхняя граница для BUY (не перекупленность)
    _RSI_LO_S = 22   # RSI нижняя граница для SELL (не перепроданность)
    _RSI_HI_S = 45   # RSI верхняя граница для SELL
    _VOL_MULT = 1.8  # объём × среднего
    _ATR_EXP  = 1.08 # ATR расширение (8%)
    _ADX_MIN  = 18   # минимальный ADX для торговли
    _MIN_BARS = 80

    # Частичные TP (метаданные)
    _TP1_PCT = 0.7   # 40% позиции
    _TP2_PCT = 1.5   # 30% позиции
    _TP3_PCT = 2.5   # 30% позиции

    # Защита от потерь
    _MAX_DAILY_TRADES   = 6      # максимум сделок в день
    _MIN_WR_THRESHOLD   = 0.58   # минимальный WR за 20 сделок (иначе пауза)
    _BLOCK_LOSSES       = 3      # убытков подряд → 180мин блок
    _BLOCK_MIN          = 180    # минут блокировки

    def __init__(self, symbol: str = "OPUSDT", **kwargs):
        super().__init__(
            symbol=symbol,
            timeframe="1",
            leverage=kwargs.pop("leverage", 7),
            stop_loss_pct=kwargs.pop("stop_loss_pct", 0.6),
            take_profit_pct=kwargs.pop("take_profit_pct", self._TP1_PCT),
            breakeven_pct=kwargs.pop("breakeven_pct", 0.35),
            trailing_stop_pct=kwargs.pop("trailing_stop_pct", 0.15),
            max_hold_minutes=kwargs.pop("max_hold_minutes", 45.0),
            edge_wr_target=0.60,
            **kwargs,
        )
        from datetime import datetime, timezone
        self._daily_trades: int = 0
        self._daily_date:   Optional[str]      = None
        self._blocked_until: Optional[datetime] = None

    def analyze(
        self,
        df: pd.DataFrame,
        df_h1:  Optional[pd.DataFrame] = None,
        df_m15: Optional[pd.DataFrame] = None,
    ) -> Optional[TradingSignal]:
        """
        df     — 1m свечи (обязательно)
        df_m15 — 15m свечи для тренд-фильтра
        df_h1  — не используется (для совместимости)
        """
        if df is None or len(df) < self._MIN_BARS:
            return None

        # ── Защита S15: блок / дневной лимит / min WR ─────────────────────
        _now = datetime.now(timezone.utc)

        if self._blocked_until and _now < self._blocked_until:
            _rem = int((self._blocked_until - _now).total_seconds() / 60)
            logger.info(f"S15: заблокирован ещё {_rem} мин (3 убытка подряд)")
            return None

        # Сброс дневного счётчика при смене даты
        _today = _now.strftime("%Y-%m-%d")
        if self._daily_date != _today:
            self._daily_date   = _today
            self._daily_trades = 0

        if self._daily_trades >= self._MAX_DAILY_TRADES:
            logger.info(f"S15: дневной лимит {self._MAX_DAILY_TRADES} сделок исчерпан")
            return None

        # Минимальный WR после накопления статистики (≥20 сделок)
        if self.trades >= 20 and self.rolling_wr_20 < self._MIN_WR_THRESHOLD:
            logger.info(
                f"S15: WR {self.rolling_wr_20:.0%} < {self._MIN_WR_THRESHOLD:.0%} → пауза"
            )
            return None

        # Блокировка после N убытков подряд
        if self.consecutive_losses >= self._BLOCK_LOSSES:
            self._blocked_until     = _now + timedelta(minutes=self._BLOCK_MIN)
            self.consecutive_losses = 0
            logger.warning(f"S15: {self._BLOCK_LOSSES} убытка подряд → блок {self._BLOCK_MIN} мин")
            return None

        close  = df["close"].astype(float)
        high   = df["high"].astype(float)
        low    = df["low"].astype(float)
        volume = df["volume"].astype(float)

        # ── Индикаторы ────────────────────────────────────────────────────
        e9  = _ema(close, 9)
        e21 = _ema(close, 21)
        e50 = _ema(close, 50)
        rsi = _rsi(close, 7)
        atr_s    = _atr(high, low, close, 14)
        vwap_s   = _vwap(high, low, close, volume)
        avg_vol  = volume.rolling(20).mean()

        c0      = close.iloc[-1]
        e9_0    = e9.iloc[-1]
        e21_0   = e21.iloc[-1]
        e50_0   = e50.iloc[-1]
        rsi_0   = rsi.iloc[-1]
        atr_0   = atr_s.iloc[-1]
        atr_avg = float(atr_s.iloc[-50:].mean()) if len(atr_s) >= 50 else atr_0
        vwap_0  = vwap_s.iloc[-1]
        vol_0   = volume.iloc[-1]
        vol_avg = avg_vol.iloc[-1]
        vol_ratio = vol_0 / (vol_avg + 1e-9)
        atr_ratio = atr_0 / (atr_avg + 1e-9)

        # ADX на 1m
        adx_val = _adx(high, low, close, 14)

        # ── Рыночный режим ────────────────────────────────────────────────
        regime = _detect_regime(adx_val, atr_ratio, vol_ratio, e9_0, e21_0, e50_0)
        if regime == "flat":
            return None  # не торгуем во флете

        # ── M15 тренд ─────────────────────────────────────────────────────
        m15_ctx = _m15_trend(df_m15)

        # ── Условия LONG ──────────────────────────────────────────────────
        ema_bull  = e9_0 > e21_0 > e50_0
        ema_bear  = e9_0 < e21_0 < e50_0
        rsi_bull  = self._RSI_LO <= rsi_0 <= self._RSI_HI
        rsi_bear  = self._RSI_LO_S <= rsi_0 <= self._RSI_HI_S
        above_vwap = c0 > vwap_0
        below_vwap = c0 < vwap_0
        vol_ok     = vol_ratio >= self._VOL_MULT
        atr_exp    = atr_ratio >= self._ATR_EXP
        adx_ok     = adx_val >= self._ADX_MIN

        buy = (
            ema_bull
            and rsi_bull
            and above_vwap
            and vol_ok
            and atr_exp
            and adx_ok
            and m15_ctx["direction"] in ("BUY", "NEUTRAL")
        )
        sell = (
            ema_bear
            and rsi_bear
            and below_vwap
            and vol_ok
            and atr_exp
            and adx_ok
            and m15_ctx["direction"] in ("SELL", "NEUTRAL")
        )

        if not buy and not sell:
            return None

        action = "BUY" if buy else "SELL"
        entry  = c0

        # ── SL / TP ───────────────────────────────────────────────────────
        if action == "BUY":
            sl  = round(entry * (1 - self.stop_loss_pct  / 100), 8)
            tp1 = round(entry * (1 + self._TP1_PCT / 100), 8)
            tp2 = round(entry * (1 + self._TP2_PCT / 100), 8)
            tp3 = round(entry * (1 + self._TP3_PCT / 100), 8)
        else:
            sl  = round(entry * (1 + self.stop_loss_pct  / 100), 8)
            tp1 = round(entry * (1 - self._TP1_PCT / 100), 8)
            tp2 = round(entry * (1 - self._TP2_PCT / 100), 8)
            tp3 = round(entry * (1 - self._TP3_PCT / 100), 8)

        # ── Оценка сигнала (0-100) ────────────────────────────────────────
        score, bonus_parts = self._calc_score(
            rsi_0, vol_ratio, atr_ratio, adx_val, regime,
            m15_ctx, action, above_vwap if action == "BUY" else below_vwap,
        )

        # < 75 — не торгуем
        if score < 75:
            return None

        # 75-85: half size, 85+: full size
        size_mult = 0.5 if score < 85 else 1.0
        confidence = round(min(0.93, score / 100), 3)

        # ── Reason ────────────────────────────────────────────────────────
        reason_parts = [
            f"S15 {action}",
            f"Score={score:.0f}",
            f"RSI7={rsi_0:.1f}",
            f"Vol×{vol_ratio:.1f}",
            f"ATR×{atr_ratio:.2f}",
            f"ADX={adx_val:.0f}",
            f"Regime={regime}",
            m15_ctx["reason"],
        ]
        reason = " | ".join(reason_parts)

        self._daily_trades += 1
        return TradingSignal(
            action=action,
            symbol=self.symbol,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp1,
            confidence=confidence,
            reason=reason,
            filters_passed={
                # Основные индикаторы
                "ema_stack":      True,
                "rsi_value":      round(rsi_0, 1),
                "vwap_side":      "above" if above_vwap else "below",
                "vol_ratio":      round(vol_ratio, 2),
                "atr_ratio":      round(atr_ratio, 3),
                "adx_value":      round(adx_val, 1),
                # Режим и тренд
                "market_regime":  regime,
                "m15_direction":  m15_ctx["direction"],
                "m15_adx":        m15_ctx["adx"],
                # Частичные TP
                "tp1":            tp1,
                "tp2":            tp2,
                "tp3":            tp3,
                "tp_pcts":        [self._TP1_PCT, self._TP2_PCT, self._TP3_PCT],
                "partial_pcts":   [0.40, 0.30, 0.30],
                # Управление размером
                "signal_score":   round(score, 1),
                "size_multiplier": size_mult,
                "score_bonuses":  bonus_parts,
                # Риск
                "leverage":       self.leverage,
                "risk_pct":       round(self.stop_loss_pct, 2),
                "max_positions":  5,
            },
        )

    def _calc_score(
        self,
        rsi: float,
        vol_ratio: float,
        atr_ratio: float,
        adx: float,
        regime: str,
        m15_ctx: Dict,
        action: str,
        vwap_ok: bool,
    ) -> Tuple[float, list]:
        """
        Оценка сигнала 0-100.
        Базовая оценка: 70 (если прошли все фильтры).
        Бонусы: до 30 дополнительных очков.
        """
        score  = 70.0
        bonuses = []

        # M15 подтверждает направление → +7
        if m15_ctx["direction"] == action:
            score += 7
            bonuses.append("M15_confirm+7")

        # ADX силён → тренд чёткий
        if adx >= 30:
            score += 5
            bonuses.append("ADX≥30+5")
        elif adx >= 25:
            score += 3
            bonuses.append("ADX≥25+3")

        # RSI в оптимальной зоне
        if action == "BUY" and 60 <= rsi <= 72:
            score += 4
            bonuses.append("RSI_optimal+4")
        elif action == "SELL" and 28 <= rsi <= 40:
            score += 4
            bonuses.append("RSI_optimal+4")

        # Большой объём
        if vol_ratio >= 2.5:
            score += 5
            bonuses.append(f"Vol×{vol_ratio:.1f}+5")
        elif vol_ratio >= 2.0:
            score += 3
            bonuses.append(f"Vol×{vol_ratio:.1f}+3")

        # Режим trending лучше
        if regime == "trending":
            score += 4
            bonuses.append("Trending+4")
        elif regime == "volatile":
            score += 2
            bonuses.append("Volatile+2")

        # Сильное ATR расширение
        if atr_ratio >= 1.3:
            score += 3
            bonuses.append("ATR_exp+3")

        # M15 ADX силён
        if m15_ctx.get("adx", 0) >= 30:
            score += 2
            bonuses.append("M15_ADX+2")

        return min(100.0, score), bonuses
