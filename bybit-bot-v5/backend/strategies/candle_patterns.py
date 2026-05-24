"""
CandlestickPatternFilter — свечные паттерны как фильтр качества входа.

НЕ открывает сделки сам по себе.
Усиливает или блокирует уже существующий сигнал стратегии.

Паттерны:
  Бычьи:    Hammer, BullishEngulfing, MorningStar, ThreeWhiteSoldiers
  Медвежьи: ShootingStar, HangingMan, BearishEngulfing, EveningStar, ThreeBlackCrows
  Нейтр.:   Doji, SpinningTop

FakeBreakoutFilter:
  Пробой уровня только тенью с возвратом → сигнал разворота.
  После FakeBreakout UP  → искать SELL.
  После FakeBreakout DOWN → искать BUY.

Candle score [-3, +3]:
  BUY:  +2 бычий паттерн | +1 FakeBreakout DOWN | +1 near_support
        -2 медвежий паттерн | -1 doji/spinning
  SELL: +2 медвежий паттерн | +1 FakeBreakout UP | +1 near_resistance
        -2 бычий паттерн | -1 doji/spinning

Пороги (candle_score >= threshold для разрешения входа):
  STRICT   =  0  (S5, скальперы)
  ADVISORY = -1  (S10, S15)
  OFF      = -99 (не применять)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ── Пороги по типу стратегии ─────────────────────────────────────────────────
THRESHOLD_STRICT   = 0    # S5, SC_* (скальперы) — нейтральный или позитивный
THRESHOLD_ADVISORY = -1   # S10, S15 — блокируем только сильный контр-сигнал
THRESHOLD_OFF      = -99  # S1-S4, S6-S9, S11-S14 — не блокируем


# ─────────────────────────────────────────────────────────────────────────────
# 1. Геометрия свечи
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CandleGeom:
    """Геометрические характеристики одной свечи."""
    open_: float
    high:  float
    low:   float
    close: float
    volume: float = 0.0

    @property
    def body(self) -> float:
        return abs(self.close - self.open_)

    @property
    def is_bull(self) -> bool:
        return self.close >= self.open_

    @property
    def is_bear(self) -> bool:
        return self.close < self.open_

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open_, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open_, self.close) - self.low

    @property
    def candle_range(self) -> float:
        return max(self.high - self.low, 1e-9)

    @property
    def body_ratio(self) -> float:
        return self.body / self.candle_range

    @property
    def upper_wick_ratio(self) -> float:
        return self.upper_wick / self.candle_range

    @property
    def lower_wick_ratio(self) -> float:
        return self.lower_wick / self.candle_range

    @property
    def mid(self) -> float:
        return (self.open_ + self.close) / 2

    @property
    def is_doji(self) -> bool:
        return self.body_ratio < 0.10

    @property
    def is_spinning_top(self) -> bool:
        return (
            0.10 <= self.body_ratio <= 0.30
            and self.upper_wick_ratio >= 0.20
            and self.lower_wick_ratio >= 0.20
        )


def _row_to_candle(row) -> CandleGeom:
    return CandleGeom(
        open_=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        volume=float(row.get("volume", 0.0)),
    )


def _extract_candles(df: pd.DataFrame, n: int = 5) -> List[CandleGeom]:
    """Возвращает последние n свечей в порядке [новейшая=0, ..., старейшая=n-1]."""
    tail = df.iloc[-n:].iloc[::-1]
    return [_row_to_candle(row) for _, row in tail.iterrows()]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Бычьи паттерны
# ─────────────────────────────────────────────────────────────────────────────

def _hammer(c0: CandleGeom, c1: CandleGeom, c2: CandleGeom,
            near_support: bool) -> bool:
    """Молот: длинная нижняя тень, маленькая верхняя, возле поддержки, после снижения."""
    if c0.body < 1e-9:
        return False
    return (
        c0.lower_wick >= 2.0 * c0.body
        and c0.upper_wick <= 0.35 * c0.body
        and near_support
        and c1.close < c2.close          # перед этим было снижение
    )


def _bullish_engulfing(c0: CandleGeom, c1: CandleGeom) -> bool:
    """Бычье поглощение: предыдущая красная, текущая зелёная перекрывает."""
    return (
        c1.is_bear
        and c0.is_bull
        and c0.open_ <= c1.close
        and c0.close >= c1.open_
    )


def _morning_star(c0: CandleGeom, c1: CandleGeom, c2: CandleGeom) -> bool:
    """
    Утренняя звезда: c2=сильная красная, c1=маленькая/doji, c0=зелёная
    выше середины c2.
    """
    return (
        c2.is_bear and c2.body_ratio >= 0.50
        and (c1.is_doji or c1.body_ratio <= 0.30)
        and c0.is_bull
        and c0.close > c2.mid
    )


def _three_white_soldiers(c0: CandleGeom, c1: CandleGeom,
                          c2: CandleGeom) -> bool:
    """Три белых солдата: 3 зелёные подряд, каждая выше предыдущей."""
    return (
        c2.is_bull and c1.is_bull and c0.is_bull
        and c1.close > c2.close
        and c0.close > c1.close
        and c2.body_ratio >= 0.40
        and c1.body_ratio >= 0.40
        and c0.body_ratio >= 0.40
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Медвежьи паттерны
# ─────────────────────────────────────────────────────────────────────────────

def _shooting_star(c0: CandleGeom, c1: CandleGeom, c2: CandleGeom,
                   near_resistance: bool) -> bool:
    """Падающая звезда: длинная верхняя тень, маленькая нижняя, после роста."""
    if c0.body < 1e-9:
        return False
    return (
        c0.upper_wick >= 2.0 * c0.body
        and c0.lower_wick <= 0.35 * c0.body
        and near_resistance
        and c1.close > c2.close          # перед этим был рост
    )


def _hanging_man(c0: CandleGeom, c1: CandleGeom, c2: CandleGeom,
                 near_resistance: bool) -> bool:
    """Повешенный: как молот, но после роста возле сопротивления."""
    if c0.body < 1e-9:
        return False
    return (
        c0.lower_wick >= 2.0 * c0.body
        and c0.upper_wick <= 0.35 * c0.body
        and near_resistance
        and c1.close > c2.close
    )


def _bearish_engulfing(c0: CandleGeom, c1: CandleGeom) -> bool:
    """Медвежье поглощение: предыдущая зелёная, текущая красная перекрывает."""
    return (
        c1.is_bull
        and c0.is_bear
        and c0.open_ >= c1.close
        and c0.close <= c1.open_
    )


def _evening_star(c0: CandleGeom, c1: CandleGeom, c2: CandleGeom) -> bool:
    """
    Вечерняя звезда: c2=сильная зелёная, c1=маленькая/doji, c0=красная
    ниже середины c2.
    """
    return (
        c2.is_bull and c2.body_ratio >= 0.50
        and (c1.is_doji or c1.body_ratio <= 0.30)
        and c0.is_bear
        and c0.close < c2.mid
    )


def _three_black_crows(c0: CandleGeom, c1: CandleGeom,
                       c2: CandleGeom) -> bool:
    """Три чёрных вороны: 3 красные подряд, каждая ниже предыдущей."""
    return (
        c2.is_bear and c1.is_bear and c0.is_bear
        and c1.close < c2.close
        and c0.close < c1.close
        and c2.body_ratio >= 0.40
        and c1.body_ratio >= 0.40
        and c0.body_ratio >= 0.40
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Ложные пробои (FakeBreakoutFilter)
# ─────────────────────────────────────────────────────────────────────────────

def detect_fake_breakout(
    c0:     CandleGeom,
    levels: Optional[Dict],
    atr:    float,
) -> Tuple[bool, str]:
    """
    Определяет ложный пробой на последней свече.

    Fake UP:   high пробил сопротивление, но close вернулся ниже → SELL setup.
    Fake DOWN: low пробил поддержку, но close вернулся выше → BUY setup.

    Returns:
        (detected: bool, direction: "UP" | "DOWN" | "")
    """
    if not levels or atr <= 0:
        return False, ""

    min_pierce = atr * 0.10   # минимум 10% ATR чтобы считать пробоем

    for r in levels.get("resistances", []):
        if c0.high > r and c0.close < r and (c0.high - r) >= min_pierce:
            return True, "UP"

    for s in levels.get("supports", []):
        if c0.low < s and c0.close > s and (s - c0.low) >= min_pierce:
            return True, "DOWN"

    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# 5. CandlestickPatternFilter — основной публичный класс
# ─────────────────────────────────────────────────────────────────────────────

class CandlestickPatternFilter:
    """
    Оценивает качество свечной картины перед открытием позиции.

    Не открывает сделки сам по себе — только усиляет или блокирует сигнал.
    """

    @classmethod
    def assess(
        cls,
        df:     pd.DataFrame,
        action: str,                        # "BUY" или "SELL"
        levels: Optional[Dict]   = None,    # LevelBuilder.build()
        atr:    Optional[float]  = None,    # ATR на входе
    ) -> Dict:
        """
        Анализирует свечи и возвращает оценку.

        Returns:
            candle_score:            int [-3, +3]
            patterns:                List[str]
            bullish_patterns:        List[str]
            bearish_patterns:        List[str]
            fake_breakout:           bool
            fake_breakout_direction: "UP" | "DOWN" | ""
            is_doji:                 bool
            is_spinning_top:         bool
            near_support:            bool
            near_resistance:         bool
            confidence_modifier:     float  (+/-0.02 за балл)
            allow_entry:             bool
            wait_for_confirmation:   bool
            reason:                  str
        """
        if df is None or len(df) < 4:
            return cls._neutral("no_data")

        candles = _extract_candles(df, n=min(5, len(df)))
        if len(candles) < 3:
            return cls._neutral("too_few_candles")

        c0 = candles[0]
        c1 = candles[1] if len(candles) > 1 else c0
        c2 = candles[2] if len(candles) > 2 else c1

        # ATR из данных если не передан
        if atr is None or atr <= 0:
            atr = cls._calc_atr(df)

        price = c0.close

        # ── Близость к уровням ────────────────────────────────────────────────
        near_support    = False
        near_resistance = False
        if levels and atr > 0:
            for s in levels.get("supports", []):
                if abs(price - s) <= atr * 1.5:
                    near_support = True
                    break
            for r in levels.get("resistances", []):
                if abs(price - r) <= atr * 1.5:
                    near_resistance = True
                    break

        # ── Нейтральные свечи ─────────────────────────────────────────────────
        is_doji         = c0.is_doji
        is_spinning_top = c0.is_spinning_top

        # ── Ложный пробой ─────────────────────────────────────────────────────
        fb_detected, fb_dir = detect_fake_breakout(c0, levels, atr)

        # ── Паттерны ──────────────────────────────────────────────────────────
        bull: List[str] = []
        bear: List[str] = []

        if _hammer(c0, c1, c2, near_support):
            bull.append("Hammer")
        if _bullish_engulfing(c0, c1):
            bull.append("BullishEngulfing")
        if _morning_star(c0, c1, c2):
            bull.append("MorningStar")
        if _three_white_soldiers(c0, c1, c2):
            bull.append("ThreeWhiteSoldiers")

        if _shooting_star(c0, c1, c2, near_resistance):
            bear.append("ShootingStar")
        if _hanging_man(c0, c1, c2, near_resistance):
            bear.append("HangingMan")
        if _bearish_engulfing(c0, c1):
            bear.append("BearishEngulfing")
        if _evening_star(c0, c1, c2):
            bear.append("EveningStar")
        if _three_black_crows(c0, c1, c2):
            bear.append("ThreeBlackCrows")

        # ── Скоринг ───────────────────────────────────────────────────────────
        raw = 0
        if action == "BUY":
            if bull:             raw += 2
            if fb_detected and fb_dir == "DOWN":
                                 raw += 1
            if near_support:     raw += 1
            if bear:             raw -= 2
            if is_doji or is_spinning_top:
                                 raw -= 1
        else:  # SELL
            if bear:             raw += 2
            if fb_detected and fb_dir == "UP":
                                 raw += 1
            if near_resistance:  raw += 1
            if bull:             raw -= 2
            if is_doji or is_spinning_top:
                                 raw -= 1

        candle_score = max(-3, min(3, raw))

        # Confidence modifier: ±0.02 за балл (диапазон ±0.06)
        conf_mod = round(candle_score * 0.02, 3)

        # Ждать подтверждения: нейтральная свеча без паттернов
        wait = (is_doji or is_spinning_top) and not bull and not bear

        # Строка с объяснением
        parts = []
        if bull:
            parts.append(f"bull:{'+'.join(bull)}")
        if bear:
            parts.append(f"bear:{'+'.join(bear)}")
        if fb_detected:
            parts.append(f"FakeBreakout{fb_dir}")
        if near_support:
            parts.append("near_support")
        if near_resistance:
            parts.append("near_resistance")
        if is_doji:
            parts.append("Doji")
        if is_spinning_top:
            parts.append("SpinningTop")
        if not parts:
            parts.append("no_pattern")

        return {
            "candle_score":            candle_score,
            "patterns":                bull + bear,
            "bullish_patterns":        bull,
            "bearish_patterns":        bear,
            "fake_breakout":           fb_detected,
            "fake_breakout_direction": fb_dir,
            "is_doji":                 is_doji,
            "is_spinning_top":         is_spinning_top,
            "near_support":            near_support,
            "near_resistance":         near_resistance,
            "confidence_modifier":     conf_mod,
            "wait_for_confirmation":   wait,
            "reason":                  " | ".join(parts),
        }

    @classmethod
    def check_early_exit(
        cls,
        df:           pd.DataFrame,
        position_side: str,           # "Buy" или "Sell" (из current_position)
        progress_pct:  float,
        levels:        Optional[Dict] = None,
        atr:           Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Триггер досрочного закрытия позиции (Early TP).

        Возвращает (should_exit: bool, reason: str).

        Активируется если прогресс >= 50% и появился:
        - противоположный паттерн
        - Doji / SpinningTop
        - ложный пробой против позиции
        """
        if progress_pct < 50.0:
            return False, ""

        # Для Early TP оцениваем контр-направление
        action = "BUY" if position_side == "Buy" else "SELL"
        r      = cls.assess(df, action, levels, atr)

        if position_side == "Buy":
            trigger = (
                bool(r["bearish_patterns"])
                or r["is_doji"]
                or (r["fake_breakout"] and r["fake_breakout_direction"] == "UP")
            )
        else:
            trigger = (
                bool(r["bullish_patterns"])
                or r["is_doji"]
                or (r["fake_breakout"] and r["fake_breakout_direction"] == "DOWN")
            )

        if trigger:
            return True, f"EarlyTP_candle:{r['reason']}"
        return False, ""

    # ── Вспомогательные ───────────────────────────────────────────────────────

    @staticmethod
    def _calc_atr(df: pd.DataFrame, period: int = 14) -> float:
        c = df["close"].astype(float)
        h = df["high"].astype(float)
        lo = df["low"].astype(float)
        tr = pd.concat([
            h - lo,
            (h - c.shift(1)).abs(),
            (lo - c.shift(1)).abs(),
        ], axis=1).max(axis=1)
        return float(tr.ewm(span=period, adjust=False).mean().iloc[-1])

    @staticmethod
    def _neutral(reason: str = "neutral") -> Dict:
        return {
            "candle_score":            0,
            "patterns":                [],
            "bullish_patterns":        [],
            "bearish_patterns":        [],
            "fake_breakout":           False,
            "fake_breakout_direction": "",
            "is_doji":                 False,
            "is_spinning_top":         False,
            "near_support":            False,
            "near_resistance":         False,
            "confidence_modifier":     0.0,
            "wait_for_confirmation":   False,
            "reason":                  reason,
        }
