"""
Утилиты анализа тренда и уровни Фибоначчи + две стратегии на их основе.

TrendAnalyzer  — свинги, HH/HL / LH/LL структура, сила тренда
FibonacciLevels — уровни ретрейсмента и расширения по свинговым точкам

S8: TREND MOMENTUM  — вход на откате в тренде (EMA-стек + ADX + структура)
S9: FIBONACCI RETRACEMENT — вход на ключевых уровнях Фибоначчи (38.2 / 50 / 61.8)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

import pandas_ta as ta
from .base import BaseStrategy, TradingSignal


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ УТИЛИТЫ
# ============================================================

class TrendAnalyzer:
    """
    Определяет тренд через структуру свингов (HH/HL, LH/LL).
    Возвращает готовый dict который можно включить в features для ML.
    """

    @staticmethod
    def find_swings(
        df: pd.DataFrame,
        window: int = 5,
    ) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """
        Возвращает (swing_highs, swing_lows) — список (index, price).
        Свинг-хай в точке i: high[i] максимален среди ±window соседей.
        """
        highs = df["high"].values
        lows  = df["low"].values
        n = len(df)

        sh: List[Tuple[int, float]] = []
        sl: List[Tuple[int, float]] = []

        for i in range(window, n - window):
            seg_h = highs[i - window: i + window + 1]
            seg_l = lows[i - window: i + window + 1]
            if highs[i] == seg_h.max():
                sh.append((i, float(highs[i])))
            if lows[i] == seg_l.min():
                sl.append((i, float(lows[i])))

        return sh, sl

    @classmethod
    def analyze(
        cls,
        df: pd.DataFrame,
        swing_window: int = 5,
        min_swings: int = 2,
    ) -> Dict:
        """
        Полный анализ тренда.

        Возвращает:
            direction   : 'uptrend' | 'downtrend' | 'flat'
            strength    : 0.0 – 1.0  (отношение подтверждённых свингов)
            swing_high  : последний свинг-хай (цена)
            swing_low   : последний свинг-лоу (цена)
            hh_hl       : True если последние 2 хая и 2 лоя восходящие
            lh_ll       : True если последние 2 хая и 2 лоя нисходящие
            swing_highs : [(idx, price), ...]
            swing_lows  : [(idx, price), ...]
        """
        sh, sl = cls.find_swings(df, window=swing_window)

        result = {
            "direction": "flat",
            "strength":  0.0,
            "swing_high": float(df["high"].max()),
            "swing_low":  float(df["low"].min()),
            "hh_hl": False,
            "lh_ll": False,
            "swing_highs": sh,
            "swing_lows":  sl,
        }

        if sh:
            result["swing_high"] = sh[-1][1]
        if sl:
            result["swing_low"] = sl[-1][1]

        if len(sh) >= min_swings and len(sl) >= min_swings:
            # Проверяем последние 2 хая и 2 лоя
            recent_h = [p for _, p in sh[-3:]]
            recent_l = [p for _, p in sl[-3:]]

            hh = all(recent_h[i] < recent_h[i + 1] for i in range(len(recent_h) - 1))
            hl = all(recent_l[i] < recent_l[i + 1] for i in range(len(recent_l) - 1))
            lh = all(recent_h[i] > recent_h[i + 1] for i in range(len(recent_h) - 1))
            ll = all(recent_l[i] > recent_l[i + 1] for i in range(len(recent_l) - 1))

            result["hh_hl"] = bool(hh and hl)
            result["lh_ll"] = bool(lh and ll)

            confirmed_up   = sum(1 for i in range(len(recent_h) - 1) if recent_h[i] < recent_h[i + 1])
            confirmed_down = sum(1 for i in range(len(recent_h) - 1) if recent_h[i] > recent_h[i + 1])
            total = max(len(recent_h) - 1, 1)

            if result["hh_hl"]:
                result["direction"] = "uptrend"
                result["strength"]  = round(confirmed_up / total, 2)
            elif result["lh_ll"]:
                result["direction"] = "downtrend"
                result["strength"]  = round(confirmed_down / total, 2)

        return result


class FibonacciLevels:
    """
    Уровни Фибоначчи по двум свинговым точкам.

    Ретрейсмент (откат в тренде):
        uptrend   → от swing_low до swing_high, ищем поддержки при откате
        downtrend → от swing_high до swing_low, ищем сопротивления при отскоке

    Расширения (цели движения):
        127.2%, 161.8%, 261.8% относительно размаха свинга
    """

    RETRACEMENT_RATIOS = {
        "0.0":   0.000,
        "23.6":  0.236,
        "38.2":  0.382,
        "50.0":  0.500,
        "61.8":  0.618,
        "78.6":  0.786,
        "100.0": 1.000,
    }
    EXTENSION_RATIOS = {
        "127.2": 1.272,
        "161.8": 1.618,
        "261.8": 2.618,
    }
    KEY_LEVELS = {"38.2", "50.0", "61.8"}   # ключевые уровни для входа

    def __init__(self, swing_low: float, swing_high: float, direction: str = "uptrend"):
        self.swing_low  = swing_low
        self.swing_high = swing_high
        self.direction  = direction
        self._range     = swing_high - swing_low
        self.levels: Dict[str, float] = {}
        self._calculate()

    def _calculate(self):
        rng = self._range
        if rng <= 0:
            return

        if self.direction == "uptrend":
            # Откат вниз от swing_high
            for name, ratio in self.RETRACEMENT_RATIOS.items():
                self.levels[name] = round(self.swing_high - ratio * rng, 8)
            # Расширения выше swing_high
            for name, ratio in self.EXTENSION_RATIOS.items():
                self.levels[name] = round(self.swing_high + (ratio - 1.0) * rng, 8)
        else:
            # downtrend: откат вверх от swing_low
            for name, ratio in self.RETRACEMENT_RATIOS.items():
                self.levels[name] = round(self.swing_low + ratio * rng, 8)
            for name, ratio in self.EXTENSION_RATIOS.items():
                self.levels[name] = round(self.swing_low - (ratio - 1.0) * rng, 8)

    def nearest(self, price: float, tolerance_pct: float = 0.5) -> Optional[Tuple[str, float, float]]:
        """
        Возвращает (имя_уровня, цена_уровня, отклонение_%) для ближайшего
        уровня в пределах tolerance_pct. Иначе None.
        """
        best_name  = None
        best_price = 0.0
        best_dist  = float("inf")

        for name, lvl_price in self.levels.items():
            if lvl_price <= 0:
                continue
            dist_pct = abs(price - lvl_price) / lvl_price * 100
            if dist_pct < tolerance_pct and dist_pct < best_dist:
                best_dist  = dist_pct
                best_name  = name
                best_price = lvl_price

        if best_name is None:
            return None
        return (best_name, best_price, round(best_dist, 4))

    def is_near_key_level(self, price: float, tolerance_pct: float = 0.5) -> bool:
        """True если цена у одного из ключевых уровней (38.2 / 50 / 61.8)."""
        hit = self.nearest(price, tolerance_pct)
        return hit is not None and hit[0] in self.KEY_LEVELS

    def tp_level(self) -> float:
        """TP-цель: уровень 0% (возврат к свинг-хаю) для uptrend."""
        if self.direction == "uptrend":
            return self.levels.get("0.0", self.swing_high)
        return self.levels.get("0.0", self.swing_low)

    def extension_tp(self, ratio: str = "127.2") -> float:
        return self.levels.get(ratio, self.tp_level())

    def to_dict(self) -> Dict:
        return {
            "direction":  self.direction,
            "swing_low":  self.swing_low,
            "swing_high": self.swing_high,
            "range":      round(self._range, 8),
            "levels":     self.levels,
        }


# ============================================================
# S8: TREND MOMENTUM
# ============================================================
class TrendMomentumStrategy(BaseStrategy):
    """
    Вход на откате к EMA21/EMA50 в подтверждённом тренде.

    Фильтры:
      1. HH/HL структура (свинги) — тренд роста
         LH/LL структура (свинги) — тренд падения
      2. EMA-стек: 9 > 21 > 50 (uptrend) или 9 < 21 < 50 (downtrend)
      3. ADX ≥ 20 — тренд достаточно силён
      4. Цена откатилась к зоне EMA21 (в пределах 0.8 ATR)
      5. RSI в нейтральной зоне (40–65 long, 35–60 short) — не перекуплен
      6. Объём выше среднего
    """
    ID   = "S8"
    NAME = "TREND MOMENTUM"
    DESCRIPTION = "Вход на откате к EMA21/50 в подтверждённом тренде (HH/HL + ADX)"
    REGIME_PREFERENCE = ["uptrend", "downtrend"]

    def __init__(self, **kwargs):
        super().__init__(
            stop_loss_pct=1.4,
            take_profit_pct=4.2,
            edge_wr_target=0.63,
            timeframe="60",
            **kwargs,
        )

    def analyze(self, df: pd.DataFrame) -> Optional[TradingSignal]:
        if len(df) < 60:
            return None

        # Индикаторы
        df["ema9"]  = ta.ema(df["close"], length=9)
        df["ema21"] = ta.ema(df["close"], length=21)
        df["ema50"] = ta.ema(df["close"], length=50)
        df["rsi"]   = ta.rsi(df["close"], length=14)
        df["atr"]   = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["vol_ma"] = df["volume"].rolling(20).mean()
        adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
        df = df.join(adx_df)

        last = df.iloc[-1]

        # Структура тренда через свинги
        trend = TrendAnalyzer.analyze(df.tail(60).reset_index(drop=True))

        # Фильтры
        is_uptrend   = trend["hh_hl"]
        is_downtrend = trend["lh_ll"]

        ema_stack_bull = last["ema9"] > last["ema21"] > last["ema50"]
        ema_stack_bear = last["ema9"] < last["ema21"] < last["ema50"]

        adx_ok = last["ADX_14"] >= 20

        # Откат к EMA21 зоне (±0.8 ATR)
        atr = last["atr"]
        near_ema21_bull = abs(last["close"] - last["ema21"]) < atr * 0.8 and last["close"] > last["ema50"]
        near_ema21_bear = abs(last["close"] - last["ema21"]) < atr * 0.8 and last["close"] < last["ema50"]

        rsi_ok_bull = 38 < last["rsi"] < 65
        rsi_ok_bear = 35 < last["rsi"] < 62
        vol_ok = last["volume"] > last["vol_ma"] * 1.2

        long_filters = {
            "trend_structure": is_uptrend,
            "ema_stack":       ema_stack_bull,
            "adx_strength":    adx_ok,
            "pullback_zone":   near_ema21_bull,
            "rsi_zone":        rsi_ok_bull,
            "volume":          vol_ok,
        }
        short_filters = {
            "trend_structure": is_downtrend,
            "ema_stack":       ema_stack_bear,
            "adx_strength":    adx_ok,
            "pullback_zone":   near_ema21_bear,
            "rsi_zone":        rsi_ok_bear,
            "volume":          vol_ok,
        }

        long_ok  = all(long_filters.values())
        short_ok = all(short_filters.values())

        if not (long_ok or short_ok):
            return None

        side  = "BUY" if long_ok else "SELL"
        entry = float(last["close"])
        swing_low  = trend["swing_low"]
        swing_high = trend["swing_high"]

        if side == "BUY":
            # SL под последним свинг-лоу (защита структуры)
            sl = min(swing_low * 0.998, entry * (1 - self.stop_loss_pct / 100))
            tp = entry * (1 + self.take_profit_pct / 100)
        else:
            sl = max(swing_high * 1.002, entry * (1 + self.stop_loss_pct / 100))
            tp = entry * (1 - self.take_profit_pct / 100)

        confidence = 0.68 + 0.04 * trend["strength"]  # 0.68–0.72

        return TradingSignal(
            action=side, symbol=self.symbol,
            confidence=round(confidence, 2),
            entry_price=entry, stop_loss=sl, take_profit=tp,
            reason=(
                f"TrendMomentum {side}: "
                f"{'HH/HL' if long_ok else 'LH/LL'} + "
                f"EMA-stack + ADX={last['ADX_14']:.1f} + "
                f"pullback EMA21"
            ),
            filters_passed=long_filters if long_ok else short_filters,
        )


# ============================================================
# S9: TREND + FIBONACCI (объединённая)
# ============================================================
class TrendFibonacciStrategy(BaseStrategy):
    """
    Объединяет тренд роста/падения с уровнями Фибоначчи.

    Требует одновременно:
      — Подтверждённый тренд (HH/HL или LH/LL через свинги)
      — EMA-стек (9 > 21 > 50 > 200) или (9 < 21 < 50 < 200)
      — ADX ≥ 20 (сила тренда)
      — Цена на откате к ключевому уровню Фибоначчи (38.2 / 50.0 / 61.8%)
      — RSI разворачивается от экстремума
      — MACD-гистограмма меняет направление
      — Объём выше среднего

    Входная логика:
      1. Определяем тренд (свинги + EMA-стек + ADX)
      2. Находим последний значимый импульс (swing_low ↔ swing_high)
      3. Ждём откат цены к 38.2 / 50 / 61.8% уровню
      4. Добавляем RSI + MACD подтверждение

    TP  = 127.2% расширение Фибоначчи (следующий импульс)
    SL  = за свинг-лоу/хаю + 0.2% буфер
    """
    ID   = "S9"
    NAME = "TREND + FIBONACCI"
    DESCRIPTION = "Тренд (HH/HL + EMA + ADX) + откат к уровням Фибоначчи 38.2/50/61.8%"
    REGIME_PREFERENCE = ["uptrend", "downtrend"]

    FIB_TOLERANCE = 0.45   # % допуска для попадания в уровень

    def __init__(self, **kwargs):
        super().__init__(
            stop_loss_pct=1.6,
            take_profit_pct=5.5,
            edge_wr_target=0.66,
            timeframe="60",   # H1 — достаточно свингов и чёткие уровни
            **kwargs,
        )

    @staticmethod
    def _find_swing(df: pd.DataFrame, lookback: int = 50) -> Tuple[Optional[float], Optional[float]]:
        """Возвращает (swing_low, swing_high) последнего импульса."""
        window = df.tail(lookback).reset_index(drop=True)
        sh, sl = TrendAnalyzer.find_swings(window, window=4)
        if not sh or not sl:
            return None, None
        return min(p for _, p in sl), max(p for _, p in sh)

    def analyze(self, df: pd.DataFrame) -> Optional[TradingSignal]:
        if len(df) < 60:
            return None

        # ── Индикаторы ──
        df["ema9"]   = ta.ema(df["close"], length=9)
        df["ema21"]  = ta.ema(df["close"], length=21)
        df["ema50"]  = ta.ema(df["close"], length=50)
        df["ema200"] = ta.ema(df["close"], length=200)
        df["rsi"]    = ta.rsi(df["close"], length=14)
        df["atr"]    = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["vol_ma"] = df["volume"].rolling(20).mean()
        adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
        df = df.join(adx_df)
        macd_df = ta.macd(df["close"], fast=12, slow=26, signal=9)
        df = df.join(macd_df)

        last = df.iloc[-1]
        prev = df.iloc[-2]

        # ── 1. Структура тренда (свинги) ──
        trend = TrendAnalyzer.analyze(df.tail(80).reset_index(drop=True))
        is_uptrend   = trend["hh_hl"]
        is_downtrend = trend["lh_ll"]
        if not (is_uptrend or is_downtrend):
            return None

        # ── 2. EMA-стек ──
        if pd.isna(last["ema200"]):
            return None

        ema_bull = last["ema9"] > last["ema21"] > last["ema50"]
        ema_bear = last["ema9"] < last["ema21"] < last["ema50"]
        ema_200_bull = last["close"] > last["ema200"]
        ema_200_bear = last["close"] < last["ema200"]

        long_trend  = is_uptrend and ema_bull and ema_200_bull
        short_trend = is_downtrend and ema_bear and ema_200_bear

        if not (long_trend or short_trend):
            return None

        # ── 3. ADX — сила тренда ──
        if last["ADX_14"] < 20:
            return None

        direction = "uptrend" if long_trend else "downtrend"

        # ── 4. Уровни Фибоначчи ──
        swing_low, swing_high = self._find_swing(df)
        if swing_low is None or swing_high is None or swing_low <= 0 or (swing_high - swing_low) / swing_low < 0.004:
            return None

        fib = FibonacciLevels(swing_low, swing_high, direction=direction)
        hit = fib.nearest(float(last["close"]), tolerance_pct=self.FIB_TOLERANCE)
        if hit is None or hit[0] not in fib.KEY_LEVELS:
            return None

        level_name, level_price, dist_pct = hit

        # ── 5. RSI разворот от экстремума ──
        if direction == "uptrend":
            rsi_rev = prev["rsi"] < 44 and last["rsi"] > prev["rsi"]
        else:
            rsi_rev = prev["rsi"] > 56 and last["rsi"] < prev["rsi"]

        # ── 6. MACD гистограмма ──
        macd_h = "MACDh_12_26_9"
        if macd_h in df.columns:
            macd_turn = (last[macd_h] > prev[macd_h]) if direction == "uptrend" else (last[macd_h] < prev[macd_h])
        else:
            macd_turn = True

        # ── 7. Объём ──
        vol_ok = last["volume"] > last["vol_ma"] * 1.1

        filters = {
            "trend_structure": True,          # HH/HL или LH/LL
            "ema_stack":       True,          # стек EMA подтверждён
            "ema200":          True,          # выше/ниже EMA200
            "adx_strength":    True,          # ADX ≥ 20
            "fib_level":       True,          # цена у ключевого уровня
            "rsi_reversal":    rsi_rev,
            "macd_turn":       macd_turn,
            "volume":          vol_ok,
        }

        if not all(filters.values()):
            return None

        side  = "BUY" if long_trend else "SELL"
        entry = float(last["close"])

        # SL за свингом + буфер
        if side == "BUY":
            sl = swing_low * 0.998
            sl = min(sl, entry * (1 - self.stop_loss_pct / 100))
            tp = fib.extension_tp("127.2")
            if tp <= entry * 1.015:
                tp = entry * (1 + self.take_profit_pct / 100)
        else:
            sl = swing_high * 1.002
            sl = max(sl, entry * (1 + self.stop_loss_pct / 100))
            tp = fib.extension_tp("127.2")
            if tp >= entry * 0.985:
                tp = entry * (1 - self.take_profit_pct / 100)

        # Уверенность растёт с силой тренда и глубиной Фибо
        fib_bonus = {"38.2": 0.00, "50.0": 0.02, "61.8": 0.04}[level_name]
        confidence = round(0.70 + 0.03 * trend["strength"] + fib_bonus, 2)

        return TradingSignal(
            action=side, symbol=self.symbol,
            confidence=min(confidence, 0.88),
            entry_price=entry, stop_loss=sl, take_profit=tp,
            reason=(
                f"Trend+Fib {side}: "
                f"{'HH/HL' if long_trend else 'LH/LL'} | "
                f"Fib {level_name}% @ {level_price:.4f} (±{dist_pct:.2f}%) | "
                f"ADX={last['ADX_14']:.1f} | RSI={last['rsi']:.1f}"
            ),
            filters_passed=filters,
        )
