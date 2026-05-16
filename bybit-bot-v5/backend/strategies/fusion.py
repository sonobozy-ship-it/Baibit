"""
Strategy Fusion Engine — автоматическое объединение стратегий.

Логика:
  1. SignalBuffer хранит последний сигнал каждой стратегии с TTL (2 × период TF).
  2. StrategyFusion.evaluate() вызывается после каждого прохода цикла.
  3. Если 2+ стратегий согласны по направлению на одном символе →
     вычисляется fusion_score из:
       - количество согласных стратегий (confluence)
       - разнообразие групп индикаторов (diversity)  ← ключевое
       - средняя ML-уверенность сигналов
       - наличие Fibonacci / структурного подтверждения (бонус)
  4. При fusion_score ≥ MIN_FUSION_SCORE создаётся FusedSignal:
       - SL: самый консервативный из всех
       - TP: приоритет S9 (Fib), иначе взвешенное среднее
       - size_multiplier: 1.0–1.6× от базового размера позиции
  5. Фьюжн-сигналы сохраняются с strategy_id="FUSION" → обучаются отдельно.

Группы индикаторов (разные группы = выше diversity_score):
  trend_ema       — S1, S6, S8, S9
  momentum_rsi    — S1, S2, S3
  volatility_bb   — S2, S5
  divergence      — S3
  breakout        — S4
  volume          — S4
  mean_reversion  — S5
  strength_adx    — S6, S8
  structure_swing — S8, S9
  fibonacci       — S9
  multi_confirm   — S7
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import TradingSignal

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# FusedSignal
# ────────────────────────────────────────────────────────────

class FusedSignal(TradingSignal):
    """TradingSignal расширенный метаданными об объединении."""

    def __init__(
        self,
        *args,
        source_strategies: List[str],
        fusion_score: float,
        diversity_score: float,
        size_multiplier: float,
        groups_used: List[str],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.source_strategies = source_strategies
        self.fusion_score      = fusion_score
        self.diversity_score   = diversity_score
        self.size_multiplier   = size_multiplier
        self.groups_used       = groups_used
        self.is_fusion         = True


# ────────────────────────────────────────────────────────────
# SignalBuffer — хранит свежие сигналы по TTL таймфрейма
# ────────────────────────────────────────────────────────────

class SignalBuffer:
    """
    Кольцевой буфер последних сигналов от каждой стратегии.
    Сигнал считается «свежим» пока не прошло 2 × период таймфрейма.
    """

    # Период таймфрейма в минутах
    TF_MINUTES: Dict[str, int] = {
        "1": 1, "3": 3, "5": 5, "15": 15, "30": 30,
        "60": 60, "120": 120, "240": 240, "360": 360,
        "720": 720, "D": 1440, "W": 10080,
    }
    TTL_FACTOR = 2  # сигнал живёт 2 × период TF

    def __init__(self):
        # sid → {"signal": TradingSignal, "ts": datetime, "tf": str}
        self._buf: Dict[str, dict] = {}

    def update(self, sid: str, signal: TradingSignal, timeframe: str):
        self._buf[sid] = {
            "signal":    signal,
            "ts":        datetime.utcnow(),
            "timeframe": str(timeframe),
        }

    def fresh(self) -> Dict[str, TradingSignal]:
        """Возвращает только актуальные сигналы."""
        now    = datetime.utcnow()
        result = {}
        for sid, entry in list(self._buf.items()):
            tf_min  = self.TF_MINUTES.get(entry["timeframe"], 60)
            ttl_min = tf_min * self.TTL_FACTOR
            age_min = (now - entry["ts"]).total_seconds() / 60
            if age_min <= ttl_min:
                result[sid] = entry["signal"]
        return result

    def invalidate(self, sid: str):
        """Удалить сигнал стратегии (после открытия сделки)."""
        self._buf.pop(sid, None)

    def clear(self):
        self._buf.clear()

    def summary(self) -> str:
        fresh = self.fresh()
        if not fresh:
            return "buffer empty"
        parts = [f"{sid}:{sig.action}@{sig.symbol}" for sid, sig in fresh.items()]
        return " | ".join(parts)


# ────────────────────────────────────────────────────────────
# StrategyFusion
# ────────────────────────────────────────────────────────────

class StrategyFusion:
    """
    Оценивает свежие сигналы на предмет confluence.
    При нахождении сильного совпадения генерирует FusedSignal.
    """

    # Индикаторные группы: разные группы → выше diversity
    INDICATOR_GROUPS: Dict[str, List[str]] = {
        "S1":  ["trend_ema",      "momentum_rsi"],
        "S2":  ["volatility_bb",  "momentum_rsi"],
        "S3":  ["divergence",     "momentum_macd"],
        "S4":  ["breakout",       "volume"],
        "S5":  ["mean_reversion", "volatility_bb"],
        "S6":  ["trend_ema",      "strength_adx"],
        "S7":  ["multi_confirm"],
        "S8":  ["structure_swing","trend_ema",  "strength_adx"],
        "S9":  ["fibonacci",      "structure_swing", "trend_ema"],
        "S10": ["scalp_ema",      "scalp_rsi",  "volatility_bb"],
        "S11": ["pattern_candle", "momentum_rsi", "volume"],   # Dragonfly Gold: свечи+RSI+объём
    }

    # Веса стратегий (выше = заслуживают больше доверия при fusion)
    STRATEGY_WEIGHTS: Dict[str, float] = {
        "S7":  1.3,   # Multi-Confirm: сам уже объединяет много фильтров
        "S9":  1.4,   # Trend+Fibonacci: самый комплексный сигнал
        "S8":  1.2,   # Trend Momentum: структурный анализ
        "S11": 1.2,   # Dragonfly Gold: уникальный паттерн-детектор
        "S6":  1.1,   # Trend Follower: ADX + Supertrend
    }
    DEFAULT_WEIGHT = 1.0

    MIN_CONFLUENCE    = 2     # минимум стратегий для fusion
    MIN_FUSION_SCORE  = 0.52  # минимальный score для создания FusedSignal
    COOLDOWN_MINUTES  = 30    # не повторять fusion на том же символе+направлении

    # Размер позиции: каждая доп. стратегия добавляет +12%
    BASE_SIZE_MULT  = 1.00
    PER_STRAT_BONUS = 0.12
    MAX_SIZE_MULT   = 1.60

    def __init__(self):
        self._last_fusion: Dict[str, datetime] = {}   # f"{symbol}_{dir}" → ts

    # ─── публичный API ─────────────────────────────────────

    def evaluate(
        self,
        buffer: SignalBuffer,
        regime: Optional[str] = None,
    ) -> Optional[FusedSignal]:
        """
        Главный метод: оценивает буфер и возвращает FusedSignal или None.
        Вызывается один раз после обхода всех стратегий в торговом цикле.
        """
        fresh = buffer.fresh()
        if len(fresh) < self.MIN_CONFLUENCE:
            return None

        # Группируем по направлению
        longs  = {sid: s for sid, s in fresh.items() if s.action == "BUY"}
        shorts = {sid: s for sid, s in fresh.items() if s.action == "SELL"}

        for signals, direction in [(longs, "BUY"), (shorts, "SELL")]:
            result = self._try_fuse(signals, direction, regime)
            if result is not None:
                logger.info(
                    f"[Fusion] {direction} [{'+'.join(result.source_strategies)}] "
                    f"score={result.fusion_score:.3f} "
                    f"diversity={result.diversity_score:.2f} "
                    f"size×{result.size_multiplier}"
                )
                return result

        return None

    # ─── внутренняя логика ──────────────────────────────────

    def _try_fuse(
        self,
        signals: Dict[str, TradingSignal],
        direction: str,
        regime: Optional[str],
    ) -> Optional[FusedSignal]:
        if len(signals) < self.MIN_CONFLUENCE:
            return None

        # Группируем по символу, берём символ с наибольшим числом согласных
        by_symbol: Dict[str, Dict[str, TradingSignal]] = {}
        for sid, sig in signals.items():
            by_symbol.setdefault(sig.symbol, {})[sid] = sig

        best_sym     = max(by_symbol, key=lambda s: len(by_symbol[s]))
        best_signals = by_symbol[best_sym]

        if len(best_signals) < self.MIN_CONFLUENCE:
            return None

        # ── Cooldown: не открываем fusion слишком часто ──
        cooldown_key = f"{best_sym}_{direction}"
        last_ts = self._last_fusion.get(cooldown_key)
        if last_ts and (datetime.utcnow() - last_ts) < timedelta(minutes=self.COOLDOWN_MINUTES):
            return None

        # ── Diversity score ──
        groups_used: set = set()
        for sid in best_signals:
            groups_used.update(self.INDICATOR_GROUPS.get(sid, []))
        diversity_score = min(len(groups_used) / 7.0, 1.0) if groups_used else 0.0

        # ── Confluence score (взвешенный) ──
        weights  = [self.STRATEGY_WEIGHTS.get(sid, self.DEFAULT_WEIGHT) for sid in best_signals]
        w_sum    = sum(weights)
        n        = len(best_signals)
        confluence_score = min(w_sum / (5 * self.DEFAULT_WEIGHT), 1.0)

        # ── Средняя ML-уверенность ──
        confidences  = [sig.confidence for sig in best_signals.values()]
        avg_conf     = float(np.mean(confidences))

        # ── Бонус за ключевые стратегии ──
        fib_bonus   = 0.08 if "S9" in best_signals else 0.0
        multi_bonus = 0.05 if "S7" in best_signals else 0.0
        struct_bonus= 0.03 if "S8" in best_signals else 0.0

        # ── Режим рынка: совпадение повышает score ──
        regime_bonus = 0.0
        if regime:
            regime_fits = sum(
                1 for sid in best_signals
                # REGIME_PREFERENCE доступен через сигнал — нет, используем известные предпочтения
                if regime in self._regime_prefs(sid)
            )
            regime_bonus = 0.04 * regime_fits

        fusion_score = (
            0.35 * confluence_score
            + 0.30 * diversity_score
            + 0.15 * avg_conf
            + 0.08 * (n / 5.0)   # raw count
            + fib_bonus + multi_bonus + struct_bonus + regime_bonus
        )
        fusion_score = round(min(fusion_score, 1.0), 4)

        if fusion_score < self.MIN_FUSION_SCORE:
            logger.debug(
                f"[Fusion] {direction} {best_sym}: score={fusion_score:.3f} < "
                f"{self.MIN_FUSION_SCORE} (n={n}, div={diversity_score:.2f})"
            )
            return None

        # ── Слияние SL / TP ──
        entry_prices = [sig.entry_price for sig in best_signals.values()]
        entry = float(np.median(entry_prices))

        if direction == "BUY":
            # SL: максимальный (наиболее консервативный / ближний к цене)
            raw_sl = max(sig.stop_loss for sig in best_signals.values())
            sl = max(raw_sl, entry * 0.985)   # не ближе 1.5%
            # TP: S9 имеет приоритет (содержит Fibonacci расширение)
            if "S9" in best_signals:
                tp = best_signals["S9"].take_profit
            else:
                tp = float(np.average(
                    [sig.take_profit for sig in best_signals.values()],
                    weights=confidences,
                ))
        else:
            raw_sl = min(sig.stop_loss for sig in best_signals.values())
            sl = min(raw_sl, entry * 1.015)
            if "S9" in best_signals:
                tp = best_signals["S9"].take_profit
            else:
                tp = float(np.average(
                    [sig.take_profit for sig in best_signals.values()],
                    weights=confidences,
                ))

        # ── Множитель позиции ──
        size_mult = self.BASE_SIZE_MULT + self.PER_STRAT_BONUS * (n - 2)
        size_mult = round(min(size_mult + (fusion_score - 0.52) * 0.3, self.MAX_SIZE_MULT), 2)

        source_ids = sorted(best_signals.keys())
        self._last_fusion[cooldown_key] = datetime.utcnow()

        return FusedSignal(
            action=direction,
            symbol=best_sym,
            confidence=round(min(0.65 + fusion_score * 0.30, 0.94), 2),
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            reason=(
                f"FUSION {direction} [{'+'.join(source_ids)}] "
                f"score={fusion_score:.3f} | "
                f"{n} стратегий | {len(groups_used)} групп индикаторов | "
                f"diversity={diversity_score:.2f}"
            ),
            filters_passed={
                sid: sig.filters_passed for sid, sig in best_signals.items()
            },
            source_strategies=source_ids,
            fusion_score=fusion_score,
            diversity_score=diversity_score,
            size_multiplier=size_mult,
            groups_used=sorted(groups_used),
        )

    @staticmethod
    def _regime_prefs(sid: str) -> List[str]:
        """Предпочтительные режимы для каждой стратегии (дублируем константы)."""
        prefs = {
            "S1": ["uptrend", "downtrend"],
            "S2": ["volatile", "flat"],
            "S3": [],
            "S4": ["uptrend", "downtrend", "volatile"],
            "S5": ["flat"],
            "S6": ["uptrend", "downtrend"],
            "S7": [],
            "S8": ["uptrend", "downtrend"],
            "S9": ["uptrend", "downtrend"],
        }
        return prefs.get(sid, [])

    def get_fusion_features(
        self,
        fused: FusedSignal,
        base_features: dict,
    ) -> dict:
        """Добавляет fusion-специфичные фичи к базовым фичам для ML."""
        extra = {
            "fusion_n_strategies":   len(fused.source_strategies),
            "fusion_score":          fused.fusion_score,
            "fusion_diversity":      fused.diversity_score,
            "fusion_size_mult":      fused.size_multiplier,
            "fusion_has_fib":        int("S9" in fused.source_strategies),
            "fusion_has_multi":      int("S7" in fused.source_strategies),
            "fusion_has_struct":     int("S8" in fused.source_strategies),
            "fusion_avg_confidence": fused.confidence,
        }
        return {**base_features, **extra}
