"""
Адаптивный менеджер параметров стратегий.

Что делает:
  1. Отслеживает скользящую статистику по последним N сделкам каждой стратегии.
  2. Адаптирует confidence threshold — ужесточает при убыточных сериях, смягчает при прибыльных.
  3. Возвращает ATR-множители с поправкой на режим рынка и недавнюю статистику.
  4. Рассчитывает долю Келли из реальных результатов → множитель размера позиции.
  5. Логирует каждое значимое изменение параметров.

Как подключить:
  - Создать экземпляр при старте бота: state.adaptive = AdaptiveParamManager()
  - Зарегистрировать стратегии: state.adaptive.register("S1", base_confidence=0.60)
  - После закрытия сделки: state.adaptive.record("S1", pnl_r=r_multiple)
  - Перед входом: threshold = state.adaptive.confidence_threshold("S1")
  - При расчёте SL/TP: sl_mult, rr = state.adaptive.atr_params("S1", regime="trending_up")
  - Множитель позиции: mult = state.adaptive.size_multiplier("S1")
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class _StrategyStats:
    """Скользящая статистика по последним window сделкам."""

    def __init__(self, window: int = 30):
        self._pnls: deque = deque(maxlen=window)

    def add(self, pnl_r: float):
        self._pnls.append(float(pnl_r))

    @property
    def count(self) -> int:
        return len(self._pnls)

    @property
    def win_rate(self) -> float:
        if not self._pnls:
            return 0.5
        return sum(1 for p in self._pnls if p > 0) / len(self._pnls)

    @property
    def profit_factor(self) -> float:
        wins   = sum(p for p in self._pnls if p > 0)
        losses = abs(sum(p for p in self._pnls if p < 0))
        if losses == 0:
            return 2.0 if wins > 0 else 1.0
        return round(wins / losses, 3)

    @property
    def avg_win_r(self) -> float:
        wins = [p for p in self._pnls if p > 0]
        return float(np.mean(wins)) if wins else 1.0

    @property
    def avg_loss_r(self) -> float:
        losses = [abs(p) for p in self._pnls if p < 0]
        return float(np.mean(losses)) if losses else 1.0

    def kelly_fraction(self, safety: float = 0.25, cap: float = 0.50) -> float:
        """
        Четверть-Kelly: f* = (b·p − q) / b × safety.
        b = avg_win / avg_loss, p = win_rate.
        Возвращает 0.10–cap.
        """
        if self.count < 10:
            return 0.25  # консервативный дефолт пока мало данных
        wr = self.win_rate
        b  = self.avg_win_r / max(self.avg_loss_r, 0.01)
        full_kelly = max(0.0, (b * wr - (1 - wr)) / b)
        return round(min(max(full_kelly * safety, 0.10), cap), 3)


class AdaptiveParamManager:
    """
    Центральный адаптивный менеджер.

    Параметры которые адаптируются:
    ─────────────────────────────
    confidence_threshold  →  0.50–0.80 (порог ML-фильтра)
    atr_sl_mult           →  0.8–2.5   (множитель ATR для SL)
    rr_target             →  1.5–6.0   (R:R цель, масштабирует TP)
    size_multiplier       →  0.50–1.50 (доля позиции по Kelly)
    """

    # ATR-множители по режиму рынка: (sl_mult, rr_target)
    REGIME_ATR: Dict[str, Tuple[float, float]] = {
        "trending_up":   (1.5, 2.8),   # тренд вверх: стандарт
        "trending_down": (1.5, 2.8),   # тренд вниз: стандарт
        "flat":          (1.0, 1.8),   # флэт: меньше пространства
        "volatile":      (2.2, 3.5),   # волатильность: шире SL
    }
    DEFAULT_ATR: Tuple[float, float] = (1.5, 2.5)

    CONF_MIN = 0.50
    CONF_MAX = 0.80
    SIZE_MIN = 0.50
    SIZE_MAX = 1.50

    # Сколько сделок нужно для начала адаптации
    MIN_TRADES_FOR_ADAPT = 10

    def __init__(self, window: int = 30):
        self._window   = window
        self._stats:     Dict[str, _StrategyStats] = {}
        self._conf:      Dict[str, float] = {}   # текущий (адаптированный) порог
        self._base_conf: Dict[str, float] = {}   # исходный порог из стратегии

    # ── регистрация ──────────────────────────────────────────

    def register(self, strategy_id: str, base_confidence: float = 0.60):
        """Зарегистрировать стратегию. Вызывать один раз при старте."""
        if strategy_id not in self._stats:
            self._stats[strategy_id]     = _StrategyStats(self._window)
            self._conf[strategy_id]      = base_confidence
            self._base_conf[strategy_id] = base_confidence

    # ── запись результата ────────────────────────────────────

    def record(self, strategy_id: str, pnl_r: float):
        """
        Записать результат закрытой сделки (R-multiple) и пересчитать параметры.
        pnl_r > 0 = прибыль, < 0 = убыток.
        Пример: +2.0 значит заработали 2R, −1.0 значит потеряли 1R (SL).
        """
        if strategy_id not in self._stats:
            self.register(strategy_id)

        self._stats[strategy_id].add(pnl_r)
        self._recalculate(strategy_id)

    def _recalculate(self, sid: str):
        """Пересчёт confidence threshold на основе скользящей статистики."""
        stats = self._stats[sid]
        if stats.count < self.MIN_TRADES_FOR_ADAPT:
            return

        wr   = stats.win_rate
        pf   = stats.profit_factor
        base = self._base_conf.get(sid, 0.60)

        if wr < 0.40 or pf < 0.70:
            # Плохая серия — ужесточаем фильтр: требуем более высокой уверенности
            new_conf = min(base + 0.15, self.CONF_MAX)
            reason   = f"плохая серия WR={wr*100:.0f}% PF={pf:.2f}"
        elif wr < 0.50 or pf < 0.90:
            # Слабовато — немного строже
            new_conf = min(base + 0.08, self.CONF_MAX)
            reason   = f"слабая серия WR={wr*100:.0f}% PF={pf:.2f}"
        elif wr > 0.70 and pf > 1.50:
            # Отличная серия — чуть мягче, больше сигналов
            new_conf = max(base - 0.05, self.CONF_MIN)
            reason   = f"сильная серия WR={wr*100:.0f}% PF={pf:.2f}"
        else:
            # Норма — возвращаем к базовому значению
            new_conf = base
            reason   = "норма"

        old = self._conf.get(sid, base)
        if abs(new_conf - old) > 0.005:
            logger.info(
                f"[Adaptive] {sid}: порог {old:.2f} → {new_conf:.2f} "
                f"({reason}, {stats.count} сделок)"
            )
        self._conf[sid] = round(new_conf, 3)

    # ── получение параметров ─────────────────────────────────

    def confidence_threshold(self, strategy_id: str) -> float:
        """Текущий адаптированный порог для ML-фильтра."""
        return self._conf.get(strategy_id, 0.60)

    def atr_params(
        self,
        strategy_id: str,
        regime: Optional[str] = None,
    ) -> Tuple[float, float]:
        """
        Возвращает (sl_atr_mult, rr_target) — входные параметры для adaptive_sl_tp().

        Базовые значения берутся из REGIME_ATR по текущему режиму рынка,
        затем корректируются на основе profit_factor последних сделок стратегии.
        """
        sl_mult, rr = self.REGIME_ATR.get(regime or "", self.DEFAULT_ATR)

        stats = self._stats.get(strategy_id)
        if stats and stats.count >= self.MIN_TRADES_FOR_ADAPT:
            pf = stats.profit_factor
            if pf < 0.80:
                # Теряем: уменьшаем SL (меньше риск), слегка увеличиваем TP-цель
                sl_mult = max(sl_mult * 0.85, 0.8)
                rr      = min(rr * 1.10, 5.0)
            elif pf > 1.60:
                # Стабильно зарабатываем: можно давать больше пространства для TP
                rr = min(rr * 1.15, 6.0)

        return round(sl_mult, 2), round(rr, 2)

    def size_multiplier(self, strategy_id: str) -> float:
        """
        Множитель размера позиции 0.50–1.50×.

        Вычисляется как функция от Kelly-доли реальных результатов:
          kelly ≈ 0.10 → mult = 0.70×  (осторожно)
          kelly ≈ 0.25 → mult = 1.00×  (стандарт)
          kelly ≈ 0.50 → mult = 1.50×  (хорошая серия)
        """
        stats = self._stats.get(strategy_id)
        if not stats or stats.count < self.MIN_TRADES_FOR_ADAPT:
            return 1.0

        k    = stats.kelly_fraction()
        mult = 0.50 + k * 2.0
        return round(min(max(mult, self.SIZE_MIN), self.SIZE_MAX), 2)

    # ── статус для UI ────────────────────────────────────────

    def status(self) -> Dict:
        """Статус всех стратегий — для отображения в UI/API."""
        result = {}
        for sid, stats in self._stats.items():
            result[sid] = {
                "trades_tracked":      stats.count,
                "win_rate_pct":        round(stats.win_rate * 100, 1),
                "profit_factor":       stats.profit_factor,
                "confidence_base":     self._base_conf.get(sid, 0.60),
                "confidence_current":  self._conf.get(sid, 0.60),
                "size_multiplier":     self.size_multiplier(sid),
                "kelly_fraction":      stats.kelly_fraction() if stats.count >= self.MIN_TRADES_FOR_ADAPT else None,
            }
        return result

    def summary_line(self) -> str:
        """Однострочный лог-дамп для отладки."""
        parts = []
        for sid, stats in self._stats.items():
            if stats.count == 0:
                continue
            parts.append(
                f"{sid}(n={stats.count} WR={stats.win_rate*100:.0f}% "
                f"conf={self._conf.get(sid, 0.60):.2f} "
                f"size×{self.size_multiplier(sid)})"
            )
        return " | ".join(parts) if parts else "нет данных"
