"""
TimeRate — система кулдаунов, anti-overtrade и защиты от серий убытков.

Кулдаун после убытков (per symbol):
  1 убыток  → same_symbol = 15 мин
  2 убытка  → same_symbol = 45 мин
  3 убытка  → стратегия заблокирована 120 мин

После прибыли: same_symbol = 3 мин (дать рынку остыть)
После любого закрытия: min_reentry = 180 сек
После убытка: min_reentry = 900 сек (15 мин)

Anti-overtrade:
  max 4 входа в час по одному символу
  max 3 входа за 15 мин по одной стратегии

Loss streak protection:
  symbol_consecutive_losses_limit  = 2 → 45 мин кулдаун
  strategy_consecutive_losses_limit = 3 → 120 мин блок
"""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Dict, Deque, Optional

logger = logging.getLogger(__name__)


class TimeRateManager:
    """
    Центральный менеджер кулдаунов и частоты торговли.
    Один экземпляр на весь бот — хранится в BotState.

    Публичный API:
      can_trade(strategy_id, symbol)      → {"ok": bool, "reason": str}
      check_overtrade(strategy_id, symbol) → {"ok": bool, "reason": str}
      register_trade_open(strategy_id, symbol)
      register_close(strategy_id, symbol, pnl_usd)
      reset_strategy(strategy_id)
      status()                            → dict для /status
    """

    # ── кулдауны по символу ───────────────────────────────────────────────────
    COOLDOWN_1_LOSS_MIN   = 15    # 1 убыток по символу
    COOLDOWN_2_LOSSES_MIN = 45    # 2 убытка подряд по символу
    COOLDOWN_PROFIT_MIN   = 3     # после прибыли (рынку остыть)

    # ── блокировка стратегии ─────────────────────────────────────────────────
    STRAT_BLOCK_LOSSES    = 3     # сколько убытков подряд → блок
    STRAT_BLOCK_MIN       = 120   # минут блокировки стратегии

    # ── min reentry ──────────────────────────────────────────────────────────
    MIN_REENTRY_SEC       = 180   # после любого закрытия
    MIN_REENTRY_LOSS_SEC  = 900   # после убытка (15 мин)

    # ── anti-overtrade ───────────────────────────────────────────────────────
    MAX_ENTRIES_SYMBOL_HOUR  = 4
    MAX_ENTRIES_STRATEGY_15M = 3

    def __init__(self) -> None:
        # Per-symbol: метка времени последнего закрытия и была ли прибыль
        self._last_close_time:    Dict[str, datetime] = {}
        self._last_close_profit:  Dict[str, bool]     = {}

        # Per-symbol: серия убытков и кулдаун
        self._sym_consec_losses:  Dict[str, int]      = defaultdict(int)
        self._sym_cooldown_until: Dict[str, datetime] = {}

        # Per-strategy: серия убытков и блокировка
        self._strat_consec_losses: Dict[str, int]     = defaultdict(int)
        self._strat_blocked_until: Dict[str, datetime] = {}

        # Anti-overtrade: временны́е метки входов
        self._sym_entries:   Dict[str, Deque[datetime]] = defaultdict(lambda: deque(maxlen=20))
        self._strat_entries: Dict[str, Deque[datetime]] = defaultdict(lambda: deque(maxlen=10))

    # ── публичные методы ──────────────────────────────────────────────────────

    def can_trade(self, strategy_id: str, symbol: str) -> Dict:
        """
        Можно ли открыть новую позицию для данной стратегии/символа?

        Returns {"ok": bool, "reason": str, "cooldown_min": float}
        """
        now = datetime.now(timezone.utc)

        # 1. Стратегия заблокирована (3 убытка подряд)?
        strat_block = self._strat_blocked_until.get(strategy_id)
        if strat_block and now < strat_block:
            rem = round((strat_block - now).total_seconds() / 60, 1)
            return {
                "ok":           False,
                "reason":       f"STRAT_BLOCKED:{rem}min ({strategy_id}, 3×SL)",
                "cooldown_min": rem,
            }

        # 2. Символ в кулдауне (убытки)?
        sym_cool = self._sym_cooldown_until.get(symbol)
        if sym_cool and now < sym_cool:
            rem = round((sym_cool - now).total_seconds() / 60, 1)
            return {
                "ok":           False,
                "reason":       f"SYM_COOLDOWN:{rem}min ({symbol})",
                "cooldown_min": rem,
            }

        # 3. Минимальный reentry после закрытия
        last_close = self._last_close_time.get(symbol)
        if last_close:
            elapsed_sec = (now - last_close).total_seconds()
            was_profit  = self._last_close_profit.get(symbol, True)
            min_sec     = self.MIN_REENTRY_LOSS_SEC if not was_profit else self.MIN_REENTRY_SEC
            if elapsed_sec < min_sec:
                rem = round((min_sec - elapsed_sec) / 60, 1)
                label = "LOSS_REENTRY" if not was_profit else "MIN_REENTRY"
                return {
                    "ok":           False,
                    "reason":       f"{label}:{rem}min ({symbol})",
                    "cooldown_min": rem,
                }

        return {"ok": True, "reason": "ok", "cooldown_min": 0.0}

    def check_overtrade(self, strategy_id: str, symbol: str) -> Dict:
        """
        Anti-overtrade проверка.

        Блокирует если:
          - > 4 входа по одному символу за последний час
          - > 3 входа по одной стратегии за последние 15 мин
        """
        now         = datetime.now(timezone.utc)
        cutoff_hour = now - timedelta(hours=1)
        cutoff_15m  = now - timedelta(minutes=15)

        sym_recent   = [t for t in self._sym_entries.get(symbol, [])       if t > cutoff_hour]
        strat_recent = [t for t in self._strat_entries.get(strategy_id, []) if t > cutoff_15m]

        if len(sym_recent) >= self.MAX_ENTRIES_SYMBOL_HOUR:
            return {
                "ok":     False,
                "reason": f"OVERTRADE:sym {symbol} {len(sym_recent)}/{self.MAX_ENTRIES_SYMBOL_HOUR}/h",
            }
        if len(strat_recent) >= self.MAX_ENTRIES_STRATEGY_15M:
            return {
                "ok":     False,
                "reason": f"OVERTRADE:strat {strategy_id} {len(strat_recent)}/{self.MAX_ENTRIES_STRATEGY_15M}/15m",
            }

        return {"ok": True, "reason": "ok"}

    def register_trade_open(self, strategy_id: str, symbol: str) -> None:
        """Регистрируем факт открытия сделки для anti-overtrade трекинга."""
        now = datetime.now(timezone.utc)
        self._sym_entries[symbol].append(now)
        self._strat_entries[strategy_id].append(now)

    def register_close(self, strategy_id: str, symbol: str, pnl_usd: float) -> None:
        """
        Регистрируем закрытие позиции и начисляем кулдаун.

        pnl_usd < -0.01  → убыток → кулдаун + счётчик серии
        pnl_usd >= -0.01 → прибыль (с дедзоной на комиссии) → короткий кулдаун + сброс серии
        """
        now     = datetime.now(timezone.utc)
        is_loss = pnl_usd < -0.01

        self._last_close_time[symbol]   = now
        self._last_close_profit[symbol] = not is_loss

        if is_loss:
            # Серия убытков по символу
            self._sym_consec_losses[symbol] += 1
            sym_losses = self._sym_consec_losses[symbol]

            # Серия убытков по стратегии
            self._strat_consec_losses[strategy_id] += 1
            strat_losses = self._strat_consec_losses[strategy_id]

            # Кулдаун по символу
            cool_min = (
                self.COOLDOWN_2_LOSSES_MIN if sym_losses >= 2
                else self.COOLDOWN_1_LOSS_MIN
            )
            self._sym_cooldown_until[symbol] = now + timedelta(minutes=cool_min)
            logger.info(
                f"[TimeRate] {strategy_id}/{symbol}: убыток #{sym_losses} → кулдаун {cool_min}мин"
            )

            # Блокировка стратегии после STRAT_BLOCK_LOSSES убытков подряд
            if strat_losses >= self.STRAT_BLOCK_LOSSES:
                self._strat_blocked_until[strategy_id] = now + timedelta(minutes=self.STRAT_BLOCK_MIN)
                self._strat_consec_losses[strategy_id] = 0
                logger.warning(
                    f"[TimeRate] {strategy_id}: {self.STRAT_BLOCK_LOSSES} убытка подряд → "
                    f"блок {self.STRAT_BLOCK_MIN}мин"
                )
        else:
            # Прибыль — сброс серий, короткий кулдаун
            self._sym_consec_losses[symbol]    = 0
            self._strat_consec_losses[strategy_id] = 0
            self._sym_cooldown_until[symbol]   = now + timedelta(minutes=self.COOLDOWN_PROFIT_MIN)

    def reset_strategy(self, strategy_id: str) -> None:
        """Ручной сброс блокировки стратегии (например через /reset команду)."""
        self._strat_blocked_until.pop(strategy_id, None)
        self._strat_consec_losses[strategy_id] = 0
        logger.info(f"[TimeRate] {strategy_id}: ручной сброс блокировки")

    def reset_symbol(self, symbol: str) -> None:
        """Ручной сброс кулдауна символа."""
        self._sym_cooldown_until.pop(symbol, None)
        self._sym_consec_losses[symbol] = 0

    def status(self) -> Dict:
        """Текущее состояние для /status эндпоинта."""
        now = datetime.now(timezone.utc)
        return {
            "blocked_strategies": {
                k: round((v - now).total_seconds() / 60, 1)
                for k, v in self._strat_blocked_until.items()
                if v > now
            },
            "cooled_symbols": {
                k: round((v - now).total_seconds() / 60, 1)
                for k, v in self._sym_cooldown_until.items()
                if v > now
            },
            "sym_consec_losses":   dict(self._sym_consec_losses),
            "strat_consec_losses": dict(self._strat_consec_losses),
        }
