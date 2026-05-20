"""
Position Monitor — умное управление открытыми позициями.

Логика закрытия (проверяется в порядке приоритета):
  1. tp_progress   — достигли 50% от запланированной прибыли
  2. near_tp       — цена прошла 80% пути к TP
  3. trailing      — прибыль откатила на 30% от максимума
  4. profit_return — прибыль была высокой, потом резко упала
  5. timeout       — позиция слишком долго с маленькой прибылью

Всё настраивается через ENV (см. config.py).
Поддерживает paper и real режимы.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import config as cfg

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Структура состояния одной позиции (хранится в памяти, не в БД)
# ─────────────────────────────────────────────────────────────────────────────
class PositionState:
    """Внутреннее состояние позиции для мониторинга."""

    def __init__(self):
        self.max_pnl: float = 0.0            # максимальный наблюдавшийся PnL
        self.trailing_armed: bool = False     # включён трейлинг прибыли
        self.anti_return_armed: bool = False  # включён anti profit-return
        self.opened_at: Optional[datetime] = None


class CloseDecision:
    """Решение о закрытии позиции."""

    def __init__(self, should_close: bool, reason: str = "", details: str = ""):
        self.should_close = should_close
        self.reason = reason          # машинный ключ: tp_progress, trailing, near_tp, …
        self.details = details        # человекочитаемый текст для Telegram

    def __bool__(self) -> bool:
        return self.should_close


# ─────────────────────────────────────────────────────────────────────────────
# Расчёт unrealized PnL
# ─────────────────────────────────────────────────────────────────────────────

def calc_unrealized_pnl(
    side: str,
    entry_price: float,
    current_price: float,
    qty: float,
    leverage: int = 1,
) -> float:
    """
    Считает unrealized PnL с учётом плеча.
    side: "Buy" или "Sell" (или "BUY"/"SELL")
    """
    direction = 1 if side.lower() in ("buy", "long") else -1
    return direction * (current_price - entry_price) * qty * leverage


def calc_tp_profit_usdt(
    side: str,
    entry_price: float,
    take_profit: float,
    qty: float,
    leverage: int = 1,
) -> float:
    """Планируемая прибыль при достижении TP."""
    return abs(take_profit - entry_price) * qty * leverage


# ─────────────────────────────────────────────────────────────────────────────
# Функции проверки (чистые, без side-эффектов)
# ─────────────────────────────────────────────────────────────────────────────

def should_close_tp_progress(
    current_pnl: float,
    tp_profit_usdt: float,
    threshold_pct: float = None,
) -> CloseDecision:
    """
    Закрыть если заработали TP_CLOSE_PERCENT от запланированной прибыли.
    Пример: TP = 0.92 USDT, порог 50% → закрыть при PnL >= 0.46 USDT.
    """
    pct = threshold_pct if threshold_pct is not None else cfg.TP_CLOSE_PERCENT
    if tp_profit_usdt <= 0:
        return CloseDecision(False)
    threshold = tp_profit_usdt * pct / 100.0
    if current_pnl >= threshold:
        return CloseDecision(
            True, "tp_progress",
            f"Достигнуто {pct:.0f}% от TP: PnL {current_pnl:.4f} ≥ {threshold:.4f} USDT",
        )
    return CloseDecision(False)


def should_close_trailing_profit(
    current_pnl: float,
    pos_state: PositionState,
    min_arm_usdt: float = None,
    drop_pct: float = None,
) -> CloseDecision:
    """
    Trailing profit:
    - Вооружается при PnL >= MIN_PROFIT_ARM_USDT
    - Закрывает если PnL откатил на TRAILING_DROP_PERCENT от max_pnl
    """
    min_arm = min_arm_usdt if min_arm_usdt is not None else cfg.MIN_PROFIT_ARM_USDT
    drop = drop_pct if drop_pct is not None else cfg.TRAILING_DROP_PERCENT

    # Обновляем максимум
    if current_pnl > pos_state.max_pnl:
        pos_state.max_pnl = current_pnl

    # Вооружаемся
    if not pos_state.trailing_armed and pos_state.max_pnl >= min_arm:
        pos_state.trailing_armed = True
        logger.debug(f"Trailing profit armed at max_pnl={pos_state.max_pnl:.4f}")

    if not pos_state.trailing_armed or pos_state.max_pnl <= 0:
        return CloseDecision(False)

    # Проверяем откат
    drop_threshold = pos_state.max_pnl * (1 - drop / 100.0)
    if current_pnl <= drop_threshold:
        return CloseDecision(
            True, "trailing_profit",
            f"Trailing: PnL {current_pnl:.4f} упал на {drop:.0f}% от макс {pos_state.max_pnl:.4f} USDT",
        )
    return CloseDecision(False)


def should_close_near_tp(
    side: str,
    entry_price: float,
    current_price: float,
    take_profit: float,
    current_pnl: float,
    near_pct: float = None,
) -> CloseDecision:
    """
    Закрыть если цена прошла CLOSE_NEAR_TP% пути к TP (и PnL положительный).
    """
    pct = near_pct if near_pct is not None else cfg.CLOSE_NEAR_TP
    if current_pnl <= 0:
        return CloseDecision(False)

    tp_distance = abs(take_profit - entry_price)
    if tp_distance <= 0:
        return CloseDecision(False)

    if side.upper() in ("BUY", "LONG"):
        progress_pct = (current_price - entry_price) / tp_distance * 100
    else:
        progress_pct = (entry_price - current_price) / tp_distance * 100

    if progress_pct >= pct:
        return CloseDecision(
            True, "near_tp",
            f"Цена прошла {progress_pct:.1f}% пути к TP (порог {pct:.0f}%)",
        )
    return CloseDecision(False)


def should_close_timeout(
    pos_state: PositionState,
    current_pnl: float,
    max_minutes: int = None,
    min_profit: float = None,
) -> CloseDecision:
    """
    Закрыть зависшие позиции: открыты > MAX_POSITION_MINUTES и PnL < MIN_EXPECTED_PROFIT.
    """
    max_min = max_minutes if max_minutes is not None else cfg.MAX_POSITION_MINUTES
    min_pnl = min_profit if min_profit is not None else cfg.MIN_EXPECTED_PROFIT

    if pos_state.opened_at is None:
        return CloseDecision(False)

    now = datetime.now(timezone.utc)
    if pos_state.opened_at.tzinfo is None:
        opened = pos_state.opened_at.replace(tzinfo=timezone.utc)
    else:
        opened = pos_state.opened_at

    held_min = (now - opened).total_seconds() / 60.0
    if held_min >= max_min and current_pnl < min_pnl:
        return CloseDecision(
            True, "timeout",
            f"Позиция {held_min:.0f} мин, PnL {current_pnl:.4f} < {min_pnl} USDT",
        )
    return CloseDecision(False)


def should_close_profit_return(
    current_pnl: float,
    pos_state: PositionState,
    arm_usdt: float = None,
    drop_pct: float = None,
) -> CloseDecision:
    """
    Anti profit-return: если прибыль была > arm_usdt, затем упала на drop_pct% → закрыть.
    Дополняет trailing (срабатывает при резком откате после высокого пика).
    """
    arm = arm_usdt if arm_usdt is not None else cfg.ANTI_RETURN_ARM_USDT
    drop = drop_pct if drop_pct is not None else cfg.ANTI_RETURN_DROP_PCT

    if not pos_state.anti_return_armed and pos_state.max_pnl >= arm:
        pos_state.anti_return_armed = True

    if not pos_state.anti_return_armed:
        return CloseDecision(False)

    drop_threshold = pos_state.max_pnl * (1 - drop / 100.0)
    if current_pnl <= drop_threshold and pos_state.max_pnl >= arm:
        return CloseDecision(
            True, "profit_return",
            f"Прибыль откатила: макс {pos_state.max_pnl:.4f} → {current_pnl:.4f} USDT "
            f"(падение >{drop:.0f}%)",
        )
    return CloseDecision(False)


# ─────────────────────────────────────────────────────────────────────────────
# Главный класс
# ─────────────────────────────────────────────────────────────────────────────

class PositionMonitor:
    """
    Мониторит все открытые позиции и принимает решения о закрытии.
    Используется в trading_loop как отдельный шаг.
    """

    def __init__(self):
        # sid → PositionState
        self._states: Dict[str, PositionState] = {}

    def get_state(self, sid: str) -> PositionState:
        if sid not in self._states:
            self._states[sid] = PositionState()
        return self._states[sid]

    def reset_state(self, sid: str):
        self._states.pop(sid, None)

    def check_position(
        self,
        sid: str,
        pos: Dict[str, Any],
        current_price: float,
    ) -> Optional[CloseDecision]:
        """
        Проверяет одну позицию и возвращает решение о закрытии (или None).

        pos — словарь current_position стратегии:
            side, entry, tp, sl, qty, leverage, opened_at
        """
        side       = pos.get("side", "Buy")
        entry      = float(pos.get("entry", 0))
        take_profit= float(pos.get("tp", 0))
        qty        = float(pos.get("qty", 0))
        leverage   = int(pos.get("leverage", 1))
        opened_at  = pos.get("opened_at")

        if entry <= 0 or qty <= 0:
            return None

        state = self.get_state(sid)

        # Инициализируем время входа
        if state.opened_at is None and opened_at:
            try:
                if isinstance(opened_at, str):
                    dt = datetime.fromisoformat(opened_at.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    state.opened_at = dt
                elif isinstance(opened_at, datetime):
                    if opened_at.tzinfo is None:
                        state.opened_at = opened_at.replace(tzinfo=timezone.utc)
                    else:
                        state.opened_at = opened_at
            except Exception:
                pass

        # Рассчитываем текущий PnL
        current_pnl = calc_unrealized_pnl(side, entry, current_price, qty, leverage)
        tp_profit   = calc_tp_profit_usdt(side, entry, take_profit, qty, leverage)

        # Проверяем в порядке приоритета
        checks = [
            should_close_tp_progress(current_pnl, tp_profit),
            should_close_near_tp(side, entry, current_price, take_profit, current_pnl),
            should_close_trailing_profit(current_pnl, state),
            should_close_profit_return(current_pnl, state),
            should_close_timeout(state, current_pnl),
        ]
        for decision in checks:
            if decision.should_close:
                logger.info(f"[Monitor] {sid} → CLOSE ({decision.reason}): {decision.details}")
                return decision

        return None

    def format_close_message(
        self,
        sid: str,
        symbol: str,
        side: str,
        pnl: float,
        margin: float,
        held_min: float,
        reason: str,
        entry: float,
        exit_price: float,
    ) -> str:
        """Форматирует Telegram-сообщение о закрытии позиции."""
        emoji = "✅" if pnl >= 0 else "❌"
        reason_labels = {
            "tp_progress":    "🎯 Достигнут % TP",
            "trailing_profit":"📈 Трейлинг прибыли",
            "near_tp":        "🏁 Близко к TP",
            "timeout":        "⏰ Таймаут",
            "profit_return":  "🛡 Защита прибыли",
            "risk_limit":     "🚨 Лимит риска",
        }
        reason_str = reason_labels.get(reason, reason)
        pnl_str = f"+{pnl:.4f}" if pnl >= 0 else f"{pnl:.4f}"
        side_emoji = "🟢" if side.upper() in ("BUY", "LONG") else "🔴"

        return (
            f"{emoji} <b>Закрыта позиция</b>\n\n"
            f"Пара: <b>{symbol}</b>\n"
            f"Side: {side_emoji} <b>{side.upper()}</b>\n"
            f"PnL: <b>{pnl_str} USDT</b>\n"
            f"Маржа: <b>{margin:.4f} USDT</b>\n"
            f"Время удержания: <i>{held_min:.0f} мин</i>\n"
            f"Вход: <code>{entry:.6g}</code> → Выход: <code>{exit_price:.6g}</code>\n"
            f"Причина: <b>{reason_str}</b>"
        )
