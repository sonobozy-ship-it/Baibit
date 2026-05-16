"""
Risk Manager — защита депозита.
Контролирует дневные лимиты, количество позиций, паузы после убытков.
"""
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(
        self,
        daily_max_loss_pct: float = 5.0,
        max_open_positions: int = 4,
        risk_per_trade_pct: float = 1.0,
        cooldown_after_loss_min: int = 15,
        max_consecutive_losses: int = 3,
        max_leverage_cap: int = 5,
        max_total_notional_pct: float = 300.0,  # макс суммарная notional как % от баланса
    ):
        self.daily_max_loss_pct = daily_max_loss_pct
        self.max_open_positions = max_open_positions
        self.risk_per_trade_pct = risk_per_trade_pct
        self.cooldown_min = cooldown_after_loss_min
        self.max_consecutive_losses = max_consecutive_losses
        self.max_leverage_cap = max_leverage_cap
        self.max_total_notional_pct = max_total_notional_pct

        # Состояние
        self.daily_start_balance: Optional[float] = None
        self.daily_pnl = 0.0
        self.daily_reset_at = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        self.kill_switch = False                # глобальный стоп
        self.kill_switch_reason = ""
        self.strategy_cooldowns: Dict[str, datetime] = {}
        self.strategy_losses: Dict[str, int] = defaultdict(int)
        self.open_positions_count = 0
        self._open_notional: Dict[str, float] = {}  # strategy_id -> notional USDT

    def reset_daily(self, current_balance: float):
        """Сброс ежедневных счётчиков (вызывается раз в сутки)."""
        self.daily_start_balance = current_balance
        self.daily_pnl = 0.0
        self.kill_switch = False
        self.kill_switch_reason = ""
        self.strategy_losses.clear()
        self.daily_reset_at = datetime.utcnow().replace(hour=0, minute=0, second=0) + timedelta(days=1)
        logger.info(f"💚 Дневной сброс. Старт баланс: {current_balance} USDT")

    def check_daily_reset(self, current_balance: float):
        """Проверяет, не нужно ли сделать дневной сброс."""
        if datetime.utcnow() >= self.daily_reset_at or self.daily_start_balance is None:
            self.reset_daily(current_balance)

    def check_leverage(self, strategy_id: str, leverage: int, balance: float) -> Dict:
        """Проверяет плечо на превышение лимита и суммарную notional экспозицию."""
        # Hard cap на плечо
        effective_lev = min(leverage, self.max_leverage_cap)
        capped = effective_lev < leverage
        if capped:
            logger.warning(f"{strategy_id}: плечо {leverage}x → ограничено до {effective_lev}x")

        # Суммарная notional не должна превышать max_total_notional_pct от баланса
        total_notional = sum(self._open_notional.values())
        max_notional = balance * (self.max_total_notional_pct / 100)
        if total_notional >= max_notional:
            return {
                "allowed": False,
                "effective_leverage": effective_lev,
                "reason": f"Суммарная notional {total_notional:.0f} ≥ лимит {max_notional:.0f} USDT",
            }

        return {"allowed": True, "effective_leverage": effective_lev, "capped": capped}

    def calculate_position_size(
        self,
        balance: float,
        entry_price: float,
        stop_loss_price: float,
        leverage: int = 1,
    ) -> float:
        """
        Расчёт размера позиции по риску.
        Например: депо 1000, риск 1% → потеря не больше 10 USDT при срабатывании SL.
        Плечо ограничивается max_leverage_cap.
        """
        effective_lev = min(leverage, self.max_leverage_cap)
        risk_usd = balance * (self.risk_per_trade_pct / 100)
        sl_distance = abs(entry_price - stop_loss_price) / entry_price
        if sl_distance == 0:
            return 0
        qty = risk_usd / (sl_distance * entry_price)
        return round(qty, 4)

    def can_open_trade(self, strategy_id: str, balance: float) -> Dict:
        """
        Главная проверка: можно ли открывать сделку?
        Возвращает {allowed: bool, reason: str}
        """
        self.check_daily_reset(balance)

        # 1. Глобальный kill switch
        if self.kill_switch:
            return {"allowed": False, "reason": f"🛑 KILL SWITCH: {self.kill_switch_reason}"}

        # 2. Дневной лимит убытков
        if self.daily_start_balance:
            current_drawdown_pct = ((balance - self.daily_start_balance) / self.daily_start_balance) * 100
            if current_drawdown_pct <= -self.daily_max_loss_pct:
                self.kill_switch = True
                self.kill_switch_reason = f"Дневной убыток {current_drawdown_pct:.2f}% > лимит {self.daily_max_loss_pct}%"
                logger.critical(self.kill_switch_reason)
                return {"allowed": False, "reason": self.kill_switch_reason}

        # 3. Лимит открытых позиций
        if self.open_positions_count >= self.max_open_positions:
            return {
                "allowed": False,
                "reason": f"Достигнут лимит позиций ({self.max_open_positions})"
            }

        # 4. Cooldown стратегии после убытка
        if strategy_id in self.strategy_cooldowns:
            if datetime.utcnow() < self.strategy_cooldowns[strategy_id]:
                remaining = (self.strategy_cooldowns[strategy_id] - datetime.utcnow()).total_seconds() / 60
                return {
                    "allowed": False,
                    "reason": f"Cooldown {remaining:.1f} мин"
                }

        # 5. Серия убытков
        if self.strategy_losses[strategy_id] >= self.max_consecutive_losses:
            return {
                "allowed": False,
                "reason": f"Серия {self.max_consecutive_losses} убытков — пауза"
            }

        return {"allowed": True, "reason": "OK"}

    def register_trade_result(self, strategy_id: str, pnl: float):
        """Записать результат сделки."""
        self.daily_pnl += pnl
        if pnl < 0:
            self.strategy_losses[strategy_id] += 1
            # Cooldown после убытка
            self.strategy_cooldowns[strategy_id] = datetime.utcnow() + timedelta(minutes=self.cooldown_min)
            logger.warning(f"📉 {strategy_id}: убыток {pnl:.2f} USDT. Cooldown {self.cooldown_min} мин")
        else:
            self.strategy_losses[strategy_id] = 0  # сбрасываем серию
            logger.info(f"📈 {strategy_id}: прибыль +{pnl:.2f} USDT")

    def register_position_open(self, strategy_id: str = "", notional_usd: float = 0.0):
        self.open_positions_count += 1
        if strategy_id and notional_usd > 0:
            self._open_notional[strategy_id] = notional_usd

    def register_position_close(self, strategy_id: str = ""):
        self.open_positions_count = max(0, self.open_positions_count - 1)
        self._open_notional.pop(strategy_id, None)

    def get_status(self) -> Dict:
        """Текущий статус для отображения в UI."""
        return {
            "daily_start_balance": self.daily_start_balance,
            "daily_pnl": self.daily_pnl,
            "daily_pnl_pct": (self.daily_pnl / self.daily_start_balance * 100) if self.daily_start_balance else 0,
            "daily_max_loss_pct": self.daily_max_loss_pct,
            "open_positions": self.open_positions_count,
            "max_positions": self.max_open_positions,
            "kill_switch": self.kill_switch,
            "kill_switch_reason": self.kill_switch_reason,
            "cooldowns": {
                sid: (cd - datetime.utcnow()).total_seconds() / 60
                for sid, cd in self.strategy_cooldowns.items()
                if cd > datetime.utcnow()
            },
            "max_leverage_cap": self.max_leverage_cap,
            "total_notional_usd": sum(self._open_notional.values()),
            "max_notional_usd": (self.daily_start_balance or 0) * (self.max_total_notional_pct / 100),
        }

    def manual_reset_kill_switch(self):
        """Ручной сброс kill switch (только осознанно!)."""
        self.kill_switch = False
        self.kill_switch_reason = ""
        logger.warning("⚠️  Kill switch сброшен вручную")
