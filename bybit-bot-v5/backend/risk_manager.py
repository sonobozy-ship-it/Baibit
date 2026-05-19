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
        max_total_notional_pct: float = 300.0,
        max_daily_trades: int = 0,              # 0 = без лимита по количеству
        max_daily_losses: int = 3,              # глобальный стоп после N убытков за день
        max_strategy_daily_losses: int = 10,    # лимит убытков в день на каждую стратегию
    ):
        self.daily_max_loss_pct = daily_max_loss_pct
        self.max_open_positions = max_open_positions
        self.risk_per_trade_pct = risk_per_trade_pct
        self.cooldown_min = cooldown_after_loss_min
        self.max_consecutive_losses = max_consecutive_losses
        self.max_leverage_cap = max_leverage_cap
        self.max_total_notional_pct = max_total_notional_pct
        self.max_daily_trades = max_daily_trades
        self.max_daily_losses = max_daily_losses
        self.max_strategy_daily_losses = max_strategy_daily_losses

        # Состояние
        self.daily_start_balance: Optional[float] = None
        self.daily_pnl = 0.0
        self.daily_trades_count = 0
        self.daily_losses_count = 0             # сколько убыточных сделок сегодня
        self.strategy_daily_losses: Dict[str, int] = defaultdict(int)  # убытков на стратегию
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
        self.daily_trades_count = 0
        self.daily_losses_count = 0
        self.kill_switch = False
        self.kill_switch_reason = ""
        self.strategy_losses.clear()
        self.strategy_daily_losses.clear()
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

    def adaptive_risk_pct(self, balance: float) -> float:
        """
        Адаптивный % риска на сделку в зависимости от размера баланса.
        Малый депозит торгует агрессивнее чтобы расти, крупный — консервативнее.
        """
        if balance < 300:
            return 2.0    # 200 USDT → 4 USDT риска / сделку
        elif balance < 1000:
            return 1.5    # 500 USDT → 7.5 USDT риска
        elif balance < 5000:
            return 1.0    # 2000 USDT → 20 USDT риска
        else:
            return 0.75   # 10000 USDT → 75 USDT риска

    def max_positions_for_balance(self, balance: float) -> int:
        """Лимит одновременных позиций по размеру депозита."""
        if balance < 300:
            return 2
        elif balance < 1000:
            return 3
        else:
            return self.max_open_positions

    def calculate_position_size(
        self,
        balance: float,
        entry_price: float,
        stop_loss_price: float,
        leverage: int = 1,
        min_notional: float = 5.0,
        atr: float = 0.0,           # ATR от рынка (если 0 — не используется)
        atr_multiplier: float = 1.5, # минимальный SL = ATR × multiplier
    ) -> float:
        """
        Расчёт размера позиции по рыночной волатильности (ATR).
        При срабатывании SL теряем adaptive_risk_pct% от баланса.
        ATR не даёт занизить дистанцию SL ниже реальной волатильности,
        что предотвращает открытие слишком больших позиций на тихом рынке.
        """
        risk_pct = self.adaptive_risk_pct(balance)
        risk_usd = balance * (risk_pct / 100)

        signal_sl_dist = abs(entry_price - stop_loss_price) / entry_price if entry_price else 0

        # Рыночная дистанция SL по ATR (минимальный порог волатильности)
        if atr and atr > 0 and entry_price > 0:
            atr_sl_dist = (atr * atr_multiplier) / entry_price
            # Используем максимум — нельзя занижать риск ниже рыночной волатильности
            effective_sl_dist = max(signal_sl_dist, atr_sl_dist)
        else:
            effective_sl_dist = signal_sl_dist

        if effective_sl_dist == 0:
            return 0

        qty = risk_usd / (effective_sl_dist * entry_price)

        # Ограничиваем маржу: не более 1% баланса на сделку
        # (при балансе 180 USDT → max маржа 1.8 USDT)
        max_margin = balance * 0.01
        max_notional = max_margin * leverage
        if qty * entry_price > max_notional:
            qty = max_notional / entry_price

        qty = round(qty, 4)
        if qty * entry_price < min_notional:
            return 0
        return qty

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
        if self.daily_start_balance is not None and self.daily_start_balance > 0:
            current_drawdown_pct = ((balance - self.daily_start_balance) / self.daily_start_balance) * 100
            if current_drawdown_pct <= -self.daily_max_loss_pct:
                self.kill_switch = True
                self.kill_switch_reason = f"Дневной убыток {current_drawdown_pct:.2f}% > лимит {self.daily_max_loss_pct}%"
                logger.critical(self.kill_switch_reason)
                return {"allowed": False, "reason": self.kill_switch_reason}

        # 3. Стоп по дневным убыткам
        if self.daily_losses_count >= self.max_daily_losses:
            self.kill_switch = True
            self.kill_switch_reason = f"Достигнут лимит убытков за день: {self.daily_losses_count}/{self.max_daily_losses}"
            logger.critical(self.kill_switch_reason)
            return {"allowed": False, "reason": self.kill_switch_reason}

        # 4. Лимит сделок в день (0 = без лимита)
        if self.max_daily_trades > 0 and self.daily_trades_count >= self.max_daily_trades:
            return {
                "allowed": False,
                "reason": f"Дневной лимит сделок {self.max_daily_trades} достигнут"
            }

        # 5. Лимит открытых позиций (адаптивный по балансу)
        pos_limit = self.max_positions_for_balance(balance)
        if self.open_positions_count >= pos_limit:
            return {
                "allowed": False,
                "reason": f"Достигнут лимит позиций ({pos_limit})"
            }

        # 6. Cooldown стратегии после убытка
        if strategy_id in self.strategy_cooldowns:
            if datetime.utcnow() < self.strategy_cooldowns[strategy_id]:
                remaining = (self.strategy_cooldowns[strategy_id] - datetime.utcnow()).total_seconds() / 60
                return {
                    "allowed": False,
                    "reason": f"Cooldown {remaining:.1f} мин"
                }

        # 7. Серия убытков одной стратегии (подряд)
        if self.strategy_losses[strategy_id] >= self.max_consecutive_losses:
            return {
                "allowed": False,
                "reason": f"Серия {self.max_consecutive_losses} убытков — пауза"
            }

        # 8. Дневной лимит убытков на стратегию
        if self.strategy_daily_losses[strategy_id] >= self.max_strategy_daily_losses:
            return {
                "allowed": False,
                "reason": (
                    f"{strategy_id}: дневной лимит {self.max_strategy_daily_losses} "
                    f"убытков достигнут — стратегия остановлена до завтра"
                ),
            }

        return {"allowed": True, "reason": "OK"}

    def register_trade_result(self, strategy_id: str, pnl: float):
        """Записать результат сделки."""
        self.daily_pnl += pnl
        if pnl < 0:
            self.daily_losses_count += 1
            self.strategy_losses[strategy_id] += 1
            self.strategy_daily_losses[strategy_id] += 1
            self.strategy_cooldowns[strategy_id] = datetime.utcnow() + timedelta(minutes=self.cooldown_min)
            logger.warning(
                f"📉 {strategy_id}: убыток {pnl:.2f} USDT. "
                f"Дневных убытков: {self.daily_losses_count}/{self.max_daily_losses}. "
                f"Убытков стратегии сегодня: {self.strategy_daily_losses[strategy_id]}/{self.max_strategy_daily_losses}. "
                f"Cooldown {self.cooldown_min} мин"
            )
        else:
            self.strategy_losses[strategy_id] = 0
            logger.info(f"📈 {strategy_id}: прибыль +{pnl:.2f} USDT")

    def register_position_open(self, strategy_id: str = "", notional_usd: float = 0.0):
        self.open_positions_count += 1
        self.daily_trades_count += 1
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
            "daily_trades_count": self.daily_trades_count,
            "daily_losses_count": self.daily_losses_count,
            "max_daily_losses": self.max_daily_losses,
            "max_daily_trades": self.max_daily_trades,
            "open_positions": self.open_positions_count,
            "max_positions": self.max_open_positions,
            "kill_switch": self.kill_switch,
            "kill_switch_reason": self.kill_switch_reason,
            "cooldowns": {
                sid: (cd - datetime.utcnow()).total_seconds() / 60
                for sid, cd in self.strategy_cooldowns.items()
                if cd > datetime.utcnow()
            },
            "strategy_daily_losses": dict(self.strategy_daily_losses),
            "max_strategy_daily_losses": self.max_strategy_daily_losses,
            "max_leverage_cap": self.max_leverage_cap,
            "total_notional_usd": sum(self._open_notional.values()),
            "max_notional_usd": (self.daily_start_balance or 0) * (self.max_total_notional_pct / 100),
        }

    def manual_reset_kill_switch(self):
        """Ручной сброс kill switch (только осознанно!)."""
        self.kill_switch = False
        self.kill_switch_reason = ""
        logger.warning("⚠️  Kill switch сброшен вручную")
