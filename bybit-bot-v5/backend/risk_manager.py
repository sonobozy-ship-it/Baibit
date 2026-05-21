"""
Risk Manager — защита депозита.
Контролирует дневные/недельные лимиты, количество позиций, паузы после убытков.

Дефолты по quant-требованиям:
  risk_per_trade   = 0.5%
  daily_max_loss   = 2.0%
  weekly_max_loss  = 5.0%
  max_position_size = 5% баланса (margin)
  max_leverage     = 5
  3 убытка подряд  = 12ч пауза стратегии (не permanent disable)
"""
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(
        self,
        daily_max_loss_pct: float = 2.0,        # 2% дневной лимит (жёсткий стоп)
        daily_pause_loss_pct: float = 0.0,       # % для паузы (0 = выключено), напр. 3.0
        daily_pause_hours: float = 1.0,          # длина паузы в часах при -3%
        weekly_max_loss_pct: float = 5.0,        # 5% недельный лимит
        max_open_positions: int = 4,
        risk_per_trade_pct: float = 0.5,         # 0.5% риска на сделку
        cooldown_after_loss_min: int = 15,
        consecutive_loss_pause_hours: float = 12.0,  # пауза после серии (было disable)
        max_consecutive_losses: int = 3,
        max_leverage_cap: int = 5,
        max_position_size_pct: float = 5.0,      # макс маржа = 5% баланса
        max_total_notional_pct: float = 300.0,
        max_daily_trades: int = 0,
        max_daily_losses: int = 3,
        max_strategy_daily_losses: int = 10,
        min_trade_usdt: float = 5.0,             # мин. размер сделки в USDT
    ):
        self.daily_max_loss_pct        = daily_max_loss_pct
        self.daily_pause_loss_pct      = daily_pause_loss_pct
        self.daily_pause_hours         = daily_pause_hours
        self.weekly_max_loss_pct       = weekly_max_loss_pct
        self.max_open_positions        = max_open_positions
        self.risk_per_trade_pct        = risk_per_trade_pct
        self.cooldown_min              = cooldown_after_loss_min
        self.consecutive_loss_pause_h  = consecutive_loss_pause_hours
        self.max_consecutive_losses    = max_consecutive_losses
        self.max_leverage_cap          = max_leverage_cap
        self.max_position_size_pct     = max_position_size_pct
        self.max_total_notional_pct    = max_total_notional_pct
        self.max_daily_trades          = max_daily_trades
        self.max_daily_losses          = max_daily_losses
        self.max_strategy_daily_losses = max_strategy_daily_losses
        self.risk_hard_cap_pct         = min(risk_per_trade_pct, 1.0)
        self.min_trade_usdt            = min_trade_usdt

        # Дневное состояние
        self.daily_start_balance: Optional[float] = None
        self.daily_pnl           = 0.0
        self.daily_trades_count  = 0
        self.daily_losses_count  = 0
        self.strategy_daily_losses: Dict[str, int] = defaultdict(int)
        self.daily_reset_at = (
            datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            + timedelta(days=1)
        )

        # Недельное состояние
        self.weekly_start_balance: Optional[float] = None
        self.weekly_pnl       = 0.0
        self.weekly_reset_at  = self._next_monday()

        # Пауза при -3% (мягкий стоп)
        self.daily_pause_until: Optional[datetime] = None

        # Общее состояние
        self.kill_switch        = False
        self.kill_switch_reason = ""
        self.strategy_cooldowns: Dict[str, datetime] = {}
        self.strategy_losses:    Dict[str, int] = defaultdict(int)
        self.open_positions_count = 0
        self.last_balance         = 0.0
        self._open_notional:     Dict[str, float] = {}

    @staticmethod
    def _next_monday() -> datetime:
        now  = datetime.utcnow()
        days = (7 - now.weekday()) % 7 or 7
        return (now + timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)

    def reset_daily(self, current_balance: float):
        """Сброс ежедневных счётчиков (вызывается раз в сутки)."""
        self.daily_start_balance = current_balance
        self.last_balance        = current_balance
        self.daily_pnl           = 0.0
        self.daily_trades_count  = 0
        self.daily_losses_count  = 0
        self.kill_switch         = False
        self.kill_switch_reason  = ""
        self.daily_pause_until   = None
        self.strategy_losses.clear()
        self.strategy_daily_losses.clear()
        self.daily_reset_at = (
            datetime.utcnow().replace(hour=0, minute=0, second=0) + timedelta(days=1)
        )
        logger.info(f"💚 Дневной сброс. Старт баланс: {current_balance} USDT")

    def reset_weekly(self, current_balance: float):
        """Сброс недельных счётчиков (раз в неделю, понедельник UTC)."""
        self.weekly_start_balance = current_balance
        self.weekly_pnl           = 0.0
        self.weekly_reset_at      = self._next_monday()
        logger.info(f"📅 Недельный сброс. Старт баланс: {current_balance} USDT")

    def check_daily_reset(self, current_balance: float):
        """Проверяет, не нужно ли сделать дневной/недельный сброс."""
        self.last_balance = current_balance
        if datetime.utcnow() >= self.daily_reset_at or self.daily_start_balance is None:
            self.reset_daily(current_balance)
        if datetime.utcnow() >= self.weekly_reset_at or self.weekly_start_balance is None:
            self.reset_weekly(current_balance)

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
        """Риск на сделку: не более hard_cap_pct от депозита при любом балансе."""
        return min(self.risk_per_trade_pct, self.risk_hard_cap_pct)

    def max_positions_for_balance(self, balance: float) -> int:
        """Лимит одновременных позиций — растёт с балансом, потолок max_open_positions."""
        if balance < 100:
            return 1
        elif balance < 300:
            return 3
        elif balance < 600:
            return 6
        elif balance < 1000:
            return 10
        else:
            return self.max_open_positions   # 20 при 1000+ USDT

    def _weekly_drawdown_pct(self, balance: float) -> float:
        if self.weekly_start_balance and self.weekly_start_balance > 0:
            return (balance - self.weekly_start_balance) / self.weekly_start_balance * 100
        return 0.0

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

        if effective_sl_dist == 0 or entry_price <= 0:
            return 0

        qty = risk_usd / (effective_sl_dist * entry_price)

        # Лимит размера позиции: margin ≤ max_position_size_pct % от баланса
        max_margin_usd  = balance * (self.max_position_size_pct / 100)
        max_qty_by_size = (max_margin_usd * leverage) / entry_price if entry_price > 0 else qty
        if qty > max_qty_by_size:
            qty = max_qty_by_size

        qty = round(qty, 4)

        # min_notional — МИНИМАЛЬНЫЙ размер позиции (не отвержение, а масштабирование вверх)
        if qty * entry_price < min_notional:
            scaled_qty = min_notional / entry_price
            # Проверяем что масштабированный объём не превышает лимит по марже
            scaled_margin = (scaled_qty * entry_price) / leverage if leverage > 0 else scaled_qty * entry_price
            if scaled_margin > max_margin_usd:
                return 0  # min_notional требует больше маржи чем разрешено — пропускаем
            qty = round(scaled_qty, 4)

        # Hard cap: риск в $ не превышает risk_hard_cap_pct от баланса
        hard_cap_usd = balance * (self.risk_hard_cap_pct / 100)
        if effective_sl_dist > 0 and qty * entry_price * effective_sl_dist > hard_cap_usd * 1.1:
            qty = hard_cap_usd / (effective_sl_dist * entry_price)
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

        # 1b. Мягкая пауза при -3% (временная, до конца часа)
        if self.daily_pause_until is not None and datetime.utcnow() < self.daily_pause_until:
            remaining_min = int((self.daily_pause_until - datetime.utcnow()).total_seconds() / 60)
            return {"allowed": False, "reason": f"⏸ Пауза −3% дневного PnL, осталось {remaining_min} мин"}

        # 2. Дневной лимит убытков (жёсткий стоп)
        if self.daily_start_balance is not None and self.daily_start_balance > 0:
            dd_pct = (balance - self.daily_start_balance) / self.daily_start_balance * 100

            # 2a. Мягкий порог −3%: пауза на daily_pause_hours
            if (
                self.daily_pause_loss_pct > 0
                and dd_pct <= -self.daily_pause_loss_pct
                and self.daily_pause_until is None
            ):
                self.daily_pause_until = datetime.utcnow() + timedelta(hours=self.daily_pause_hours)
                msg = (
                    f"⏸ Дневной убыток {dd_pct:.2f}% ≤ -{self.daily_pause_loss_pct}%: "
                    f"пауза на {self.daily_pause_hours:.0f}ч до {self.daily_pause_until.strftime('%H:%M')} UTC"
                )
                logger.warning(msg)
                return {"allowed": False, "reason": msg}

            if dd_pct <= -self.daily_max_loss_pct:
                self.kill_switch = True
                self.kill_switch_reason = (
                    f"Дневной убыток {dd_pct:.2f}% > лимит {self.daily_max_loss_pct}%"
                )
                logger.critical(self.kill_switch_reason)
                return {"allowed": False, "reason": self.kill_switch_reason}

        # 2b. Недельный лимит убытков 5%
        weekly_dd = self._weekly_drawdown_pct(balance)
        if weekly_dd <= -self.weekly_max_loss_pct:
            self.kill_switch = True
            self.kill_switch_reason = (
                f"Недельный убыток {weekly_dd:.2f}% > лимит {self.weekly_max_loss_pct}%"
            )
            logger.critical(self.kill_switch_reason)
            return {"allowed": False, "reason": self.kill_switch_reason}

        # 3. Стоп по дневным убыткам (0 = отключено)
        if self.max_daily_losses > 0 and self.daily_losses_count >= self.max_daily_losses:
            self.kill_switch = True
            self.kill_switch_reason = (
                f"Достигнут лимит убытков за день: {self.daily_losses_count}/{self.max_daily_losses}"
            )
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

        # 7. Серия убытков подряд → 12-часовая пауза (не permanent disable)
        if self.strategy_losses[strategy_id] >= self.max_consecutive_losses:
            pause_key = f"__consec__{strategy_id}"
            if pause_key not in self.strategy_cooldowns or \
                    self.strategy_cooldowns[pause_key] <= datetime.utcnow():
                # Устанавливаем 12ч cooldown при первом срабатывании
                pause_until = datetime.utcnow() + timedelta(hours=self.consecutive_loss_pause_h)
                self.strategy_cooldowns[pause_key] = pause_until
                logger.warning(
                    f"⏸ {strategy_id}: {self.max_consecutive_losses} убытка подряд → "
                    f"пауза {self.consecutive_loss_pause_h:.0f}ч до {pause_until.strftime('%H:%M UTC')}"
                )
            if datetime.utcnow() < self.strategy_cooldowns[pause_key]:
                remaining_h = (
                    self.strategy_cooldowns[pause_key] - datetime.utcnow()
                ).total_seconds() / 3600
                return {
                    "allowed": False,
                    "reason": (
                        f"⏸ Серия {self.max_consecutive_losses} убытков → "
                        f"пауза ещё {remaining_h:.1f}ч"
                    ),
                }
            # Cooldown истёк — сбрасываем серию
            self.strategy_losses[strategy_id] = 0

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
        self.daily_pnl  += pnl
        self.weekly_pnl += pnl
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
        effective_balance = self.last_balance or self.daily_start_balance or 0.0
        effective_max_positions = self.max_positions_for_balance(effective_balance) if effective_balance > 0 else self.max_open_positions
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
            "max_positions": effective_max_positions,
            "max_positions_configured": self.max_open_positions,
            "effective_balance": effective_balance,
            "daily_pause_loss_pct": self.daily_pause_loss_pct,
            "daily_pause_until": self.daily_pause_until.isoformat() if self.daily_pause_until else None,
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
