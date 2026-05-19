"""
Базовый класс стратегии. От него наследуются все 7 стратегий.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class TradingSignal:
    """Объект торгового сигнала."""
    def __init__(
        self,
        action: str,            # "BUY", "SELL", "HOLD", "CLOSE"
        symbol: str,
        confidence: float,      # 0-1, доверие сигналу
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        reason: str,            # почему открываем
        filters_passed: Dict,   # какие фильтры сработали
    ):
        self.action = action
        self.symbol = symbol
        self.confidence = confidence
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.reason = reason
        self.filters_passed = filters_passed
        self.timestamp = pd.Timestamp.now()


class BaseStrategy(ABC):
    """Базовый класс для всех стратегий."""

    # Должны быть переопределены в наследниках
    ID: str = ""
    NAME: str = ""
    DESCRIPTION: str = ""
    # Список предпочтительных рыночных режимов (пустой = все режимы).
    # Значения: "flat", "uptrend", "downtrend", "volatile"
    REGIME_PREFERENCE: list = []

    def __init__(
        self,
        symbol: str,
        timeframe: str = "15",
        leverage: int = 5,
        stop_loss_pct: float = 1.5,
        take_profit_pct: float = 3.0,
        breakeven_pct: float = 1.0,         # % движения для переноса стопа в безубыток
        trailing_stop_pct: float = 0.5,     # шаг трейлинг-стопа
        edge_wr_target: float = 0.55,       # ожидаемый WR после фильтров
        **kwargs,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.leverage = leverage
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.breakeven_pct = breakeven_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.edge_wr_target = edge_wr_target

        # Состояние
        self.enabled = True
        self.auto_disabled = False
        self.pnl = 0.0
        self.trades = 0
        self.wins = 0
        self.losses = 0
        self.consecutive_losses = 0
        self.current_position = None       # {'side': 'Buy', 'entry': 67000, 'sl': 66000, 'tp': 69000, 'be_moved': False}
        self.history = []                  # последние PnL для графика

        # Параметры стратегии (переопределяются в наследниках)
        self.params = kwargs

    @abstractmethod
    def analyze(self, df: pd.DataFrame) -> Optional[TradingSignal]:
        """
        Анализ свечей. Должен вернуть TradingSignal или None.
        df: pandas DataFrame с колонками: timestamp, open, high, low, close, volume
        """
        pass

    def check_breakeven(self, current_price: float) -> Optional[float]:
        """
        Проверяет, нужно ли переносить стоп в безубыток.
        Возвращает новый SL или None.
        """
        if not self.current_position or self.current_position.get("be_moved"):
            return None

        entry = self.current_position["entry"]
        side = self.current_position["side"]

        if side == "Buy":
            move_pct = ((current_price - entry) / entry) * 100
            if move_pct >= self.breakeven_pct:
                self.current_position["be_moved"] = True
                logger.info(f"{self.ID} {self.symbol}: 🛡 SL перенесён в безубыток @ {entry}")
                return entry
        else:  # Sell
            move_pct = ((entry - current_price) / entry) * 100
            if move_pct >= self.breakeven_pct:
                self.current_position["be_moved"] = True
                logger.info(f"{self.ID} {self.symbol}: 🛡 SL перенесён в безубыток @ {entry}")
                return entry
        return None

    def check_trailing_stop(self, current_price: float) -> Optional[float]:
        """
        Трейлинг-стоп. Активируется после переноса в безубыток.
        Возвращает новый SL или None.
        """
        if not self.current_position or not self.current_position.get("be_moved"):
            return None

        entry = self.current_position["entry"]
        side = self.current_position["side"]
        current_sl = self.current_position["sl"]

        if side == "Buy":
            # Трейлинг для лонга: SL подтягивается за ценой
            new_sl = current_price * (1 - self.trailing_stop_pct / 100)
            if new_sl > current_sl:
                self.current_position["sl"] = new_sl
                logger.info(f"{self.ID} {self.symbol}: 🎯 Trailing SL → {new_sl:.4f}")
                return new_sl
        else:  # Sell
            new_sl = current_price * (1 + self.trailing_stop_pct / 100)
            if new_sl < current_sl:
                self.current_position["sl"] = new_sl
                logger.info(f"{self.ID} {self.symbol}: 🎯 Trailing SL → {new_sl:.4f}")
                return new_sl
        return None

    def check_early_tp(self, current_price: float, threshold_pct: float = 85.0) -> Optional[float]:
        """
        Ранний выход: если цена прошла >= threshold_pct% пути от входа до TP —
        возвращает текущую цену (сигнал закрыть прямо сейчас).
        0 или отрицательный порог — отключает функцию.
        """
        if not self.current_position or threshold_pct <= 0:
            return None

        entry = self.current_position["entry"]
        tp    = self.current_position["tp"]
        side  = self.current_position["side"]

        if side == "Buy":
            tp_dist = tp - entry
            progress = (current_price - entry) / tp_dist * 100 if tp_dist > 0 else 0
        else:
            tp_dist = entry - tp
            progress = (entry - current_price) / tp_dist * 100 if tp_dist > 0 else 0

        if progress >= threshold_pct:
            logger.info(
                f"{self.ID} {self.symbol}: 💰 Ранний TP @ {current_price:.6f} "
                f"({progress:.0f}% от TP)"
            )
            return current_price
        return None

    def register_position(self, side: str, entry: float, sl: float, tp: float):
        """Регистрация открытой позиции."""
        self.current_position = {
            "side": side,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "initial_sl": sl,  # хранится для корректного R-multiple после trailing/BE
            "be_moved": False,
            "opened_at": pd.Timestamp.utcnow(),
        }

    def close_position(self, exit_price: float, qty: float = 1.0, fees_pct: float = 0.06) -> Dict:
        """
        Закрытие позиции с КОРРЕКТНЫМ PnL.
        Возвращает {pnl_usd, pnl_pct, r_multiple}
        """
        if not self.current_position:
            return {"pnl_usd": 0, "pnl_pct": 0, "r_multiple": 0}

        entry = self.current_position["entry"]
        side = self.current_position["side"]
        sl = self.current_position["sl"]
        initial_sl = self.current_position.get("initial_sl", sl)

        # PnL = qty * price_diff минус комиссии
        if side == "Buy":
            gross = qty * (exit_price - entry)
            r_dist = entry - initial_sl  # исходный риск, не смещается trailing/BE
            r_gain = exit_price - entry
        else:
            gross = qty * (entry - exit_price)
            r_dist = initial_sl - entry  # исходный риск; для SELL initial_sl > entry
            r_gain = entry - exit_price

        notional_in = qty * entry
        notional_out = qty * exit_price
        fees = (notional_in + notional_out) * (fees_pct / 100)
        pnl_usd = gross - fees
        pnl_pct = (pnl_usd / notional_in * 100) if notional_in > 0 else 0
        r_multiple = (r_gain / r_dist) if r_dist > 0 else 0

        self.pnl += pnl_usd
        self.trades += 1
        self.history.append(round(pnl_usd, 2))
        if len(self.history) > 50:
            self.history.pop(0)

        if pnl_usd > 0:
            self.wins += 1
            self.consecutive_losses = 0
        else:
            self.losses += 1
            self.consecutive_losses += 1

        self.current_position = None
        return {
            "pnl_usd": round(pnl_usd, 4),
            "pnl_pct": round(pnl_pct, 4),
            "r_multiple": round(r_multiple, 3),
            "fees": round(fees, 4),
        }

    @property
    def rolling_wr_20(self) -> float:
        """WR за последние 20 сделок (для ML фич, без leakage)."""
        if not self.history:
            return 0.5
        recent = self.history[-20:]
        wins = sum(1 for p in recent if p > 0)
        return wins / len(recent)

    @property
    def rolling_pnl_20(self) -> float:
        """PnL за последние 20 сделок."""
        return sum(self.history[-20:])

    def get_rolling_stats(self) -> Dict:
        """Снимок статистики НА МОМЕНТ сигнала (используется feature_extractor)."""
        return {
            "rolling_wr_20": self.rolling_wr_20,
            "rolling_pnl_20": self.rolling_pnl_20,
            "consecutive_losses": self.consecutive_losses,
            "trades": self.trades,
        }

    @property
    def win_rate(self) -> float:
        return (self.wins / self.trades * 100) if self.trades > 0 else 0

    @property
    def rr_ratio(self) -> float:
        return self.take_profit_pct / self.stop_loss_pct if self.stop_loss_pct > 0 else 0

    @property
    def expected_value(self) -> float:
        """Матожидание сделки: WR*TP - (1-WR)*SL."""
        wr = self.edge_wr_target
        return wr * self.take_profit_pct - (1 - wr) * self.stop_loss_pct

    def to_dict(self) -> Dict:
        """Сериализация для отправки в UI."""
        return {
            "id": self.ID,
            "name": self.NAME,
            "description": self.DESCRIPTION,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "leverage": self.leverage,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "enabled": self.enabled,
            "auto_disabled": self.auto_disabled,
            "pnl": round(self.pnl, 2),
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 1),
            "edge_wr_target": round(self.edge_wr_target * 100, 1),
            "rr_ratio": round(self.rr_ratio, 2),
            "expected_value": round(self.expected_value, 3),
            "consecutive_losses": self.consecutive_losses,
            "current_position": self.current_position,
            "history": self.history[-15:],
        }
