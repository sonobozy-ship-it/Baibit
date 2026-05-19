"""
Paper Trading — виртуальная торговля без реальных денег.
Открывает виртуальные позиции, следит за SL/TP, считает PnL.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from strategies.base import TradingSignal

logger = logging.getLogger(__name__)

_STATE_FILE = Path("data/paper_state.json")


class PaperTrader:
    """Виртуальный кошелёк для тестовой торговли параллельно с реальной."""

    def __init__(self, initial_balance: float = 1000.0, fee_pct: float = 0.06):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.fee_pct = fee_pct
        self.positions: Dict[str, Dict] = {}   # symbol -> position
        self.trades_history: List[Dict] = []
        self._load_state()

    def _load_state(self):
        """Загрузить состояние с диска (баланс, позиции, история)."""
        try:
            if _STATE_FILE.exists():
                data = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
                self.balance = data.get("balance", self.initial_balance)
                self.positions = data.get("positions", {})
                self.trades_history = data.get("trades_history", [])
                logger.info(
                    f"[PaperTrader] Восстановлено: баланс={self.balance:.2f} USDT, "
                    f"позиций={len(self.positions)}, сделок={len(self.trades_history)}"
                )
        except Exception as e:
            logger.warning(f"[PaperTrader] Не удалось загрузить state: {e}")

    def _save_state(self):
        """Сохранить состояние на диск."""
        try:
            _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            _STATE_FILE.write_text(
                json.dumps({
                    "balance": self.balance,
                    "positions": self.positions,
                    "trades_history": self.trades_history,
                    "saved_at": datetime.utcnow().isoformat(),
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"[PaperTrader] Не удалось сохранить state: {e}")

    def open_position(self, signal: TradingSignal, strategy_id: str, qty: float, leverage: int = 1):
        """Виртуально открыть позицию."""
        if signal.symbol in self.positions:
            return {"success": False, "reason": "Уже есть позиция на этом символе"}

        cost = qty * signal.entry_price / leverage
        if cost > self.balance:
            return {"success": False, "reason": "Недостаточно виртуальных средств"}

        fee = cost * (self.fee_pct / 100)
        self.balance -= fee

        self.positions[signal.symbol] = {
            "strategy_id": strategy_id,
            "side": "Buy" if signal.action == "BUY" else "Sell",
            "entry_price": signal.entry_price,
            "qty": qty,
            "leverage": leverage,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "opened_at": datetime.utcnow().isoformat(),
            "be_moved": False,
        }
        self._save_state()
        logger.info(f"[PAPER] {strategy_id} OPEN {signal.action} {signal.symbol} @ {signal.entry_price}")
        return {"success": True, "position": self.positions[signal.symbol]}

    def check_positions(self, current_prices: Dict[str, float]):
        """Проверка SL/TP по всем позициям."""
        closed = []
        for symbol, pos in list(self.positions.items()):
            if symbol not in current_prices:
                continue
            price = current_prices[symbol]
            side = pos["side"]

            hit_sl = (side == "Buy" and price <= pos["stop_loss"]) or \
                     (side == "Sell" and price >= pos["stop_loss"])
            hit_tp = (side == "Buy" and price >= pos["take_profit"]) or \
                     (side == "Sell" and price <= pos["take_profit"])

            if hit_sl or hit_tp:
                exit_price = pos["take_profit"] if hit_tp else pos["stop_loss"]
                closed.append(self._close_position(symbol, exit_price, "TP" if hit_tp else "SL"))
        return closed

    def _close_position(self, symbol: str, exit_price: float, reason: str) -> Dict:
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return {"pnl_usd": 0, "pnl_pct": 0, "symbol": symbol, "exit_reason": reason}
        entry = pos["entry_price"]
        qty = pos["qty"]
        side = pos["side"]
        lev = pos["leverage"]

        if side == "Buy":
            pnl_usd = qty * (exit_price - entry)
        else:
            pnl_usd = qty * (entry - exit_price)

        pnl_pct = ((exit_price - entry) / entry * 100 if side == "Buy" else (entry - exit_price) / entry * 100) * lev
        # Комиссия на notional обеих сторон сделки (как на реальной бирже)
        fee = qty * (entry + exit_price) * (self.fee_pct / 100)
        pnl_usd -= fee

        self.balance += pnl_usd

        trade = {
            **pos,
            "symbol": symbol,
            "exit_price": exit_price,
            "exit_reason": reason,
            "pnl_usd": round(pnl_usd, 2),
            "pnl_pct": round(pnl_pct, 2),
            "closed_at": datetime.utcnow().isoformat(),
        }
        self.trades_history.append(trade)
        self._save_state()
        logger.info(f"[PAPER] CLOSE {symbol} @ {exit_price} → {pnl_usd:+.2f} USDT ({reason})")
        return trade

    def get_stats(self) -> Dict:
        if not self.trades_history:
            return {
                "balance": self.balance,
                "initial_balance": self.initial_balance,
                "trades": 0,
                "open_positions": len(self.positions),
            }
        pnls = [t["pnl_usd"] for t in self.trades_history]
        wins = [p for p in pnls if p > 0]
        return {
            "balance": round(self.balance, 2),
            "initial_balance": self.initial_balance,
            "roi_pct": round((self.balance - self.initial_balance) / self.initial_balance * 100, 2),
            "trades": len(self.trades_history),
            "wins": len(wins),
            "win_rate": round(len(wins) / len(pnls) * 100, 2),
            "total_pnl": round(sum(pnls), 2),
            "open_positions": len(self.positions),
        }
