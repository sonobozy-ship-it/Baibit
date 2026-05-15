"""
Фильтр коррелированных позиций.
Не открываем 2 одинаковых направления на коррелированных парах (BTC + ETH).
"""
import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class CorrelationFilter:
    """
    Вычисляет корреляцию между активами и блокирует двойные позиции.

    Например:
    - BTC и ETH коррелируют > 0.8 → нельзя открывать обе на LONG
    - BTC и DOGE могут двигаться по-разному → можно
    """

    def __init__(self, correlation_threshold: float = 0.75):
        self.threshold = correlation_threshold
        self.correlations: Dict[tuple, float] = {}

    def calculate_correlation(self, prices: Dict[str, pd.DataFrame]) -> Dict[tuple, float]:
        """
        prices: {"BTCUSDT": df, "ETHUSDT": df, ...}
        Возвращает корреляции между всеми парами.
        """
        # Объединяем close-цены
        returns = {}
        for symbol, df in prices.items():
            if len(df) < 50:
                continue
            returns[symbol] = df["close"].pct_change().dropna()

        if len(returns) < 2:
            return {}

        # Выравниваем по длине
        min_len = min(len(r) for r in returns.values())
        returns_aligned = {sym: r.iloc[-min_len:].reset_index(drop=True) for sym, r in returns.items()}
        df_returns = pd.DataFrame(returns_aligned)
        corr_matrix = df_returns.corr()

        result = {}
        symbols = list(corr_matrix.columns)
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                s1, s2 = symbols[i], symbols[j]
                result[(s1, s2)] = round(corr_matrix.loc[s1, s2], 3)

        self.correlations = result
        return result

    def is_correlated(self, symbol1: str, symbol2: str) -> bool:
        """Считается ли пара коррелированной."""
        key = tuple(sorted([symbol1, symbol2]))
        for (s1, s2), corr in self.correlations.items():
            if tuple(sorted([s1, s2])) == key:
                return abs(corr) >= self.threshold
        return False

    def can_open(self, new_symbol: str, new_side: str, open_positions: List[Dict]) -> Dict:
        """
        Проверяет, можно ли открыть новую позицию.
        new_side: "BUY" или "SELL"
        open_positions: [{"symbol": "BTCUSDT", "side": "BUY"}, ...]

        Возвращает {"allowed": bool, "reason": str}
        """
        for pos in open_positions:
            pos_symbol = pos["symbol"]
            pos_side = pos["side"]

            if pos_symbol == new_symbol:
                # Уже есть позиция на этом символе
                return {
                    "allowed": False,
                    "reason": f"Уже открыта позиция на {new_symbol}"
                }

            if self.is_correlated(pos_symbol, new_symbol):
                # Если коррелированы и направление одинаковое — блокируем
                if pos_side == new_side:
                    corr = self._get_corr(pos_symbol, new_symbol)
                    return {
                        "allowed": False,
                        "reason": f"{pos_symbol} и {new_symbol} коррелируют (ρ={corr:.2f}), нет смысла дублировать {new_side}"
                    }

        return {"allowed": True, "reason": "OK"}

    def _get_corr(self, s1: str, s2: str) -> float:
        key = tuple(sorted([s1, s2]))
        for (a, b), corr in self.correlations.items():
            if tuple(sorted([a, b])) == key:
                return corr
        return 0
