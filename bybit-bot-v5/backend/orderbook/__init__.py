"""
orderbook/ — стаканный торговый движок для ORDERBOOK_ONLY режима.

Включается через env-переменную:
  ORDERBOOK_ONLY=true

В этом режиме все обычные стратегии S1–S15, AI-сигналы и MTF-фильтры
отключены. Работает только OrderbookEngine.
"""
from .orderbook_engine import OrderbookEngine, ObEngineConfig  # noqa: F401
