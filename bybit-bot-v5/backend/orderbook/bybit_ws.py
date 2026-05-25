"""
Bybit WebSocket клиент для orderbook.50 — полная глубина стакана.
Поддерживает snapshot + incremental delta updates.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional, Set

import websockets

logger = logging.getLogger(__name__)

_WS_URL        = "wss://stream.bybit.com/v5/public/linear"
_RECONNECT_SEC = 3
_PING_INTERVAL = 20
_DEPTH         = 50   # orderbook.50 — 50 уровней с каждой стороны


# ─────────────────────────────────────────────────────────────────────────────
# Структуры данных
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BookLevel:
    price: float
    size:  float

    @property
    def usdt(self) -> float:
        return self.price * self.size


@dataclass
class OrderbookSnapshot:
    """Актуальный снимок стакана по одному символу."""
    symbol:      str
    bids:        List[BookLevel]   # отсортированы по убыванию цены
    asks:        List[BookLevel]   # отсортированы по возрастанию цены
    ts_exchange: int
    ts_local:    float

    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 0.0

    @property
    def spread_abs(self) -> float:
        return max(0.0, self.best_ask - self.best_bid)

    @property
    def spread_pct(self) -> float:
        m = self.mid_price
        return (self.spread_abs / m * 100) if m > 0 else 0.0

    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2

    def bid_depth_usdt(self, max_pct: float = 0.3) -> float:
        """Объём на бидах в пределах max_pct% от лучшего бида (USDT)."""
        if not self.bids:
            return 0.0
        threshold = self.best_bid * (1.0 - max_pct / 100.0)
        return sum(lvl.usdt for lvl in self.bids if lvl.price >= threshold)

    def ask_depth_usdt(self, max_pct: float = 0.3) -> float:
        """Объём на асках в пределах max_pct% от лучшего аска (USDT)."""
        if not self.asks:
            return 0.0
        threshold = self.best_ask * (1.0 + max_pct / 100.0)
        return sum(lvl.usdt for lvl in self.asks if lvl.price <= threshold)

    @property
    def imbalance(self) -> float:
        """Дисбаланс стакана: >0.5 — давление снизу (bid сильнее), <0.5 — наоборот."""
        bd = self.bid_depth_usdt(0.3)
        ad = self.ask_depth_usdt(0.3)
        total = bd + ad
        return bd / total if total > 0 else 0.5

    @property
    def bid_wall_usdt(self) -> float:
        """Крупнейший одиночный ордер на биде (top-10 уровней)."""
        return max((lvl.usdt for lvl in self.bids[:10]), default=0.0)

    @property
    def ask_wall_usdt(self) -> float:
        """Крупнейший одиночный ордер на аске (top-10 уровней)."""
        return max((lvl.usdt for lvl in self.asks[:10]), default=0.0)

    @property
    def bid_size(self) -> float:
        return self.bids[0].size if self.bids else 0.0

    @property
    def ask_size(self) -> float:
        return self.asks[0].size if self.asks else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Поддержание состояния стакана
# ─────────────────────────────────────────────────────────────────────────────

class BookState:
    """Поддерживает полный стакан одного символа (dict price→size)."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self._bids: Dict[float, float] = {}
        self._asks: Dict[float, float] = {}

    def apply_snapshot(self, bids: list, asks: list, ts: int) -> Optional[OrderbookSnapshot]:
        self._bids.clear()
        self._asks.clear()
        for px_s, sz_s in bids:
            px, sz = float(px_s), float(sz_s)
            if sz > 0:
                self._bids[px] = sz
        for px_s, sz_s in asks:
            px, sz = float(px_s), float(sz_s)
            if sz > 0:
                self._asks[px] = sz
        return self._build()

    def apply_delta(self, bids: list, asks: list, ts: int) -> Optional[OrderbookSnapshot]:
        for px_s, sz_s in bids:
            px, sz = float(px_s), float(sz_s)
            if sz == 0.0:
                self._bids.pop(px, None)
            else:
                self._bids[px] = sz
        for px_s, sz_s in asks:
            px, sz = float(px_s), float(sz_s)
            if sz == 0.0:
                self._asks.pop(px, None)
            else:
                self._asks[px] = sz
        return self._build()

    def _build(self) -> Optional[OrderbookSnapshot]:
        if not self._bids or not self._asks:
            return None
        sorted_bids = sorted(self._bids.items(), key=lambda x: -x[0])
        sorted_asks = sorted(self._asks.items(), key=lambda x:  x[0])
        bb, ba = sorted_bids[0][0], sorted_asks[0][0]
        if bb >= ba:  # crossed book — skip
            return None
        return OrderbookSnapshot(
            symbol=self.symbol,
            bids=[BookLevel(p, s) for p, s in sorted_bids[:_DEPTH]],
            asks=[BookLevel(p, s) for p, s in sorted_asks[:_DEPTH]],
            ts_exchange=0,
            ts_local=time.time(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket менеджер
# ─────────────────────────────────────────────────────────────────────────────

SnapshotCb = Callable[[OrderbookSnapshot], Awaitable[None]]


class OrderbookWsManager:
    """
    Подключается к Bybit WebSocket orderbook.{_DEPTH}.<symbol> для каждого символа.
    Поддерживает актуальный стакан и вызывает on_snapshot при каждом обновлении.
    """

    def __init__(self, symbols: List[str], on_snapshot: SnapshotCb):
        self._symbols     = list(symbols)
        self._on_snapshot = on_snapshot
        self._books: Dict[str, BookState] = {s: BookState(s) for s in symbols}
        self._running     = False

    async def start(self) -> None:
        self._running = True
        while self._running:
            try:
                await self._connect()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                if self._running:
                    logger.warning(
                        f"[OB/WS] disconnected ({exc}); retry in {_RECONNECT_SEC}s"
                    )
                    await asyncio.sleep(_RECONNECT_SEC)

    async def stop(self) -> None:
        self._running = False

    async def _connect(self) -> None:
        logger.info(f"[OB/WS] connecting → {_WS_URL}")
        async with websockets.connect(
            _WS_URL,
            ping_interval=_PING_INTERVAL,
            ping_timeout=10,
            max_size=4 * 1024 * 1024,
        ) as ws:
            args = [f"orderbook.{_DEPTH}.{s}" for s in self._symbols]
            await ws.send(json.dumps({"op": "subscribe", "args": args}))
            logger.info(f"[OB/WS] subscribed depth={_DEPTH}: {self._symbols}")
            async for raw in ws:
                if not self._running:
                    break
                try:
                    await self._handle(json.loads(raw))
                except Exception as exc:
                    logger.debug(f"[OB/WS] handle error: {exc}")

    async def _handle(self, msg: dict) -> None:
        topic = msg.get("topic", "")
        if not topic.startswith("orderbook."):
            return
        data  = msg.get("data", {})
        sym   = data.get("s", "")
        book  = self._books.get(sym)
        if book is None:
            return

        ts    = int(msg.get("ts", 0))
        mtype = msg.get("type", "snapshot")
        bids  = data.get("b", [])
        asks  = data.get("a", [])

        if mtype == "snapshot":
            snap = book.apply_snapshot(bids, asks, ts)
        else:
            snap = book.apply_delta(bids, asks, ts)

        if snap:
            asyncio.ensure_future(self._on_snapshot(snap))

    def get_snapshot(self, symbol: str) -> Optional[OrderbookSnapshot]:
        """Возвращает текущий стакан без ожидания WS-обновления."""
        book = self._books.get(symbol)
        return book._build() if book else None
