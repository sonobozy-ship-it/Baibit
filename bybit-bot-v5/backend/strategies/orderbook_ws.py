"""
Bybit public WebSocket client для orderbook.1 (best bid/ask per symbol).
Используется исключительно стратегией OB_SCALPER.

Подписывается на orderbook.1.<symbol> и вызывает async-коллбэк on_tick
при каждом обновлении стакана.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, List, Optional

import websockets

logger = logging.getLogger(__name__)

_WS_URL        = "wss://stream.bybit.com/v5/public/linear"
_RECONNECT_SEC = 3
_PING_INTERVAL = 20


# ─────────────────────────────────────────────────────────────────────────────
# Тик стакана
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OrderbookTick:
    symbol:      str
    best_bid:    float
    best_ask:    float
    bid_size:    float    # базовая валюта
    ask_size:    float
    ts_exchange: int      # exchange timestamp ms
    ts_local:    float    # time.time()

    @property
    def spread_abs(self) -> float:
        return max(0.0, self.best_ask - self.best_bid)

    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread_pct(self) -> float:
        m = self.mid_price
        return (self.spread_abs / m * 100) if m > 0 else 0.0

    @property
    def imbalance(self) -> float:
        total = self.bid_size + self.ask_size
        return self.bid_size / total if total > 0 else 0.5

    @property
    def bid_depth_usdt(self) -> float:
        return self.bid_size * self.best_bid

    @property
    def ask_depth_usdt(self) -> float:
        return self.ask_size * self.best_ask


AsyncTickCb = Callable[[OrderbookTick], Awaitable[None]]


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket клиент
# ─────────────────────────────────────────────────────────────────────────────

class OrderbookWsClient:
    """
    Подключается к Bybit orderbook.1 и поддерживает best bid/ask per symbol.
    Вызывает on_tick (async) при каждом обновлении.
    Автоматически переподключается при обрыве соединения.
    """

    def __init__(self, symbols: List[str], on_tick: AsyncTickCb):
        self._symbols = list(symbols)
        self._on_tick = on_tick
        # per-symbol book: {"bid": (price, size) | None, "ask": ...}
        self._books: Dict[str, Dict[str, Optional[tuple]]] = {
            s: {"bid": None, "ask": None} for s in symbols
        }
        self._running = False

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
        ) as ws:
            args = [f"orderbook.1.{s}" for s in self._symbols]
            await ws.send(json.dumps({"op": "subscribe", "args": args}))
            logger.info(f"[OB/WS] subscribed: {self._symbols}")

            async for raw in ws:
                if not self._running:
                    break
                try:
                    await self._handle(json.loads(raw))
                except Exception as exc:
                    logger.debug(f"[OB/WS] parse error: {exc}")

    async def _handle(self, msg: dict) -> None:
        topic = msg.get("topic", "")
        if not topic.startswith("orderbook."):
            return

        data  = msg.get("data", {})
        sym   = data.get("s", "")
        if sym not in self._books:
            return

        mtype = msg.get("type", "snapshot")
        ts    = int(msg.get("ts", 0))
        book  = self._books[sym]

        if mtype == "snapshot":
            book["bid"] = None
            book["ask"] = None

        # orderbook.1 has at most 1 level per side
        for px_s, sz_s in data.get("b", []):
            px, sz = float(px_s), float(sz_s)
            book["bid"] = (px, sz) if sz > 0 else None

        for px_s, sz_s in data.get("a", []):
            px, sz = float(px_s), float(sz_s)
            book["ask"] = (px, sz) if sz > 0 else None

        if book["bid"] is None or book["ask"] is None:
            return

        bb, bs  = book["bid"]
        ba, as_ = book["ask"]

        if ba <= bb:  # sanity: crossed book → skip
            return

        tick = OrderbookTick(
            symbol=sym,
            best_bid=bb, best_ask=ba,
            bid_size=bs, ask_size=as_,
            ts_exchange=ts,
            ts_local=time.time(),
        )
        asyncio.ensure_future(self._on_tick(tick))
