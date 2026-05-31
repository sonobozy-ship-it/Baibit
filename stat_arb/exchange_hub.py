"""
ExchangeHub — unified async wrapper over multiple ccxt exchanges.
All network calls are dispatched to a ThreadPoolExecutor so the
asyncio event loop never blocks.
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import ccxt

from .config import ArbConfig, ExchangeCreds
from .models import Ticker

logger = logging.getLogger(__name__)


def _ccxt_symbol(symbol: str) -> str:
    """Convert BTCUSDT → BTC/USDT."""
    if "/" in symbol:
        return symbol
    if symbol.endswith("USDT"):
        return symbol[:-4] + "/USDT"
    return symbol


class ExchangeHub:
    """
    Manages ccxt exchange instances and provides async wrappers for:
      - ticker fetching (parallel across all exchanges)
      - order-book depth checks
      - order placement & cancellation
      - OHLCV (for trend filter)
    """

    def __init__(self, cfg: ArbConfig):
        self._cfg  = cfg
        self._pool = ThreadPoolExecutor(max_workers=20, thread_name_prefix="ccxt")

        self._exchanges: Dict[str, ccxt.Exchange] = {}
        for name, creds in cfg.creds.items():
            options = cfg.ccxt_options.get(name, {})
            init = {
                "apiKey":          creds.api_key,
                "secret":          creds.api_secret,
                "enableRateLimit": True,
                **options,
            }
            if creds.passphrase:
                init["password"] = creds.passphrase
            cls = getattr(ccxt, name)
            self._exchanges[name] = cls(init)

    # ── Tickers ───────────────────────────────────────────────────────────────

    def _fetch_tickers_sync(
        self, name: str, symbols: List[str]
    ) -> Dict[str, Ticker]:
        ex  = self._exchanges[name]
        out: Dict[str, Ticker] = {}
        try:
            cc_syms = [_ccxt_symbol(s) for s in symbols]
            raw = ex.fetch_tickers(cc_syms)
            for orig, cc in zip(symbols, cc_syms):
                t = raw.get(cc) or raw.get(cc.replace("/", "") )
                if not t:
                    continue
                bid = t.get("bid")
                ask = t.get("ask")
                if bid and ask:
                    out[orig] = Ticker(exchange=name, symbol=orig,
                                       bid=float(bid), ask=float(ask))
        except Exception as exc:
            logger.debug(f"[{name}] fetch_tickers: {exc}")
            # Fallback: fetch individually
            for sym in symbols:
                try:
                    t = ex.fetch_ticker(_ccxt_symbol(sym))
                    bid, ask = t.get("bid"), t.get("ask")
                    if bid and ask:
                        out[sym] = Ticker(exchange=name, symbol=sym,
                                          bid=float(bid), ask=float(ask))
                except Exception:
                    pass
        return out

    async def get_all_tickers(
        self, symbols: List[str]
    ) -> Dict[str, Dict[str, Ticker]]:
        """
        Returns {symbol: {exchange_name: Ticker}} for all exchanges in parallel.
        """
        loop = asyncio.get_event_loop()
        tasks = {
            name: loop.run_in_executor(
                self._pool, self._fetch_tickers_sync, name, symbols
            )
            for name in self._exchanges
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        by_symbol: Dict[str, Dict[str, Ticker]] = {s: {} for s in symbols}
        for name, res in zip(tasks.keys(), results):
            if isinstance(res, dict):
                for sym, ticker in res.items():
                    by_symbol[sym][name] = ticker
        return by_symbol

    # ── Order book depth ─────────────────────────────────────────────────────

    def _orderbook_depth_usdt_sync(
        self, exchange: str, symbol: str, side: str, notional: float
    ) -> float:
        """Sum up USDT depth on ask (for long) or bid (for short) side up to notional*5."""
        ex  = self._exchanges[exchange]
        try:
            ob  = ex.fetch_order_book(_ccxt_symbol(symbol), limit=20)
            key = "asks" if side == "long" else "bids"
            total = 0.0
            for price, size in ob[key]:
                total += float(price) * float(size)
            return total
        except Exception:
            return 0.0

    async def check_liquidity(
        self, long_ex: str, short_ex: str, symbol: str, notional: float
    ) -> Tuple[float, float]:
        """Returns (long_depth_usdt, short_depth_usdt)."""
        loop = asyncio.get_event_loop()
        long_d, short_d = await asyncio.gather(
            loop.run_in_executor(
                self._pool, self._orderbook_depth_usdt_sync,
                long_ex, symbol, "long", notional,
            ),
            loop.run_in_executor(
                self._pool, self._orderbook_depth_usdt_sync,
                short_ex, symbol, "short", notional,
            ),
        )
        return long_d, short_d

    # ── Order placement ───────────────────────────────────────────────────────

    def _place_order_sync(
        self,
        exchange:   str,
        symbol:     str,
        side:       str,     # "buy" | "sell"
        qty:        float,
        leverage:   int,
        reduce_only: bool = False,
    ) -> Optional[Dict]:
        ex = self._exchanges[exchange]
        try:
            # Set leverage
            try:
                ex.set_leverage(leverage, _ccxt_symbol(symbol),
                                params={"marginMode": "isolated"})
            except Exception:
                pass

            params: dict = {}
            if reduce_only:
                params["reduceOnly"] = True

            order = ex.create_market_order(
                _ccxt_symbol(symbol), side, qty, params=params
            )
            return order
        except Exception as exc:
            logger.error(f"[{exchange}] place_order {side} {symbol} qty={qty}: {exc}")
            return None

    async def open_both_legs(
        self,
        long_exchange:  str,
        short_exchange: str,
        symbol:         str,
        qty:            float,
        leverage:       int,
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Fire long and short orders simultaneously. Returns (long_order, short_order)."""
        loop  = asyncio.get_event_loop()
        long_f  = loop.run_in_executor(
            self._pool, self._place_order_sync,
            long_exchange, symbol, "buy", qty, leverage, False,
        )
        short_f = loop.run_in_executor(
            self._pool, self._place_order_sync,
            short_exchange, symbol, "sell", qty, leverage, False,
        )
        long_res, short_res = await asyncio.gather(long_f, short_f, return_exceptions=True)

        if isinstance(long_res, Exception):
            long_res = None
        if isinstance(short_res, Exception):
            short_res = None

        return long_res, short_res

    async def close_both_legs(
        self,
        long_exchange:  str,
        short_exchange: str,
        symbol:         str,
        qty:            float,
    ) -> None:
        """Close both legs simultaneously (reduceOnly market)."""
        loop  = asyncio.get_event_loop()
        await asyncio.gather(
            loop.run_in_executor(
                self._pool, self._place_order_sync,
                long_exchange, symbol, "sell", qty, 1, True,
            ),
            loop.run_in_executor(
                self._pool, self._place_order_sync,
                short_exchange, symbol, "buy", qty, 1, True,
            ),
            return_exceptions=True,
        )

    async def cancel_order(self, exchange: str, symbol: str, order_id: str) -> None:
        ex = self._exchanges[exchange]
        loop = asyncio.get_event_loop()
        def _sync():
            try:
                ex.cancel_order(order_id, _ccxt_symbol(symbol))
            except Exception:
                pass
        await loop.run_in_executor(self._pool, _sync)

    # ── OHLCV (trend filter) ─────────────────────────────────────────────────

    async def get_klines(
        self,
        exchange: str,
        symbol:   str,
        timeframe: str = "1m",
        limit:    int = 210,
    ) -> List[List]:
        ex   = self._exchanges[exchange]
        loop = asyncio.get_event_loop()
        def _sync():
            try:
                return ex.fetch_ohlcv(_ccxt_symbol(symbol), timeframe, limit=limit)
            except Exception:
                return []
        return await loop.run_in_executor(self._pool, _sync)

    def close(self):
        self._pool.shutdown(wait=False)
