"""
SpreadDetector — calculates cross-exchange spread and enforces the
5-second hold requirement before flagging an opportunity as actionable.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict, Optional, Tuple

from .config import ArbConfig
from .models import Ticker, SpreadSnapshot


class PriceHistory:
    """Rolling per-symbol/per-exchange price history for volatility detection."""

    def __init__(self, window_sec: float = 60.0):
        self._window  = window_sec
        # {symbol: {exchange: deque[(ts, price)]}}
        self._hist: Dict[str, Dict[str, Deque[Tuple[float, float]]]] = {}

    def record(self, tickers: Dict[str, Dict[str, Ticker]]) -> None:
        now = time.time()
        cutoff = now - self._window
        for symbol, by_ex in tickers.items():
            for ex, t in by_ex.items():
                hist = self._hist.setdefault(symbol, {}).setdefault(ex, deque())
                hist.append((now, t.mid))
                while hist and hist[0][0] < cutoff:
                    hist.popleft()

    def volatility_pct(self, symbol: str) -> float:
        """Max price change across all exchanges in the last 60s."""
        sym_hist = self._hist.get(symbol, {})
        max_vol = 0.0
        for hist in sym_hist.values():
            if len(hist) < 2:
                continue
            oldest = hist[0][1]
            newest = hist[-1][1]
            if oldest > 0:
                vol = abs(newest - oldest) / oldest * 100
                max_vol = max(max_vol, vol)
        return max_vol


class SpreadDetector:
    """
    Detects cross-exchange spread opportunities and tracks how long
    each opportunity has been above the entry threshold.
    """

    def __init__(self, cfg: ArbConfig):
        self._cfg = cfg
        # When spread for (symbol, long_ex, short_ex) was first seen above threshold
        self._first_seen: Dict[str, float] = {}

    def _key(self, symbol: str, long_ex: str, short_ex: str) -> str:
        return f"{symbol}:{long_ex}→{short_ex}"

    def scan(
        self,
        symbol:  str,
        tickers: Dict[str, Ticker],   # {exchange_name: Ticker}
    ) -> Optional[SpreadSnapshot]:
        """
        Find the best long/short pair for this symbol.
        Returns a SpreadSnapshot if spread exceeds the threshold, else None.
        Updates internal timing state.
        """
        if len(tickers) < 2:
            return None

        min_spread = self._cfg.min_spread_for(symbol)

        # Best pair: buy at the cheapest ask, sell at the most expensive bid
        cheap_ex    = min(tickers.items(), key=lambda x: x[1].ask)
        expensive_ex = max(tickers.items(), key=lambda x: x[1].bid)

        long_name, long_t    = cheap_ex
        short_name, short_t  = expensive_ex

        if long_name == short_name:
            return None

        long_ask  = long_t.ask
        short_bid = short_t.bid

        if short_bid <= long_ask:
            return None

        spread_pct = (short_bid - long_ask) / long_ask * 100
        key = self._key(symbol, long_name, short_name)

        if spread_pct < min_spread:
            # Below threshold — reset timer
            self._first_seen.pop(key, None)
            return None

        # Start timer on first sighting
        if key not in self._first_seen:
            self._first_seen[key] = time.time()

        return SpreadSnapshot(
            symbol         = symbol,
            long_exchange  = long_name,
            short_exchange = short_name,
            long_ask       = long_ask,
            short_bid      = short_bid,
            spread_pct     = spread_pct,
        )

    def is_mature(self, snap: SpreadSnapshot) -> bool:
        """True if this spread has been above threshold for >= hold_seconds."""
        key = self._key(snap.symbol, snap.long_exchange, snap.short_exchange)
        first = self._first_seen.get(key)
        if first is None:
            return False
        return (time.time() - first) >= self._cfg.spread_hold_seconds

    def reset(self, symbol: str, long_ex: str, short_ex: str) -> None:
        self._first_seen.pop(self._key(symbol, long_ex, short_ex), None)
