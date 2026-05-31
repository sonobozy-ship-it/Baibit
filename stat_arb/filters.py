"""
Entry filters for the stat-arb bot.
Each filter returns (passed: bool, reason: str).
"""
from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from .config import ArbConfig
from .exchange_hub import ExchangeHub
from .models import SpreadSnapshot
from .spread_detector import PriceHistory

logger = logging.getLogger(__name__)


def _ema(prices: List[float], period: int) -> float:
    if not prices:
        return 0.0
    k   = 2 / (period + 1)
    val = prices[0]
    for p in prices[1:]:
        val = p * k + val * (1 - k)
    return val


class EntryFilters:
    """Aggregates all entry filters."""

    def __init__(self, hub: ExchangeHub, cfg: ArbConfig):
        self._hub = hub
        self._cfg = cfg
        self._events: List[datetime] = []
        self._load_events()

    def _load_events(self) -> None:
        path = Path(self._cfg.news_events_file)
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            for ts_str in data.get("events", []):
                self._events.append(datetime.fromisoformat(ts_str.replace("Z", "+00:00")))
        except Exception as exc:
            logger.warning(f"[Filters] events file load error: {exc}")

    # ── Volatility ────────────────────────────────────────────────────────────

    def check_volatility(
        self, snap: SpreadSnapshot, history: PriceHistory
    ) -> Tuple[bool, str]:
        vol = history.volatility_pct(snap.symbol)
        if vol >= self._cfg.max_volatility_60s_pct:
            return False, f"volatility={vol:.2f}%>={self._cfg.max_volatility_60s_pct}%"
        return True, ""

    # ── Liquidity ─────────────────────────────────────────────────────────────

    async def check_liquidity(
        self, snap: SpreadSnapshot, notional_usdt: float
    ) -> Tuple[bool, str]:
        required = notional_usdt * self._cfg.min_liquidity_ratio
        long_d, short_d = await self._hub.check_liquidity(
            snap.long_exchange, snap.short_exchange, snap.symbol, notional_usdt
        )
        if long_d < required:
            return False, f"long_depth={long_d:.0f}<{required:.0f}"
        if short_d < required:
            return False, f"short_depth={short_d:.0f}<{required:.0f}"
        return True, ""

    # ── Trend ─────────────────────────────────────────────────────────────────

    async def check_trend(self, snap: SpreadSnapshot) -> Tuple[bool, str]:
        if not self._cfg.trend_filter_enabled:
            return True, ""

        klines_long, klines_short = [], []
        try:
            klines_long  = await self._hub.get_klines(
                snap.long_exchange,  snap.symbol, "1m", limit=210
            )
            klines_short = await self._hub.get_klines(
                snap.short_exchange, snap.symbol, "1m", limit=210
            )
        except Exception:
            return True, ""   # can't check → allow

        def strong_trend(klines) -> bool:
            if len(klines) < self._cfg.ema_long:
                return False
            closes = [float(k[4]) for k in klines]
            ema50  = _ema(closes[-self._cfg.ema_short:], self._cfg.ema_short)
            ema200 = _ema(closes[-self._cfg.ema_long:],  self._cfg.ema_long)
            if ema200 <= 0:
                return False
            deviation = abs(ema50 - ema200) / ema200 * 100
            return deviation >= self._cfg.trend_deviation_pct

        # Both exchanges showing strong trend in same direction
        trend_long  = strong_trend(klines_long)
        trend_short = strong_trend(klines_short)

        if trend_long and trend_short:
            # Check same direction
            if klines_long and klines_short:
                closes_l = [float(k[4]) for k in klines_long[-5:]]
                closes_s = [float(k[4]) for k in klines_short[-5:]]
                dir_l = closes_l[-1] > closes_l[0]
                dir_s = closes_s[-1] > closes_s[0]
                if dir_l == dir_s:
                    return False, "strong_trend_both_exchanges"

        return True, ""

    # ── News ─────────────────────────────────────────────────────────────────

    def check_news(self) -> Tuple[bool, str]:
        if not self._cfg.news_filter_enabled or not self._events:
            return True, ""

        now    = datetime.now(timezone.utc)
        buf_s  = self._cfg.news_buffer_minutes * 60
        for ev in self._events:
            delta = abs((now - ev).total_seconds())
            if delta <= buf_s:
                return False, f"near_event={ev.isoformat()}"
        return True, ""

    # ── Composite check ───────────────────────────────────────────────────────

    async def check_all(
        self,
        snap:          SpreadSnapshot,
        history:       PriceHistory,
        notional_usdt: float,
    ) -> Tuple[bool, str]:
        ok, reason = self.check_volatility(snap, history)
        if not ok:
            return False, reason

        ok, reason = self.check_news()
        if not ok:
            return False, reason

        ok, reason = await self.check_liquidity(snap, notional_usdt)
        if not ok:
            return False, reason

        ok, reason = await self.check_trend(snap)
        if not ok:
            return False, reason

        return True, ""
