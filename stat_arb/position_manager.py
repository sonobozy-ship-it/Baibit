"""
PositionManager — lifecycle of arb positions.

Opening:  fire both orders simultaneously (or simulate in paper mode).
          If one leg fails → cancel/rollback the other.
Monitoring: check exit conditions on every tick.
Closing:  fire both close orders simultaneously.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple

from .config import ArbConfig
from .exchange_hub import ExchangeHub
from .models import ArbPosition, ExitReason, SpreadSnapshot, Ticker
from .paper_engine import PaperEngine

logger = logging.getLogger(__name__)


class PositionManager:
    def __init__(
        self,
        hub:    ExchangeHub,
        cfg:    ArbConfig,
        paper:  Optional[PaperEngine] = None,
    ):
        self._hub   = hub
        self._cfg   = cfg
        self._paper = paper
        self.positions: List[ArbPosition] = []

    # ── Open ──────────────────────────────────────────────────────────────────

    async def open(
        self,
        snap:     SpreadSnapshot,
        notional: float,
        leverage: int,
        qty:      float,
    ) -> Optional[ArbPosition]:
        if self._paper:
            pos = self._paper.open_position(snap, qty, notional, leverage)
            self.positions.append(pos)
            return pos

        # Live — both legs simultaneously
        t0 = time.time()
        long_order, short_order = await self._hub.open_both_legs(
            snap.long_exchange, snap.short_exchange,
            snap.symbol, qty, leverage,
        )
        elapsed_ms = (time.time() - t0) * 1000
        logger.info(f"[PM] open both legs in {elapsed_ms:.0f}ms")

        # Rollback if one leg failed
        if not long_order or not short_order:
            await self._rollback(snap, long_order, short_order, qty)
            logger.error(f"[PM] OPEN FAILED {snap.symbol} — rolled back")
            return None

        # Check 300ms constraint
        if elapsed_ms > self._cfg.max_execution_gap_ms:
            logger.warning(
                f"[PM] {snap.symbol}: order gap {elapsed_ms:.0f}ms > "
                f"{self._cfg.max_execution_gap_ms}ms"
            )

        long_fill  = float(long_order.get("average")  or long_order.get("price")  or snap.long_ask)
        short_fill = float(short_order.get("average") or short_order.get("price") or snap.short_bid)

        pos = ArbPosition(
            symbol           = snap.symbol,
            long_exchange    = snap.long_exchange,
            short_exchange   = snap.short_exchange,
            long_entry       = long_fill,
            short_entry      = short_fill,
            qty              = qty,
            notional_usdt    = notional,
            entry_spread_pct = snap.spread_pct,
            expected_pnl_usdt= self._expected_pnl(snap, notional, qty),
            leverage         = leverage,
            long_order_id    = long_order.get("id"),
            short_order_id   = short_order.get("id"),
        )
        self.positions.append(pos)
        return pos

    # ── Monitor ───────────────────────────────────────────────────────────────

    def check_exit(
        self,
        pos:     ArbPosition,
        tickers: Dict[str, Dict[str, Ticker]],
    ) -> Optional[str]:
        """
        Returns ExitReason string if position should be closed, else None.
        """
        by_ex    = tickers.get(pos.symbol, {})
        long_t   = by_ex.get(pos.long_exchange)
        short_t  = by_ex.get(pos.short_exchange)

        if not long_t or not short_t:
            return None   # no fresh tickers

        # Current spread: to close, we need to reverse — sell long (at bid), buy short (at ask)
        curr_spread = pos.current_spread_pct(short_t.bid, long_t.ask)
        cfg = self._cfg

        # TP: spread converged to ≤ 50% of entry
        if curr_spread <= pos.entry_spread_pct * cfg.take_profit_ratio:
            pos.exit_spread_pct = curr_spread
            return ExitReason.TAKE_PROFIT

        # Aggressive TP: floating PnL >= 70% of expected
        unrealized = pos.unrealized_pnl(long_t.bid, short_t.ask)
        if pos.expected_pnl_usdt > 0 and unrealized >= pos.expected_pnl_usdt * cfg.aggressive_profit_ratio:
            pos.exit_spread_pct = curr_spread
            return ExitReason.AGGRESSIVE_TP

        # SL: spread expanded to ≥ 150% of entry
        if curr_spread >= pos.entry_spread_pct * cfg.stop_loss_ratio:
            pos.exit_spread_pct = curr_spread
            return ExitReason.STOP_LOSS

        return None

    # ── Close ─────────────────────────────────────────────────────────────────

    async def close(
        self,
        pos:    ArbPosition,
        reason: str,
        tickers: Dict[str, Dict[str, Ticker]],
    ) -> float:
        """Close both legs. Returns realized PnL."""
        by_ex   = tickers.get(pos.symbol, {})
        long_t  = by_ex.get(pos.long_exchange)
        short_t = by_ex.get(pos.short_exchange)

        if self._paper:
            long_bid  = long_t.bid  if long_t  else pos.long_entry
            short_ask = short_t.ask if short_t else pos.short_entry
            pnl = self._paper.close_position(pos, long_bid, short_ask, reason)
        else:
            await self._hub.close_both_legs(
                pos.long_exchange, pos.short_exchange, pos.symbol, pos.qty
            )
            long_exit  = long_t.bid  if long_t  else pos.long_entry
            short_exit = short_t.ask if short_t else pos.short_entry
            long_pnl   = (long_exit  - pos.long_entry)  * pos.qty
            short_pnl  = (pos.short_entry - short_exit) * pos.qty
            exit_fees  = (long_exit + short_exit) * pos.qty * 0.001 * 2   # ~0.1% × 2 legs
            pnl        = long_pnl + short_pnl - exit_fees

            pos.ts_close          = time.time()
            pos.long_exit         = long_exit
            pos.short_exit        = short_exit
            pos.realized_pnl_usdt = round(pnl, 6)
            pos.exit_reason       = reason

        if pos in self.positions:
            self.positions.remove(pos)
        return pnl

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _rollback(self, snap, long_order, short_order, qty):
        if long_order and not short_order:
            await self._hub.close_both_legs(
                snap.long_exchange, snap.long_exchange,  # dummy — won't matter
                snap.symbol, qty
            )
        elif short_order and not long_order:
            await self._hub.close_both_legs(
                snap.short_exchange, snap.short_exchange,
                snap.symbol, qty
            )

    def _expected_pnl(
        self, snap: SpreadSnapshot, notional: float, qty: float
    ) -> float:
        gross = snap.spread_pct / 100 * notional
        fees  = (snap.long_ask + snap.short_bid) * qty * 0.001 * 4   # 4 legs total
        return round(gross - fees, 6)
