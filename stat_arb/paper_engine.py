"""
Paper trading engine — simulates fills and tracks PnL without real orders.
Market orders are assumed to fill at the current bid/ask.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Optional

from .config import ArbConfig
from .models import ArbPosition, SpreadSnapshot, Ticker

logger = logging.getLogger(__name__)


class PaperEngine:
    def __init__(self, cfg: ArbConfig):
        self._cfg     = cfg
        self.balance  = cfg.capital_usdt

    def open_position(
        self,
        snap:         SpreadSnapshot,
        qty:          float,
        notional:     float,
        leverage:     int,
    ) -> ArbPosition:
        """Simulate simultaneous market entry on both legs."""
        # In paper mode, fills are instant at the snapshot prices
        long_fill  = snap.long_ask     # buy at ask
        short_fill = snap.short_bid    # sell at bid

        # Taker fees on entry
        entry_fees = (long_fill + short_fill) * qty * self._fee_pct * 2 / 100

        # Expected gross PnL at 50% convergence
        expected_gross = snap.spread_pct / 100 * notional
        expected_fees  = entry_fees + (long_fill + short_fill) * qty * self._fee_pct * 2 / 100
        expected_net   = expected_gross - expected_fees

        self.balance -= entry_fees   # deduct entry fees immediately

        pos = ArbPosition(
            symbol           = snap.symbol,
            long_exchange    = snap.long_exchange,
            short_exchange   = snap.short_exchange,
            long_entry       = long_fill,
            short_entry      = short_fill,
            qty              = qty,
            notional_usdt    = notional,
            entry_spread_pct = snap.spread_pct,
            expected_pnl_usdt= round(expected_net, 4),
            leverage         = leverage,
        )
        logger.info(
            f"[PAPER] OPEN {snap.symbol} long@{long_fill} short@{short_fill} "
            f"spread={snap.spread_pct:.3f}% expected_net≈{expected_net:.4f} USDT"
        )
        return pos

    def close_position(
        self,
        pos:         ArbPosition,
        long_bid:    float,    # current bid on long exchange (to exit long)
        short_ask:   float,    # current ask on short exchange (to exit short)
        reason:      str,
    ) -> float:
        """Simulate simultaneous market exit. Returns realized net PnL."""
        long_pnl  = (long_bid  - pos.long_entry)  * pos.qty
        short_pnl = (pos.short_entry - short_ask)  * pos.qty
        gross_pnl = long_pnl + short_pnl

        # Taker fees on exit
        exit_fees = (long_bid + short_ask) * pos.qty * self._fee_pct * 2 / 100
        net_pnl   = gross_pnl - exit_fees

        self.balance += net_pnl

        pos.ts_close         = time.time()
        pos.long_exit        = long_bid
        pos.short_exit       = short_ask
        pos.realized_pnl_usdt = round(net_pnl, 6)
        pos.exit_reason      = reason

        logger.info(
            f"[PAPER] CLOSE {pos.symbol} pnl={net_pnl:+.5f} USDT "
            f"hold={pos.ts_close - pos.ts_open:.1f}s reason={reason}"
        )
        return net_pnl

    @property
    def _fee_pct(self) -> float:
        return self._cfg.creds.get("binance", type("", (), {"taker_fee_pct": 0.10})()).taker_fee_pct
