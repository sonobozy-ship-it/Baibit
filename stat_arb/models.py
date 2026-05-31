"""Data models for the stat-arb bot."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ExitReason(str, Enum):
    TAKE_PROFIT   = "take_profit"
    AGGRESSIVE_TP = "aggressive_tp"
    STOP_LOSS     = "stop_loss"
    MANUAL        = "manual"
    ERROR         = "error"


@dataclass
class Ticker:
    exchange: str
    symbol:   str
    bid:      float
    ask:      float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2


@dataclass
class SpreadSnapshot:
    """A detected spread opportunity at a single point in time."""
    symbol:         str
    long_exchange:  str     # cheap — we go LONG here
    short_exchange: str     # expensive — we go SHORT here
    long_ask:       float   # price we pay to enter long (cheap ask)
    short_bid:      float   # price we receive to enter short (expensive bid)
    spread_pct:     float   # (short_bid - long_ask) / long_ask * 100
    ts:             float = field(default_factory=time.time)

    @property
    def fee_adjusted_pct(self) -> float:
        """Rough net spread after round-trip fees (assumes 0.1% × 4 legs)."""
        return self.spread_pct - 0.40


@dataclass
class ArbPosition:
    """A live pair position (one long leg + one short leg)."""
    symbol:          str
    long_exchange:   str
    short_exchange:  str
    long_entry:      float     # fill price for long leg
    short_entry:     float     # fill price for short leg
    qty:             float     # base asset quantity per leg
    notional_usdt:   float     # USDT notional per leg
    entry_spread_pct: float
    expected_pnl_usdt: float   # max PnL if spread → 0 (minus fees)
    leverage:        int
    long_order_id:   Optional[str] = None
    short_order_id:  Optional[str] = None
    ts_open:         float = field(default_factory=time.time)

    # Filled on close
    ts_close:        float = 0.0
    long_exit:       float = 0.0
    short_exit:      float = 0.0
    exit_spread_pct: float = 0.0
    realized_pnl_usdt: float = 0.0
    exit_reason:     str   = ""

    def unrealized_pnl(self, current_long_bid: float, current_short_ask: float) -> float:
        """Estimate current floating PnL (before exit fees)."""
        long_pnl  = (current_long_bid  - self.long_entry)  * self.qty
        short_pnl = (self.short_entry  - current_short_ask) * self.qty
        return long_pnl + short_pnl

    def current_spread_pct(self, current_short_bid: float, current_long_ask: float) -> float:
        if current_long_ask <= 0:
            return 0.0
        return (current_short_bid - current_long_ask) / current_long_ask * 100
