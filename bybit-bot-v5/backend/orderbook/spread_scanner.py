"""
SpreadScanner — оценивает рыночную возможность по текущему снимку стакана.
Проверяет:
  - спред >= MIN_SPREAD_PCT
  - net_edge после комиссий > min_net_profit
  - spread_to_fee_ratio >= 3.0
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .bybit_ws import OrderbookSnapshot


@dataclass
class SpreadOpportunity:
    symbol:          str
    side:            str    # "Buy" | "Sell" | "Both"
    entry_price:     float
    exit_price:      float
    spread_abs:      float
    spread_pct:      float
    gross_usdt:      float
    net_usdt:        float
    fees_usdt:       float
    ratio:           float  # spread_to_fee_ratio
    score:           float  # 0.0–1.0 (качество возможности)
    skip_reason:     str = ""

    @property
    def is_valid(self) -> bool:
        return not self.skip_reason


@dataclass
class ScanConfig:
    min_spread_pct:          float = 0.08     # %
    spread_to_fee_ratio_min: float = 3.0
    min_net_profit_usdt:     float = 0.015    # USDT
    maker_fee_pct:           float = 0.02
    slippage_pct:            float = 0.01
    safety_pct:              float = 0.01
    max_position_usdt:       float = 25.0
    enable_short:            bool  = False


class SpreadScanner:
    """
    Вычисляет SpreadOpportunity для одного символа на основе OrderbookSnapshot.
    Не хранит состояние — чистая функция scan().
    """

    @staticmethod
    def scan(
        snap:  OrderbookSnapshot,
        cfg:   ScanConfig,
        qty:   Optional[float] = None,  # если None — вычисляется из max_position_usdt
    ) -> SpreadOpportunity:
        best_bid = snap.best_bid
        best_ask = snap.best_ask

        if best_bid <= 0 or best_ask <= 0:
            return SpreadOpportunity(
                symbol=snap.symbol, side="Buy",
                entry_price=0, exit_price=0,
                spread_abs=0, spread_pct=0,
                gross_usdt=0, net_usdt=-999, fees_usdt=0, ratio=0, score=0,
                skip_reason="no_price",
            )

        # Направление — по дисбалансу стакана
        imb = snap.imbalance
        if cfg.enable_short and imb < 0.35:
            side  = "Sell"
            entry = best_ask
        else:
            side  = "Buy"
            entry = best_bid

        # Количество
        if qty is None or qty <= 0:
            qty = cfg.max_position_usdt / entry if entry > 0 else 0.0
        if qty <= 0:
            return SpreadOpportunity(
                symbol=snap.symbol, side=side,
                entry_price=entry, exit_price=entry,
                spread_abs=snap.spread_abs, spread_pct=snap.spread_pct,
                gross_usdt=0, net_usdt=-999, fees_usdt=0, ratio=0, score=0,
                skip_reason="qty_zero",
            )

        # Минимальная цена выхода
        costs_per_unit = (
            2 * entry * cfg.maker_fee_pct / 100
            + entry * cfg.slippage_pct / 100
            + entry * cfg.safety_pct / 100
            + cfg.min_net_profit_usdt / qty
        )
        if side == "Buy":
            exit_price = entry + costs_per_unit
        else:
            exit_price = entry - costs_per_unit

        gross     = abs(exit_price - entry) * qty
        fees      = (entry + exit_price) * qty * cfg.maker_fee_pct / 100
        slippage  = entry * qty * cfg.slippage_pct / 100
        safety    = entry * qty * cfg.safety_pct / 100
        net       = gross - fees - slippage - safety
        ratio     = (gross / fees) if fees > 0 else 0.0

        # Фильтры
        if snap.spread_pct < cfg.min_spread_pct:
            skip = f"spread_pct={snap.spread_pct:.4f}%<{cfg.min_spread_pct}%"
        elif ratio < cfg.spread_to_fee_ratio_min:
            skip = f"ratio={ratio:.2f}<{cfg.spread_to_fee_ratio_min}"
        elif net < cfg.min_net_profit_usdt:
            skip = f"net={net:.5f}<{cfg.min_net_profit_usdt}"
        else:
            skip = ""

        # Score: нормализованный спред×ratio (0–1)
        score = min(1.0, (snap.spread_pct / max(cfg.min_spread_pct, 0.001)) *
                    (ratio / max(cfg.spread_to_fee_ratio_min, 1.0)) * 0.25)

        return SpreadOpportunity(
            symbol=snap.symbol, side=side,
            entry_price=entry,  exit_price=exit_price,
            spread_abs=snap.spread_abs, spread_pct=snap.spread_pct,
            gross_usdt=round(gross, 6),
            net_usdt=round(net, 6),
            fees_usdt=round(fees, 6),
            ratio=round(ratio, 2),
            score=round(score, 4),
            skip_reason=skip,
        )
