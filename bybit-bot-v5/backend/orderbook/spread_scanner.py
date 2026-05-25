"""
SpreadScanner — оценивает рыночную возможность по текущему снимку стакана.

ЛОГИКА ЗАХВАТА СПРЕДА:
  Вход:  BUY limit на best_bid  (мы — маркет-мейкер на стороне покупки)
  Выход: SELL limit на best_ask (мы — маркет-мейкер на стороне продажи)
  Прибыль = (ask - bid) × qty − maker_fees × 2 − slippage − safety

exit_price = текущий best_ask (для BUY) или best_bid (для SELL).
Заранее вычисленные «costs_per_unit» НЕ определяют exit — выход ставится
ВНУТРЬ спреда, а не выше аска (что потребовало бы движения цены вверх).

Фильтры отбрасывают возможности, где спред недостаточен для покрытия
комиссий, проскальзывания и минимальной прибыли.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .bybit_ws import OrderbookSnapshot


@dataclass
class SpreadOpportunity:
    symbol:      str
    side:        str    # "Buy" | "Sell"
    entry_price: float
    exit_price:  float  # всегда market ask/bid — не «bid + costs»
    spread_abs:  float
    spread_pct:  float
    gross_usdt:  float
    net_usdt:    float
    fees_usdt:   float
    ratio:       float  # gross / fees
    score:       float  # 0.0–1.0
    skip_reason: str = ""

    @property
    def is_valid(self) -> bool:
        return not self.skip_reason


@dataclass
class ScanConfig:
    # Минимальный спред для входа.
    # Нужен ≥ 2*maker_fee + slippage + safety + min_net_profit/position.
    # Для позиции 25 USDT с дефолтными комиссиями это ~0.12%; ставим 0.10%
    # как лёгкий запас — net-фильтр отсечёт невыгодные случаи точнее.
    min_spread_pct:          float = 0.10     # %
    spread_to_fee_ratio_min: float = 2.0      # gross/fees >= 2 (spread > 2× fees)
    min_net_profit_usdt:     float = 0.005    # USDT после всех затрат
    maker_fee_pct:           float = 0.02     # % за одну сторону
    slippage_pct:            float = 0.01     # % резерв на проскальзывание
    safety_pct:              float = 0.01     # % страховой запас
    max_position_usdt:       float = 25.0
    enable_short:            bool  = False


class SpreadScanner:
    """
    Оценивает SpreadOpportunity по снимку стакана.
    exit_price = текущий best_ask (BUY) / best_bid (SELL) — настоящий захват спреда.
    """

    @staticmethod
    def scan(
        snap: OrderbookSnapshot,
        cfg:  ScanConfig,
        qty:  Optional[float] = None,
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
            side       = "Sell"
            entry      = best_ask   # входим на аске (продаём)
            exit_price = best_bid   # выходим на биде (закрываем покупателем)
        else:
            side       = "Buy"
            entry      = best_bid   # входим на биде (покупаем)
            exit_price = best_ask   # выходим на аске (продаём тем, кто покупает)

        # Количество
        if qty is None or qty <= 0:
            qty = cfg.max_position_usdt / entry if entry > 0 else 0.0
        if qty <= 0:
            return SpreadOpportunity(
                symbol=snap.symbol, side=side,
                entry_price=entry, exit_price=exit_price,
                spread_abs=snap.spread_abs, spread_pct=snap.spread_pct,
                gross_usdt=0, net_usdt=-999, fees_usdt=0, ratio=0, score=0,
                skip_reason="qty_zero",
            )

        # P&L на основе РЕАЛЬНОГО спреда (exit = market ask/bid)
        gross    = abs(exit_price - entry) * qty        # = spread × qty
        fees     = (entry + exit_price) * qty * cfg.maker_fee_pct / 100
        slippage = entry * qty * cfg.slippage_pct / 100
        safety   = entry * qty * cfg.safety_pct / 100
        net      = gross - fees - slippage - safety
        ratio    = (gross / fees) if fees > 0 else 0.0

        # Фильтры
        if snap.spread_pct < cfg.min_spread_pct:
            skip = f"spread={snap.spread_pct:.4f}%<{cfg.min_spread_pct}%"
        elif ratio < cfg.spread_to_fee_ratio_min:
            skip = f"ratio={ratio:.2f}<{cfg.spread_to_fee_ratio_min}"
        elif net < cfg.min_net_profit_usdt:
            skip = f"net={net:.5f}<{cfg.min_net_profit_usdt}"
        else:
            skip = ""

        # Score: качество возможности (ширина спреда × ratio), 0–1
        score = min(1.0, (snap.spread_pct / max(cfg.min_spread_pct, 0.001)) *
                    (ratio / max(cfg.spread_to_fee_ratio_min, 1.0)) * 0.25)

        return SpreadOpportunity(
            symbol=snap.symbol, side=side,
            entry_price=entry, exit_price=exit_price,
            spread_abs=snap.spread_abs, spread_pct=snap.spread_pct,
            gross_usdt=round(gross, 6),
            net_usdt=round(net, 6),
            fees_usdt=round(fees, 6),
            ratio=round(ratio, 2),
            score=round(score, 4),
            skip_reason=skip,
        )
