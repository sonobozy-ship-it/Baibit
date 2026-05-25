"""
LiquidityFilter — проверяет глубину и качество ликвидности стакана.

Блокирует вход если:
  - недостаточная глубина бида/аска
  - "стена" против позиции (крупный ордер блокирует выход)
  - сильный перекос стакана против направления
  - расширяющийся спред (признак нестабильности)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .bybit_ws import OrderbookSnapshot


@dataclass
class LiquidityConfig:
    min_book_depth_usdt:   float = 5_000.0   # мин. суммарная глубина в 0.3%
    max_wall_ratio:        float = 0.60       # стена > 60% глубины → блок
    min_imbalance_long:    float = 0.35       # для BUY: bid_ratio >= 0.35
    max_imbalance_short:   float = 0.65       # для SELL: bid_ratio <= 0.65
    max_spread_velocity:   float = 0.05       # макс. изменение spread_pct за тик
    min_best_bid_size_usdt: float = 200.0     # мин. объём на лучшем биде (USDT)
    min_best_ask_size_usdt: float = 200.0
    depth_check_pct:        float = 0.30      # % от цены для измерения глубины


@dataclass
class LiquidityResult:
    passed:         bool
    reason:         str
    bid_depth_usdt: float
    ask_depth_usdt: float
    imbalance:      float
    bid_wall_usdt:  float
    ask_wall_usdt:  float
    score:          float    # 0.0–1.0


class LiquidityFilter:
    """
    Проверяет один снимок стакана для заданного направления входа.
    Хранит историю снимков для фильтра velocity.
    """

    def __init__(self, symbols: List[str], history_len: int = 5):
        # История spread_pct per symbol (для velocity)
        self._spread_hist: Dict[str, List[float]] = {s: [] for s in symbols}
        self._max_hist = history_len

    def update_history(self, snap: OrderbookSnapshot) -> None:
        hist = self._spread_hist.setdefault(snap.symbol, [])
        hist.append(snap.spread_pct)
        if len(hist) > self._max_hist:
            hist.pop(0)

    def check(
        self,
        snap: OrderbookSnapshot,
        side: str,                  # "Buy" | "Sell"
        cfg:  LiquidityConfig,
        position_usdt: float = 25.0,
    ) -> LiquidityResult:

        bid_depth = snap.bid_depth_usdt(cfg.depth_check_pct)
        ask_depth = snap.ask_depth_usdt(cfg.depth_check_pct)
        imbalance = snap.imbalance
        bid_wall  = snap.bid_wall_usdt
        ask_wall  = snap.ask_wall_usdt

        # 1. Достаточная глубина с обеих сторон
        if bid_depth < cfg.min_book_depth_usdt:
            return self._fail(
                f"bid_depth={bid_depth:.0f}<{cfg.min_book_depth_usdt:.0f}",
                bid_depth, ask_depth, imbalance, bid_wall, ask_wall,
            )
        if ask_depth < cfg.min_book_depth_usdt:
            return self._fail(
                f"ask_depth={ask_depth:.0f}<{cfg.min_book_depth_usdt:.0f}",
                bid_depth, ask_depth, imbalance, bid_wall, ask_wall,
            )

        # 2. Объём на лучшем биде/аске достаточен
        best_bid_usdt = snap.bids[0].usdt if snap.bids else 0.0
        best_ask_usdt = snap.asks[0].usdt if snap.asks else 0.0
        if side == "Buy" and best_bid_usdt < cfg.min_best_bid_size_usdt:
            return self._fail(
                f"best_bid_size={best_bid_usdt:.0f}<{cfg.min_best_bid_size_usdt:.0f}",
                bid_depth, ask_depth, imbalance, bid_wall, ask_wall,
            )
        if side == "Sell" and best_ask_usdt < cfg.min_best_ask_size_usdt:
            return self._fail(
                f"best_ask_size={best_ask_usdt:.0f}<{cfg.min_best_ask_size_usdt:.0f}",
                bid_depth, ask_depth, imbalance, bid_wall, ask_wall,
            )

        # 3. Стена против позиции (блокирует выход)
        if side == "Buy":
            # Выходим через ask → крупная стена на аске плохо
            wall_ratio = ask_wall / ask_depth if ask_depth > 0 else 0.0
            if wall_ratio > cfg.max_wall_ratio:
                return self._fail(
                    f"ask_wall_ratio={wall_ratio:.2f}>{cfg.max_wall_ratio}",
                    bid_depth, ask_depth, imbalance, bid_wall, ask_wall,
                )
            # Дисбаланс: для BUY нужен минимальный support на биде
            if imbalance < cfg.min_imbalance_long:
                return self._fail(
                    f"imbalance={imbalance:.3f}<{cfg.min_imbalance_long} (bid too weak)",
                    bid_depth, ask_depth, imbalance, bid_wall, ask_wall,
                )
        else:
            wall_ratio = bid_wall / bid_depth if bid_depth > 0 else 0.0
            if wall_ratio > cfg.max_wall_ratio:
                return self._fail(
                    f"bid_wall_ratio={wall_ratio:.2f}>{cfg.max_wall_ratio}",
                    bid_depth, ask_depth, imbalance, bid_wall, ask_wall,
                )
            if imbalance > cfg.max_imbalance_short:
                return self._fail(
                    f"imbalance={imbalance:.3f}>{cfg.max_imbalance_short} (ask too weak)",
                    bid_depth, ask_depth, imbalance, bid_wall, ask_wall,
                )

        # 4. Velocity — расширяющийся спред
        hist = self._spread_hist.get(snap.symbol, [])
        if len(hist) >= 2:
            velocity = abs(hist[-1] - hist[-2])
            if velocity > cfg.max_spread_velocity:
                return self._fail(
                    f"spread_velocity={velocity:.4f}>{cfg.max_spread_velocity}",
                    bid_depth, ask_depth, imbalance, bid_wall, ask_wall,
                )

        # 5. Позиция не превышает доступную ликвидность
        exit_depth = ask_depth if side == "Buy" else bid_depth
        if exit_depth < position_usdt * 3:
            return self._fail(
                f"exit_depth={exit_depth:.0f}<{position_usdt*3:.0f} (3× position)",
                bid_depth, ask_depth, imbalance, bid_wall, ask_wall,
            )

        # Всё прошло — рассчитываем score
        depth_score  = min(1.0, min(bid_depth, ask_depth) / (cfg.min_book_depth_usdt * 5))
        imbal_score  = (imbalance - 0.5) * 2 if side == "Buy" else (0.5 - imbalance) * 2
        score = round(max(0.0, min(1.0, depth_score * 0.7 + imbal_score * 0.3)), 3)

        return LiquidityResult(
            passed=True, reason="ok",
            bid_depth_usdt=round(bid_depth, 2),
            ask_depth_usdt=round(ask_depth, 2),
            imbalance=round(imbalance, 4),
            bid_wall_usdt=round(bid_wall, 2),
            ask_wall_usdt=round(ask_wall, 2),
            score=score,
        )

    @staticmethod
    def _fail(reason: str, bd: float, ad: float,
              imb: float, bw: float, aw: float) -> LiquidityResult:
        return LiquidityResult(
            passed=False, reason=reason,
            bid_depth_usdt=round(bd, 2),
            ask_depth_usdt=round(ad, 2),
            imbalance=round(imb, 4),
            bid_wall_usdt=round(bw, 2),
            ask_wall_usdt=round(aw, 2),
            score=0.0,
        )
