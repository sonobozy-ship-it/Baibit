"""
PairSelector — фильтрует пары из whitelist по ликвидности и стабильности.

Исключает:
  - Пары с недостаточной глубиной стакана
  - Пары с аномальным расширением спреда
  - Пары с резким pump/dump (velocity > threshold)
  - Пары на кулдауне после убытков (из RiskGuard)
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from .bybit_ws import OrderbookSnapshot
from .risk_guard import RiskGuard


@dataclass
class PairSelectorConfig:
    whitelist:               List[str] = field(default_factory=lambda: [
        "DOGEUSDT", "XRPUSDT", "TRXUSDT", "ADAUSDT", "SOLUSDT", "BTCUSDT", "ETHUSDT",
    ])
    max_pairs_active:        int   = 7
    min_book_depth_usdt:     float = 3_000.0
    max_spread_pct:          float = 0.50    # если спред слишком широкий — пропуск
    max_pump_pct:            float = 1.0     # макс. изменение mid_price за 60 сек
    pump_window_sec:         float = 60.0


class PairSelector:
    """
    Выбирает торгуемые пары из whitelist, отсеивая нестабильные.
    """

    def __init__(self, cfg: PairSelectorConfig):
        self._cfg = cfg
        # История mid_price для pump detection (deque для O(1) удаления слева)
        self._mid_hist: Dict[str, Deque[Tuple[float, float]]] = {
            s: deque() for s in cfg.whitelist
        }  # [(timestamp, mid_price)]

    def update(self, snap: OrderbookSnapshot) -> None:
        """Вызывается при каждом обновлении стакана."""
        hist = self._mid_hist.setdefault(snap.symbol, deque())
        now  = time.time()
        hist.append((now, snap.mid_price))
        # Очищаем историю старше 2× pump_window_sec (popleft — O(1))
        cutoff = now - self._cfg.pump_window_sec * 2
        while hist and hist[0][0] < cutoff:
            hist.popleft()

    def select(
        self,
        snapshots: Dict[str, OrderbookSnapshot],
        risk:      Optional[RiskGuard] = None,
    ) -> List[str]:
        """
        Возвращает список торгуемых пар (≤ max_pairs_active) из whitelist,
        прошедших все проверки.
        """
        cfg = self._cfg
        active: List[str] = []

        for sym in cfg.whitelist:
            snap = snapshots.get(sym)
            if snap is None:
                continue

            # Проверка глубины
            bd = snap.bid_depth_usdt(0.3)
            ad = snap.ask_depth_usdt(0.3)
            if min(bd, ad) < cfg.min_book_depth_usdt:
                continue

            # Слишком широкий спред — признак нестабильности
            if snap.spread_pct > cfg.max_spread_pct:
                continue

            # Pump/dump detection
            if self._is_pumping(sym, snap):
                continue

            # RiskGuard кулдаун
            if risk is not None:
                ok, _ = risk.can_open(sym, 1.0)   # проверяем только кулдауны
                if not ok:
                    continue

            active.append(sym)
            if len(active) >= cfg.max_pairs_active:
                break

        return active

    def _is_pumping(self, symbol: str, snap: OrderbookSnapshot) -> bool:
        cfg  = self._cfg
        hist = self._mid_hist.get(symbol)
        if not hist:
            return False
        now  = time.time()
        window_start = now - cfg.pump_window_sec
        # Записи внутри окна pump_window_sec (deque отсортирован по времени)
        window_entries = [(t, p) for t, p in hist if t >= window_start]
        if len(window_entries) < 2:
            return False
        oldest_price = window_entries[0][1]
        if oldest_price <= 0:
            return False
        change_pct = abs(snap.mid_price - oldest_price) / oldest_price * 100
        return change_pct > cfg.max_pump_pct

    def status(self, snapshots: Dict[str, OrderbookSnapshot]) -> str:
        lines = ["Пары в whitelist:"]
        for sym in self._cfg.whitelist:
            snap = snapshots.get(sym)
            if snap:
                pumping = "⚡" if self._is_pumping(sym, snap) else ""
                lines.append(
                    f"  {sym}: spread={snap.spread_pct:.3f}% "
                    f"bid_d={snap.bid_depth_usdt(0.3):.0f} "
                    f"ask_d={snap.ask_depth_usdt(0.3):.0f} USDT {pumping}"
                )
            else:
                lines.append(f"  {sym}: нет данных")
        return "\n".join(lines)
