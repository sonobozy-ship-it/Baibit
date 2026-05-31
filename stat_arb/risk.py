"""Risk manager — position sizing and deployment limits."""
from __future__ import annotations

from .config import ArbConfig
from .models import ArbPosition


class RiskManager:
    def __init__(self, cfg: ArbConfig):
        self._cfg         = cfg
        self._open: list  = []   # List[ArbPosition] — reference from position_manager

    def attach(self, positions: list) -> None:
        self._open = positions

    def can_open(self) -> tuple[bool, str]:
        max_pos = self._cfg.max_positions()
        if len(self._open) >= max_pos:
            return False, f"max_positions={max_pos}"

        deployed = sum(p.notional_usdt * 2 for p in self._open)   # 2 legs
        max_dep  = self._cfg.capital_usdt * self._cfg.max_deployed_pct / 100
        notional = self._cfg.position_notional()
        if deployed + notional * 2 > max_dep:
            return False, f"max_deployed={max_dep:.0f}"

        return True, ""

    def position_notional(self) -> float:
        return self._cfg.position_notional()

    def leverage(self) -> int:
        lev = self._cfg.default_leverage
        return min(lev, self._cfg.max_leverage)

    def qty_for_notional(self, notional_usdt: float, price: float) -> float:
        if price <= 0:
            return 0.0
        return notional_usdt / price
