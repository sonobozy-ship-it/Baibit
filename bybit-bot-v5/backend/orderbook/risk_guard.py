"""
RiskGuard — глобальные ограничения риска для ORDERBOOK_ONLY режима.

Контролирует:
  - суммарную экспозицию (max_total_exposure_usdt)
  - количество открытых позиций
  - дневной лимит убытков
  - кулдаун после убытка/серии убытков
  - дневной счётчик комиссий
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple


@dataclass
class RiskConfig:
    max_open_positions:      int   = 3
    max_total_exposure_usdt: float = 100.0
    max_position_usdt:       float = 25.0
    daily_loss_limit_usdt:   float = 25.0
    max_consecutive_losses:  int   = 3
    cooldown_after_loss_sec: float = 120.0
    cooldown_cancel_sec:     float = 10.0
    max_slippage_pct:        float = 0.03    # %


@dataclass
class SymbolRisk:
    open_positions: int   = 0
    exposure_usdt:  float = 0.0
    cons_losses:    int   = 0
    cooldown_until: float = 0.0
    total_trades:   int   = 0
    total_wins:     int   = 0
    total_pnl:      float = 0.0
    total_fees:     float = 0.0


class RiskGuard:
    """Единая точка контроля риска для всего стаканного движка."""

    def __init__(self, cfg: RiskConfig):
        self._cfg         = cfg
        self._symbols:    Dict[str, SymbolRisk] = {}
        self._daily_loss:  float = 0.0
        self._daily_date: str   = ""
        self._daily_fees: float = 0.0
        self._total_exp:  float = 0.0   # текущая суммарная экспозиция
        self._open_count: int   = 0

        # Статистика пропусков
        self._skips: Dict[str, int] = {}

    def _reset_day(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._daily_date:
            self._daily_date = today
            self._daily_loss  = 0.0
            self._daily_fees = 0.0

    def _sym(self, symbol: str) -> SymbolRisk:
        if symbol not in self._symbols:
            self._symbols[symbol] = SymbolRisk()
        return self._symbols[symbol]

    # ── Проверка перед входом ─────────────────────────────────────────────────

    def can_open(self, symbol: str, position_usdt: float) -> Tuple[bool, str]:
        self._reset_day()
        cfg = self._cfg
        sr  = self._sym(symbol)

        if self._daily_loss <= -cfg.daily_loss_limit_usdt:
            return False, f"daily_loss_limit({self._daily_loss:.2f})"
        if self._open_count >= cfg.max_open_positions:
            return False, f"max_positions({self._open_count})"
        if self._total_exp + position_usdt > cfg.max_total_exposure_usdt:
            return False, f"max_exposure({self._total_exp:.0f}+{position_usdt:.0f}>{cfg.max_total_exposure_usdt:.0f})"
        if time.time() < sr.cooldown_until:
            rem = sr.cooldown_until - time.time()
            return False, f"cooldown_{rem:.0f}s"
        if sr.cons_losses >= cfg.max_consecutive_losses:
            return False, f"cons_losses({sr.cons_losses})"

        return True, "ok"

    def record_skip(self, reason: str) -> None:
        self._skips[reason] = self._skips.get(reason, 0) + 1

    # ── Регистрация событий ───────────────────────────────────────────────────

    def register_open(self, symbol: str, position_usdt: float) -> None:
        sr = self._sym(symbol)
        sr.open_positions += 1
        sr.exposure_usdt  += position_usdt
        self._open_count  += 1
        self._total_exp   += position_usdt

    def register_close(
        self,
        symbol:    str,
        net_pnl:   float,
        fees:      float,
        position_usdt: float,
        cancelled: bool = False,
    ) -> None:
        self._reset_day()
        sr = self._sym(symbol)
        sr.open_positions = max(0, sr.open_positions - 1)
        sr.exposure_usdt  = max(0.0, sr.exposure_usdt - position_usdt)
        self._open_count  = max(0, self._open_count - 1)
        self._total_exp   = max(0.0, self._total_exp - position_usdt)

        if cancelled:
            sr.cooldown_until = time.time() + self._cfg.cooldown_cancel_sec
            return

        self._daily_loss  += min(0.0, net_pnl)
        self._daily_fees += fees
        sr.total_trades  += 1
        sr.total_pnl     += net_pnl
        sr.total_fees    += fees

        if net_pnl > 0:
            sr.total_wins += 1
            sr.cons_losses = 0
        else:
            sr.cons_losses   += 1
            sr.cooldown_until = time.time() + self._cfg.cooldown_after_loss_sec

    # ── Метрики ───────────────────────────────────────────────────────────────

    @property
    def daily_pnl(self) -> float:
        """Накопленные убытки за день (≤ 0). Выигрыши не учитываются — это loss-accumulator."""
        self._reset_day()
        return self._daily_loss

    @property
    def daily_fees(self) -> float:
        self._reset_day()
        return self._daily_fees

    @property
    def open_positions(self) -> int:
        return self._open_count

    @property
    def total_exposure(self) -> float:
        return self._total_exp

    def summary(self) -> dict:
        self._reset_day()
        total_trades = sum(s.total_trades for s in self._symbols.values())
        total_wins   = sum(s.total_wins   for s in self._symbols.values())
        total_pnl    = sum(s.total_pnl    for s in self._symbols.values())
        wr = total_wins / total_trades if total_trades > 0 else 0.0
        return {
            "open_positions":    self._open_count,
            "total_exposure":    round(self._total_exp, 2),
            "daily_pnl":         round(self._daily_loss, 5),
            "daily_fees":        round(self._daily_fees, 5),
            "total_trades":      total_trades,
            "winrate":           round(wr, 3),
            "total_pnl":         round(total_pnl, 5),
            "skips":             dict(self._skips),
            "by_symbol": {
                s: {
                    "total":   r.total_trades,
                    "wr":      round(r.total_wins / r.total_trades, 3) if r.total_trades > 0 else 0.0,
                    "pnl":     round(r.total_pnl, 5),
                    "fees":    round(r.total_fees, 5),
                    "cons_loss": r.cons_losses,
                    "exposure":  round(r.exposure_usdt, 2),
                }
                for s, r in self._symbols.items()
            },
        }
