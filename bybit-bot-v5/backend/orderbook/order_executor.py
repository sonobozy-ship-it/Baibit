"""
OrderExecutor — жизненный цикл ордеров для одного символа.

Стейт-машина:
  IDLE      → нет открытых ордеров/позиций
  ENTRY     → ордер входа выставлен, ждём исполнения
  POSITION  → вход исполнен, выставляем ордер выхода
  EXIT      → ордер выхода выставлен, ждём исполнения
  EMERGENCY → принудительный выход

Принципы:
  • Только POST_ONLY LIMIT ордера на вход.
  • Если ордер входа не исполнился за max_order_lifetime_sec — отменить.
  • После исполнения входа: немедленно выставить ордер выхода.
  • Если цена пошла против на max_adverse_move_pct → emergency exit.
  • Dry-run: ничего не отправляем, только логируем.
  • Paper: симулируем исполнение без реальных ордеров.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from .bybit_ws import OrderbookSnapshot

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Конфиг
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExecutorConfig:
    max_order_lifetime_sec: float = 2.0
    max_exit_wait_sec:      float = 8.0
    emergency_exit_sec:     float = 20.0
    max_adverse_move_pct:   float = 0.15    # % против позиции → emergency
    maker_fee_pct:          float = 0.02
    dry_run:                bool  = False
    paper:                  bool  = True


# ─────────────────────────────────────────────────────────────────────────────
# Запись сделки
# ─────────────────────────────────────────────────────────────────────────────

class ExecStatus(str, Enum):
    OPEN      = "open"
    WIN       = "win"
    LOSS      = "loss"
    CANCELLED = "cancelled"
    EMERGENCY = "emergency"
    DRY_RUN   = "dry_run"


@dataclass
class TradeRecord:
    symbol:      str
    side:        str     # "Buy" | "Sell"
    entry_price: float = 0.0
    exit_price:  float = 0.0
    qty:         float = 0.0
    gross_pnl:   float = 0.0
    fees_usdt:   float = 0.0
    net_pnl:     float = 0.0
    spread_entry: float = 0.0
    imbalance:   float = 0.5
    bid_depth:   float = 0.0
    ask_depth:   float = 0.0
    latency_ms:  float = 0.0
    hold_sec:    float = 0.0
    status:      ExecStatus = ExecStatus.OPEN
    exit_reason: str  = ""
    ts_open:     float = field(default_factory=time.time)
    ts_close:    float = 0.0

    def finalize(self, exit_price: float, fees: float,
                 status: ExecStatus, reason: str) -> None:
        self.exit_price  = exit_price
        self.ts_close    = time.time()
        self.hold_sec    = self.ts_close - self.ts_open
        self.fees_usdt   = fees
        self.status      = status
        self.exit_reason = reason
        if self.side == "Buy":
            self.gross_pnl = (exit_price - self.entry_price) * self.qty
        else:
            self.gross_pnl = (self.entry_price - exit_price) * self.qty
        self.net_pnl = self.gross_pnl - fees


# ─────────────────────────────────────────────────────────────────────────────
# Paper order simulator
# ─────────────────────────────────────────────────────────────────────────────

class PaperOrder:
    def __init__(self, side: str, price: float, qty: float):
        self.side        = side
        self.price       = price
        self.qty         = qty
        self.filled      = False
        self.fill_price: Optional[float] = None

    def try_fill(self, snap: OrderbookSnapshot) -> bool:
        if self.filled:
            return True
        # Maker BUY: fills when best_ask drops to our bid price
        if self.side == "Buy"  and snap.best_ask <= self.price:
            self.filled = True
            self.fill_price = snap.best_ask
        elif self.side == "Sell" and snap.best_bid >= self.price:
            self.filled = True
            self.fill_price = snap.best_bid
        return self.filled


# ─────────────────────────────────────────────────────────────────────────────
# State machine per symbol
# ─────────────────────────────────────────────────────────────────────────────

class Phase(str, Enum):
    IDLE      = "idle"
    ENTRY     = "entry"
    POSITION  = "position"
    EXIT      = "exit"
    EMERGENCY = "emergency"


@dataclass
class ExecState:
    symbol:          str
    phase:           Phase = Phase.IDLE
    trade:           Optional[TradeRecord] = None
    side:            Optional[str] = None
    entry_order_id:  Optional[str] = None
    exit_order_id:   Optional[str] = None
    paper_entry:     Optional[PaperOrder] = None
    paper_exit:      Optional[PaperOrder] = None
    entry_placed_at: float = 0.0
    exit_placed_at:  float = 0.0
    last_poll:       float = 0.0

    def reset(self) -> None:
        self.phase          = Phase.IDLE
        self.trade          = None
        self.side           = None
        self.entry_order_id = None
        self.exit_order_id  = None
        self.paper_entry    = None
        self.paper_exit     = None
        self.entry_placed_at = 0.0
        self.exit_placed_at  = 0.0
        self.last_poll       = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Основной executor
# ─────────────────────────────────────────────────────────────────────────────

import asyncio


class OrderExecutor:
    """
    Управляет жизненным циклом ордеров для одного символа.
    Работает в asyncio-контексте.

    Все публичные методы защищены asyncio.Lock: высокочастотные WS-тики
    могут запускать on_tick() параллельно через ensure_future(), поэтому
    состояние стейт-машины должно обновляться атомарно.
    """

    def __init__(
        self,
        symbol:      str,
        cfg:         ExecutorConfig,
        bybit_client = None,         # BybitClient (pybit-обёртка)
    ):
        self._symbol = symbol
        self._cfg    = cfg
        self._bybit  = bybit_client
        self._state  = ExecState(symbol=symbol)
        self._lock   = asyncio.Lock()   # защита от конкурентных on_tick / open

    @property
    def phase(self) -> Phase:
        return self._state.phase

    @property
    def trade(self) -> Optional[TradeRecord]:
        return self._state.trade

    @property
    def is_idle(self) -> bool:
        return self._state.phase == Phase.IDLE

    # ── Открытие позиции ──────────────────────────────────────────────────────

    async def open(
        self,
        side:        str,          # "Buy" | "Sell"
        entry_price: float,
        exit_price:  float,
        qty:         float,
        snap:        OrderbookSnapshot,
    ) -> bool:
        """Выставить ордер входа. Возвращает True если ордер создан."""
        async with self._lock:
            return await self._open_locked(side, entry_price, exit_price, qty, snap)

    async def _open_locked(
        self, side: str, entry_price: float,
        exit_price: float, qty: float, snap: OrderbookSnapshot,
    ) -> bool:
        st  = self._state
        cfg = self._cfg
        now = time.time()

        # Повторная проверка под локом — avoid double-open
        if st.phase != Phase.IDLE:
            return False

        trade = TradeRecord(
            symbol=self._symbol, side=side,
            entry_price=entry_price, exit_price=exit_price,
            qty=qty,
            spread_entry=snap.spread_abs,
            imbalance=snap.imbalance,
            bid_depth=snap.bid_depth_usdt(0.3),
            ask_depth=snap.ask_depth_usdt(0.3),
            latency_ms=(now - snap.ts_local) * 1000,
            ts_open=now,
        )

        if cfg.dry_run:
            logger.info(
                f"[DRY_RUN] {self._symbol}: WOULD ENTER {side} @ {entry_price:.8f} "
                f"exit={exit_price:.8f} qty={qty:.4f}"
            )
            trade.finalize(exit_price, 0.0, ExecStatus.DRY_RUN, "dry_run")
            return False   # не считается настоящим открытием

        # Устанавливаем trade и phase атомарно — phase=ENTRY сразу
        st.trade = trade
        st.side  = side
        st.phase = Phase.ENTRY      # ← выставляем ДО любого await
        st.entry_placed_at = now

        if cfg.paper:
            st.paper_entry = PaperOrder(side, entry_price, qty)
            return True

        # Live — REST после установки phase, чтобы is_idle() уже видел ENTRY
        oid = await self._place_post_only(side, entry_price, qty)
        if not oid:
            st.reset()
            return False
        st.entry_order_id = oid
        return True

    # ── Обработка тика ────────────────────────────────────────────────────────

    async def on_tick(self, snap: OrderbookSnapshot) -> Optional[TradeRecord]:
        """
        Обновляет стейт-машину на каждом тике.
        Возвращает TradeRecord если сделка завершена (для CSV-логирования).
        Защищён локом — безопасен при параллельных вызовах из ensure_future.
        """
        if self._state.phase == Phase.IDLE:
            return None   # быстрый путь без лока
        async with self._lock:
            return await self._on_tick_locked(snap)

    async def _on_tick_locked(self, snap: OrderbookSnapshot) -> Optional[TradeRecord]:
        st = self._state
        if st.phase == Phase.IDLE:
            return None

        if st.phase == Phase.ENTRY:
            return await self._tick_entry(snap)
        if st.phase == Phase.POSITION:
            return await self._tick_place_exit(snap)
        if st.phase == Phase.EXIT:
            return await self._tick_exit(snap)
        if st.phase == Phase.EMERGENCY:
            return await self._tick_emergency(snap)
        return None

    # ── Фаза ENTRY ────────────────────────────────────────────────────────────

    async def _tick_entry(self, snap: OrderbookSnapshot) -> Optional[TradeRecord]:
        st      = self._state
        cfg     = self._cfg
        now     = time.time()
        elapsed = now - st.entry_placed_at
        filled  = False

        if cfg.paper:
            if st.paper_entry and st.paper_entry.try_fill(snap):
                filled = True
                st.trade.entry_price = st.paper_entry.fill_price or st.trade.entry_price
        else:
            if now - st.last_poll >= 0.4:
                st.last_poll = now
                if await self._order_filled(st.entry_order_id):
                    filled = True

        if filled:
            logger.info(
                f"[OB] {self._symbol}: ENTRY FILLED @ {st.trade.entry_price:.8f} "
                f"({st.side}) in {elapsed:.2f}s"
            )
            st.phase = Phase.POSITION
            return None

        # Таймаут
        if elapsed >= cfg.max_order_lifetime_sec:
            logger.info(f"[OB] {self._symbol}: entry CANCELLED (timeout {elapsed:.2f}s)")
            if not cfg.paper and st.entry_order_id:
                await self._cancel_order(st.entry_order_id)
            fees = 0.0
            st.trade.finalize(
                st.trade.entry_price, fees, ExecStatus.CANCELLED, "entry_timeout"
            )
            rec = st.trade
            st.reset()
            return rec
        return None

    # ── Фаза POSITION → выставляем выход ─────────────────────────────────────

    async def _tick_place_exit(self, snap: OrderbookSnapshot) -> Optional[TradeRecord]:
        st      = self._state
        cfg     = self._cfg
        exit_sd = "Sell" if st.side == "Buy" else "Buy"
        now     = time.time()

        # Проверяем adverse move
        if self._is_adverse(snap):
            logger.warning(
                f"[OB] {self._symbol}: adverse move detected → emergency exit"
            )
            st.phase = Phase.EMERGENCY
            return None

        # Захват спреда: выход на текущем рынке, а не на цене из сканера.
        # К моменту исполнения входа цена могла сдвинуться, обновляем.
        # Для BUY: выходим на best_ask (мы лучший продавец внутри спреда).
        # Минимальный выход: entry + 2×maker_fee (хотя бы покрыть комиссии).
        if st.side == "Buy":
            market_exit = snap.best_ask
            min_exit    = st.trade.entry_price * (1 + 2 * cfg.maker_fee_pct / 100)
            exit_price  = max(market_exit, min_exit)
        else:
            market_exit = snap.best_bid
            max_exit    = st.trade.entry_price * (1 - 2 * cfg.maker_fee_pct / 100)
            exit_price  = min(market_exit, max_exit)
        st.trade.exit_price = exit_price

        if cfg.paper:
            st.paper_exit    = PaperOrder(exit_sd, exit_price, st.trade.qty)
            st.phase         = Phase.EXIT
            st.exit_placed_at = now
        else:
            oid = await self._place_post_only(exit_sd, exit_price, st.trade.qty)
            if not oid:
                st.phase = Phase.EMERGENCY
                return None
            st.exit_order_id  = oid
            st.phase          = Phase.EXIT
            st.exit_placed_at = now

        logger.info(
            f"[OB] {self._symbol}: EXIT {exit_sd} placed @ {exit_price:.8f}"
        )
        return None

    # ── Фаза EXIT ─────────────────────────────────────────────────────────────

    async def _tick_exit(self, snap: OrderbookSnapshot) -> Optional[TradeRecord]:
        st      = self._state
        cfg     = self._cfg
        now     = time.time()
        elapsed = now - st.exit_placed_at
        filled  = False

        if cfg.paper:
            if st.paper_exit and st.paper_exit.try_fill(snap):
                filled = True
                st.trade.exit_price = st.paper_exit.fill_price or st.trade.exit_price
        else:
            if now - st.last_poll >= 0.4:
                st.last_poll = now
                if await self._order_filled(st.exit_order_id):
                    filled = True

        if filled:
            return self._close_trade(snap, "exit_filled")

        # Adverse move → emergency
        if self._is_adverse(snap):
            st.phase = Phase.EMERGENCY
            return None

        # Переставляем выход ближе к рынку
        if elapsed >= cfg.max_exit_wait_sec:
            exit_sd   = "Sell" if st.side == "Buy" else "Buy"
            new_price = snap.best_bid if st.side == "Buy" else snap.best_ask
            if not cfg.paper:
                if st.exit_order_id:
                    await self._cancel_order(st.exit_order_id)
                new_oid = await self._place_post_only(exit_sd, new_price, st.trade.qty)
                st.exit_order_id = new_oid
            else:
                st.paper_exit = PaperOrder(exit_sd, new_price, st.trade.qty)
            st.trade.exit_price = new_price
            st.exit_placed_at   = now
            logger.info(f"[OB] {self._symbol}: exit repriced → {new_price:.8f}")

        if elapsed >= cfg.emergency_exit_sec:
            st.phase = Phase.EMERGENCY
        return None

    # ── Фаза EMERGENCY ────────────────────────────────────────────────────────

    async def _tick_emergency(self, snap: OrderbookSnapshot) -> Optional[TradeRecord]:
        st      = self._state
        cfg     = self._cfg
        exit_sd = "Sell" if st.side == "Buy" else "Buy"
        # Агрессивный лимит (не маркет)
        if st.side == "Buy":
            price = round(snap.best_bid * 0.9995, 8)
        else:
            price = round(snap.best_ask * 1.0005, 8)

        logger.warning(f"[OB] {self._symbol}: EMERGENCY {exit_sd} @ {price:.8f}")
        if not cfg.paper:
            if st.exit_order_id:
                await self._cancel_order(st.exit_order_id)
            await self._place_post_only(exit_sd, price, st.trade.qty)

        st.trade.exit_price = price
        return self._close_trade(snap, "emergency_exit")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _close_trade(self, snap: OrderbookSnapshot, reason: str) -> TradeRecord:
        st    = self._state
        trade = st.trade
        fees  = (trade.entry_price + trade.exit_price) * trade.qty * self._cfg.maker_fee_pct / 100

        if reason == "emergency_exit":
            status = ExecStatus.EMERGENCY
        elif ((trade.side == "Buy"  and trade.exit_price > trade.entry_price) or
              (trade.side == "Sell" and trade.exit_price < trade.entry_price)):
            status = ExecStatus.WIN
        else:
            status = ExecStatus.LOSS

        trade.finalize(trade.exit_price, fees, status, reason)
        rec = trade
        st.reset()
        return rec

    def _is_adverse(self, snap: OrderbookSnapshot) -> bool:
        st  = self._state
        cfg = self._cfg
        if not st.trade or st.trade.entry_price <= 0:
            return False
        if st.side == "Buy":
            move_pct = (st.trade.entry_price - snap.best_bid) / st.trade.entry_price * 100
        else:
            move_pct = (snap.best_ask - st.trade.entry_price) / st.trade.entry_price * 100
        return move_pct > cfg.max_adverse_move_pct

    # ── REST API (live mode) ──────────────────────────────────────────────────

    async def _place_post_only(self, side: str, price: float, qty: float) -> Optional[str]:
        if not self._bybit:
            return None
        def _sync():
            return self._bybit.session.place_order(
                category="linear", symbol=self._symbol,
                side=side, orderType="Limit",
                qty=str(qty), price=str(price),
                timeInForce="PostOnly", reduceOnly=False,
            )
        try:
            resp = await asyncio.to_thread(_sync)
            if resp.get("retCode") == 0:
                return resp["result"]["orderId"]
            logger.warning(f"[OB] place_order failed: {resp.get('retMsg')}")
            return None
        except Exception as exc:
            logger.error(f"[OB] place_order error: {exc}")
            return None

    async def _cancel_order(self, order_id: str) -> bool:
        if not self._bybit:
            return False
        def _sync():
            return self._bybit.session.cancel_order(
                category="linear", symbol=self._symbol, orderId=order_id,
            )
        try:
            resp = await asyncio.to_thread(_sync)
            return resp.get("retCode") == 0
        except Exception as exc:
            logger.error(f"[OB] cancel_order error: {exc}")
            return False

    async def _order_filled(self, order_id: Optional[str]) -> bool:
        if not order_id or not self._bybit:
            return False
        def _sync():
            return self._bybit.session.get_open_orders(
                category="linear", symbol=self._symbol,
            )
        try:
            resp = await asyncio.to_thread(_sync)
            if resp.get("retCode") == 0:
                ids = {o["orderId"] for o in resp["result"].get("list", [])}
                return order_id not in ids
            return False
        except Exception as exc:
            logger.error(f"[OB] order_status error: {exc}")
            return False
