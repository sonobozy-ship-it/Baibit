"""
OB_SCALPER — Orderbook Spread Scalper для Bybit.

Стратегия читает стакан через WebSocket и ищет возможности забрать спред
(maker buy + maker sell) с положительным net_expected_usdt.

Принципы:
  • Только POST_ONLY LIMIT ордера (maker).
  • Вход ТОЛЬКО если net_expected_usdt >= min_net_profit_usdt.
  • spread_to_fee_ratio >= spread_to_fee_ratio_min (по умолчанию 3.0).
  • Никаких market-ордеров на вход.
  • Paper mode: CSV-лог, полная симуляция без реальных ордеров.
  • Не является BaseStrategy — живёт вне основного торгового цикла.

Интеграция:
  Запускается как отдельная asyncio-задача.
  Настраивается через env-переменные ENABLE_OB_SCALPER, OB_SCALPER_*.
"""
from __future__ import annotations

import asyncio
import csv
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .orderbook_ws import OrderbookTick, OrderbookWsClient

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. min_spread_abs defaults (USDT)
# ─────────────────────────────────────────────────────────────────────────────

_MIN_SPREAD_ABS: Dict[str, float] = {
    "DOGEUSDT": 0.00003,
    "XRPUSDT":  0.0003,
    "TRXUSDT":  0.00003,
    "ADAUSDT":  0.0002,
    "SOLUSDT":  0.01,
    "BTCUSDT":  1.0,
    "ETHUSDT":  0.2,
}

_DEFAULT_SYMBOLS = list(_MIN_SPREAD_ABS.keys())


# ─────────────────────────────────────────────────────────────────────────────
# 2. Конфигурация
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ObScalperConfig:
    symbols:      List[str] = field(default_factory=lambda: list(_DEFAULT_SYMBOLS))
    paper_mode:   bool      = True
    enable_short: bool      = False

    # Ограничения риска
    max_active_symbols:     int   = 3
    max_position_usdt:      float = 25.0
    risk_per_trade_usdt:    float = 3.0
    daily_loss_limit_usdt:  float = 10.0
    max_consecutive_losses: int   = 3

    # Net edge
    min_net_profit_usdt:     float = 0.02
    min_spread_pct:          float = 0.03
    spread_to_fee_ratio_min: float = 3.0

    # Комиссии и буферы (%)
    maker_fee_pct: float = 0.02
    taker_fee_pct: float = 0.055
    slippage_pct:  float = 0.01
    safety_pct:    float = 0.01

    # Тайминги (секунды)
    cancel_entry_sec:    float = 1.5
    max_exit_wait_sec:   float = 8.0
    emergency_exit_sec:  float = 20.0
    cooldown_loss_sec:   float = 60.0
    cooldown_cancel_sec: float = 5.0

    # Фильтры стакана
    max_price_move_pct: float = 0.05   # макс. движение между тиками
    min_depth_usdt:     float = 10.0   # мин. глубина стакана (USDT)

    # Переопределение min_spread_abs по символу (None → defaults)
    min_spread_abs: Dict[str, float] = field(default_factory=dict)

    def get_min_spread_abs(self, symbol: str) -> float:
        return self.min_spread_abs.get(symbol) or _MIN_SPREAD_ABS.get(symbol, 0.0001)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Trade record
# ─────────────────────────────────────────────────────────────────────────────

class TradeStatus(str, Enum):
    OPEN      = "open"
    WIN       = "win"
    LOSS      = "loss"
    CANCELLED = "cancelled"
    EMERGENCY = "emergency"


@dataclass
class TradeRecord:
    symbol:      str
    side:        str       # "Buy" | "Sell"
    entry_price: float = 0.0
    exit_price:  float = 0.0
    qty:         float = 0.0
    gross_pnl:   float = 0.0
    fees_usdt:   float = 0.0
    net_pnl:     float = 0.0
    spread_entry: float = 0.0
    imbalance:   float = 0.5
    latency_ms:  float = 0.0
    hold_sec:    float = 0.0
    status:      TradeStatus = TradeStatus.OPEN
    exit_reason: str  = ""
    ts_open:     float = field(default_factory=time.time)
    ts_close:    float = 0.0

    def finalize(
        self,
        exit_price: float,
        fees: float,
        status: TradeStatus,
        reason: str,
    ) -> None:
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
# 4. Net edge calculator
# ─────────────────────────────────────────────────────────────────────────────

class NetEdge:
    """
    Все расчёты прибыльности в USDT.

    gross_spread_usdt = |exit - entry| × qty
    fees_usdt         = (entry + exit) × qty × maker_fee_pct / 100
    slippage_usdt     = entry × qty × slippage_pct / 100
    safety_usdt       = entry × qty × safety_pct / 100
    net_expected_usdt = gross - fees - slippage - safety
    """

    @staticmethod
    def fees(entry: float, exit_: float, qty: float, maker_pct: float) -> float:
        return (entry + exit_) * qty * maker_pct / 100

    @staticmethod
    def calc(
        entry: float, exit_: float, qty: float,
        maker_pct: float, slip_pct: float, safe_pct: float,
    ) -> Tuple[float, float, float, float]:
        """Returns (gross_usdt, total_costs_usdt, net_usdt, spread_to_fee_ratio)."""
        if qty <= 0 or entry <= 0:
            return 0.0, 0.0, -999.0, 0.0
        gross = abs(exit_ - entry) * qty
        fees  = (entry + exit_) * qty * maker_pct / 100
        slip  = entry * qty * slip_pct / 100
        safe  = entry * qty * safe_pct / 100
        net   = gross - fees - slip - safe
        ratio = (gross / fees) if fees > 0 else 0.0
        return gross, fees + slip + safe, net, ratio

    @staticmethod
    def calc_exit_price(
        entry: float, qty: float,
        maker_pct: float, slip_pct: float, safe_pct: float,
        min_net_usdt: float, side: str,
    ) -> float:
        """Минимальная цена выхода для покрытия всех расходов + min_net_profit."""
        if qty <= 0 or entry <= 0:
            return entry
        costs = (
            2 * entry * qty * maker_pct / 100   # две стороны комиссии (приближение)
            + entry * qty * slip_pct / 100
            + entry * qty * safe_pct / 100
            + min_net_usdt
        )
        delta = costs / qty
        return (entry + delta) if side == "Buy" else (entry - delta)

    @staticmethod
    def round_qty(qty: float, step: float) -> float:
        if step <= 0:
            return qty
        precision = max(0, -int(math.floor(math.log10(step)))) if 0 < step < 1 else 0
        return round(math.floor(qty / step) * step, precision)

    @staticmethod
    def round_price(price: float, tick_size: float) -> float:
        if tick_size <= 0:
            return price
        precision = max(0, -int(math.floor(math.log10(tick_size)))) if 0 < tick_size < 1 else 0
        return round(round(price / tick_size) * tick_size, precision)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Risk manager
# ─────────────────────────────────────────────────────────────────────────────

class ObRiskManager:
    def __init__(self, cfg: ObScalperConfig):
        self._cfg        = cfg
        self._daily_pnl: float = 0.0
        self._daily_date: str  = ""
        self._cons_loss: Dict[str, int]   = {}
        self._cooldown:  Dict[str, float] = {}
        self._skipped:   Dict[str, Dict[str, int]] = {}

    def _reset_day(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._daily_date:
            self._daily_date = today
            self._daily_pnl  = 0.0

    def can_trade(self, symbol: str) -> Tuple[bool, str]:
        self._reset_day()
        if self._daily_pnl <= -self._cfg.daily_loss_limit_usdt:
            return False, "daily_loss_limit"
        now = time.time()
        cd  = self._cooldown.get(symbol, 0)
        if now < cd:
            return False, f"cooldown_{cd - now:.0f}s"
        if self._cons_loss.get(symbol, 0) >= self._cfg.max_consecutive_losses:
            return False, f"cons_losses_{self._cons_loss[symbol]}"
        return True, "ok"

    def record_skip(self, symbol: str, reason: str) -> None:
        self._skipped.setdefault(symbol, {})
        self._skipped[symbol][reason] = self._skipped[symbol].get(reason, 0) + 1

    def register(self, symbol: str, net_pnl: float, cancelled: bool) -> None:
        self._reset_day()
        if cancelled:
            self._cooldown[symbol] = time.time() + self._cfg.cooldown_cancel_sec
            return
        self._daily_pnl += min(0.0, net_pnl)
        if net_pnl < 0:
            self._cons_loss[symbol] = self._cons_loss.get(symbol, 0) + 1
            self._cooldown[symbol]  = time.time() + self._cfg.cooldown_loss_sec
        else:
            self._cons_loss[symbol] = 0

    @property
    def daily_pnl(self) -> float:
        self._reset_day()
        return self._daily_pnl

    def skipped_summary(self) -> Dict[str, Dict[str, int]]:
        return dict(self._skipped)


# ─────────────────────────────────────────────────────────────────────────────
# 6. CSV logger
# ─────────────────────────────────────────────────────────────────────────────

_CSV_FIELDS = [
    "timestamp_open", "timestamp_close", "symbol", "side",
    "entry_price", "exit_price", "qty",
    "gross_pnl", "fees", "net_pnl",
    "hold_time_sec", "spread_entry", "imbalance_entry",
    "latency_ms", "status", "exit_reason",
]


class ObCsvLogger:
    def __init__(self, path: str = "logs/obs_orderbook_trades.csv"):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists() or self._path.stat().st_size == 0:
            with self._path.open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=_CSV_FIELDS).writeheader()

    def log(self, rec: TradeRecord) -> None:
        row = {
            "timestamp_open":  datetime.fromtimestamp(rec.ts_open,  tz=timezone.utc).isoformat(),
            "timestamp_close": datetime.fromtimestamp(rec.ts_close, tz=timezone.utc).isoformat(),
            "symbol":          rec.symbol,
            "side":            rec.side,
            "entry_price":     round(rec.entry_price,  8),
            "exit_price":      round(rec.exit_price,   8),
            "qty":             round(rec.qty,           8),
            "gross_pnl":       round(rec.gross_pnl,    6),
            "fees":            round(rec.fees_usdt,     6),
            "net_pnl":         round(rec.net_pnl,       6),
            "hold_time_sec":   round(rec.hold_sec,      2),
            "spread_entry":    round(rec.spread_entry,  8),
            "imbalance_entry": round(rec.imbalance,     4),
            "latency_ms":      round(rec.latency_ms,    1),
            "status":          rec.status.value,
            "exit_reason":     rec.exit_reason,
        }
        with self._path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=_CSV_FIELDS).writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Paper order simulator
# ─────────────────────────────────────────────────────────────────────────────

class PaperOrder:
    """
    Симуляция POST_ONLY LIMIT ордера.
    Maker BUY fills when best_ask <= price (seller hits our bid).
    Maker SELL fills when best_bid >= price (buyer hits our ask).
    """

    def __init__(self, side: str, price: float, qty: float):
        self.side        = side
        self.price       = price
        self.qty         = qty
        self.filled      = False
        self.fill_price: Optional[float] = None
        self.placed_at   = time.time()

    def try_fill(self, tick: OrderbookTick) -> bool:
        if self.filled:
            return True
        if self.side == "Buy"  and tick.best_ask <= self.price:
            self.filled     = True
            self.fill_price = tick.best_ask
        elif self.side == "Sell" and tick.best_bid >= self.price:
            self.filled     = True
            self.fill_price = tick.best_bid
        return self.filled


# ─────────────────────────────────────────────────────────────────────────────
# 8. Per-symbol state machine
# ─────────────────────────────────────────────────────────────────────────────

class Phase(str, Enum):
    IDLE      = "idle"
    ENTRY     = "entry"       # ордер входа выставлен
    POSITION  = "position"    # вход исполнен, выставляем выход
    EXIT      = "exit"        # ордер выхода выставлен
    EMERGENCY = "emergency"   # принудительный выход


class SymbolState:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.phase:           Phase = Phase.IDLE
        self.trade:           Optional[TradeRecord] = None
        self.side:            Optional[str] = None
        # Live mode order IDs
        self.entry_order_id:  Optional[str] = None
        self.exit_order_id:   Optional[str] = None
        self._last_poll:      float = 0.0   # REST poll rate-limit
        # Paper mode orders
        self.paper_entry:     Optional[PaperOrder] = None
        self.paper_exit:      Optional[PaperOrder] = None
        # Phase timestamps
        self.entry_placed_at: float = 0.0
        self.exit_placed_at:  float = 0.0
        # Impulse filter
        self.last_tick:       Optional[OrderbookTick] = None

    def reset(self) -> None:
        self.phase           = Phase.IDLE
        self.trade           = None
        self.side            = None
        self.entry_order_id  = None
        self.exit_order_id   = None
        self.paper_entry     = None
        self.paper_exit      = None
        self.entry_placed_at = 0.0
        self.exit_placed_at  = 0.0
        self._last_poll      = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 9. Главная стратегия
# ─────────────────────────────────────────────────────────────────────────────

NotifyFn = Callable[[str], Any]  # может быть sync или async


class OrderbookSpreadScalper:
    """
    OB_SCALPER — Orderbook Spread Scalper для Bybit.

    Запускается как отдельная asyncio-задача (asyncio.create_task).
    Не является BaseStrategy — не участвует в основном торговом цикле.
    Читает стакан через WebSocket и торгует исключительно maker-ордерами.
    """

    ID   = "OB_SCALPER"
    NAME = "ORDERBOOK SPREAD SCALPER"

    def __init__(
        self,
        cfg:         ObScalperConfig,
        bybit_client = None,           # BybitClient (bybit_client.py)
        notify_fn:   Optional[NotifyFn] = None,
    ):
        self._cfg    = cfg
        self._bybit  = bybit_client
        self._notify = notify_fn

        self._states: Dict[str, SymbolState] = {s: SymbolState(s) for s in cfg.symbols}
        self._risk   = ObRiskManager(cfg)
        self._csv    = ObCsvLogger()
        self._running = False
        self._ws_task: Optional[asyncio.Task] = None

        # Кеш instrument_info (qty_step, tick_size, min_qty)
        self._inst_info: Dict[str, Dict] = {}

        # Статистика по символам
        self._stats: Dict[str, Dict] = {
            s: {"total": 0, "wins": 0, "losses": 0, "cancels": 0,
                "net_pnl": 0.0, "skipped": 0}
            for s in cfg.symbols
        }

    # ── Старт / остановка ─────────────────────────────────────────────────────

    async def start(self) -> None:
        self._running = True
        mode = "PAPER" if self._cfg.paper_mode else "LIVE"
        logger.info(f"[OB_SCALPER] starting [{mode}] symbols={self._cfg.symbols}")

        # Загружаем instrument info для корректного округления qty/price
        if self._bybit:
            for sym in self._cfg.symbols:
                try:
                    info = await asyncio.to_thread(self._bybit.get_instrument_info, sym)
                    self._inst_info[sym] = info
                    logger.debug(f"[OB_SCALPER] {sym} info: {info}")
                except Exception as exc:
                    logger.warning(f"[OB_SCALPER] instrument_info {sym}: {exc}")

        ws = OrderbookWsClient(
            symbols=self._cfg.symbols,
            on_tick=self._on_tick,
        )
        self._ws_task = asyncio.create_task(ws.start())

        await self._notify_msg(
            f"🚀 OB_SCALPER STARTED [{mode}]\n"
            f"Symbols: {', '.join(self._cfg.symbols)}\n"
            f"Max position: {self._cfg.max_position_usdt} USDT | "
            f"Min net profit: {self._cfg.min_net_profit_usdt} USDT | "
            f"Daily loss limit: {self._cfg.daily_loss_limit_usdt} USDT"
        )

    async def stop(self) -> None:
        self._running = False
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
        logger.info("[OB_SCALPER] stopped")

    # ── WS callback ───────────────────────────────────────────────────────────

    async def _on_tick(self, tick: OrderbookTick) -> None:
        if not self._running:
            return
        st = self._states.get(tick.symbol)
        if st is None:
            return
        try:
            await self._process(st, tick)
        except Exception as exc:
            logger.exception(f"[OB_SCALPER] {tick.symbol}: tick error: {exc}")
        finally:
            st.last_tick = tick

    # ── State machine dispatcher ───────────────────────────────────────────────

    async def _process(self, st: SymbolState, tick: OrderbookTick) -> None:
        if st.phase == Phase.IDLE:
            await self._try_entry(st, tick)
        elif st.phase == Phase.ENTRY:
            await self._check_entry_fill(st, tick)
        elif st.phase == Phase.POSITION:
            await self._place_exit_order(st, tick)
        elif st.phase == Phase.EXIT:
            await self._check_exit_fill(st, tick)
        elif st.phase == Phase.EMERGENCY:
            await self._do_emergency_exit(st, tick)

    # ── Фаза IDLE: поиск входа ────────────────────────────────────────────────

    async def _try_entry(self, st: SymbolState, tick: OrderbookTick) -> None:
        sym = tick.symbol
        cfg = self._cfg

        # Риск-проверка
        ok, reason = self._risk.can_trade(sym)
        if not ok:
            return

        # Лимит активных позиций
        active = sum(1 for s in self._states.values() if s.phase != Phase.IDLE)
        if active >= cfg.max_active_symbols:
            return

        # Импульс-фильтр: пропускаем при резком движении цены
        if st.last_tick and st.last_tick.mid_price > 0:
            move_pct = (
                abs(tick.mid_price - st.last_tick.mid_price)
                / st.last_tick.mid_price * 100
            )
            if move_pct > cfg.max_price_move_pct:
                self._stats[sym]["skipped"] += 1
                self._risk.record_skip(sym, "impulse")
                return

        # Фильтры спреда
        skip_reason = self._check_spread(sym, tick)
        if skip_reason:
            self._stats[sym]["skipped"] += 1
            self._risk.record_skip(sym, skip_reason)
            return

        # Определяем направление
        side = "Buy"
        if cfg.enable_short and tick.imbalance < 0.3:
            side = "Sell"

        # Цена входа
        entry = tick.best_bid if side == "Buy" else tick.best_ask

        # Qty с учётом min_qty и qty_step
        info  = self._inst_info.get(sym, {})
        step  = info.get("qty_step", 0.001)
        min_q = info.get("min_qty",  0.001)
        raw_qty = cfg.max_position_usdt / entry if entry > 0 else 0.0
        qty = NetEdge.round_qty(raw_qty, step)
        if qty < min_q:
            qty = min_q

        # Минимальная цена выхода для покрытия всех расходов
        exit_price = NetEdge.calc_exit_price(
            entry=entry, qty=qty,
            maker_pct=cfg.maker_fee_pct,
            slip_pct=cfg.slippage_pct,
            safe_pct=cfg.safety_pct,
            min_net_usdt=cfg.min_net_profit_usdt,
            side=side,
        )

        # Проверяем net edge
        gross, total_costs, net, ratio = NetEdge.calc(
            entry=entry, exit_=exit_price, qty=qty,
            maker_pct=cfg.maker_fee_pct,
            slip_pct=cfg.slippage_pct,
            safe_pct=cfg.safety_pct,
        )

        if net < cfg.min_net_profit_usdt:
            self._stats[sym]["skipped"] += 1
            self._risk.record_skip(sym, "net_edge_negative")
            logger.debug(
                f"[OB_SCALPER] {sym}: skip net={net:.5f} < {cfg.min_net_profit_usdt} USDT"
            )
            return

        if ratio < cfg.spread_to_fee_ratio_min:
            self._stats[sym]["skipped"] += 1
            self._risk.record_skip(sym, "spread_to_fee_ratio")
            return

        # Все фильтры прошли — выставляем entry
        now      = time.time()
        latency  = (now - tick.ts_local) * 1000
        tick_sz  = info.get("tick_size", 0.0001)
        r_entry  = NetEdge.round_price(entry, tick_sz)
        r_exit   = NetEdge.round_price(exit_price, tick_sz)

        trade = TradeRecord(
            symbol=sym, side=side,
            entry_price=r_entry, exit_price=r_exit,
            qty=qty, spread_entry=tick.spread_abs,
            imbalance=tick.imbalance, latency_ms=latency,
            ts_open=now,
        )
        st.trade = trade
        st.side  = side

        logger.info(
            f"[OB_SCALPER] {sym}: ENTRY {side} @ {r_entry:.8f} "
            f"exit={r_exit:.8f} qty={qty} net_exp={net:.5f} USDT "
            f"ratio={ratio:.1f} spread={tick.spread_pct:.3f}%"
        )

        if cfg.paper_mode:
            st.paper_entry    = PaperOrder(side, r_entry, qty)
            st.phase          = Phase.ENTRY
            st.entry_placed_at = now
            await self._notify_msg(
                f"📋 OB_SCALPER PAPER ENTRY\n"
                f"Symbol: {sym} | {side}\n"
                f"Bid/Ask: {tick.best_bid:.8f} / {tick.best_ask:.8f}\n"
                f"Spread: {tick.spread_abs:.8f} ({tick.spread_pct:.3f}%)\n"
                f"Imbalance: {tick.imbalance:.3f}\n"
                f"Entry: {r_entry:.8f} | Exit: {r_exit:.8f}\n"
                f"Qty: {qty} | Gross: {gross:.5f} USDT | Net expected: {net:.5f} USDT"
            )
        else:
            oid = await self._place_post_only(sym, side, r_entry, qty)
            if not oid:
                st.reset()
                return
            st.entry_order_id  = oid
            st.phase           = Phase.ENTRY
            st.entry_placed_at = now
            await self._notify_msg(
                f"📤 OB_SCALPER ENTRY ORDER\n"
                f"Symbol: {sym} | {side}\n"
                f"Bid/Ask: {tick.best_bid:.8f} / {tick.best_ask:.8f}\n"
                f"Entry: {r_entry:.8f} | Exit: {r_exit:.8f}\n"
                f"Net expected: {net:.5f} USDT | orderId: {oid}"
            )

    # ── Фаза ENTRY: ждём исполнения ───────────────────────────────────────────

    async def _check_entry_fill(self, st: SymbolState, tick: OrderbookTick) -> None:
        now     = time.time()
        elapsed = now - st.entry_placed_at
        sym     = tick.symbol
        filled  = False

        if self._cfg.paper_mode:
            if st.paper_entry and st.paper_entry.try_fill(tick):
                filled = True
                st.trade.entry_price = st.paper_entry.fill_price or st.trade.entry_price
        else:
            # Опрашиваем REST не чаще 2 раз в секунду
            if now - st._last_poll >= 0.5:
                st._last_poll = now
                if await self._order_filled(sym, st.entry_order_id):
                    filled = True

        if filled:
            st.phase = Phase.POSITION
            logger.info(f"[OB_SCALPER] {sym}: entry FILLED @ {st.trade.entry_price:.8f}")
            await self._notify_msg(
                f"✅ OB_SCALPER ENTRY FILLED\n"
                f"Symbol: {sym} | {st.side}\n"
                f"Entry price: {st.trade.entry_price:.8f}"
            )
            return

        # Таймаут — отменяем ордер входа
        if elapsed >= self._cfg.cancel_entry_sec:
            if not self._cfg.paper_mode and st.entry_order_id:
                await self._cancel_order(sym, st.entry_order_id)
            logger.info(
                f"[OB_SCALPER] {sym}: entry CANCELLED (timeout {elapsed:.2f}s)"
            )
            await self._notify_msg(
                f"❌ OB_SCALPER ENTRY CANCELLED\n"
                f"Symbol: {sym} | reason: timeout {elapsed:.1f}s"
            )
            st.trade.finalize(
                st.trade.entry_price, 0.0, TradeStatus.CANCELLED, "entry_timeout"
            )
            self._csv.log(st.trade)
            self._risk.register(sym, 0.0, cancelled=True)
            self._stats[sym]["cancels"] += 1
            st.reset()

    # ── Фаза POSITION: выставляем выход ───────────────────────────────────────

    async def _place_exit_order(self, st: SymbolState, tick: OrderbookTick) -> None:
        sym     = tick.symbol
        cfg     = self._cfg
        exit_sd = "Sell" if st.side == "Buy" else "Buy"

        # Пересчитываем exit по фактической цене входа
        info    = self._inst_info.get(sym, {})
        tick_sz = info.get("tick_size", 0.0001)
        raw_exit = NetEdge.calc_exit_price(
            entry=st.trade.entry_price, qty=st.trade.qty,
            maker_pct=cfg.maker_fee_pct,
            slip_pct=cfg.slippage_pct,
            safe_pct=cfg.safety_pct,
            min_net_usdt=cfg.min_net_profit_usdt,
            side=st.side,
        )
        exit_price = NetEdge.round_price(raw_exit, tick_sz)
        st.trade.exit_price = exit_price

        now = time.time()

        if cfg.paper_mode:
            st.paper_exit     = PaperOrder(exit_sd, exit_price, st.trade.qty)
            st.phase          = Phase.EXIT
            st.exit_placed_at = now
        else:
            oid = await self._place_post_only(sym, exit_sd, exit_price, st.trade.qty)
            if not oid:
                st.phase = Phase.EMERGENCY
                return
            st.exit_order_id  = oid
            st.phase          = Phase.EXIT
            st.exit_placed_at = now

        logger.info(
            f"[OB_SCALPER] {sym}: EXIT {exit_sd} placed @ {exit_price:.8f}"
        )
        await self._notify_msg(
            f"📤 OB_SCALPER EXIT ORDER\n"
            f"Symbol: {sym} | {exit_sd}\n"
            f"Exit: {exit_price:.8f}"
        )

    # ── Фаза EXIT: ждём исполнения ────────────────────────────────────────────

    async def _check_exit_fill(self, st: SymbolState, tick: OrderbookTick) -> None:
        now     = time.time()
        elapsed = now - st.exit_placed_at
        sym     = tick.symbol
        filled  = False
        cfg     = self._cfg

        if cfg.paper_mode:
            if st.paper_exit and st.paper_exit.try_fill(tick):
                filled = True
                st.trade.exit_price = st.paper_exit.fill_price or st.trade.exit_price
        else:
            if now - st._last_poll >= 0.5:
                st._last_poll = now
                if await self._order_filled(sym, st.exit_order_id):
                    filled = True

        if filled:
            await self._close_trade(st, tick, "exit_filled")
            return

        # Переставляем exit ближе к рынку если завис
        if elapsed >= cfg.max_exit_wait_sec:
            exit_sd  = "Sell" if st.side == "Buy" else "Buy"
            new_exit = tick.best_bid if st.side == "Buy" else tick.best_ask
            info     = self._inst_info.get(sym, {})
            tick_sz  = info.get("tick_size", 0.0001)
            new_exit = NetEdge.round_price(new_exit, tick_sz)

            if not cfg.paper_mode:
                if st.exit_order_id:
                    await self._cancel_order(sym, st.exit_order_id)
                new_oid = await self._place_post_only(sym, exit_sd, new_exit, st.trade.qty)
                st.exit_order_id = new_oid
            else:
                st.paper_exit = PaperOrder(exit_sd, new_exit, st.trade.qty)

            st.exit_placed_at   = now
            st.trade.exit_price = new_exit
            logger.info(
                f"[OB_SCALPER] {sym}: exit repriced → {new_exit:.8f} (elapsed={elapsed:.1f}s)"
            )

        # Принудительный выход если emergency_exit_sec истёк
        if elapsed >= cfg.emergency_exit_sec:
            st.phase = Phase.EMERGENCY

    # ── Фаза EMERGENCY: агрессивный выход ────────────────────────────────────

    async def _do_emergency_exit(self, st: SymbolState, tick: OrderbookTick) -> None:
        sym     = tick.symbol
        exit_sd = "Sell" if st.side == "Buy" else "Buy"
        # Чуть агрессивнее best bid/ask (не market order)
        if st.side == "Buy":
            price = NetEdge.round_price(
                tick.best_bid * 0.9995, self._inst_info.get(sym, {}).get("tick_size", 0.0001)
            )
        else:
            price = NetEdge.round_price(
                tick.best_ask * 1.0005, self._inst_info.get(sym, {}).get("tick_size", 0.0001)
            )

        logger.warning(
            f"[OB_SCALPER] {sym}: EMERGENCY EXIT {exit_sd} @ {price:.8f}"
        )

        if not self._cfg.paper_mode:
            if st.exit_order_id:
                await self._cancel_order(sym, st.exit_order_id)
            await self._place_post_only(sym, exit_sd, price, st.trade.qty)

        st.trade.exit_price = price
        await self._close_trade(st, tick, "emergency_exit")

    # ── Закрытие сделки ───────────────────────────────────────────────────────

    async def _close_trade(
        self, st: SymbolState, tick: OrderbookTick, reason: str
    ) -> None:
        sym   = tick.symbol
        trade = st.trade
        fees  = NetEdge.fees(
            trade.entry_price, trade.exit_price, trade.qty, self._cfg.maker_fee_pct
        )

        if reason == "emergency_exit":
            status = TradeStatus.EMERGENCY
        elif (trade.side == "Buy"  and trade.exit_price > trade.entry_price) or \
             (trade.side == "Sell" and trade.exit_price < trade.entry_price):
            status = TradeStatus.WIN
        else:
            status = TradeStatus.LOSS

        trade.finalize(trade.exit_price, fees, status, reason)
        self._csv.log(trade)
        self._risk.register(sym, trade.net_pnl, cancelled=False)

        s = self._stats[sym]
        s["total"] += 1
        if trade.net_pnl > 0:
            s["wins"]   += 1
        else:
            s["losses"] += 1
        s["net_pnl"] = round(s["net_pnl"] + trade.net_pnl, 6)

        emoji = "✅" if trade.net_pnl > 0 else ("⚡" if reason == "emergency_exit" else "❌")
        await self._notify_msg(
            f"{emoji} OB_SCALPER CLOSED\n"
            f"Symbol: {sym} | {trade.side}\n"
            f"Entry: {trade.entry_price:.8f} → Exit: {trade.exit_price:.8f}\n"
            f"Gross: {trade.gross_pnl:+.5f} | Fees: {fees:.5f} | "
            f"Net: {trade.net_pnl:+.5f} USDT\n"
            f"Hold: {trade.hold_sec:.1f}s | Reason: {reason} | "
            f"Spread: {trade.spread_entry:.8f}"
        )

        if self._risk.daily_pnl <= -self._cfg.daily_loss_limit_usdt:
            await self._notify_msg(
                f"🚨 OB_SCALPER DAILY LOSS LIMIT REACHED\n"
                f"Daily PnL: {self._risk.daily_pnl:.4f} USDT ≤ "
                f"-{self._cfg.daily_loss_limit_usdt} USDT\n"
                f"Trading paused until tomorrow (UTC midnight)."
            )

        logger.info(
            f"[OB_SCALPER] {sym}: CLOSED net={trade.net_pnl:+.5f} USDT "
            f"[{status.value}] reason={reason}"
        )
        st.reset()

    # ── Фильтр спреда ─────────────────────────────────────────────────────────

    def _check_spread(self, symbol: str, tick: OrderbookTick) -> Optional[str]:
        """Возвращает причину пропуска или None если спред OK."""
        cfg = self._cfg

        if tick.spread_abs < cfg.get_min_spread_abs(symbol):
            return "spread_abs_too_small"
        if tick.spread_pct < cfg.min_spread_pct:
            return "spread_pct_too_small"
        if tick.bid_depth_usdt < cfg.min_depth_usdt:
            return "bid_depth_insufficient"
        if tick.ask_depth_usdt < cfg.min_depth_usdt:
            return "ask_depth_insufficient"

        # Быстрая проверка spread_to_fee_ratio по текущей mid-цене
        fee_rt = tick.mid_price * (cfg.maker_fee_pct / 100) * 2
        if fee_rt > 0 and (tick.spread_abs / fee_rt) < cfg.spread_to_fee_ratio_min:
            return "spread_to_fee_ratio"

        return None

    # ── Управление ордерами (live mode) ──────────────────────────────────────

    async def _place_post_only(
        self, symbol: str, side: str, price: float, qty: float
    ) -> Optional[str]:
        if not self._bybit:
            return None

        def _sync():
            return self._bybit.session.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Limit",
                qty=str(qty),
                price=str(price),
                timeInForce="PostOnly",
                reduceOnly=False,
            )

        try:
            resp = await asyncio.to_thread(_sync)
            if resp.get("retCode") == 0:
                return resp["result"]["orderId"]
            logger.warning(
                f"[OB_SCALPER] place_order failed: {resp.get('retMsg')} "
                f"({symbol} {side} {price} {qty})"
            )
            return None
        except Exception as exc:
            logger.error(f"[OB_SCALPER] place_order error: {exc}")
            return None

    async def _cancel_order(self, symbol: str, order_id: str) -> bool:
        if not self._bybit:
            return False

        def _sync():
            return self._bybit.session.cancel_order(
                category="linear",
                symbol=symbol,
                orderId=order_id,
            )

        try:
            resp = await asyncio.to_thread(_sync)
            ok = resp.get("retCode") == 0
            if not ok:
                logger.warning(
                    f"[OB_SCALPER] cancel_order failed: {resp.get('retMsg')} "
                    f"({symbol} {order_id})"
                )
            return ok
        except Exception as exc:
            logger.error(f"[OB_SCALPER] cancel_order error: {exc}")
            return False

    async def _order_filled(self, symbol: str, order_id: Optional[str]) -> bool:
        """
        True если ордер НЕ в открытых (исполнен или отменён биржей).
        Используется как индикатор исполнения.
        """
        if not order_id or not self._bybit:
            return False

        def _sync():
            return self._bybit.session.get_open_orders(
                category="linear",
                symbol=symbol,
            )

        try:
            resp = await asyncio.to_thread(_sync)
            if resp.get("retCode") == 0:
                open_ids = {o["orderId"] for o in resp["result"].get("list", [])}
                return order_id not in open_ids
            return False
        except Exception as exc:
            logger.error(f"[OB_SCALPER] order_status error: {exc}")
            return False

    # ── Telegram уведомления ──────────────────────────────────────────────────

    async def _notify_msg(self, msg: str) -> None:
        if not self._notify:
            return
        try:
            result = self._notify(msg)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            pass

    # ── Метрики и статус ──────────────────────────────────────────────────────

    def metrics(self) -> dict:
        total = sum(s["total"]   for s in self._stats.values())
        wins  = sum(s["wins"]    for s in self._stats.values())
        net   = sum(s["net_pnl"] for s in self._stats.values())
        skips = sum(s["skipped"] for s in self._stats.values())
        cxl   = sum(s["cancels"] for s in self._stats.values())
        wr    = wins / total if total > 0 else 0.0
        return {
            "total_trades":       total,
            "winrate":            round(wr, 3),
            "total_net_pnl":      round(net, 5),
            "skipped_count":      skips,
            "cancel_count":       cxl,
            "daily_pnl":          round(self._risk.daily_pnl, 5),
            "skipped_by_reason":  self._risk.skipped_summary(),
            "by_symbol": {
                s: {
                    "total":   v["total"],
                    "wr":      round(v["wins"] / v["total"], 3) if v["total"] > 0 else 0.0,
                    "net_pnl": round(v["net_pnl"], 5),
                    "phase":   self._states[s].phase.value,
                }
                for s, v in self._stats.items()
            },
        }

    def status_text(self) -> str:
        m    = self.metrics()
        mode = "📄 PAPER" if self._cfg.paper_mode else "🔴 LIVE"
        lines = [
            f"📊 OB_SCALPER [{mode}]",
            f"Symbols: {', '.join(self._cfg.symbols)}",
            f"Trades: {m['total_trades']} | WR: {m['winrate']:.1%} | "
            f"Net: {m['total_net_pnl']:+.5f} USDT",
            f"Daily PnL: {m['daily_pnl']:+.5f} USDT | "
            f"Skipped: {m['skipped_count']} | Cancelled: {m['cancel_count']}",
            "",
            "По символам:",
        ]
        for sym, s in m["by_symbol"].items():
            lines.append(
                f"  {sym}: {s['total']} trades | WR={s['wr']:.1%} | "
                f"Net={s['net_pnl']:+.5f} | [{s['phase']}]"
            )

        # Top skip reasons
        all_skips: Dict[str, int] = {}
        for by_sym in m["skipped_by_reason"].values():
            for r, c in by_sym.items():
                all_skips[r] = all_skips.get(r, 0) + c
        if all_skips:
            lines.append("")
            lines.append("Причины пропуска (top):")
            for r, c in sorted(all_skips.items(), key=lambda x: -x[1])[:5]:
                lines.append(f"  {r}: {c}")

        return "\n".join(lines)
