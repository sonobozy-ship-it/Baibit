"""
OrderbookEngine — главный координатор ORDERBOOK_ONLY режима.

Связывает:
  OrderbookWsManager → PairSelector → SpreadScanner →
  LiquidityFilter → OrderExecutor × N → RiskGuard → CSV-лог → Telegram

Режимы:
  dry_run=True  → только логирует решения, никаких ордеров / симуляций
  paper=True    → симулирует исполнение, пишет CSV (нет реальных ордеров)
  live          → реальные POST_ONLY LIMIT-ордера через pybit
"""
from __future__ import annotations

import asyncio
import csv
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Coroutine, Dict, List, Optional

from .bybit_ws import OrderbookSnapshot, OrderbookWsManager
from .liquidity_filter import LiquidityConfig, LiquidityFilter
from .order_executor import (
    ExecutorConfig,
    ExecStatus,
    OrderExecutor,
    TradeRecord,
)
from .pair_selector import PairSelector, PairSelectorConfig
from .risk_guard import RiskConfig, RiskGuard
from .spread_scanner import ScanConfig, SpreadScanner

logger = logging.getLogger(__name__)

NotifyFn = Callable[[str], Coroutine]


# ─────────────────────────────────────────────────────────────────────────────
# Конфиг движка
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ObEngineConfig:
    # Режим
    dry_run: bool = False
    paper:   bool = True

    # WebSocket
    ws_symbols: List[str] = field(default_factory=lambda: [
        "DOGEUSDT", "XRPUSDT", "TRXUSDT", "ADAUSDT",
        "SOLUSDT", "BTCUSDT", "ETHUSDT",
    ])

    # Scan loop
    scan_interval_sec:  float = 1.0    # пауза между проходами сканера
    notify_every_n_dry: int   = 100    # раз в N dry-run решений → Telegram

    # CSV
    csv_path: str = "logs/orderbook_engine_trades.csv"

    # Sub-configs (переопределяемые через env)
    pair_cfg:      PairSelectorConfig = field(default_factory=PairSelectorConfig)
    scan_cfg:      ScanConfig         = field(default_factory=ScanConfig)
    liquidity_cfg: LiquidityConfig    = field(default_factory=LiquidityConfig)
    risk_cfg:      RiskConfig         = field(default_factory=RiskConfig)
    exec_cfg:      ExecutorConfig     = field(default_factory=ExecutorConfig)


# ─────────────────────────────────────────────────────────────────────────────
# CSV-логгер
# ─────────────────────────────────────────────────────────────────────────────

_CSV_FIELDS = [
    "ts_open", "ts_close", "symbol", "side",
    "entry_price", "exit_price", "qty",
    "gross_pnl", "fees_usdt", "net_pnl",
    "spread_entry", "imbalance", "bid_depth", "ask_depth",
    "latency_ms", "hold_sec", "status", "exit_reason",
]


class ObCsvLogger:
    def __init__(self, path: str):
        self._path = path
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=_CSV_FIELDS).writeheader()

    def write(self, rec: TradeRecord) -> None:
        row = {
            "ts_open":     datetime.fromtimestamp(rec.ts_open, tz=timezone.utc).isoformat(),
            "ts_close":    datetime.fromtimestamp(rec.ts_close, tz=timezone.utc).isoformat() if rec.ts_close else "",
            "symbol":      rec.symbol,
            "side":        rec.side,
            "entry_price": rec.entry_price,
            "exit_price":  rec.exit_price,
            "qty":         rec.qty,
            "gross_pnl":   round(rec.gross_pnl, 6),
            "fees_usdt":   round(rec.fees_usdt, 6),
            "net_pnl":     round(rec.net_pnl, 6),
            "spread_entry": round(rec.spread_entry, 8),
            "imbalance":   round(rec.imbalance, 4),
            "bid_depth":   round(rec.bid_depth, 2),
            "ask_depth":   round(rec.ask_depth, 2),
            "latency_ms":  round(rec.latency_ms, 2),
            "hold_sec":    round(rec.hold_sec, 3),
            "status":      rec.status.value,
            "exit_reason": rec.exit_reason,
        }
        with open(self._path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=_CSV_FIELDS).writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# Движок
# ─────────────────────────────────────────────────────────────────────────────

class OrderbookEngine:
    """
    Главный координатор ORDERBOOK_ONLY режима.
    Запускается одним asyncio-таском: await engine.start()
    """

    def __init__(
        self,
        cfg:          ObEngineConfig,
        bybit_client  = None,
        notify_fn:    Optional[NotifyFn] = None,
    ):
        self._cfg       = cfg
        self._bybit     = bybit_client
        self._notify_fn = notify_fn

        # Пометка режима в exec_cfg
        cfg.exec_cfg.dry_run = cfg.dry_run
        cfg.exec_cfg.paper   = cfg.paper and not cfg.dry_run

        self._pair_sel  = PairSelector(cfg.pair_cfg)
        self._scanner   = SpreadScanner()
        self._liq_filter = LiquidityFilter(cfg.ws_symbols)
        self._risk      = RiskGuard(cfg.risk_cfg)
        self._csv       = ObCsvLogger(cfg.csv_path)

        self._ws_mgr: Optional[OrderbookWsManager] = None
        self._snapshots: Dict[str, OrderbookSnapshot] = {}

        # Executor per symbol (создаётся для каждой пары)
        self._executors: Dict[str, OrderExecutor] = {
            sym: OrderExecutor(sym, cfg.exec_cfg, bybit_client)
            for sym in cfg.ws_symbols
        }

        # Состояние
        self._running    = False
        self._paused     = False
        self._scan_task: Optional[asyncio.Task] = None

        # Метрики
        self._total_decisions  = 0
        self._dry_run_count    = 0
        self._trades_completed = 0
        self._trades_won       = 0
        self._total_net_pnl    = 0.0
        self._start_ts         = 0.0

    # ── Жизненный цикл ───────────────────────────────────────────────────────

    async def start(self) -> None:
        self._running  = True
        self._start_ts = time.time()
        mode = "DRY-RUN" if self._cfg.dry_run else ("PAPER" if self._cfg.paper else "LIVE")
        logger.info(f"[OBEngine] Запуск [{mode}] symbols={self._cfg.ws_symbols}")
        await self._notify(
            f"📊 OrderbookEngine запущен [{mode}]\n"
            f"Пары: {', '.join(self._cfg.ws_symbols)}"
        )

        self._ws_mgr = OrderbookWsManager(
            symbols     = self._cfg.ws_symbols,
            on_snapshot = self._on_snapshot,
        )

        self._scan_task = asyncio.create_task(self._scan_loop())
        await self._ws_mgr.start()         # блокирует, пока WS жив

    async def stop(self) -> None:
        self._running = False
        if self._scan_task and not self._scan_task.done():
            self._scan_task.cancel()
        if self._ws_mgr:
            await self._ws_mgr.stop()
        logger.info("[OBEngine] Остановлен")

    def pause(self) -> None:
        self._paused = True
        logger.info("[OBEngine] ⏸ Пауза")

    def resume(self) -> None:
        self._paused = False
        logger.info("[OBEngine] ▶ Возобновлён")

    # ── Callback: новый снимок стакана ────────────────────────────────────────

    async def _on_snapshot(self, snap: OrderbookSnapshot) -> None:
        """Вызывается из WS-потока при каждом обновлении (async для ensure_future)."""
        self._snapshots[snap.symbol] = snap
        self._pair_sel.update(snap)
        self._liq_filter.update_history(snap)

        # Диспатч тика в executor этого символа (fire-and-forget с обработкой ошибок)
        if self._running and not self._paused:
            task = asyncio.ensure_future(self._tick_executor(snap))
            task.add_done_callback(self._on_executor_task_done)

    def _on_executor_task_done(self, task: "asyncio.Task[None]") -> None:
        exc = task.exception() if not task.cancelled() else None
        if exc:
            logger.error(f"[OBEngine] _tick_executor failed: {exc}", exc_info=exc)

    async def _tick_executor(self, snap: OrderbookSnapshot) -> None:
        exc = self._executors.get(snap.symbol)
        if not exc:
            return
        try:
            rec = await exc.on_tick(snap)
        except Exception as e:
            logger.error(f"[OBEngine] on_tick {snap.symbol} error: {e}", exc_info=True)
            return
        if rec is not None:
            await self._on_trade_closed(rec)

    # ── Scan loop: ищем новые входы ──────────────────────────────────────────

    async def _scan_loop(self) -> None:
        while self._running:
            try:
                if not self._paused:
                    await self._scan_once()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"[OBEngine] scan_loop error: {exc}", exc_info=True)
            await asyncio.sleep(self._cfg.scan_interval_sec)

    async def _scan_once(self) -> None:
        cfg  = self._cfg
        snaps = self._snapshots

        # Выбираем активные пары
        active = self._pair_sel.select(snaps, self._risk)

        for sym in active:
            snap = snaps.get(sym)
            if snap is None:
                continue

            exc = self._executors[sym]
            if not exc.is_idle:
                continue   # уже в позиции

            # Проверка RiskGuard
            pos_usdt = cfg.scan_cfg.max_position_usdt
            ok, reason = self._risk.can_open(sym, pos_usdt)
            if not ok:
                self._risk.record_skip(reason)
                continue

            # SpreadScanner
            opp = self._scanner.scan(snap, cfg.scan_cfg)
            if not opp.is_valid:
                self._risk.record_skip(f"scan:{opp.skip_reason}")
                continue

            # LiquidityFilter
            liq = self._liq_filter.check(snap, opp.side, cfg.liquidity_cfg, pos_usdt)
            if not liq.passed:
                self._risk.record_skip(f"liq:{liq.reason}")
                continue

            self._total_decisions += 1

            # Qty (целые лоты не обязательны на линейных перп, но для paper OK)
            qty = pos_usdt / opp.entry_price if opp.entry_price > 0 else 0.0
            if qty <= 0:
                continue

            if cfg.dry_run:
                self._dry_run_count += 1
                logger.info(
                    f"[DRY_RUN] {sym} {opp.side} entry={opp.entry_price:.8f} "
                    f"exit={opp.exit_price:.8f} net≈{opp.net_usdt:.5f} "
                    f"spread={snap.spread_pct:.4f}% ratio={opp.ratio:.2f}"
                )
                # Каждые N решений — уведомление в Telegram
                if self._dry_run_count % cfg.notify_every_n_dry == 0:
                    await self._notify(
                        f"🔍 [DRY-RUN] {self._dry_run_count} решений\n"
                        f"Пример: {sym} {opp.side} entry={opp.entry_price:.6f} "
                        f"net≈{opp.net_usdt:.5f} USDT\n"
                        + self._risk_summary_short()
                    )
                continue

            # Paper / Live — открываем
            try:
                opened = await exc.open(
                    side        = opp.side,
                    entry_price = opp.entry_price,
                    exit_price  = opp.exit_price,
                    qty         = qty,
                    snap        = snap,
                )
            except Exception as e:
                logger.error(f"[OBEngine] exc.open {sym} error: {e}", exc_info=True)
                continue
            if opened:
                self._risk.register_open(sym, pos_usdt)
                logger.info(
                    f"[OBEngine] OPEN {sym} {opp.side} @ {opp.entry_price:.8f} "
                    f"qty={qty:.4f} net≈{opp.net_usdt:.5f}"
                )

    # ── Закрытие сделки ───────────────────────────────────────────────────────

    async def _on_trade_closed(self, rec: TradeRecord) -> None:
        cfg      = self._cfg
        pos_usdt = cfg.scan_cfg.max_position_usdt
        cancelled = rec.status == ExecStatus.CANCELLED

        self._risk.register_close(
            symbol        = rec.symbol,
            net_pnl       = rec.net_pnl,
            fees          = rec.fees_usdt,
            position_usdt = pos_usdt,
            cancelled     = cancelled,
        )

        if not cancelled and rec.status != ExecStatus.DRY_RUN:
            self._trades_completed += 1
            if rec.status == ExecStatus.WIN:
                self._trades_won += 1
            self._total_net_pnl += rec.net_pnl
            self._csv.write(rec)

            emoji = "✅" if rec.status == ExecStatus.WIN else (
                "🆘" if rec.status == ExecStatus.EMERGENCY else "❌"
            )
            pnl_str = f"{rec.net_pnl:+.5f}"
            await self._notify(
                f"{emoji} {rec.symbol} {rec.side} "
                f"[{rec.status.value.upper()}] "
                f"net={pnl_str} USDT "
                f"hold={rec.hold_sec:.1f}s | {rec.exit_reason}"
            )
            logger.info(
                f"[OBEngine] CLOSE {rec.symbol} {rec.side} "
                f"status={rec.status.value} net={pnl_str} USDT"
            )

    # ── Метрики и статус ─────────────────────────────────────────────────────

    def metrics(self) -> dict:
        risk_sum = self._risk.summary()
        wr = self._trades_won / self._trades_completed if self._trades_completed > 0 else 0.0
        uptime = time.time() - self._start_ts if self._start_ts else 0
        return {
            "mode":             "dry_run" if self._cfg.dry_run else ("paper" if self._cfg.paper else "live"),
            "running":          self._running,
            "paused":           self._paused,
            "uptime_sec":       round(uptime, 0),
            "total_decisions":  self._total_decisions,
            "dry_run_count":    self._dry_run_count,
            "trades_completed": self._trades_completed,
            "trades_won":       self._trades_won,
            "winrate":          round(wr, 3),
            "total_net_pnl":    round(self._total_net_pnl, 5),
            "risk":             risk_sum,
            "active_pairs":     self._pair_sel.select(self._snapshots, self._risk),
        }

    def status_text(self) -> str:
        m   = self.metrics()
        r   = m["risk"]
        uptime_h = m["uptime_sec"] / 3600
        mode_tag = m["mode"].upper()
        lines = [
            f"📊 OrderbookEngine [{mode_tag}]",
            f"⏱ Uptime: {uptime_h:.1f}ч  {'⏸' if m['paused'] else '▶'}",
            f"🔢 Решений: {m['total_decisions']}  Dry-run: {m['dry_run_count']}",
            f"📈 Сделок: {m['trades_completed']}  WR: {m['winrate']:.1%}",
            f"💰 Net PnL: {m['total_net_pnl']:+.4f} USDT",
            f"📉 Daily PnL: {r['daily_pnl']:+.4f} USDT",
            f"🔓 Открытых: {r['open_positions']}  Экспоз: {r['total_exposure']:.1f} USDT",
            f"⏭ Пропусков: {sum(r['skips'].values())} "
            f"({', '.join(f'{k}:{v}' for k,v in list(r['skips'].items())[:3])})",
            f"📋 Пары: {', '.join(m['active_pairs']) or 'нет'}",
        ]
        return "\n".join(lines)

    def pairs_status_text(self) -> str:
        return self._pair_sel.status(self._snapshots)

    def _risk_summary_short(self) -> str:
        r = self._risk.summary()
        return (
            f"daily={r['daily_pnl']:+.4f} trades={r['total_trades']} "
            f"wr={r['winrate']:.1%} open={r['open_positions']}"
        )

    # ── Telegram helper ───────────────────────────────────────────────────────

    async def _notify(self, msg: str) -> None:
        if self._notify_fn:
            try:
                await self._notify_fn(msg)
            except Exception as exc:
                logger.warning(f"[OBEngine] notify error: {exc}")
