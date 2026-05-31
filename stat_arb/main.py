"""
Main event loop for the cross-exchange statistical arbitrage bot.

Flow per tick:
  1. Fetch all prices from all exchanges in parallel
  2. Monitor open positions → exit if TP / aggressive-TP / SL triggered
  3. Scan for new spread opportunities
  4. Apply entry filters (volatility, liquidity, trend, news)
  5. Open position if all checks pass
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
from typing import Dict, List

from dotenv import load_dotenv

from .config import ArbConfig, load_config
from .db_logger import DbLogger
from .exchange_hub import ExchangeHub
from .filters import EntryFilters
from .models import ArbPosition, Ticker
from .notifier import Notifier
from .paper_engine import PaperEngine
from .position_manager import PositionManager
from .risk import RiskManager
from .spread_detector import PriceHistory, SpreadDetector

logger = logging.getLogger(__name__)


async def run(cfg: ArbConfig) -> None:
    mode = "PAPER" if cfg.paper_mode else "LIVE"
    logger.info(f"[StatArb] Starting [{mode}] cap={cfg.capital_usdt}$")

    hub      = ExchangeHub(cfg)
    paper    = PaperEngine(cfg) if cfg.paper_mode else None
    pos_mgr  = PositionManager(hub, cfg, paper)
    risk     = RiskManager(cfg)
    risk.attach(pos_mgr.positions)
    detector = SpreadDetector(cfg)
    history  = PriceHistory()
    filters  = EntryFilters(hub, cfg)
    db       = DbLogger(cfg.db_path)
    notifier = Notifier(cfg)

    running      = True
    last_status  = 0.0
    STATUS_INTERVAL = 3600   # hourly status ping

    def _stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)

    await notifier._send(
        f"🚀 <b>Stat-Arb запущен [{mode}]</b>\n"
        f"Капитал: {cfg.capital_usdt}$ | "
        f"Монет: {len(cfg.allowed_symbols)} | "
        f"Leverage: ×{cfg.default_leverage}"
    )

    while running:
        t0 = time.monotonic()

        try:
            # ── 1. Fetch prices ───────────────────────────────────────────────
            all_tickers: Dict[str, Dict[str, Ticker]] = await hub.get_all_tickers(
                cfg.allowed_symbols
            )
            history.record(all_tickers)

            # ── 2. Monitor open positions ─────────────────────────────────────
            for pos in list(pos_mgr.positions):
                exit_reason = pos_mgr.check_exit(pos, all_tickers)
                if exit_reason:
                    logger.info(f"[StatArb] EXIT {pos.symbol} reason={exit_reason}")
                    await pos_mgr.close(pos, exit_reason, all_tickers)
                    await notifier.send_exit(pos)
                    db.log(pos)

            # ── 3. Scan for new opportunities ─────────────────────────────────
            can_open, reason = risk.can_open()
            if can_open:
                for symbol in cfg.allowed_symbols:
                    tickers_for_sym = all_tickers.get(symbol, {})
                    if len(tickers_for_sym) < 2:
                        continue

                    # Already have a position in this symbol?
                    if any(p.symbol == symbol for p in pos_mgr.positions):
                        continue

                    snap = detector.scan(symbol, tickers_for_sym)
                    if snap is None:
                        continue

                    if not detector.is_mature(snap):
                        logger.debug(
                            f"[StatArb] {symbol} spread={snap.spread_pct:.3f}% "
                            f"holding... (need {cfg.spread_hold_seconds}s)"
                        )
                        continue

                    notional = risk.position_notional()
                    leverage = risk.leverage()
                    qty      = risk.qty_for_notional(notional, snap.long_ask)

                    if qty <= 0:
                        continue

                    # ── 4. Filters ────────────────────────────────────────────
                    ok, filter_reason = await filters.check_all(snap, history, notional)
                    if not ok:
                        logger.info(
                            f"[StatArb] {symbol} spread={snap.spread_pct:.3f}% "
                            f"FILTERED: {filter_reason}"
                        )
                        detector.reset(symbol, snap.long_exchange, snap.short_exchange)
                        continue

                    # ── 5. Open ───────────────────────────────────────────────
                    logger.info(
                        f"[StatArb] OPENING {symbol} "
                        f"LONG {snap.long_exchange} / SHORT {snap.short_exchange} "
                        f"spread={snap.spread_pct:.3f}% qty={qty:.4f} notional={notional:.1f}$"
                    )
                    pos = await pos_mgr.open(snap, notional, leverage, qty)
                    if pos:
                        await notifier.send_entry(pos)
                        detector.reset(symbol, snap.long_exchange, snap.short_exchange)
                    else:
                        await notifier.send_error(
                            f"Не удалось открыть {symbol} {snap.long_exchange}/{snap.short_exchange}"
                        )

            # ── Periodic status ───────────────────────────────────────────────
            now = time.time()
            if now - last_status >= STATUS_INTERVAL:
                last_status = now
                bal = paper.balance if paper else 0.0
                await notifier.send_status(bal, len(pos_mgr.positions), db.get_summary())

        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.error(f"[StatArb] main loop error: {exc}", exc_info=True)

        # Maintain scan interval
        elapsed = time.monotonic() - t0
        sleep_t = max(0.0, cfg.scan_interval_sec - elapsed)
        await asyncio.sleep(sleep_t)

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("[StatArb] Shutting down...")
    if paper:
        logger.info(f"[StatArb] Final paper balance: {paper.balance:.4f}$")

    summary = db.get_summary()
    logger.info(
        f"[StatArb] Session: trades={summary.get('total',0)} "
        f"pnl={summary.get('pnl_usdt',0):+.4f}$ "
        f"wr={summary.get('winrate',0):.1%}"
    )
    hub.close()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler("logs/stat_arb.log"),
            logging.StreamHandler(),
        ],
    )
    import os
    os.makedirs("logs", exist_ok=True)

    cfg = load_config()
    asyncio.run(run(cfg))


if __name__ == "__main__":
    main()
