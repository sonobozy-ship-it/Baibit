"""SQLite trade logger."""
from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import List, Optional

from .models import ArbPosition

logger = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS trades (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol        TEXT NOT NULL,
    long_exchange TEXT NOT NULL,
    short_exchange TEXT NOT NULL,
    entry_spread_pct  REAL,
    exit_spread_pct   REAL,
    long_entry    REAL,
    short_entry   REAL,
    long_exit     REAL,
    short_exit    REAL,
    qty           REAL,
    notional_usdt REAL,
    pnl_usdt      REAL,
    duration_sec  REAL,
    exit_reason   TEXT,
    ts_open       TEXT,
    ts_close      TEXT
);
"""


class DbLogger:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self._path = db_path
        with self._conn() as c:
            c.execute(_DDL)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    def log(self, pos: ArbPosition) -> None:
        def ts(t: float) -> str:
            return datetime.fromtimestamp(t, tz=timezone.utc).isoformat()

        duration = (pos.ts_close - pos.ts_open) if pos.ts_close else 0.0
        row = (
            pos.symbol, pos.long_exchange, pos.short_exchange,
            pos.entry_spread_pct, pos.exit_spread_pct,
            pos.long_entry, pos.short_entry, pos.long_exit, pos.short_exit,
            pos.qty, pos.notional_usdt, pos.realized_pnl_usdt, duration,
            pos.exit_reason, ts(pos.ts_open), ts(pos.ts_close or pos.ts_open),
        )
        try:
            with self._conn() as c:
                c.execute(
                    "INSERT INTO trades "
                    "(symbol,long_exchange,short_exchange,"
                    "entry_spread_pct,exit_spread_pct,"
                    "long_entry,short_entry,long_exit,short_exit,"
                    "qty,notional_usdt,pnl_usdt,duration_sec,exit_reason,ts_open,ts_close)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    row,
                )
        except Exception as exc:
            logger.error(f"[DB] log error: {exc}")

    def get_summary(self) -> dict:
        try:
            with self._conn() as c:
                row = c.execute(
                    "SELECT COUNT(*), COALESCE(SUM(pnl_usdt),0), "
                    "COALESCE(SUM(CASE WHEN pnl_usdt>0 THEN 1 ELSE 0 END),0) "
                    "FROM trades"
                ).fetchone()
                total, pnl, wins = row
                wr = wins / total if total else 0.0
                return {"total": total, "pnl_usdt": round(pnl, 4),
                        "wins": wins, "winrate": round(wr, 3)}
        except Exception:
            return {}
