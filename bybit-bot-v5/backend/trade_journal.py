"""
Журнал всех сделок — MySQL / SQLite через DBPool.
Экспорт в CSV и Excel, heatmap-статистика.
"""
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import logging

from db_pool import DBPool

logger = logging.getLogger(__name__)

_CREATE_TRADES = """
CREATE TABLE IF NOT EXISTS trades (
    id              {ai_pk},
    timestamp       VARCHAR(32) NOT NULL,
    strategy_id     VARCHAR(16) NOT NULL,
    strategy_name   VARCHAR(64),
    symbol          VARCHAR(20) NOT NULL,
    side            VARCHAR(8)  NOT NULL,
    entry_price     {real}      NOT NULL,
    exit_price      {real},
    qty             {real}      NOT NULL,
    leverage        INT,
    stop_loss       {real},
    take_profit     {real},
    initial_sl      {real},
    initial_tp      {real},
    atr_at_entry    {real},
    be_triggered    TINYINT DEFAULT 0,
    trailing_triggered TINYINT DEFAULT 0,
    r_multiple      {real},
    signal_reason   TEXT,
    pnl_usd         {real},
    pnl_pct         {real},
    exit_reason     VARCHAR(32),
    duration_sec    INT,
    fees            {real},
    filters_passed  TEXT,
    ai_score        {real},
    paper_trading   TINYINT DEFAULT 0,
    opened_at       VARCHAR(32),
    closed_at       VARCHAR(32)
)
"""

# Новые колонки для миграции существующих БД
_MIGRATION_COLUMNS = [
    ("initial_sl",          "{real}"),
    ("initial_tp",          "{real}"),
    ("atr_at_entry",        "{real}"),
    ("be_triggered",        "TINYINT DEFAULT 0"),
    ("trailing_triggered",  "TINYINT DEFAULT 0"),
    ("r_multiple",          "{real}"),
    ("signal_reason",       "TEXT"),
]


class TradeJournal:
    def __init__(self, db_pool: Optional[DBPool] = None):
        """
        db_pool — общий пул (MySQL или SQLite).
        Если не передан — создаёт свой SQLite-пул в logs/trades.db.
        """
        if db_pool:
            self.pool = db_pool
        else:
            self.pool = DBPool("logs/trades.db")
        self._init_db()

    def _init_db(self):
        real = DBPool.real_type()
        sql = _CREATE_TRADES.format(ai_pk=DBPool.ai_pk(), real=real)
        with self.pool.cursor() as c:
            c.execute(sql)
            if not self.pool.is_mysql:
                c.execute("CREATE INDEX IF NOT EXISTS idx_strategy ON trades(strategy_id)")
                c.execute("CREATE INDEX IF NOT EXISTS idx_symbol    ON trades(symbol)")
                c.execute("CREATE INDEX IF NOT EXISTS idx_ts        ON trades(timestamp)")
            else:
                for idx, col in [("idx_strategy", "strategy_id"),
                                  ("idx_symbol",   "symbol"),
                                  ("idx_ts",        "timestamp")]:
                    try:
                        c.execute(f"CREATE INDEX {idx} ON trades({col})")
                    except Exception:
                        pass

        # Миграция: добавляем новые колонки в уже существующие таблицы
        self._migrate_columns(real)

    def _migrate_columns(self, real: str):
        """Добавляет новые колонки в таблицу trades если их ещё нет."""
        for col_name, col_type in _MIGRATION_COLUMNS:
            col_def = col_type.replace("{real}", real)
            try:
                with self.pool.cursor() as c:
                    c.execute(f"ALTER TABLE trades ADD COLUMN {col_name} {col_def}")
                logger.info(f"[Journal] Миграция: добавлена колонка {col_name}")
            except Exception:
                pass  # колонка уже существует

    # ── запись ────────────────────────────────────────────────

    def log_trade(self, trade: Dict) -> int:
        """Записать сделку. Возвращает trade_id."""
        sql = self.pool.adapt("""
            INSERT INTO trades (
                timestamp, strategy_id, strategy_name, symbol, side,
                entry_price, exit_price, qty, leverage, stop_loss, take_profit,
                initial_sl, initial_tp, atr_at_entry,
                be_triggered, trailing_triggered, r_multiple, signal_reason,
                pnl_usd, pnl_pct, exit_reason, duration_sec, fees,
                filters_passed, ai_score, paper_trading, opened_at, closed_at
            ) VALUES (?,?,?,?,?, ?,?,?,?,?,?, ?,?,?, ?,?,?,?, ?,?,?,?,?, ?,?,?,?,?)
        """)
        values = (
            trade.get("timestamp", datetime.utcnow().isoformat()),
            trade["strategy_id"],
            trade.get("strategy_name", ""),
            trade["symbol"],
            trade["side"],
            trade["entry_price"],
            trade.get("exit_price"),
            trade["qty"],
            trade.get("leverage", 1),
            trade.get("stop_loss"),
            trade.get("take_profit"),
            # Начальные уровни (до BE/trailing)
            trade.get("initial_sl",   trade.get("stop_loss")),
            trade.get("initial_tp",   trade.get("take_profit")),
            trade.get("atr_at_entry"),
            1 if trade.get("be_triggered")       else 0,
            1 if trade.get("trailing_triggered") else 0,
            trade.get("r_multiple"),
            trade.get("signal_reason", ""),
            trade.get("pnl_usd", 0),
            trade.get("pnl_pct", 0),
            trade.get("exit_reason", ""),
            trade.get("duration_sec", 0),
            trade.get("fees", 0),
            json.dumps(trade.get("filters_passed", {}), default=str),
            trade.get("ai_score"),
            1 if trade.get("paper_trading") else 0,
            trade.get("opened_at"),
            trade.get("closed_at"),
        )
        with self.pool.cursor() as c:
            c.execute(sql, values)
            return c.lastrowid

    def update_trade_levels(self, trade_id: int, **kwargs):
        """
        Обновляет уровни сделки в процессе жизни позиции.
        Принимает любые поля: stop_loss, take_profit, be_triggered, trailing_triggered.
        """
        if not trade_id or not kwargs:
            return
        allowed = {"stop_loss", "take_profit", "be_triggered", "trailing_triggered"}
        fields = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields:
            return
        set_clause = ", ".join(f"{k}=?" for k in fields)
        sql = self.pool.adapt(f"UPDATE trades SET {set_clause} WHERE id=?")
        with self.pool.cursor() as c:
            c.execute(sql, list(fields.values()) + [trade_id])

    def update_trade_close(self, trade_id: int, exit_price: float, pnl_usd: float,
                           pnl_pct: float, exit_reason: str, fees: float = 0,
                           r_multiple: float = 0):
        sql = self.pool.adapt("""
            UPDATE trades
            SET exit_price=?, pnl_usd=?, pnl_pct=?,
                exit_reason=?, fees=?, r_multiple=?, closed_at=?
            WHERE id=?
        """)
        with self.pool.cursor() as c:
            c.execute(sql, (exit_price, pnl_usd, pnl_pct, exit_reason,
                            fees, r_multiple, datetime.utcnow().isoformat(), trade_id))

    # ── чтение ────────────────────────────────────────────────

    def get_trades(self, strategy_id: Optional[str] = None,
                   symbol: Optional[str] = None,
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None,
                   limit: int = 1000) -> List[Dict]:
        query = "SELECT * FROM trades WHERE 1=1"
        params: list = []
        if strategy_id:
            query += " AND strategy_id = ?"
            params.append(strategy_id)
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self.pool.connection() as conn:
            if self.pool.is_mysql:
                with conn.cursor() as c:
                    c.execute(self.pool.adapt(query), params)
                    return list(c.fetchall())
            else:
                import sqlite3
                conn.row_factory = sqlite3.Row
                return [dict(r) for r in conn.execute(self.pool.adapt(query), params).fetchall()]

    def get_open_trade(self, strategy_id: str, symbol: str) -> Optional[Dict]:
        """Найти последнюю незакрытую запись в журнале для strategy_id + symbol."""
        query = self.pool.adapt(
            "SELECT * FROM trades WHERE strategy_id=? AND symbol=? "
            "AND exit_price IS NULL ORDER BY id DESC LIMIT 1"
        )
        try:
            with self.pool.connection() as conn:
                if self.pool.is_mysql:
                    with conn.cursor() as c:
                        c.execute(query, [strategy_id, symbol])
                        row = c.fetchone()
                        return dict(row) if row else None
                else:
                    import sqlite3
                    conn.row_factory = sqlite3.Row
                    row = conn.execute(query, [strategy_id, symbol]).fetchone()
                    return dict(row) if row else None
        except Exception as e:
            logger.warning(f"[Journal] get_open_trade error: {e}")
            return None

    def restore_strategy_stats(self, strategy_id: str) -> Optional[Dict]:
        """
        Восстанавливает статистику стратегии из БД без pandas (raw cursor).
        Возвращает агрегаты + последние 50 PnL для history[] + consecutive_losses.
        Надёжнее pd.read_sql при любой версии pandas / SQLite.
        """
        query = self.pool.adapt(
            "SELECT pnl_usd FROM trades "
            "WHERE strategy_id=? AND exit_price IS NOT NULL "
            "ORDER BY id ASC"
        )
        try:
            with self.pool.connection() as conn:
                if self.pool.is_mysql:
                    with conn.cursor() as c:
                        c.execute(query, [strategy_id])
                        rows = c.fetchall()
                        pnl_list = [float(r["pnl_usd"] or 0) for r in rows]
                else:
                    rows = conn.execute(query, [strategy_id]).fetchall()
                    pnl_list = [float(r[0] or 0) for r in rows]

            if not pnl_list:
                return None

            wins = sum(1 for p in pnl_list if p > 0)
            consecutive = 0
            for p in reversed(pnl_list):
                if p < 0:
                    consecutive += 1
                else:
                    break

            return {
                "trades":             len(pnl_list),
                "wins":               wins,
                "losses":             len(pnl_list) - wins,
                "total_pnl":          round(sum(pnl_list), 4),
                "history":            [round(p, 4) for p in pnl_list[-50:]],
                "consecutive_losses": consecutive,
            }
        except Exception as e:
            logger.warning(f"[Journal] restore_strategy_stats({strategy_id}): {e}")
            return None

    def get_stats_by_strategy(self) -> List[Dict]:
        sql = "SELECT * FROM trades WHERE exit_price IS NOT NULL"
        if self.pool.is_mysql:
            with self.pool.connection() as conn:
                df = pd.read_sql(sql, conn)
        else:
            import sqlite3 as _sq3
            conn = _sq3.connect(self.pool.db_path, check_same_thread=False, timeout=10)
            try:
                df = pd.read_sql(sql, conn)
            finally:
                conn.close()
        if df.empty:
            return []
        df["pnl_usd"] = pd.to_numeric(df["pnl_usd"], errors="coerce").fillna(0.0)
        stats = []
        for sid, g in df.groupby("strategy_id"):
            wins = g[g["pnl_usd"] > 0]
            losses = g[g["pnl_usd"] < 0]
            _name_raw = g["strategy_name"].iloc[0]
            _name = str(_name_raw) if (pd.notna(_name_raw) and _name_raw) else sid
            stats.append({
                "strategy_id": sid,
                "strategy_name": _name,
                "trades": len(g),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": round(len(wins) / len(g) * 100, 2) if len(g) else 0,
                "total_pnl": round(g["pnl_usd"].sum(), 2),
                "avg_win": round(wins["pnl_usd"].mean(), 2) if len(wins) else 0,
                "avg_loss": round(losses["pnl_usd"].mean(), 2) if len(losses) else 0,
                "profit_factor": round(
                    abs(wins["pnl_usd"].sum() / losses["pnl_usd"].sum()), 2
                ) if len(losses) and losses["pnl_usd"].sum() != 0 else 0,
                "best_trade": round(g["pnl_usd"].max(), 2),
                "worst_trade": round(g["pnl_usd"].min(), 2),
            })
        return stats

    # ── экспорт ───────────────────────────────────────────────

    def _build_filter_sql(
        self,
        strategy_id: Optional[str] = None,
        symbol: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        closed_only: bool = True,
    ):
        """Строит WHERE-клаузу и список параметров для фильтрации."""
        where, params = ["1=1"], []
        if closed_only:
            where.append("exit_price IS NOT NULL")
        if strategy_id:
            where.append("strategy_id = ?")
            params.append(strategy_id)
        if symbol:
            where.append("symbol = ?")
            params.append(symbol)
        if from_date:
            where.append("timestamp >= ?")
            params.append(from_date)
        if to_date:
            where.append("timestamp <= ?")
            params.append(to_date)
        return " AND ".join(where), params

    def _load_df(
        self,
        strategy_id: Optional[str] = None,
        symbol: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        closed_only: bool = True,
    ) -> pd.DataFrame:
        where, params = self._build_filter_sql(
            strategy_id, symbol, from_date, to_date, closed_only
        )
        sql = self.pool.adapt(f"SELECT * FROM trades WHERE {where} ORDER BY timestamp ASC")
        if self.pool.is_mysql:
            with self.pool.connection() as conn:
                return pd.read_sql(sql, conn, params=params)
        else:
            # Свежее соединение БЕЗ row_factory — pd.read_sql несовместим с sqlite3.Row
            import sqlite3 as _sq3
            conn = _sq3.connect(self.pool.db_path, check_same_thread=False, timeout=10)
            try:
                df = pd.read_sql(sql, conn, params=params)
            finally:
                conn.close()
            return df

    def export_to_csv(
        self,
        output_path: str = "logs/trades_export.csv",
        strategy_id: Optional[str] = None,
        symbol: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> str:
        df = self._load_df(strategy_id, symbol, from_date, to_date, closed_only=False)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return output_path

    def export_to_excel(
        self,
        output_path: str = "logs/trades_export.xlsx",
        strategy_id: Optional[str] = None,
        symbol: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> str:
        df = self._load_df(strategy_id, symbol, from_date, to_date, closed_only=False)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="All Trades", index=False)
            if not df.empty:
                closed = df[df["exit_price"].notna()].copy()
                pd.DataFrame(self.get_stats_by_strategy()).to_excel(
                    writer, sheet_name="By Strategy", index=False
                )
                if not closed.empty:
                    closed["timestamp_dt"] = pd.to_datetime(closed["timestamp"])
                    closed["hour"] = closed["timestamp_dt"].dt.hour
                    closed["dayofweek"] = closed["timestamp_dt"].dt.day_name()
                    closed.pivot_table(
                        values="pnl_usd", index="dayofweek", columns="hour",
                        aggfunc="sum", fill_value=0,
                    ).to_excel(writer, sheet_name="Heatmap")
                    # Equity curve
                    eq = closed.sort_values("timestamp").copy()
                    eq["cumulative_pnl"] = eq["pnl_usd"].cumsum()
                    eq[["timestamp", "strategy_id", "symbol", "pnl_usd", "cumulative_pnl"]].to_excel(
                        writer, sheet_name="Equity Curve", index=False
                    )
        return output_path

    def export_snapshots_csv(
        self,
        output_path: str = "logs/snapshots_export.csv",
        strategy_id: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> str:
        """Экспортирует signal_snapshots (ML-фичи + исходы) в CSV."""
        where, params = ["1=1"], []
        if strategy_id:
            where.append("strategy_id = ?")
            params.append(strategy_id)
        if from_date:
            where.append("timestamp >= ?")
            params.append(from_date)
        if to_date:
            where.append("timestamp <= ?")
            params.append(to_date)
        sql = "SELECT * FROM signal_snapshots WHERE {} ORDER BY timestamp ASC".format(
            " AND ".join(where)
        )
        try:
            with self.pool.connection() as conn:
                df = pd.read_sql(self.pool.adapt(sql), conn, params=params)
        except Exception:
            df = pd.DataFrame()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return output_path

    def get_equity_curve(
        self,
        strategy_id: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> List[Dict]:
        """Нарастающий PnL по времени для построения equity-кривой."""
        df = self._load_df(strategy_id=strategy_id, from_date=from_date, to_date=to_date)
        if df.empty:
            return []
        df = df.sort_values("timestamp").copy()
        df["cumulative_pnl"] = df["pnl_usd"].cumsum()
        return df[["timestamp", "strategy_id", "symbol", "pnl_usd", "cumulative_pnl"]].to_dict(orient="records")

    def get_heatmap_data(self) -> Dict:
        with self.pool.connection() as conn:
            df = pd.read_sql(
                "SELECT * FROM trades WHERE exit_price IS NOT NULL", conn
            )
        if df.empty:
            return {}
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
        df["hour"] = df["timestamp_dt"].dt.hour
        df["dayofweek"] = df["timestamp_dt"].dt.day_name()
        return {
            sid: g.pivot_table(
                values="pnl_usd", index="dayofweek", columns="hour",
                aggfunc="sum", fill_value=0,
            ).to_dict()
            for sid, g in df.groupby("strategy_id")
        }
