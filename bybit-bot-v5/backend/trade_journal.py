"""
Журнал всех сделок — MySQL / SQLite через DBPool.
Экспорт в CSV и Excel, heatmap-статистика.
"""
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import logging

from db_pool import DBPool

logger = logging.getLogger(__name__)

_CREATE_TRADES = """
CREATE TABLE IF NOT EXISTS trades (
    id          {ai_pk},
    timestamp   VARCHAR(32) NOT NULL,
    strategy_id VARCHAR(16) NOT NULL,
    strategy_name VARCHAR(64),
    symbol      VARCHAR(20) NOT NULL,
    side        VARCHAR(8)  NOT NULL,
    entry_price {real}      NOT NULL,
    exit_price  {real},
    qty         {real}      NOT NULL,
    leverage    INT,
    stop_loss   {real},
    take_profit {real},
    pnl_usd     {real},
    pnl_pct     {real},
    exit_reason VARCHAR(32),
    duration_sec INT,
    fees        {real},
    filters_passed TEXT,
    ai_score    {real},
    paper_trading TINYINT DEFAULT 0,
    opened_at   VARCHAR(32),
    closed_at   VARCHAR(32)
)
"""


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
        sql = _CREATE_TRADES.format(
            ai_pk=DBPool.ai_pk(),
            real=DBPool.real_type(),
        )
        with self.pool.cursor() as c:
            c.execute(sql)
            if not self.pool.is_mysql:
                c.execute("CREATE INDEX IF NOT EXISTS idx_strategy ON trades(strategy_id)")
                c.execute("CREATE INDEX IF NOT EXISTS idx_symbol    ON trades(symbol)")
                c.execute("CREATE INDEX IF NOT EXISTS idx_ts        ON trades(timestamp)")
            else:
                # MySQL: CREATE INDEX IF NOT EXISTS — через INFORMATION_SCHEMA
                for idx, col in [("idx_strategy", "strategy_id"),
                                  ("idx_symbol",   "symbol"),
                                  ("idx_ts",        "timestamp")]:
                    try:
                        c.execute(f"CREATE INDEX {idx} ON trades({col})")
                    except Exception:
                        pass  # уже существует

    # ── запись ────────────────────────────────────────────────

    def log_trade(self, trade: Dict) -> int:
        """Записать сделку. Возвращает trade_id."""
        sql = self.pool.adapt("""
            INSERT INTO trades (
                timestamp, strategy_id, strategy_name, symbol, side,
                entry_price, exit_price, qty, leverage, stop_loss, take_profit,
                pnl_usd, pnl_pct, exit_reason, duration_sec, fees,
                filters_passed, ai_score, paper_trading, opened_at, closed_at
            ) VALUES (?,?,?,?,?, ?,?,?,?,?,?, ?,?,?,?,?, ?,?,?,?,?)
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
            trade.get("pnl_usd", 0),
            trade.get("pnl_pct", 0),
            trade.get("exit_reason", ""),
            trade.get("duration_sec", 0),
            trade.get("fees", 0),
            str(trade.get("filters_passed", {})),
            trade.get("ai_score"),
            1 if trade.get("paper_trading") else 0,
            trade.get("opened_at"),
            trade.get("closed_at"),
        )
        with self.pool.cursor() as c:
            c.execute(sql, values)
            return c.lastrowid

    def update_trade_close(self, trade_id: int, exit_price: float, pnl_usd: float,
                           pnl_pct: float, exit_reason: str, fees: float = 0):
        sql = self.pool.adapt("""
            UPDATE trades
            SET exit_price=?, pnl_usd=?, pnl_pct=?,
                exit_reason=?, fees=?, closed_at=?
            WHERE id=?
        """)
        with self.pool.cursor() as c:
            c.execute(sql, (exit_price, pnl_usd, pnl_pct, exit_reason,
                            fees, datetime.utcnow().isoformat(), trade_id))

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

    def get_stats_by_strategy(self) -> List[Dict]:
        with self.pool.connection() as conn:
            df = pd.read_sql(
                "SELECT * FROM trades WHERE exit_price IS NOT NULL",
                conn,
            )
        if df.empty:
            return []
        stats = []
        for sid, g in df.groupby("strategy_id"):
            wins = g[g["pnl_usd"] > 0]
            losses = g[g["pnl_usd"] < 0]
            stats.append({
                "strategy_id": sid,
                "strategy_name": g["strategy_name"].iloc[0],
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

    def export_to_csv(self, output_path: str = "logs/trades_export.csv") -> str:
        with self.pool.connection() as conn:
            df = pd.read_sql("SELECT * FROM trades ORDER BY timestamp DESC", conn)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return output_path

    def export_to_excel(self, output_path: str = "logs/trades_export.xlsx") -> str:
        with self.pool.connection() as conn:
            df = pd.read_sql("SELECT * FROM trades ORDER BY timestamp DESC", conn)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="All Trades", index=False)
            if not df.empty:
                pd.DataFrame(self.get_stats_by_strategy()).to_excel(
                    writer, sheet_name="By Strategy", index=False
                )
                df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
                df["hour"] = df["timestamp_dt"].dt.hour
                df["dayofweek"] = df["timestamp_dt"].dt.day_name()
                df.pivot_table(
                    values="pnl_usd", index="dayofweek", columns="hour",
                    aggfunc="sum", fill_value=0,
                ).to_excel(writer, sheet_name="Heatmap")
        return output_path

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
