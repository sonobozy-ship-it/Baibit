"""
Журнал всех сделок с экспортом в CSV/Excel и SQLite.
"""
import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class TradeJournal:
    def __init__(self, db_path: str = "logs/trades.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                strategy_id TEXT NOT NULL,
                strategy_name TEXT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                qty REAL NOT NULL,
                leverage INTEGER,
                stop_loss REAL,
                take_profit REAL,
                pnl_usd REAL,
                pnl_pct REAL,
                exit_reason TEXT,
                duration_sec INTEGER,
                fees REAL,
                filters_passed TEXT,
                ai_score REAL,
                paper_trading INTEGER DEFAULT 0,
                opened_at TEXT,
                closed_at TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_strategy ON trades(strategy_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON trades(symbol)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON trades(timestamp)")
        conn.commit()
        conn.close()

    def log_trade(self, trade: Dict) -> int:
        """Записать сделку. Возвращает trade_id."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO trades (
                timestamp, strategy_id, strategy_name, symbol, side,
                entry_price, exit_price, qty, leverage, stop_loss, take_profit,
                pnl_usd, pnl_pct, exit_reason, duration_sec, fees,
                filters_passed, ai_score, paper_trading, opened_at, closed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
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
        ))
        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return trade_id

    def update_trade_close(self, trade_id: int, exit_price: float, pnl_usd: float,
                          pnl_pct: float, exit_reason: str, fees: float = 0):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            UPDATE trades SET exit_price=?, pnl_usd=?, pnl_pct=?, exit_reason=?,
                              fees=?, closed_at=?
            WHERE id=?
        """, (exit_price, pnl_usd, pnl_pct, exit_reason, fees,
              datetime.utcnow().isoformat(), trade_id))
        conn.commit()
        conn.close()

    def get_trades(self, strategy_id: Optional[str] = None,
                   symbol: Optional[str] = None,
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None,
                   limit: int = 1000) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        query = "SELECT * FROM trades WHERE 1=1"
        params = []
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
        rows = [dict(row) for row in conn.execute(query, params).fetchall()]
        conn.close()
        return rows

    def get_stats_by_strategy(self) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql("SELECT * FROM trades WHERE exit_price IS NOT NULL", conn)
        conn.close()
        if df.empty:
            return []
        stats = []
        for sid, group in df.groupby("strategy_id"):
            wins = group[group["pnl_usd"] > 0]
            losses = group[group["pnl_usd"] < 0]
            stats.append({
                "strategy_id": sid,
                "strategy_name": group["strategy_name"].iloc[0] if len(group) else "",
                "trades": len(group),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": round(len(wins) / len(group) * 100, 2) if len(group) else 0,
                "total_pnl": round(group["pnl_usd"].sum(), 2),
                "avg_win": round(wins["pnl_usd"].mean(), 2) if len(wins) else 0,
                "avg_loss": round(losses["pnl_usd"].mean(), 2) if len(losses) else 0,
                "profit_factor": round(abs(wins["pnl_usd"].sum() / losses["pnl_usd"].sum()), 2)
                                 if len(losses) and losses["pnl_usd"].sum() != 0 else 0,
                "best_trade": round(group["pnl_usd"].max(), 2),
                "worst_trade": round(group["pnl_usd"].min(), 2),
            })
        return stats

    def export_to_csv(self, output_path: str = "logs/trades_export.csv") -> str:
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql("SELECT * FROM trades ORDER BY timestamp DESC", conn)
        conn.close()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Экспорт CSV: {output_path} ({len(df)} строк)")
        return output_path

    def export_to_excel(self, output_path: str = "logs/trades_export.xlsx") -> str:
        conn = sqlite3.connect(self.db_path)
        df_trades = pd.read_sql("SELECT * FROM trades ORDER BY timestamp DESC", conn)
        conn.close()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df_trades.to_excel(writer, sheet_name="All Trades", index=False)

            # Сводка по стратегиям
            if not df_trades.empty:
                stats_df = pd.DataFrame(self.get_stats_by_strategy())
                stats_df.to_excel(writer, sheet_name="By Strategy", index=False)

                # Heatmap: PnL по часам × дням недели
                df_trades["timestamp_dt"] = pd.to_datetime(df_trades["timestamp"])
                df_trades["hour"] = df_trades["timestamp_dt"].dt.hour
                df_trades["dayofweek"] = df_trades["timestamp_dt"].dt.day_name()
                heatmap = df_trades.pivot_table(
                    values="pnl_usd", index="dayofweek", columns="hour",
                    aggfunc="sum", fill_value=0,
                )
                heatmap.to_excel(writer, sheet_name="Heatmap")

        logger.info(f"Экспорт Excel: {output_path}")
        return output_path

    def get_heatmap_data(self) -> Dict:
        """Тепловая карта производительности: стратегия × час × день недели."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql("SELECT * FROM trades WHERE exit_price IS NOT NULL", conn)
        conn.close()
        if df.empty:
            return {}
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
        df["hour"] = df["timestamp_dt"].dt.hour
        df["dayofweek"] = df["timestamp_dt"].dt.day_name()

        result = {}
        for sid, group in df.groupby("strategy_id"):
            heatmap = group.pivot_table(
                values="pnl_usd", index="dayofweek", columns="hour",
                aggfunc="sum", fill_value=0,
            ).to_dict()
            result[sid] = heatmap
        return result
