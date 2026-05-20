"""
Гибридное хранилище ML-данных:
- MySQL или SQLite (через DBPool): метаданные сигналов, модели, предсказания
- Parquet: исторические свечи (быстро для ML)
"""
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging
import sys
import os

# Добавляем родительскую директорию в путь чтобы найти db_pool
sys.path.insert(0, str(Path(__file__).parent.parent))
from db_pool import DBPool

logger = logging.getLogger(__name__)


class MLDataStore:
    """
    Главное ML-хранилище.

    Структура:
    - ml_data.db (SQLite) - signals, features, predictions, outcomes
    - data/candles/{symbol}_{timeframe}.parquet - свечи
    - data/features/{strategy_id}.parquet - извлечённые фичи (для batch обучения)
    """

    def __init__(self, db_path: str = "data/ml_data.db",
                 candles_dir: str = "data/candles",
                 features_dir: str = "data/features"):
        Path(candles_dir).mkdir(parents=True, exist_ok=True)
        Path(features_dir).mkdir(parents=True, exist_ok=True)

        self.db_path = db_path
        self.candles_dir = Path(candles_dir)
        self.features_dir = Path(features_dir)
        self.pool = DBPool(db_path)
        self._init_db()

    def _init_db(self):
        """Инициализация таблиц (MySQL или SQLite)."""
        ai = DBPool.ai_pk()
        real = DBPool.real_type()
        tjson = DBPool.text_json()
        # VARCHAR не принимает CURRENT_TIMESTAMP в MySQL — используем пустую строку
        now = "DEFAULT ''"

        tables = [
            f"""CREATE TABLE IF NOT EXISTS signal_snapshots (
                id               {ai},
                timestamp        VARCHAR(32) NOT NULL,
                strategy_id      VARCHAR(16) NOT NULL,
                symbol           VARCHAR(20) NOT NULL,
                timeframe        VARCHAR(8),
                action           VARCHAR(8)  NOT NULL,
                entry_price      {real}      NOT NULL,
                stop_loss        {real},
                take_profit      {real},
                features_json    {tjson}     NOT NULL,
                ml_prediction    {real},
                ml_confidence    {real},
                ml_model_version VARCHAR(64),
                trade_taken      TINYINT     DEFAULT 0,
                trade_id         INT,
                outcome          VARCHAR(8),
                pnl_r            {real},
                exit_reason      VARCHAR(16),
                duration_min     INT,
                created_at       VARCHAR(32) {now}
            )""",
            f"""CREATE TABLE IF NOT EXISTS ml_models (
                version          VARCHAR(64) PRIMARY KEY,
                strategy_id      VARCHAR(16),
                model_type       VARCHAR(32),
                trained_at       VARCHAR(32),
                samples_count    INT,
                accuracy         {real},
                precision_val    {real},
                recall_val       {real},
                f1_score         {real},
                roc_auc          {real},
                feature_importance_json {tjson},
                hyperparams_json {tjson},
                file_path        TEXT,
                active           TINYINT DEFAULT 0
            )""",
            f"""CREATE TABLE IF NOT EXISTS market_regimes (
                id           {ai},
                timestamp    VARCHAR(32) NOT NULL,
                symbol       VARCHAR(20) NOT NULL,
                regime_id    INT         NOT NULL,
                regime_name  VARCHAR(32),
                features_summary TEXT,
                created_at   VARCHAR(32) {now}
            )""",
            f"""CREATE TABLE IF NOT EXISTS optimization_runs (
                id           {ai},
                run_id       VARCHAR(16) UNIQUE,
                strategy_id  VARCHAR(16),
                symbol       VARCHAR(20),
                started_at   VARCHAR(32),
                completed_at VARCHAR(32),
                best_params_json TEXT,
                best_score   {real},
                method       VARCHAR(32),
                trials       INT
            )""",
        ]
        indexes = [
            ("idx_snap_strategy", "signal_snapshots", "strategy_id"),
            ("idx_snap_outcome",  "signal_snapshots", "outcome"),
            ("idx_snap_timestamp","signal_snapshots", "timestamp"),
        ]

        with self.pool.cursor() as c:
            for ddl in tables:
                c.execute(ddl)
            for idx_name, tbl, col in indexes:
                if self.pool.is_mysql:
                    try:
                        c.execute(f"CREATE INDEX {idx_name} ON {tbl}({col})")
                    except Exception:
                        pass
                else:
                    c.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {tbl}({col})")

        logger.info(f"ML Data Store готов ({'MySQL' if self.pool.is_mysql else self.db_path})")

    # ========== СВЕЧИ (PARQUET) ==========
    def save_candles(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """Сохранение свечей в Parquet (append-friendly)."""
        if df.empty:
            return
        path = self.candles_dir / f"{symbol}_{timeframe}.parquet"
        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
            combined = combined.sort_values("timestamp").reset_index(drop=True)
            combined.to_parquet(path, index=False)
        else:
            df.to_parquet(path, index=False)

    def load_candles(self, symbol: str, timeframe: str,
                     start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        """Загрузка свечей из Parquet."""
        path = self.candles_dir / f"{symbol}_{timeframe}.parquet"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(path)
        if start:
            df = df[df["timestamp"] >= start]
        if end:
            df = df[df["timestamp"] <= end]
        return df.reset_index(drop=True)

    # ========== СИГНАЛЫ И ФИЧИ ==========
    def save_signal_snapshot(self, snapshot: Dict) -> int:
        sql = self.pool.adapt("""
            INSERT INTO signal_snapshots (
                timestamp, strategy_id, symbol, timeframe, action,
                entry_price, stop_loss, take_profit, features_json,
                ml_prediction, ml_confidence, ml_model_version, trade_taken
            ) VALUES (?,?,?,?,?, ?,?,?,?, ?,?,?,?)
        """)
        values = (
            snapshot.get("timestamp", datetime.utcnow().isoformat()),
            snapshot["strategy_id"],
            snapshot["symbol"],
            snapshot.get("timeframe", ""),
            snapshot["action"],
            snapshot["entry_price"],
            snapshot.get("stop_loss"),
            snapshot.get("take_profit"),
            json.dumps(snapshot["features"]),
            snapshot.get("ml_prediction"),
            snapshot.get("ml_confidence"),
            snapshot.get("ml_model_version"),
            1 if snapshot.get("trade_taken") else 0,
        )
        with self.pool.cursor() as c:
            c.execute(sql, values)
            return c.lastrowid

    def update_signal_outcome(self, snapshot_id: int, outcome: str,
                              pnl_r: float, exit_reason: str,
                              duration_min: int, trade_id: Optional[int] = None):
        sql = self.pool.adapt("""
            UPDATE signal_snapshots
            SET outcome=?, pnl_r=?, exit_reason=?, duration_min=?, trade_id=?
            WHERE id=?
        """)
        with self.pool.cursor() as c:
            c.execute(sql, (outcome, pnl_r, exit_reason, duration_min, trade_id, snapshot_id))

    def get_training_data(self, strategy_id: Optional[str] = None,
                          min_samples: int = 50,
                          only_taken_trades: bool = False) -> pd.DataFrame:
        query = "SELECT * FROM signal_snapshots WHERE outcome IS NOT NULL AND outcome != ''"
        params: list = []
        if strategy_id:
            query += " AND strategy_id = ?"
            params.append(strategy_id)
        if only_taken_trades:
            query += " AND trade_taken = 1"

        adapted = self.pool.adapt(query)
        if self.pool.is_mysql:
            with self.pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(adapted, params)
                    cols = [d[0] for d in cur.description]
                    df = pd.DataFrame([[row[c] for c in cols] for row in cur.fetchall()], columns=cols)
        else:
            import sqlite3 as _sq3
            conn = _sq3.connect(self.pool.db_path, check_same_thread=False, timeout=10)
            try:
                cur = conn.cursor()
                cur.execute(adapted, params)
                cols = [d[0] for d in cur.description]
                df = pd.DataFrame(cur.fetchall(), columns=cols)
            finally:
                conn.close()

        if df.empty or len(df) < min_samples:
            return pd.DataFrame()

        def _safe_loads(s):
            try:
                return json.loads(s)
            except Exception:
                return {}
        features_df = pd.json_normalize(df["features_json"].apply(_safe_loads)).reset_index(drop=True)
        df = df.reset_index(drop=True)
        features_df["label"]       = (df["outcome"] == "win").astype(int).values
        features_df["pnl_r"]       = df["pnl_r"].values
        features_df["strategy_id"] = df["strategy_id"].values
        features_df["symbol"]      = df["symbol"].values
        features_df["timestamp"]   = df["timestamp"].values
        features_df["snapshot_id"] = df["id"].values
        return features_df

    # ========== ML МОДЕЛИ ==========
    def register_model(self, version: str, strategy_id: str, model_type: str,
                       metrics: Dict, hyperparams: Dict, feature_importance: Dict,
                       file_path: str, samples_count: int, set_active: bool = True):
        with self.pool.cursor() as c:
            if set_active:
                c.execute(self.pool.adapt("UPDATE ml_models SET active=0 WHERE strategy_id=?"),
                          (strategy_id,))
            upsert = self.pool.adapt("""
                INSERT INTO ml_models (
                    version, strategy_id, model_type, trained_at, samples_count,
                    accuracy, precision_val, recall_val, f1_score, roc_auc,
                    feature_importance_json, hyperparams_json, file_path, active
                ) VALUES (?,?,?,?,?, ?,?,?,?,?, ?,?,?,?)
                AS new_vals ON DUPLICATE KEY UPDATE
                    trained_at=new_vals.trained_at, samples_count=new_vals.samples_count,
                    accuracy=new_vals.accuracy, f1_score=new_vals.f1_score,
                    roc_auc=new_vals.roc_auc, file_path=new_vals.file_path,
                    active=new_vals.active
            """) if self.pool.is_mysql else self.pool.adapt("""
                INSERT OR REPLACE INTO ml_models (
                    version, strategy_id, model_type, trained_at, samples_count,
                    accuracy, precision_val, recall_val, f1_score, roc_auc,
                    feature_importance_json, hyperparams_json, file_path, active
                ) VALUES (?,?,?,?,?, ?,?,?,?,?, ?,?,?,?)
            """)
            c.execute(upsert, (
                version, strategy_id, model_type, datetime.utcnow().isoformat(),
                samples_count,
                metrics.get("accuracy", 0), metrics.get("precision", 0),
                metrics.get("recall", 0), metrics.get("f1", 0), metrics.get("roc_auc", 0),
                json.dumps(feature_importance), json.dumps(hyperparams),
                file_path, 1 if set_active else 0,
            ))
        logger.info(f"Модель {version} зарегистрирована для {strategy_id}")

    def get_active_model(self, strategy_id: str) -> Optional[Dict]:
        sql = self.pool.adapt(
            "SELECT * FROM ml_models WHERE strategy_id=? AND active=1 "
            "ORDER BY trained_at DESC LIMIT 1"
        )
        with self.pool.connection() as conn:
            if self.pool.is_mysql:
                with conn.cursor() as c:
                    c.execute(sql, (strategy_id,))
                    return c.fetchone()
            else:
                import sqlite3
                conn.row_factory = sqlite3.Row
                row = conn.execute(sql, (strategy_id,)).fetchone()
                return dict(row) if row else None

    def list_models(self, strategy_id: Optional[str] = None) -> List[Dict]:
        query = "SELECT * FROM ml_models"
        params: list = []
        if strategy_id:
            query += " WHERE strategy_id=?"
            params.append(strategy_id)
        query += " ORDER BY trained_at DESC"
        with self.pool.connection() as conn:
            if self.pool.is_mysql:
                with conn.cursor() as c:
                    c.execute(self.pool.adapt(query), params)
                    return list(c.fetchall())
            else:
                import sqlite3
                conn.row_factory = sqlite3.Row
                return [dict(r) for r in conn.execute(self.pool.adapt(query), params).fetchall()]

    # ========== СТАТИСТИКА ==========
    def get_ml_stats(self) -> Dict:
        def scalar(conn, sql):
            if self.pool.is_mysql:
                with conn.cursor() as c:
                    c.execute(sql)
                    r = c.fetchone()
                    return list(r.values())[0] if r else 0
            else:
                row = conn.execute(sql).fetchone()
                return row[0] if row is not None else 0

        with self.pool.connection() as conn:
            total   = scalar(conn, "SELECT COUNT(*) FROM signal_snapshots")
            labeled = scalar(conn, "SELECT COUNT(*) FROM signal_snapshots WHERE outcome IS NOT NULL")
            taken   = scalar(conn, "SELECT COUNT(*) FROM signal_snapshots WHERE trade_taken=1")
            wins    = scalar(conn, "SELECT COUNT(*) FROM signal_snapshots WHERE outcome='win'")
            n_mod   = scalar(conn, "SELECT COUNT(*) FROM ml_models")
            act_mod = scalar(conn, "SELECT COUNT(*) FROM ml_models WHERE active=1")

            per_strat_sql = """
                SELECT strategy_id,
                       COUNT(*) as total,
                       SUM(CASE WHEN outcome IS NOT NULL THEN 1 ELSE 0 END) as labeled,
                       SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END) as wins,
                       AVG(CASE WHEN outcome IS NOT NULL THEN pnl_r ELSE NULL END) as avg_r
                FROM signal_snapshots
                GROUP BY strategy_id
            """
            if self.pool.is_mysql:
                with conn.cursor() as c:
                    c.execute(per_strat_sql)
                    rows = c.fetchall()
                per_strategy = [
                    {
                        "strategy_id": r["strategy_id"],
                        "total": r["total"],
                        "labeled": r["labeled"] or 0,
                        "wins": r["wins"] or 0,
                        "wr": round((r["wins"] or 0) / (r["labeled"] or 1) * 100, 2),
                        "avg_r": round(r["avg_r"] or 0, 3),
                    } for r in rows
                ]
            else:
                rows = conn.execute(per_strat_sql).fetchall()
                per_strategy = [
                    {
                        "strategy_id": r[0], "total": r[1],
                        "labeled": r[2] or 0, "wins": r[3] or 0,
                        "wr": round((r[3] or 0) / (r[2] or 1) * 100, 2),
                        "avg_r": round(r[4] or 0, 3),
                    } for r in rows
                ]

        return {
            "total_signals": total,
            "labeled_signals": labeled,
            "trades_taken": taken,
            "wins": wins,
            "overall_wr": round(wins / labeled * 100, 2) if labeled else 0,
            "per_strategy": per_strategy,
            "models_total": n_mod,
            "active_models": act_mod,
        }
