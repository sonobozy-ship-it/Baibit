"""
Гибридное хранилище:
- SQLite: метаданные сделок, фичи, ML-предсказания
- Parquet: исторические свечи (быстро для ML)
"""
import sqlite3
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging

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
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(candles_dir).mkdir(parents=True, exist_ok=True)
        Path(features_dir).mkdir(parents=True, exist_ok=True)

        self.db_path = db_path
        self.candles_dir = Path(candles_dir)
        self.features_dir = Path(features_dir)
        self._init_db()

    def _init_db(self):
        """Инициализация таблиц SQLite."""
        conn = sqlite3.connect(self.db_path)

        # Снимки сигналов перед открытием
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signal_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                strategy_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT,
                action TEXT NOT NULL,
                entry_price REAL NOT NULL,
                stop_loss REAL,
                take_profit REAL,
                features_json TEXT NOT NULL,
                ml_prediction REAL,
                ml_confidence REAL,
                ml_model_version TEXT,
                trade_taken INTEGER DEFAULT 0,
                trade_id INTEGER,
                outcome TEXT,
                pnl_r REAL,
                exit_reason TEXT,
                duration_min INTEGER,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_snap_strategy ON signal_snapshots(strategy_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_snap_outcome ON signal_snapshots(outcome)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_snap_timestamp ON signal_snapshots(timestamp)")

        # Версии моделей
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ml_models (
                version TEXT PRIMARY KEY,
                strategy_id TEXT,
                model_type TEXT,
                trained_at TEXT,
                samples_count INTEGER,
                accuracy REAL,
                precision_val REAL,
                recall_val REAL,
                f1_score REAL,
                roc_auc REAL,
                feature_importance_json TEXT,
                hyperparams_json TEXT,
                file_path TEXT,
                active INTEGER DEFAULT 0
            )
        """)

        # Рыночные режимы (кластеры)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS market_regimes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                regime_id INTEGER NOT NULL,
                regime_name TEXT,
                features_summary TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)

        # История оптимизаций
        conn.execute("""
            CREATE TABLE IF NOT EXISTS optimization_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT UNIQUE,
                strategy_id TEXT,
                symbol TEXT,
                started_at TEXT,
                completed_at TEXT,
                best_params_json TEXT,
                best_score REAL,
                method TEXT,
                trials INTEGER
            )
        """)

        conn.commit()
        conn.close()
        logger.info(f"ML Data Store initialized: {self.db_path}")

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
        """
        Сохранить снимок сигнала перед открытием сделки.
        Возвращает snapshot_id для последующего обновления outcome.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO signal_snapshots (
                timestamp, strategy_id, symbol, timeframe, action,
                entry_price, stop_loss, take_profit, features_json,
                ml_prediction, ml_confidence, ml_model_version, trade_taken
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
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
        ))
        snapshot_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return snapshot_id

    def update_signal_outcome(self, snapshot_id: int, outcome: str,
                              pnl_r: float, exit_reason: str,
                              duration_min: int, trade_id: Optional[int] = None):
        """Обновить outcome для снимка после закрытия сделки."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            UPDATE signal_snapshots
            SET outcome = ?, pnl_r = ?, exit_reason = ?,
                duration_min = ?, trade_id = ?
            WHERE id = ?
        """, (outcome, pnl_r, exit_reason, duration_min, trade_id, snapshot_id))
        conn.commit()
        conn.close()

    def get_training_data(self, strategy_id: Optional[str] = None,
                          min_samples: int = 50,
                          only_taken_trades: bool = False) -> pd.DataFrame:
        """
        Получить размеченные данные для обучения.
        Возвращает DataFrame с фичами + label (1=win, 0=loss).
        """
        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT * FROM signal_snapshots
            WHERE outcome IS NOT NULL AND outcome != ''
        """
        params = []
        if strategy_id:
            query += " AND strategy_id = ?"
            params.append(strategy_id)
        if only_taken_trades:
            query += " AND trade_taken = 1"

        df = pd.read_sql(query, conn, params=params)
        conn.close()

        if df.empty or len(df) < min_samples:
            return pd.DataFrame()

        # Разворачиваем JSON фичи в колонки
        features_df = pd.json_normalize(df["features_json"].apply(json.loads))

        # Метки: win=1, loss=0
        features_df["label"] = (df["outcome"] == "win").astype(int)
        features_df["pnl_r"] = df["pnl_r"]
        features_df["strategy_id"] = df["strategy_id"]
        features_df["symbol"] = df["symbol"]
        features_df["timestamp"] = df["timestamp"]
        features_df["snapshot_id"] = df["id"]

        return features_df

    # ========== ML МОДЕЛИ ==========
    def register_model(self, version: str, strategy_id: str, model_type: str,
                       metrics: Dict, hyperparams: Dict, feature_importance: Dict,
                       file_path: str, samples_count: int, set_active: bool = True):
        """Регистрация новой обученной модели."""
        conn = sqlite3.connect(self.db_path)
        # Деактивируем старые модели этой стратегии
        if set_active:
            conn.execute("UPDATE ml_models SET active = 0 WHERE strategy_id = ?", (strategy_id,))
        conn.execute("""
            INSERT OR REPLACE INTO ml_models (
                version, strategy_id, model_type, trained_at, samples_count,
                accuracy, precision_val, recall_val, f1_score, roc_auc,
                feature_importance_json, hyperparams_json, file_path, active
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            version, strategy_id, model_type, datetime.utcnow().isoformat(),
            samples_count,
            metrics.get("accuracy", 0),
            metrics.get("precision", 0),
            metrics.get("recall", 0),
            metrics.get("f1", 0),
            metrics.get("roc_auc", 0),
            json.dumps(feature_importance),
            json.dumps(hyperparams),
            file_path,
            1 if set_active else 0,
        ))
        conn.commit()
        conn.close()
        logger.info(f"Модель {version} зарегистрирована для {strategy_id}")

    def get_active_model(self, strategy_id: str) -> Optional[Dict]:
        """Получить активную модель для стратегии."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM ml_models WHERE strategy_id = ? AND active = 1 ORDER BY trained_at DESC LIMIT 1",
            (strategy_id,)
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def list_models(self, strategy_id: Optional[str] = None) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        query = "SELECT * FROM ml_models"
        params = []
        if strategy_id:
            query += " WHERE strategy_id = ?"
            params.append(strategy_id)
        query += " ORDER BY trained_at DESC"
        rows = [dict(r) for r in conn.execute(query, params).fetchall()]
        conn.close()
        return rows

    # ========== СТАТИСТИКА ==========
    def get_ml_stats(self) -> Dict:
        """Общая статистика ML-системы."""
        conn = sqlite3.connect(self.db_path)
        total_signals = conn.execute("SELECT COUNT(*) FROM signal_snapshots").fetchone()[0]
        labeled = conn.execute(
            "SELECT COUNT(*) FROM signal_snapshots WHERE outcome IS NOT NULL"
        ).fetchone()[0]
        taken = conn.execute(
            "SELECT COUNT(*) FROM signal_snapshots WHERE trade_taken = 1"
        ).fetchone()[0]
        wins = conn.execute(
            "SELECT COUNT(*) FROM signal_snapshots WHERE outcome = 'win'"
        ).fetchone()[0]

        per_strategy = conn.execute("""
            SELECT strategy_id,
                   COUNT(*) as total,
                   SUM(CASE WHEN outcome IS NOT NULL THEN 1 ELSE 0 END) as labeled,
                   SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                   AVG(CASE WHEN outcome IS NOT NULL THEN pnl_r ELSE NULL END) as avg_r
            FROM signal_snapshots
            GROUP BY strategy_id
        """).fetchall()

        models_count = conn.execute("SELECT COUNT(*) FROM ml_models").fetchone()[0]
        active_models = conn.execute("SELECT COUNT(*) FROM ml_models WHERE active = 1").fetchone()[0]

        conn.close()

        return {
            "total_signals": total_signals,
            "labeled_signals": labeled,
            "trades_taken": taken,
            "wins": wins,
            "overall_wr": round(wins / labeled * 100, 2) if labeled else 0,
            "per_strategy": [
                {
                    "strategy_id": r[0],
                    "total": r[1],
                    "labeled": r[2],
                    "wins": r[3],
                    "wr": round(r[3] / r[2] * 100, 2) if r[2] else 0,
                    "avg_r": round(r[4], 3) if r[4] is not None else 0,
                } for r in per_strategy
            ],
            "models_total": models_count,
            "active_models": active_models,
        }
