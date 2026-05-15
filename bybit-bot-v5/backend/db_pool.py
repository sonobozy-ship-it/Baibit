"""
Database manager — persistent SQLite connection с пулом.
Решает проблему частых connect/close в горячем цикле.
"""
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DBPool:
    """Thread-safe SQLite connection с retry и WAL режимом."""

    _instances = {}
    _lock = threading.Lock()

    def __new__(cls, db_path: str):
        with cls._lock:
            if db_path not in cls._instances:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instances[db_path] = instance
            return cls._instances[db_path]

    def __init__(self, db_path: str):
        if self._initialized:
            return
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._initialized = True
        # WAL для лучшей concurrency
        conn = sqlite3.connect(db_path, check_same_thread=False, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")  # 64MB
        conn.execute("PRAGMA foreign_keys=ON")
        conn.close()
        logger.info(f"DB pool инициализирован: {db_path} (WAL mode)")

    def _get_conn(self) -> sqlite3.Connection:
        """Thread-local connection (один на поток)."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                self.db_path, check_same_thread=False, timeout=10
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    @contextmanager
    def cursor(self):
        """Context manager для cursor с авто-commit."""
        conn = self._get_conn()
        cur = conn.cursor()
        try:
            yield cur
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"DB error: {e}")
            raise
        finally:
            cur.close()

    @contextmanager
    def connection(self):
        """Прямой доступ к connection (для pd.read_sql и т.п.)."""
        yield self._get_conn()

    def close_all(self):
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
