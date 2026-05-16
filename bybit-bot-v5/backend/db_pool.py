"""
Database connection pool.
Поддерживает MySQL (если задан MYSQL_HOST) и SQLite (по умолчанию).
Все модули бота работают через этот пул — переключение БД без изменения кода.
"""
import os
import threading
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ────── Конфигурация ──────
MYSQL_HOST = os.getenv("MYSQL_HOST", "")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "baibit")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "baibit")
USE_MYSQL = bool(MYSQL_HOST)

if USE_MYSQL:
    try:
        import pymysql
        import pymysql.cursors
        logger.info(f"Режим: MySQL {MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}")
    except ImportError:
        USE_MYSQL = False
        logger.warning("pymysql не установлен — откат на SQLite")

if not USE_MYSQL:
    import sqlite3


class DBPool:
    """
    Thread-safe пул подключений к MySQL или SQLite.
    Singleton per db_path (для SQLite) или per MySQL DSN.
    """

    _instances: dict = {}
    _lock = threading.Lock()

    def __new__(cls, db_path: str = "data/ml_data.db"):
        key = "mysql" if USE_MYSQL else db_path
        with cls._lock:
            if key not in cls._instances:
                inst = super().__new__(cls)
                inst._ready = False
                cls._instances[key] = inst
            return cls._instances[key]

    def __init__(self, db_path: str = "data/ml_data.db"):
        with self.__class__._lock:
            if self._ready:
                return
            self._ready = True  # set early under lock to prevent double-init
            self.db_path = db_path
            self._local = threading.local()

        if USE_MYSQL:
            self._verify_mysql()
        else:
            self._init_sqlite()

    # ── внутренние ───────────────────────────────────────────

    def _verify_mysql(self):
        conn = self._new_mysql()
        conn.close()
        logger.info("MySQL подключение проверено")

    def _new_mysql(self):
        return pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=False,
            connect_timeout=10,
        )

    def _init_sqlite(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.close()
        logger.info(f"SQLite инициализирован: {self.db_path} (WAL)")

    def _get_sqlite(self):
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-64000")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return self._local.conn

    # ── публичный API ─────────────────────────────────────────

    @contextmanager
    def connection(self):
        """Контекст-менеджер: возвращает соединение, автоматически commit/rollback."""
        if USE_MYSQL:
            conn = self._new_mysql()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        else:
            conn = self._get_sqlite()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    @contextmanager
    def cursor(self):
        """Контекст-менеджер: возвращает cursor с авто-commit."""
        with self.connection() as conn:
            if USE_MYSQL:
                cur = conn.cursor()
                try:
                    yield cur
                finally:
                    cur.close()
            else:
                cur = conn.cursor()
                try:
                    yield cur
                finally:
                    cur.close()

    # ── SQL-хелперы ───────────────────────────────────────────

    @property
    def is_mysql(self) -> bool:
        return USE_MYSQL

    @staticmethod
    def ph() -> str:
        """Placeholder: %s для MySQL, ? для SQLite."""
        return "%s" if USE_MYSQL else "?"

    @staticmethod
    def adapt(sql: str) -> str:
        """Заменяет ? на %s в MySQL-режиме."""
        if USE_MYSQL:
            return sql.replace("?", "%s")
        return sql

    @staticmethod
    def ai_pk() -> str:
        """AUTO_INCREMENT PRIMARY KEY для нужной БД."""
        if USE_MYSQL:
            return "INT NOT NULL AUTO_INCREMENT PRIMARY KEY"
        return "INTEGER PRIMARY KEY AUTOINCREMENT"

    @staticmethod
    def now_default() -> str:
        """DEFAULT NOW() / datetime('now')."""
        if USE_MYSQL:
            return "DEFAULT CURRENT_TIMESTAMP"
        return "DEFAULT (datetime('now'))"

    @staticmethod
    def real_type() -> str:
        return "DOUBLE" if USE_MYSQL else "REAL"

    @staticmethod
    def text_json() -> str:
        return "LONGTEXT" if USE_MYSQL else "TEXT"

    def close_all(self):
        if not USE_MYSQL and hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
