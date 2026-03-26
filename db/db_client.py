"""
db/db_client.py

MySQL connection pool — pure infrastructure, no business logic.

Key design points:
  - Pool is lazy-initialised once, thread-safe via double-checked locking.
  - autocommit=False on the pool — every write uses explicit commit/rollback.
  - execute_query   : single SELECT / INSERT / UPDATE / DELETE.
  - execute_transaction : multiple statements in ONE atomic transaction.
    If any statement fails the whole batch is rolled back.
  - cursor and connection are ALWAYS closed, even on exception.
  - rowcount returned for DML so callers can check how many rows were affected.
"""

import os
import threading
import mysql.connector
from mysql.connector import pooling
from dotenv import load_dotenv

load_dotenv()

_pool      = None
_pool_lock = threading.Lock()


# ── Pool ──────────────────────────────────────────────────────────────────────

def get_pool() -> pooling.MySQLConnectionPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = pooling.MySQLConnectionPool(
                    pool_name         = "shop_pool",
                    pool_size         = 20,
                    pool_reset_session= True,
                    connection_timeout= 10,
                    host              = os.getenv("MYSQL_HOST", "localhost"),
                    port              = int(os.getenv("MYSQL_PORT", 3306)),
                    user              = os.getenv("MYSQL_USER", "root"),
                    password          = os.getenv("MYSQL_PASSWORD", ""),
                    database          = os.getenv("MYSQL_DATABASE", "shopping_assistant"),
                    autocommit        = False,   # explicit commit on every write
                )
    return _pool


def get_connection():
    return get_pool().get_connection()


# ── Single query ──────────────────────────────────────────────────────────────

def execute_query(sql: str, params: tuple = None, fetch: bool = True):
    """
    Execute a single SQL statement.

    fetch=True  → list of row dicts        (SELECT)
    fetch=False → rowcount as int          (INSERT / UPDATE / DELETE)

    Cursor and connection are ALWAYS closed, even on exception.
    """
    conn   = get_connection()
    cursor = None
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql, params or ())
        if fetch:
            return cursor.fetchall()
        else:
            conn.commit()
            return cursor.rowcount          # number of rows affected
    except mysql.connector.Error as exc:
        conn.rollback()
        print(f"[DB ERROR] {exc}\nSQL: {sql}\nParams: {params}")
        raise
    finally:
        if cursor is not None:
            cursor.close()
        conn.close()


# ── Atomic transaction ────────────────────────────────────────────────────────

def execute_transaction(statements: list[tuple]) -> bool:
    """
    Execute multiple SQL statements as a single atomic transaction.

    Args:
        statements: list of (sql, params) tuples executed in order.

    Returns:
        True if ALL statements committed successfully.
        Raises the original exception after rolling back if any statement fails.

    Example:
        execute_transaction([
            ("UPDATE products SET stock = stock - 1 WHERE product_id = %s AND stock > 0",
             (product_id,)),
            ("INSERT INTO orders (order_id, user_id, total_amount, items_json) "
             "VALUES (%s, %s, %s, %s)",
             (order_id, user_id, total, items_json)),
        ])
    """
    conn   = get_connection()
    cursor = None
    try:
        cursor = conn.cursor(dictionary=True)
        conn.start_transaction()

        for sql, params in statements:
            cursor.execute(sql, params or ())

        conn.commit()
        return True

    except Exception as exc:
        conn.rollback()
        print(f"[DB TRANSACTION ERROR] {exc}")
        raise

    finally:
        if cursor is not None:
            cursor.close()
        conn.close()