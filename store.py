"""
Storage layer for calls and orders that must survive Render restarts.

Render free instances do not keep local disk, so production uses Postgres when
DATABASE_URL is present. Local development and tests use SQLite. Callers should
not know which SQL dialect is active, and webhook flows should never fail just
because storage is temporarily unavailable.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from contextlib import closing, nullcontext
from datetime import datetime, timedelta
from typing import Any


SQLITE_DEFAULT_PATH = "/tmp/swiggy.db"
SESSION_TTL_SECS = 21600
# SQLite has one writer, so this process-wide lock avoids "database is locked" errors.
# ponytail: one lock for every SQLite database this process touches; split into
# per-path locks if a second database ever shares it or write throughput matters.
_WRITE_LOCK = threading.Lock()


def _database_url() -> str:
    return os.getenv("DATABASE_URL", "")


def _is_postgres() -> bool:
    url = _database_url()
    return url.startswith(("postgres://", "postgresql://"))


def _write_guard():
    return _WRITE_LOCK if not _is_postgres() else nullcontext()


def _placeholder() -> str:
    return "%s" if _is_postgres() else "?"


def _sqlite_path() -> str:
    path = os.getenv("ORDER_HISTORY_DB", SQLITE_DEFAULT_PATH)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    return path


def _connect():
    if _is_postgres():
        import psycopg
        from psycopg.rows import dict_row

        return psycopg.connect(_database_url(), row_factory=dict_row)

    conn = sqlite3.connect(_sqlite_path(), timeout=30.0)
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_dict(row: Any) -> dict:
    return dict(row)


def _log_storage_warning(
    action: str, error: Exception | None = None, *, include_error: bool = True
) -> None:
    if include_error and error is not None:
        logging.warning("Storage %s failed; continuing without persistence: %s", action, error)
    else:
        logging.warning("Storage %s failed; continuing without persistence", action)


def _init() -> None:
    id_type = "BIGSERIAL PRIMARY KEY" if _is_postgres() else "INTEGER PRIMARY KEY AUTOINCREMENT"
    try:
        with closing(_connect()) as conn:
            with conn:
                conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS orders (
                        id              {id_type},
                        session_id      TEXT    NOT NULL,
                        order_type      TEXT    NOT NULL,
                        summary         TEXT    NOT NULL,
                        items_json      TEXT    NOT NULL,
                        restaurant_name TEXT    DEFAULT '',
                        total_amount    REAL    DEFAULT 0,
                        placed_at       TEXT    NOT NULL
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_session "
                    "ON orders(session_id, placed_at DESC)"
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id   TEXT PRIMARY KEY,
                        history_json TEXT NOT NULL,
                        updated_at   TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS user_tokens (
                        user_id      TEXT NOT NULL,
                        server_key   TEXT NOT NULL,
                        access_token TEXT NOT NULL,
                        refresh_token TEXT NOT NULL,
                        expires_at   REAL NOT NULL,
                        updated_at   TEXT NOT NULL,
                        PRIMARY KEY (user_id, server_key)
                    )
                    """
                )
    except Exception as error:
        _log_storage_warning("initialization", error)


def _json_items(row: dict) -> dict:
    result = dict(row)
    try:
        result["items"] = json.loads(result.pop("items_json", "[]"))
    except json.JSONDecodeError:
        result["items"] = []
    return result


def _now_iso() -> str:
    return datetime.now().isoformat()


def _session_cutoff_iso() -> str:
    ttl = int(os.getenv("SESSION_TTL_SECS", str(SESSION_TTL_SECS)))
    return (datetime.now() - timedelta(seconds=ttl)).isoformat()


def save_order(
    session_id: str,
    order_type: str,
    summary: str,
    items: list,
    restaurant_name: str = "",
    total_amount: float = 0.0,
) -> None:
    """Persist a completed order without risking the live conversation."""
    ph = _placeholder()
    try:
        with _write_guard(), closing(_connect()) as conn:
            with conn:
                conn.execute(
                    f"""
                    INSERT INTO orders
                        (session_id, order_type, summary, items_json,
                         restaurant_name, total_amount, placed_at)
                    VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})
                    """,
                    (
                        session_id,
                        order_type,
                        summary,
                        json.dumps(items),
                        restaurant_name,
                        total_amount,
                        _now_iso(),
                    ),
                )
    except Exception as error:
        _log_storage_warning("save_order", error)


def get_last_order(session_id: str) -> dict | None:
    """Return the most recent order for this session, or None."""
    ph = _placeholder()
    try:
        with closing(_connect()) as conn:
            row = conn.execute(
                f"SELECT * FROM orders WHERE session_id = {ph} "
                "ORDER BY placed_at DESC LIMIT 1",
                (session_id,),
            ).fetchone()
    except Exception as error:
        _log_storage_warning("get_last_order", error)
        return None
    if not row:
        return None
    return _json_items(_row_to_dict(row))


def get_recent_orders(session_id: str, limit: int = 5) -> list[dict]:
    """Return recent orders, newest first, preserving the legacy response shape."""
    ph = _placeholder()
    try:
        with closing(_connect()) as conn:
            rows = conn.execute(
                f"SELECT * FROM orders WHERE session_id = {ph} "
                f"ORDER BY placed_at DESC LIMIT {ph}",
                (session_id, limit),
            ).fetchall()
    except Exception as error:
        _log_storage_warning("get_recent_orders", error)
        return []
    return [_json_items(_row_to_dict(row)) for row in rows]


def get_session(session_id: str) -> list[dict]:
    """Return persisted conversation history, or an empty history on miss/failure."""
    ph = _placeholder()
    try:
        with closing(_connect()) as conn:
            row = conn.execute(
                f"SELECT history_json FROM sessions WHERE session_id = {ph}",
                (session_id,),
            ).fetchone()
    except Exception as error:
        _log_storage_warning("get_session", error)
        return []
    if not row:
        return []
    try:
        data = _row_to_dict(row)["history_json"]
        history = json.loads(data)
    except (json.JSONDecodeError, TypeError, KeyError) as error:
        _log_storage_warning("decode_session", error)
        return []
    return history if isinstance(history, list) else []


def update_session(session_id: str, history: list[dict]) -> None:
    """Upsert conversation history and opportunistically prune expired sessions."""
    ph = _placeholder()
    now = _now_iso()
    payload = json.dumps(history)
    try:
        with _write_guard(), closing(_connect()) as conn:
            with conn:
                conn.execute(
                    f"""
                    INSERT INTO sessions (session_id, history_json, updated_at)
                    VALUES ({ph}, {ph}, {ph})
                    ON CONFLICT(session_id) DO UPDATE SET
                        history_json = excluded.history_json,
                        updated_at = excluded.updated_at
                    """,
                    (session_id, payload, now),
                )
                conn.execute(
                    f"DELETE FROM sessions WHERE updated_at < {ph}",
                    (_session_cutoff_iso(),),
                )
    except Exception as error:
        _log_storage_warning("update_session", error)


def clear_session(session_id: str) -> None:
    """Delete persisted conversation history for a completed or ended session."""
    ph = _placeholder()
    try:
        with _write_guard(), closing(_connect()) as conn:
            with conn:
                conn.execute(f"DELETE FROM sessions WHERE session_id = {ph}", (session_id,))
    except Exception as error:
        _log_storage_warning("clear_session", error)


def save_user_token(user_id: str, server_key: str, record: dict[str, Any]) -> None:
    """Persist a user's Swiggy token without risking the live conversation."""
    ph = _placeholder()
    try:
        access_token = record.get("access_token")
        refresh_token = record.get("refresh_token") or ""
        expires_at = float(record.get("expires_at", 0))
        if not isinstance(access_token, str) or not access_token:
            raise ValueError("invalid access token record")
        if not isinstance(refresh_token, str):
            refresh_token = ""
        with _write_guard(), closing(_connect()) as conn:
            with conn:
                conn.execute(
                    f"""
                    INSERT INTO user_tokens
                        (user_id, server_key, access_token, refresh_token,
                         expires_at, updated_at)
                    VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph})
                    ON CONFLICT(user_id, server_key) DO UPDATE SET
                        access_token = excluded.access_token,
                        refresh_token = excluded.refresh_token,
                        expires_at = excluded.expires_at,
                        updated_at = excluded.updated_at
                    """,
                    (
                        user_id,
                        server_key,
                        access_token,
                        refresh_token,
                        expires_at,
                        _now_iso(),
                    ),
                )
    except Exception:
        _log_storage_warning("save_user_token", include_error=False)


def get_user_token(user_id: str, server_key: str) -> dict[str, Any] | None:
    """Return a user's Swiggy token record, or None on miss/failure."""
    ph = _placeholder()
    try:
        with closing(_connect()) as conn:
            row = conn.execute(
                f"""
                SELECT access_token, refresh_token, expires_at
                FROM user_tokens
                WHERE user_id = {ph} AND server_key = {ph}
                """,
                (user_id, server_key),
            ).fetchone()
    except Exception:
        _log_storage_warning("get_user_token", include_error=False)
        return None
    return _row_to_dict(row) if row else None


def clear_user_tokens(user_id: str) -> None:
    """Delete all persisted Swiggy tokens for a user."""
    ph = _placeholder()
    try:
        with _write_guard(), closing(_connect()) as conn:
            with conn:
                conn.execute(f"DELETE FROM user_tokens WHERE user_id = {ph}", (user_id,))
    except Exception:
        _log_storage_warning("clear_user_tokens", include_error=False)


_init()
