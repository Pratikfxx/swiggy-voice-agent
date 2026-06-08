"""
Order History — SQLite-backed persistent storage.

Keyed by session_id (phone number for WhatsApp, call SID for voice).
On Railway: mount a volume at /data and set ORDER_HISTORY_DB=/data/orders.db
for persistence across deploys. Falls back to /tmp (resets on restart) locally.
"""

import json
import os
import sqlite3
from datetime import datetime

# ─────────────────────────────────────────────
# DB setup
# ─────────────────────────────────────────────

_default_path = "/data/orders.db"
_fallback_path = "/tmp/orders.db"

def _db_path() -> str:
    preferred = os.getenv("ORDER_HISTORY_DB", _default_path)
    directory = os.path.dirname(preferred)
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
            return preferred
        except OSError:
            pass
    elif not directory or os.path.exists(directory):
        return preferred
    return _fallback_path


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def _init() -> None:
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT    NOT NULL,
                order_type      TEXT    NOT NULL,
                summary         TEXT    NOT NULL,
                items_json      TEXT    NOT NULL,
                restaurant_name TEXT    DEFAULT '',
                total_amount    REAL    DEFAULT 0,
                placed_at       TEXT    NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_session "
            "ON orders(session_id, placed_at DESC)"
        )

_init()


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def save_order(
    session_id: str,
    order_type: str,        # "food" | "grocery" | "dineout"
    summary: str,           # e.g. "Chicken Biryani from Biryani Blues"
    items: list,            # raw items list (dicts with name/qty/price)
    restaurant_name: str = "",
    total_amount: float = 0.0,
) -> None:
    """Persist a completed order for a user session."""
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO orders
                (session_id, order_type, summary, items_json,
                 restaurant_name, total_amount, placed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                order_type,
                summary,
                json.dumps(items),
                restaurant_name,
                total_amount,
                datetime.now().isoformat(),
            ),
        )


def get_last_order(session_id: str) -> dict | None:
    """Return the most recent order for this session, or None."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM orders WHERE session_id = ? "
            "ORDER BY placed_at DESC LIMIT 1",
            (session_id,),
        ).fetchone()
    if not row:
        return None
    result = dict(row)
    result["items"] = json.loads(result.pop("items_json", "[]"))
    return result


def get_recent_orders(session_id: str, limit: int = 5) -> list[dict]:
    """Return the N most recent orders for this session."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM orders WHERE session_id = ? "
            "ORDER BY placed_at DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    orders = []
    for row in rows:
        r = dict(row)
        r["items"] = json.loads(r.pop("items_json", "[]"))
        orders.append(r)
    return orders
