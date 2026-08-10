"""
Order history compatibility facade.

The public API is intentionally unchanged for existing agent callers, but the
actual persistence lives in store.py so production can use Render Postgres.
"""

from store import get_last_order, get_recent_orders, save_order
