import asyncio
import json
import logging
import os
import threading
import time

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from swiggy_auth import get_access_token
from swiggy_scope import ACTIVE_SWIGGY_SERVERS, SERVER_AUTH_KEYS, SWIGGY_SERVER_URLS


# Fetch addresses through an active-scope server — a token for an inactive
# scope (e.g. food while the product is Instamart-only) goes stale unnoticed.
_ADDRESS_SERVER = ACTIVE_SWIGGY_SERVERS[0]
ADDRESS_URL = SWIGGY_SERVER_URLS[_ADDRESS_SERVER]
ADDRESS_TOKEN_KEY = SERVER_AUTH_KEYS[_ADDRESS_SERVER]
TTL = int(os.getenv("DEFAULT_ADDR_TTL", "600"))

_cache = {"addr": None, "ts": 0.0}
_lock = threading.Lock()
_refreshing = False


def _get_field(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _parse_addresses_result(result) -> dict | None:
    payload = _get_field(result, "structuredContent")
    if not isinstance(payload, dict):
        payload = _get_field(result, "structured_content")

    if not isinstance(payload, dict):
        content = _get_field(result, "content", []) or []
        first = content[0] if content else None
        text = _get_field(first, "text", "")
        if not text:
            return None
        payload = json.loads(text)

    addresses = payload.get("addresses") if isinstance(payload, dict) else None
    if not addresses:
        return None

    address = addresses[0]
    if not isinstance(address, dict):
        return None

    address_id = address.get("id")
    if address_id is None:
        return None

    return {
        "id": str(address_id),
        "label": address.get("addressTag") or address.get("addressCategory") or "Home",
        "area": str(address.get("addressLine", ""))[:80],
    }


async def fetch_default_address() -> dict | None:
    try:
        token = get_access_token(ADDRESS_TOKEN_KEY)
        async with streamablehttp_client(
            ADDRESS_URL,
            headers={"Authorization": f"Bearer {token}"},
        ) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("get_addresses", {})
        return _parse_addresses_result(result)
    except Exception:
        logging.exception("default address fetch failed")
        return None


def get_cached_default() -> dict | None:
    with _lock:
        return _cache["addr"]


async def refresh_default_address() -> dict | None:
    addr = await fetch_default_address()
    if addr:
        with _lock:
            _cache["addr"] = addr
            _cache["ts"] = time.time()
        logging.info("default address refreshed: %s (%s)", addr.get("label"), addr.get("id"))
    return addr


def get_default_blocking(timeout: float = 4.0) -> dict | None:
    """Cached default address, fetching synchronously if the cache is cold.

    Cold cache means the system prompt carries no address, and the model burns
    a whole conversational turn asking which address to use — on voice that is
    the first thing a caller hears after a deploy. A short blocking fetch is
    cheaper than that turn. Runs in the agent's worker thread, so a fresh
    event loop via asyncio.run is safe here.
    """
    addr = get_cached_default()
    if addr:
        return addr
    try:
        return asyncio.run(asyncio.wait_for(refresh_default_address(), timeout))
    except Exception:
        logging.exception("blocking address fetch failed")
        return None


def maybe_background_refresh() -> None:
    global _refreshing
    try:
        with _lock:
            ts = _cache["ts"]
            if _refreshing:
                return
            if _cache["addr"] is not None and (time.time() - ts) < TTL:
                return
            _refreshing = True

        def _worker():
            global _refreshing
            try:
                asyncio.run(refresh_default_address())
            except Exception:
                logging.exception("bg address refresh failed")
            finally:
                with _lock:
                    _refreshing = False

        threading.Thread(target=_worker, daemon=True).start()
    except Exception:
        logging.exception("bg address refresh start failed")
