import asyncio
from collections import OrderedDict
import json
import logging
import os
import threading
import time

from mcp import ClientSession

from swiggy_auth import get_access_token
from swiggy_mcp import open_authenticated_mcp
from swiggy_scope import ACTIVE_SWIGGY_SERVERS, SERVER_AUTH_KEYS, SWIGGY_SERVER_URLS


# Fetch addresses through an active-scope server — a token for an inactive
# scope (e.g. food while the product is Instamart-only) goes stale unnoticed.
_ADDRESS_SERVER = ACTIVE_SWIGGY_SERVERS[0]
ADDRESS_URL = SWIGGY_SERVER_URLS[_ADDRESS_SERVER]
ADDRESS_TOKEN_KEY = SERVER_AUTH_KEYS[_ADDRESS_SERVER]
TTL = int(os.getenv("DEFAULT_ADDR_TTL", "600"))

_CACHE_MAX = 1000
_cache: OrderedDict[str, dict] = OrderedDict()
_lock = threading.Lock()
_refreshing: set[str] = set()


def _cache_key(user_id: str) -> str:
    return user_id or "__default__"


def _get_field(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _parse_addresses_result(result) -> dict | None:
    payload = _get_field(result, "structuredContent")
    if not isinstance(payload, dict):
        payload = _get_field(result, "structured_content")

    if not isinstance(payload, dict):
        # Scan every block for the first JSON one. get_addresses returns
        # human-readable prose first ("Found 29 saved addresses (page 1 of
        # 3)"), and taking block zero is what silently emptied every product
        # search when Swiggy added the same prose there.
        payload = None
        for block in _get_field(result, "content", []) or []:
            text = _get_field(block, "text", "")
            if not text:
                continue
            try:
                candidate = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(candidate, dict):
                payload = candidate
                break
        if payload is None:
            return None

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


async def fetch_default_address(user_id: str = "") -> dict | None:
    try:
        token = get_access_token(ADDRESS_TOKEN_KEY, user_id=user_id)
        async with open_authenticated_mcp(ADDRESS_URL, token) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("get_addresses", {})
        return _parse_addresses_result(result)
    except Exception:
        logging.exception("default address fetch failed")
        return None


def get_cached_default(user_id: str = "") -> dict | None:
    with _lock:
        entry = _cache.get(_cache_key(user_id))
        return entry["addr"] if entry else None


async def refresh_default_address(user_id: str = "") -> dict | None:
    addr = await fetch_default_address(user_id)
    if addr:
        with _lock:
            key = _cache_key(user_id)
            _cache.pop(key, None)
            _cache[key] = {"addr": addr, "ts": time.time()}
            while len(_cache) > _CACHE_MAX:
                _cache.popitem(last=False)
        logging.info("default address refreshed: %s (%s)", addr.get("label"), addr.get("id"))
    return addr


def get_default_blocking(user_id: str = "", timeout: float = 4.0) -> dict | None:
    """Cached default address, fetching synchronously if the cache is cold.

    Cold cache means the system prompt carries no address, and the model burns
    a whole conversational turn asking which address to use — on voice that is
    the first thing a caller hears after a deploy. A short blocking fetch is
    cheaper than that turn. Runs in the agent's worker thread, so a fresh
    event loop via asyncio.run is safe here.
    """
    addr = get_cached_default(user_id)
    if addr:
        return addr
    try:
        return asyncio.run(asyncio.wait_for(refresh_default_address(user_id), timeout))
    except Exception:
        logging.exception("blocking address fetch failed")
        return None


def maybe_background_refresh(user_id: str = "") -> None:
    key = _cache_key(user_id)
    try:
        with _lock:
            entry = _cache.get(key) or {"addr": None, "ts": 0.0}
            if key in _refreshing:
                return
            if entry["addr"] is not None and (time.time() - entry["ts"]) < TTL:
                return
            _refreshing.add(key)

        def _worker():
            try:
                asyncio.run(refresh_default_address(user_id))
            except Exception:
                logging.exception("bg address refresh failed")
            finally:
                with _lock:
                    _refreshing.discard(key)

        threading.Thread(target=_worker, daemon=True).start()
    except Exception:
        logging.exception("bg address refresh start failed")
