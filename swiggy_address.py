import asyncio
import json
import logging
import os
import threading
import time

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from swiggy_auth import get_access_token


FOOD_URL = "https://mcp.swiggy.com/food"
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
        token = get_access_token("food")
        async with streamablehttp_client(
            FOOD_URL,
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
