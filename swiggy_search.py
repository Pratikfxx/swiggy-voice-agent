"""Parallel Instamart product search.

Voice turns run under a hard ~22s deadline. When the model searches items one
at a time (each a sequential server-side MCP round trip), a 2-item order already
blows the budget and a recipe cart (~10 items) never finishes. This tool lets
the model fire ONE call with every query and fans them out concurrently against
the Instamart MCP, so N searches cost about the time of one.
"""

import asyncio
import json
import logging

from mcp import ClientSession

from swiggy_auth import get_access_token
from swiggy_mcp import open_authenticated_mcp
from swiggy_scope import ACTIVE_SWIGGY_SERVERS, SERVER_AUTH_KEYS, SWIGGY_SERVER_URLS

_IM_SERVER = ACTIVE_SWIGGY_SERVERS[0]
_IM_URL = SWIGGY_SERVER_URLS[_IM_SERVER]
_IM_TOKEN_KEY = SERVER_AUTH_KEYS[_IM_SERVER]

# Cap the fan-out so a huge list can't open dozens of connections at once.
_MAX_QUERIES = 12


def _search_payload(result) -> dict | None:
    """The first JSON block of a search_products result, or None.

    Swiggy now returns TWO content blocks: human-readable display
    instructions first, then the JSON. Taking block zero and giving up when it
    fails to parse silently emptied every multi-item cart, so scan all blocks.
    """
    for block in (getattr(result, "content", None) or []):
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _products_of(payload: dict) -> list:
    """Products from either response envelope.

    The payload moved from {"data": {"products": [...]}} to a top-level
    {"products": [...]}. Accept both so a rollback either way keeps working.
    """
    products = payload.get("products")
    if isinstance(products, list):
        return products
    return (payload.get("data") or {}).get("products") or []


def _top_match(query: str, result) -> dict:
    """Reduce a search_products result to the single best buyable variation."""
    payload = _search_payload(result)
    if payload is None:
        logging.warning("search_products returned no JSON block for %r", query)
        return {"query": query, "found": False}

    products = _products_of(payload)
    for product in products:
        for var in product.get("variations", []):
            if not var.get("isInStockAndAvailable", True):
                continue
            price = var.get("price") or {}
            return {
                "query": query,
                "found": True,
                "name": var.get("displayName") or product.get("displayName"),
                "brand": var.get("brandName") or product.get("brand"),
                "quantity": var.get("quantityDescription"),
                "price": price.get("offerPrice", price.get("mrp")),
                "skuId": var.get("skuId"),
                "spinId": var.get("spinId"),
            }
    return {"query": query, "found": False}


async def _search_one(session: ClientSession, query: str, address_id: str) -> dict:
    try:
        result = await session.call_tool(
            "search_products",
            {"addressId": address_id, "query": query, "offset": 0},
        )
        return _top_match(query, result)
    except Exception:
        logging.exception("batch search failed for %r", query)
        return {"query": query, "found": False}


async def _batch(queries: list[str], address_id: str) -> list[dict]:
    """One MCP connection, all searches multiplexed over it concurrently.

    Opening a separate streamablehttp connection per query and gathering them
    trips anyio task-group cleanup (especially inside a worker thread's loop).
    A single session multiplexes JSON-RPC requests by id, so the calls still run
    concurrently without N connections.
    """
    token = get_access_token(_IM_TOKEN_KEY)
    async with open_authenticated_mcp(_IM_URL, token) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tasks = [_search_one(session, q, address_id) for q in queries]
            return await asyncio.gather(*tasks)


def _first_text(result) -> str:
    for block in (getattr(result, "content", None) or []):
        t = getattr(block, "text", None)
        if t:
            return t
    return ""


def _tool_errored(result) -> bool:
    """Whether the MCP call itself reported an error, across package versions."""
    for attr in ("is_error", "isError"):
        flag = getattr(result, attr, None)
        if flag is not None:
            return bool(flag)
    return False


def _cart_accepted(result) -> bool:
    """Whether update_cart actually took the items.

    The response carries no "success" field — it returns the resulting cart.
    Treat a non-error call that came back with a cart body as accepted. The
    previous check string-matched '"success": true' against the FIRST text
    block, which is now Swiggy's display-instructions prose, so it read False
    even when the cart had been filled correctly.
    """
    if _tool_errored(result):
        return False
    payload = _search_payload(result)
    return bool(payload) and ("cartId" in payload or "items" in payload)


async def _search_and_cart(
    queries: list[str], address_id: str, quantity: int
) -> tuple[list[dict], bool, str]:
    token = get_access_token(_IM_TOKEN_KEY)
    async with open_authenticated_mcp(_IM_URL, token) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            found = await asyncio.gather(
                *[_search_one(session, q, address_id) for q in queries]
            )
            items = [
                {"spinId": f["spinId"], "skuId": f["skuId"], "quantity": quantity}
                for f in found
                if f.get("found") and f.get("spinId") and f.get("skuId")
            ]
            cart_ok, cart_err = False, ""
            if items:
                try:
                    res = await session.call_tool(
                        "update_cart",
                        {"selectedAddressId": address_id, "items": items},
                    )
                    cart_ok = _cart_accepted(res)
                    if not cart_ok:
                        cart_err = _first_text(res)[:200]
                except Exception as exc:
                    logging.exception("update_cart failed")
                    cart_err = str(exc)[:200]
            return found, cart_ok, cart_err


def search_and_add_to_cart(
    queries: list[str], address_id: str, quantity: int = 1
) -> dict:
    """Search many Instamart items in parallel AND add the best match of each to
    the cart in one deterministic step. Runs in a worker thread, so asyncio.run
    is safe here. Returns what was added (with prices + subtotal) and anything
    not found, so the model can summarize and ask to confirm without chaining
    more tool calls."""
    cleaned = [str(q).strip() for q in (queries or []) if str(q).strip()][:_MAX_QUERIES]
    if not cleaned:
        return {"error": "no queries provided"}
    if not address_id:
        return {"error": "address_id is required"}
    try:
        qty = max(1, int(quantity))
    except (TypeError, ValueError):
        qty = 1

    found, cart_ok, cart_err = asyncio.run(_search_and_cart(cleaned, address_id, qty))
    added = [
        {
            "item": f["query"],
            "name": f.get("name"),
            "quantity": qty,
            "price": f.get("price"),
        }
        for f in found
        if f.get("found")
    ]
    not_found = [f["query"] for f in found if not f.get("found")]
    subtotal = sum((a["price"] or 0) * qty for a in added)
    return {
        "cart_updated": cart_ok,
        "cart_error": cart_err,
        "added": added,
        "not_found": not_found,
        "subtotal": subtotal,
    }
