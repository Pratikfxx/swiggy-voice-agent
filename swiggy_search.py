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

from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

from swiggy_auth import get_access_token
from swiggy_scope import ACTIVE_SWIGGY_SERVERS, SERVER_AUTH_KEYS, SWIGGY_SERVER_URLS

_IM_SERVER = ACTIVE_SWIGGY_SERVERS[0]
_IM_URL = SWIGGY_SERVER_URLS[_IM_SERVER]
_IM_TOKEN_KEY = SERVER_AUTH_KEYS[_IM_SERVER]

# Cap the fan-out so a huge list can't open dozens of connections at once.
_MAX_QUERIES = 12


def _top_match(query: str, result) -> dict:
    """Reduce a search_products result to the single best buyable variation."""
    text = ""
    for block in (getattr(result, "content", None) or []):
        t = getattr(block, "text", None)
        if t:
            text = t
            break
    if not text:
        return {"query": query, "found": False}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {"query": query, "found": False}

    products = (payload.get("data") or {}).get("products") or []
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
    async with streamablehttp_client(
        _IM_URL, headers={"Authorization": f"Bearer {token}"}
    ) as (read, write, _):
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


async def _search_and_cart(
    queries: list[str], address_id: str, quantity: int
) -> tuple[list[dict], bool, str]:
    token = get_access_token(_IM_TOKEN_KEY)
    async with streamablehttp_client(
        _IM_URL, headers={"Authorization": f"Bearer {token}"}
    ) as (read, write, _):
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
                    text = _first_text(res)
                    cart_ok = '"success": true' in text or '"success":true' in text
                    if not cart_ok:
                        cart_err = text[:200]
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
