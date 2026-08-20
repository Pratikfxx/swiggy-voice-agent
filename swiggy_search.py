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
import re

from mcp import ClientSession

from swiggy_auth import get_access_token
from swiggy_mcp import open_authenticated_mcp
from swiggy_scope import ACTIVE_SWIGGY_SERVERS, SERVER_AUTH_KEYS, SWIGGY_SERVER_URLS

_IM_SERVER = ACTIVE_SWIGGY_SERVERS[0]
_IM_URL = SWIGGY_SERVER_URLS[_IM_SERVER]
_IM_TOKEN_KEY = SERVER_AUTH_KEYS[_IM_SERVER]

# Cap the fan-out so a huge list can't open dozens of connections at once.
_MAX_QUERIES = 12


_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12,
}
_PACK_UNITS = (
    r"(?:pack|packs|packet|packets|can|cans|bottle|bottles|box|boxes|piece|pieces"
    r"|tin|tins|litre|litres|liter|liters|kg|kgs|gram|grams|ml|dozen)"
)
# "six pack of diet coke" searched literally returns Red Bull, and
# "diet coke 6 pack" returns a prebiotic soda: the extra words wreck Swiggy's
# relevance ranking. Strip the count off the query and remember it separately.
# A bare leading number is usually part of the product — "6 seed bread",
# "7up", "5 star" — so only treat it as a count when a unit or "of" follows.
_LEADING_QTY = re.compile(
    rf"^\s*(?P<count>\d+|{'|'.join(_NUMBER_WORDS)})\s+"
    rf"(?:(?:{_PACK_UNITS})\b\s*(?:of\s+)?|of\s+)",
    re.I,
)
_TRAILING_QTY = re.compile(rf"\s*(?P<count>\d+)\s*(?:{_PACK_UNITS})\b\s*$", re.I)


def _split_pack_size(query: str) -> tuple[str, int | None]:
    """("six pack of diet coke") -> ("diet coke", 6). No count -> (query, None)."""
    text = (query or "").strip()
    count = None

    match = _TRAILING_QTY.search(text)
    if match:
        count = int(match.group("count"))
        text = text[: match.start()].strip()

    match = _LEADING_QTY.match(text)
    if match and match.group("count"):
        raw = match.group("count").lower()
        parsed = _NUMBER_WORDS.get(raw)
        if parsed is None:
            try:
                parsed = int(raw)
            except ValueError:
                parsed = None
        remainder = text[match.end():].strip()
        # Only strip when something is left — "6" alone is the product.
        if parsed is not None and remainder:
            count = count or parsed
            text = remainder

    return (text or query).strip(), count


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


def _top_match(query: str, result, pack_size: int | None = None) -> dict:
    """Reduce a search_products result to the single best buyable variation."""
    payload = _search_payload(result)
    if payload is None:
        logging.warning("search_products returned no JSON block for %r", query)
        return {"query": query, "found": False}

    products = _products_of(payload)
    for product in products:
        # Swiggy lists a product's variations largest-pack-first, so taking the
        # first one put a six-pack of Diet Coke (Rs259) in the cart when the
        # caller said "diet coke". Stay on this product — it is the most
        # relevant match — but take its cheapest in-stock pack.
        in_stock = [
            var for var in product.get("variations", [])
            if var.get("isInStockAndAvailable", True)
        ]
        if not in_stock:
            continue

        def _pack_price(var):
            price = var.get("price") or {}
            value = price.get("offerPrice", price.get("mrp"))
            try:
                return float(value)
            except (TypeError, ValueError):
                return float("inf")

        chosen = min(in_stock, key=_pack_price)
        if pack_size and pack_size > 1:
            # "six pack of diet coke" should get the six-pack, not the cheapest
            # single. Variations describe themselves as e.g. "330 ml x 6".
            wanted = re.compile(rf"x\s*{pack_size}\b", re.I)
            matching = [
                var for var in in_stock
                if wanted.search(str(var.get("quantityDescription") or ""))
            ]
            if matching:
                chosen = min(matching, key=_pack_price)

        for var in [chosen]:
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
    # Search the product, not the phrasing: "six pack of diet coke" sent
    # verbatim came back as Red Bull.
    term, pack_size = _split_pack_size(query)
    try:
        result = await session.call_tool(
            "search_products",
            {"addressId": address_id, "query": term, "offset": 0},
        )
        return _top_match(query, result, pack_size)
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


async def _existing_cart_items(session: ClientSession) -> list[dict]:
    """The cart as update_cart wants it back: spinId, skuId, quantity.

    update_cart REPLACES the whole cart with whatever it is given, so adding a
    second item without resending the first silently deletes it. Read what is
    already there and merge.
    """
    try:
        res = await session.call_tool("get_cart", {})
    except Exception:
        logging.exception("get_cart failed; adding without merge would drop the cart")
        return []
    if _tool_errored(res):
        return []
    payload = _search_payload(res) or {}
    items = []
    for entry in payload.get("items") or []:
        sku, spin = entry.get("skuId"), entry.get("spinId")
        if not sku or not spin:
            continue
        try:
            qty = int(entry.get("quantity") or 1)
        except (TypeError, ValueError):
            qty = 1
        items.append({"spinId": spin, "skuId": sku, "quantity": max(qty, 1)})
    return items


def _merge_cart_items(existing: list[dict], additions: list[dict]) -> list[dict]:
    """Existing items first, new ones appended, deduplicated by skuId."""
    merged = list(existing)
    seen = {item["skuId"] for item in merged}
    for item in additions:
        if item["skuId"] in seen:
            continue
        merged.append(item)
        seen.add(item["skuId"])
    return merged


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
                    items = _merge_cart_items(
                        await _existing_cart_items(session), items
                    )
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


def _cart_summary(items: list[dict]) -> dict:
    return {
        "items": [
            {"name": i.get("itemName"), "variant": i.get("itemVariant"),
             "quantity": i.get("quantity"), "price": i.get("discountedFinalPrice", i.get("mrp"))}
            for i in items
        ],
        "subtotal": sum(
            float(i.get("discountedFinalPrice") or i.get("mrp") or 0) * int(i.get("quantity") or 1)
            for i in items
        ),
    }


def _matches(name: str, term: str) -> bool:
    """Whole-word match of a caller's term against a catalogue name.

    Substring matching removed a Monster Energy because the caller said
    "sugar" and the product is "Monster Energy Ultra Zero Sugar350ml".
    Word boundaries make that a near-miss instead of a false positive, but
    catalogue names are still unreliable — "cinnamon" never matches
    "Supreme Harvest Cassia (Taj) Roll" — which is why keep-lists exist.
    """
    words = [w for w in re.split(r"[^a-z0-9]+", term.lower()) if w]
    if not words:
        return False
    lowered = name.lower()
    return all(re.search(rf"\b{re.escape(w)}", lowered) for w in words)


def _partition_cart(
    lines: list[dict], remove: list[str], keep_only: list[str]
) -> tuple[list[dict], list[dict]]:
    """Split cart lines into (keep, drop). Pure, so it is testable offline."""
    keep, dropped = [], []
    for line in lines:
        name = str(line.get("itemName") or "")
        if keep_only:
            hit = any(_matches(name, term) for term in keep_only)
        else:
            hit = not any(_matches(name, term) for term in remove)
        (keep if hit else dropped).append(line)
    return keep, dropped


async def _remove_from_cart(
    remove: list[str], keep_only: list[str], address_id: str
) -> dict:
    """Drop or retain cart lines by name, then write the cart back.

    Doing this through the model meant it had to read the whole cart and echo
    back every skuId it wanted to keep — a 23.6s turn that blew the voice
    deadline. Prefer keep_only: naming the two things to keep is far more
    reliable than naming the six to drop.
    """
    remove = [t.strip() for t in (remove or []) if str(t).strip()]
    keep_only = [t.strip() for t in (keep_only or []) if str(t).strip()]
    if not remove and not keep_only:
        return {"error": "name the items to remove, or the items to keep"}

    token = get_access_token(_IM_TOKEN_KEY)
    async with open_authenticated_mcp(_IM_URL, token) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            try:
                cart = await session.call_tool("get_cart", {})
            except Exception as exc:
                logging.exception("get_cart failed during removal")
                return {"error": str(exc)[:200], "cart_updated": False}

            lines = (_search_payload(cart) or {}).get("items") or []
            keep, dropped = _partition_cart(lines, remove, keep_only)

            # A keep-list that matches nothing must never mean "delete
            # everything". "keep only the drinks" matched no product — they are
            # named "Monster Energy Ultra Zero Sugar" and "Diet Coke Can" — and
            # emptied the cart. Refuse, and say what is actually in there.
            if keep_only and not keep:
                return {
                    "cart_unchanged": True,
                    "removed": [],
                    "still_in_cart": [i.get("itemName") for i in lines],
                    "error": (
                        "NOTHING WAS REMOVED and the cart is NOT empty. The "
                        "keep_only terms matched no product name — a category "
                        "word like 'drinks' will not match. Tell the user the "
                        "cart is unchanged, read out still_in_cart, and ask "
                        "which of those to keep. Do NOT say the cart is empty."
                    ),
                }

            if not dropped:
                return {"removed": [], "kept": [i.get("itemName") for i in keep],
                        "cart_updated": False, "error": "nothing in the cart matched"}

            items = [
                {"spinId": i.get("spinId"), "skuId": i.get("skuId"),
                 "quantity": int(i.get("quantity") or 1)}
                for i in keep if i.get("spinId") and i.get("skuId")
            ]
            try:
                if items:
                    res = await session.call_tool(
                        "update_cart", {"selectedAddressId": address_id, "items": items}
                    )
                    ok = _cart_accepted(res)
                else:
                    res = await session.call_tool("clear_cart", {})
                    ok = not _tool_errored(res)
            except Exception as exc:
                logging.exception("update_cart failed during removal")
                return {"error": str(exc)[:200], "cart_updated": False}

            summary = _cart_summary(keep)
            return {
                "removed": [i.get("itemName") for i in dropped],
                "kept": summary["items"],
                "subtotal": round(summary["subtotal"]),
                "cart_updated": ok,
            }


def remove_from_cart(
    address_id: str,
    remove: list[str] | None = None,
    keep_only: list[str] | None = None,
) -> dict:
    """Sync wrapper. Runs in a worker thread, so asyncio.run is safe here."""
    if not address_id:
        return {"error": "address_id is required"}
    return asyncio.run(_remove_from_cart(remove or [], keep_only or [], address_id))
