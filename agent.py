"""
Swiggy Voice Agent — Core Claude Agent

Handles multi-turn conversations across voice and chat surfaces.
Uses Claude with tool-calling for demo mode and Anthropic MCP connectors for live mode.

Surface modes:
  - "voice": concise responses, max 3 items, spoken naturally
  - "chat": rich markdown, tables, full lists
"""

import json
import os
import re
import logging

import anthropic

from recipe_engine import get_recipe_ingredients as _get_recipe_ingredients
from order_history import save_order, get_recent_orders
import swiggy_address
from swiggy_auth import get_all_access_tokens
from swiggy_tools import (
    get_saved_address,
    search_food_restaurants,
    get_restaurant_menu,
    search_instamart_products,
    place_food_order_mock,
    place_instamart_order_mock,
    search_dineout_restaurants,
    get_dineout_slots,
    book_dineout_table_mock,
)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
AGENT_MODEL = os.getenv("AGENT_MODEL", "claude-sonnet-4-6")
VOICE_MODEL = os.getenv("VOICE_MODEL", "claude-haiku-4-5")
VOICE_API_TIMEOUT_SECS = float(os.getenv("VOICE_API_TIMEOUT_SECS", "7.0"))
CHAT_API_TIMEOUT_SECS = float(os.getenv("CHAT_API_TIMEOUT_SECS", "30.0"))
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
DEFAULT_ADDRESS_ID = os.getenv("DEFAULT_ADDRESS_ID", "")
DEFAULT_ADDRESS_LABEL = os.getenv("DEFAULT_ADDRESS_LABEL", "Home")
DEFAULT_ADDRESS_AREA = os.getenv("DEFAULT_ADDRESS_AREA", "")
CONFIRM_RE = re.compile(
    r"\b(yes|yeah|yep|yup|haan|haa|ha|confirm(ed)?|theek hai|thik hai|place (it|the order)|order it|book it|go ahead|pakka|kar do|kardo|do it|sure|okay|ok|proceed)\b",
    re.I,
)

SWIGGY_SERVERS = {
    "swiggy-food": "https://mcp.swiggy.com/food",
    "swiggy-instamart": "https://mcp.swiggy.com/im",
    "swiggy-dineout": "https://mcp.swiggy.com/dineout",
}
MCP_AUTH_KEYS = {
    "swiggy-food": "food",
    "swiggy-instamart": "im",
    "swiggy-dineout": "dineout",
}
SPEND_TOOLS = {
    "swiggy-food": ["place_food_order"],
    "swiggy-instamart": ["checkout"],
    "swiggy-dineout": ["book_table"],
}


def _route_servers(user_message: str, surface: str) -> list[str]:
    """Return which Swiggy MCP servers to attach.

    Product mode is temporarily Instamart-only, so do not attach Food or Dineout.
    """
    return ["swiggy-instamart"]


def _is_confirmation(text):
    return bool(CONFIRM_RE.search(text or ""))

# ─────────────────────────────────────────────
# Tool definitions — Claude sees these
# ─────────────────────────────────────────────

TOOLS = [
    {
        "name": "get_user_address",
        "description": "Get the user's saved home address. Call this first if you need to confirm delivery location.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "search_food_restaurants",
        "description": (
            "Search Swiggy Food for restaurants that serve a specific dish or cuisine. "
            "Returns top 3 restaurants with name, rating, delivery time, distance, and active offers. "
            "Use for READY-MADE food delivery — not for buying raw ingredients."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Dish name or cuisine, e.g. 'chicken biryani', 'pizza', 'south indian'"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_restaurant_menu",
        "description": "Get menu items from a specific restaurant by its ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "restaurant_id": {"type": "string"},
                "dish_query": {"type": "string", "description": "Optional: filter by dish name"}
            },
            "required": ["restaurant_id"]
        }
    },
    {
        "name": "get_recipe_ingredients",
        "description": (
            "Get the list of raw ingredients needed to COOK a dish at home. "
            "Use ONLY when the user says things like 'items for X', 'ingredients for X', "
            "'I want to cook X', or 'what do I need to make X'. "
            "Do NOT use this for ordering ready-made food."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dish_name": {
                    "type": "string",
                    "description": "Name of the dish to cook, e.g. 'alfredo pasta', 'chicken biryani'"
                }
            },
            "required": ["dish_name"]
        }
    },
    {
        "name": "search_grocery_product",
        "description": (
            "Search Swiggy Instamart for a specific grocery product. "
            "Use for milk, eggs, bread, vegetables, spices, and any raw ingredient. "
            "Call once per product. Returns product name, price, brand, and delivery time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "product_name": {
                    "type": "string",
                    "description": "Product name, e.g. 'milk', 'eggs', 'parmesan cheese'"
                },
                "quantity_hint": {
                    "type": "string",
                    "description": "Quantity needed e.g. '1 liter', '6 pieces', '200g'",
                    "default": "1"
                }
            },
            "required": ["product_name"]
        }
    },
    {
        "name": "place_food_order",
        "description": (
            "Place a Swiggy Food order. Call ONLY after user explicitly confirms. "
            "Never call without a 'yes', 'confirm', 'haan', or 'theek hai' from the user."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "restaurant_id": {"type": "string"},
                "restaurant_name": {"type": "string"},
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "item_id": {"type": "string"},
                            "name": {"type": "string"},
                            "price": {"type": "number"},
                            "quantity": {"type": "integer"}
                        }
                    }
                },
                "total_amount": {"type": "number"}
            },
            "required": ["restaurant_id", "restaurant_name", "items", "total_amount"]
        }
    },
    {
        "name": "place_grocery_order",
        "description": (
            "Place an Instamart grocery order. Call ONLY after user explicitly confirms. "
            "Never call without a 'yes', 'confirm', 'haan', or 'theek hai' from the user."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "product_id": {"type": "string"},
                            "name": {"type": "string"},
                            "price": {"type": "number"},
                            "quantity": {"type": "integer"}
                        }
                    }
                },
                "total_amount": {"type": "number"}
            },
            "required": ["items", "total_amount"]
        }
    },
    {
        "name": "search_dineout_restaurants",
        "description": (
            "Search Swiggy Dineout for restaurants to DINE IN — sit-down meals, date nights, "
            "family dinners, celebrations. Use when user wants to GO OUT and book a table. "
            "NOT for food delivery. Returns top restaurants with deals and ratings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Cuisine or vibe: 'italian', 'chinese', 'rooftop', 'romantic', 'family'"
                },
                "guests": {"type": "integer", "description": "Number of people", "default": 2},
                "time_pref": {"type": "string", "description": "e.g. 'tonight 8pm', 'tomorrow lunch'", "default": "tonight"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_dineout_slots",
        "description": "Get available table booking slots for a specific Dineout restaurant.",
        "input_schema": {
            "type": "object",
            "properties": {
                "restaurant_id": {"type": "string"},
                "date": {"type": "string", "default": "today"}
            },
            "required": ["restaurant_id"]
        }
    },
    {
        "name": "book_dineout_table",
        "description": (
            "Book a table at a Dineout restaurant. Call ONLY after user confirms the restaurant, "
            "slot time, and number of guests. Never book without explicit confirmation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "restaurant_id": {"type": "string"},
                "restaurant_name": {"type": "string"},
                "slot_id": {"type": "string"},
                "slot_time": {"type": "string"},
                "guests": {"type": "integer"}
            },
            "required": ["restaurant_id", "restaurant_name", "slot_id", "slot_time", "guests"]
        }
    },
    {
        "name": "get_order_history",
        "description": (
            "Retrieve the user's past orders. Call this when the user says things like "
            "'order my usual', 'same as last time', 'repeat my order', 'what did I order last', "
            "'order again'. Returns the last 5 orders with items and restaurant details. "
            "Use the result to re-place the most recent order (or the one the user specifies) "
            "after confirming with them."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of past orders to retrieve (default 5, max 10)",
                    "default": 5
                }
            },
            "required": []
        }
    }
]

LIVE_LOCAL_TOOLS = [
    tool for tool in TOOLS
    if tool["name"] in {"get_recipe_ingredients", "get_order_history"}
]
LOCAL_NAMES = {tool["name"] for tool in LIVE_LOCAL_TOOLS}


# ─────────────────────────────────────────────
# Tool executor — maps tool names → functions
# ─────────────────────────────────────────────

def execute_tool(tool_name: str, tool_input: dict, session_id: str = "") -> str:
    """Execute a tool call and return JSON string result."""
    try:
        if tool_name == "get_user_address":
            result = get_saved_address()

        elif tool_name == "search_food_restaurants":
            result = search_food_restaurants(tool_input["query"])

        elif tool_name == "get_restaurant_menu":
            result = get_restaurant_menu(
                tool_input["restaurant_id"],
                tool_input.get("dish_query", "")
            )

        elif tool_name == "get_recipe_ingredients":
            result = _get_recipe_ingredients(tool_input["dish_name"])

        elif tool_name == "search_grocery_product":
            result = search_instamart_products(
                tool_input["product_name"],
                tool_input.get("quantity_hint", "1")
            )

        elif tool_name == "place_food_order":
            result = place_food_order_mock(
                tool_input["restaurant_id"],
                tool_input["items"]
            )
            # Auto-save to order history on success
            if session_id and result.get("success"):
                items = tool_input.get("items", [])
                summary = (
                    f"{', '.join(i.get('name','') for i in items[:3])}"
                    f" from {tool_input.get('restaurant_name', '')}"
                )
                save_order(
                    session_id=session_id,
                    order_type="food",
                    summary=summary,
                    items=items,
                    restaurant_name=tool_input.get("restaurant_name", ""),
                    total_amount=tool_input.get("total_amount", 0),
                )

        elif tool_name == "place_grocery_order":
            result = place_instamart_order_mock(tool_input["items"])
            # Auto-save to order history on success
            if session_id and result.get("success"):
                items = tool_input.get("items", [])
                summary = (
                    f"Groceries: {', '.join(i.get('name','') for i in items[:4])}"
                    + (" & more" if len(items) > 4 else "")
                )
                save_order(
                    session_id=session_id,
                    order_type="grocery",
                    summary=summary,
                    items=items,
                    total_amount=tool_input.get("total_amount", 0),
                )

        elif tool_name == "search_dineout_restaurants":
            result = search_dineout_restaurants(
                tool_input["query"],
                tool_input.get("guests", 2),
                tool_input.get("time_pref", "tonight")
            )

        elif tool_name == "get_dineout_slots":
            result = get_dineout_slots(
                tool_input["restaurant_id"],
                tool_input.get("date", "today")
            )

        elif tool_name == "book_dineout_table":
            result = book_dineout_table_mock(
                tool_input["restaurant_id"],
                tool_input["restaurant_name"],
                tool_input["slot_id"],
                tool_input["slot_time"],
                tool_input["guests"]
            )
            # Auto-save dineout booking on success
            if session_id and result.get("success"):
                save_order(
                    session_id=session_id,
                    order_type="dineout",
                    summary=(
                        f"Table for {tool_input.get('guests')} "
                        f"at {tool_input.get('restaurant_name')} "
                        f"@ {tool_input.get('slot_time')}"
                    ),
                    items=[],
                    restaurant_name=tool_input.get("restaurant_name", ""),
                )

        elif tool_name == "get_order_history":
            limit = min(int(tool_input.get("limit", 5)), 10)
            orders = get_recent_orders(session_id, limit=limit) if session_id else []
            if not orders:
                result = {"orders": [], "message": "No previous orders found for this user."}
            else:
                result = {"orders": orders}

        else:
            result = {"error": f"Unknown tool: {tool_name}"}

        return json.dumps(result)

    except Exception as e:
        return json.dumps({"error": str(e), "tool": tool_name})


# ─────────────────────────────────────────────
# System prompts — voice vs chat
# ─────────────────────────────────────────────

VOICE_SYSTEM_PROMPT = """You are Swiggy's Instamart-only voice ordering assistant on a live phone call.

CURRENT SCOPE:
- Instamart-only for now: groceries, snacks, drinks, household essentials, personal care, packaged foods, and ingredients.
- Do not offer cooked meal delivery or reservations.
- If the user asks for cooked meals or reservations, say: "I can help with Instamart items right now. Want groceries, drinks, snacks, or essentials instead?"

VOICE RULES — NON-NEGOTIABLE:
- Plain spoken English only. Zero emojis, markdown, bullets, or symbols.
- Use 1-2 short natural sentences, usually under 35 words. Be warm, not robotic.
- Give enough context to decide: store, main item or brand, quantity, total price, and ETA when available.
- Never narrate what you're doing ("Let me search...", "I found 3 options...") unless the search is slow or ambiguous.
- Do not say "I'll search", "Let me search", or similar process narration.
- For simple grocery requests, pick clear top matches and ask "Add these?" instead of making the user choose brands.
- Never read order IDs, tracking codes, or technical fields aloud.
- Prices: "469 rupees" not "₹469". Times: "30 minutes" not "30 mins".

ORDER FLOW:
1. User says what they want → search Instamart → give ONE strong top result or a small cart.
2. Confirm with one natural sentence → wait for yes/no.
3. On yes → checkout → say done in one sentence → hang up.
- Always use saved address. Never ask for it.
- Only place after: yes, haan, okay, confirm, theek hai.

CONFIRMATION:
- Grocery: "[Main items], [total] rupees, [ETA] on Instamart. Confirm?"
- Recipe cart: "[Main ingredients], [total] rupees. Order them?"
- If there are more than 3 grocery items, summarize the count and name the most important 2-3 items.
- Do not read individual prices unless the user asks.

AFTER ORDER (one sentence max):
"Done! Arriving in [ETA]."

REPEAT ORDER:
- "order my usual" / "phir se" → call get_order_history → repeat only Instamart orders. If the last order was not Instamart, ask what grocery or essential they want instead.

LANGUAGE: Match whatever the user speaks — Hindi, Hinglish, English all fine.
"""

CHAT_SYSTEM_PROMPT = """You are Swiggy's Instamart-only AI ordering assistant for chat/WhatsApp.

## Your mission
Help users order Instamart groceries, snacks, drinks, household essentials, personal care, packaged foods, and ingredients conversationally. Be helpful, clear, and efficient.

## Current scope
- Instamart-only for now.
- Do not offer cooked meal delivery or reservations.
- If the user asks for cooked meals or reservations, say you can only help with Instamart right now and offer close grocery/snack/ingredient alternatives.

## Chat response rules
- Use markdown formatting (bold, tables, bullet points)
- Show a focused Instamart cart as a table with item, qty, price, and ETA when available
- Always confirm before placing order: "Ready to place? Reply **yes** to confirm."
- After confirmation → place order → send confirmation with order ID

## Intent detection
- "get me milk, eggs, bread" → search each on Instamart → show cart total → confirm → place
- "items for alfredo pasta" → get_recipe_ingredients → search each on Instamart → show full cart → confirm → place
- Cooked meal or reservation requests → explain Instamart-only scope and offer ingredients, snacks, drinks, or essentials instead
- Mixed requests → handle only the Instamart items and clearly say which non-Instamart parts cannot be handled right now

## Repeat orders
- Triggers: "order my usual", "same as last time", "repeat my order", "what did I order last", "order again"
- Call get_order_history to look up past orders.
- If Instamart orders exist: show a summary of the last 1–3 Instamart orders and ask which one to repeat (or confirm the latest).
- If no history: say so and ask what they'd like instead.
- On confirmation, re-place the exact same Instamart order.

## Fridge awareness
- If the conversation contains a [FRIDGE SCAN] message, the user has shared what's already in their fridge.
- When building a grocery or ingredient cart, FIRST check the fridge list. Do NOT add items that are already there.
- Explicitly tell the user which ingredients they already have vs. what you're ordering.
- Example: "You already have eggs and butter ✓ — ordering pasta, heavy cream, parmesan, garlic."

## Confirmation format (chat)
Show a cart summary table, total, ETA, then ask for confirmation.

## Language
Respond in the same language the user writes in (English or Hindi).
"""

LIVE_SYSTEM_SUFFIX = """

## LIVE Swiggy mode
You now have LIVE Swiggy Instamart tools. These tools use the user's real Swiggy account and can spend real money.

Hard safety rule: NEVER call checkout unless the user has EXPLICITLY confirmed in their most recent message (yes/haan/confirm/etc). If they have not confirmed, summarize and ask for confirmation instead.

When fetching addresses, default to the address tagged Home or the most recently used; confirm the delivery address in one short line before placing.

ADDRESS & SPEED RULES: Do NOT call get_addresses just to search - searching for food or products needs no address, so search immediately. Only resolve a delivery address when actually placing an order, and then default to the user's Home address (or most recently used) automatically. NEVER ask the user to choose an address unless they explicitly bring it up. On voice especially, keep it to one short question max before proposing an item.
"""


def _block_value(block, key, default=None):
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


def _content_text(content) -> str:
    if isinstance(content, str):
        return content.strip()

    text_blocks = []
    for block in content or []:
        text = _block_value(block, "text")
        if text:
            text_blocks.append(text)
    return " ".join(text_blocks).strip()


def _spend_server_for_tool(tool_name: str) -> str:
    for server_name, tool_names in SPEND_TOOLS.items():
        if tool_name in tool_names:
            return server_name
    return ""


def _model_for(surface):
    return VOICE_MODEL if surface == "voice" else AGENT_MODEL


def _max_tokens_for(surface):
    return 400 if surface == "voice" else 1024


def _api_timeout_for(surface):
    return VOICE_API_TIMEOUT_SECS if surface == "voice" else CHAT_API_TIMEOUT_SECS


def _live_order_summary(tool_name: str, tool_input: dict) -> tuple[str, str, list, str, float]:
    if not isinstance(tool_input, dict):
        tool_input = {}

    server_name = _spend_server_for_tool(tool_name)
    order_type = {
        "swiggy-food": "food",
        "swiggy-instamart": "grocery",
        "swiggy-dineout": "dineout",
    }.get(server_name, "food")

    items = tool_input.get("items") or tool_input.get("cart_items") or []
    if not isinstance(items, list):
        items = []

    restaurant_name = (
        tool_input.get("restaurant_name")
        or tool_input.get("restaurantName")
        or tool_input.get("name")
        or ""
    )
    total_amount = (
        tool_input.get("total_amount")
        or tool_input.get("totalAmount")
        or tool_input.get("amount")
        or 0
    )

    if tool_name == "place_food_order":
        item_names = [str(item.get("name", "")) for item in items if isinstance(item, dict)]
        summary = ", ".join([name for name in item_names if name][:3])
        if restaurant_name:
            summary = f"{summary} from {restaurant_name}" if summary else f"Food from {restaurant_name}"
        summary = summary or "Live food order"
    elif tool_name == "checkout":
        item_names = [str(item.get("name", "")) for item in items if isinstance(item, dict)]
        summary = "Groceries"
        if item_names:
            summary = f"Groceries: {', '.join(item_names[:4])}"
            if len(item_names) > 4:
                summary += " & more"
    elif tool_name == "book_table":
        guests = tool_input.get("guests") or tool_input.get("party_size") or tool_input.get("partySize")
        slot_time = tool_input.get("slot_time") or tool_input.get("slotTime") or tool_input.get("time")
        summary = "Dineout table booking"
        if restaurant_name:
            summary = f"Table at {restaurant_name}"
        if guests:
            summary += f" for {guests}"
        if slot_time:
            summary += f" @ {slot_time}"
    else:
        summary = f"{tool_name}: {json.dumps(tool_input, default=str)[:160]}"

    try:
        total_amount = float(total_amount)
    except (TypeError, ValueError):
        total_amount = 0.0

    return order_type, summary, items, restaurant_name, total_amount


def _save_live_order_if_any(content_blocks: list, session_id: str) -> None:
    if not session_id:
        return

    pending_spend_tools = {}
    last_pending_spend_tool = None
    for block in content_blocks:
        block_type = _block_value(block, "type")
        if block_type == "mcp_tool_use":
            tool_name = _block_value(block, "name", "")
            if not _spend_server_for_tool(tool_name):
                continue
            tool_id = _block_value(block, "id")
            if tool_id:
                pending_spend_tools[tool_id] = block
            last_pending_spend_tool = block
        elif block_type == "mcp_tool_result":
            tool_use_id = _block_value(block, "tool_use_id")
            tool_use = pending_spend_tools.get(tool_use_id) or last_pending_spend_tool
            if not tool_use or _block_value(block, "is_error", False):
                continue

            try:
                order_type, summary, items, restaurant_name, total_amount = _live_order_summary(
                    _block_value(tool_use, "name", ""),
                    _block_value(tool_use, "input", {}),
                )
                save_order(
                    session_id=session_id,
                    order_type=order_type,
                    summary=summary,
                    items=items,
                    restaurant_name=restaurant_name,
                    total_amount=total_amount,
                )
            except Exception:
                logging.exception("Failed to save live Swiggy order history")
            return


# ─────────────────────────────────────────────
# Main agent runner
# ─────────────────────────────────────────────

def _run_agent_demo(
    user_message: str,
    conversation_history: list[dict],
    surface: str = "voice",
    session_id: str = ""
) -> tuple[str, list[dict]]:
    system_prompt = VOICE_SYSTEM_PROMPT if surface == "voice" else CHAT_SYSTEM_PROMPT
    messages = conversation_history + [{"role": "user", "content": user_message}]

    for _ in range(8):
        response = client.messages.create(
            model=_model_for(surface),
            max_tokens=_max_tokens_for(surface),
            system=system_prompt,
            tools=TOOLS,
            messages=messages,
            timeout=_api_timeout_for(surface),
        )
        text_blocks = [b.text for b in response.content if hasattr(b, "text")]
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn" or not tool_use_blocks:
            return " ".join(text_blocks).strip(), messages

        tool_results = [
            {
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": execute_tool(tu.name, tu.input, session_id=session_id),
            }
            for tu in tool_use_blocks
        ]
        messages.append({"role": "user", "content": tool_results})

    return "Sorry, something went wrong. Please try again.", messages


def _run_agent_live(
    user_message: str,
    conversation_history: list[dict],
    surface: str,
    session_id: str,
    tokens: dict[str, str],
) -> tuple[str, list[dict]]:
    messages = conversation_history + [{"role": "user", "content": user_message}]

    try:
        system_prompt = (
            VOICE_SYSTEM_PROMPT if surface == "voice" else CHAT_SYSTEM_PROMPT
        ) + LIVE_SYSTEM_SUFFIX
        swiggy_address.maybe_background_refresh()
        default_address = swiggy_address.get_cached_default()
        if default_address:
            addr_id = default_address["id"]
            addr_label = default_address["label"]
            addr_area = default_address.get("area", "")
        elif DEFAULT_ADDRESS_ID:
            addr_id = DEFAULT_ADDRESS_ID
            addr_label = DEFAULT_ADDRESS_LABEL
            addr_area = DEFAULT_ADDRESS_AREA
        else:
            addr_id = ""

        if addr_id:
            system_prompt += (
                f"\n\nDEFAULT DELIVERY ADDRESS: {addr_label} ({addr_area}), "
                f"addressId {addr_id}. Use this addressId directly for all orders. "
                "Do NOT call get_addresses at all unless the user explicitly asks to change, "
                "list, or pick a different address."
            )
        confirmed = _is_confirmation(user_message)
        active = _route_servers(user_message, surface)
        mcp_servers = [
            {
                "type": "url",
                "url": url,
                "name": name,
                "authorization_token": tokens[MCP_AUTH_KEYS[name]],
            }
            for name, url in SWIGGY_SERVERS.items()
            if name in active
        ]
        tools = [
            {
                "type": "mcp_toolset",
                "mcp_server_name": name,
                "default_config": {"enabled": True},
                "configs": {
                    tool: {"enabled": confirmed}
                    for tool in SPEND_TOOLS[name]
                },
            }
            for name in active
        ]
        tools.extend(LIVE_LOCAL_TOOLS)

        for _ in range(8):
            response = client.beta.messages.create(
                model=_model_for(surface),
                max_tokens=_max_tokens_for(surface),
                system=system_prompt,
                tools=tools,
                mcp_servers=mcp_servers,
                betas=["mcp-client-2025-11-20"],
                messages=messages,
                timeout=_api_timeout_for(surface),
            )
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "pause_turn":
                continue

            local_tool_uses = [
                b for b in response.content
                if getattr(b, "type", None) == "tool_use"
            ]
            if not local_tool_uses:
                break

            tool_results = [
                {
                    "type": "tool_result",
                    "tool_use_id": b.id,
                    "content": execute_tool(b.name, b.input, session_id=session_id),
                }
                for b in local_tool_uses
            ]
            messages.append({"role": "user", "content": tool_results})

        assistant_blocks = []
        for message in messages:
            if message.get("role") != "assistant":
                continue
            content = message.get("content")
            if isinstance(content, str):
                continue
            assistant_blocks.extend(content or [])
        _save_live_order_if_any(assistant_blocks, session_id)

        final_text = ""
        for message in reversed(messages):
            if message.get("role") == "assistant":
                final_text = _content_text(message.get("content"))
                break

        return final_text or "Done. Please check Swiggy for the latest status.", messages

    except Exception:
        logging.exception("Swiggy live agent failed")
        return "Sorry, I hit a problem reaching Swiggy. Please try again in a moment.", messages


def run_agent(
    user_message: str,
    conversation_history: list[dict],
    surface: str = "voice",  # "voice" or "chat"
    session_id: str = ""
) -> tuple[str, list[dict]]:
    """
    Run one turn of the agent.

    Args:
        user_message: The user's latest message
        conversation_history: Previous messages in this session
        surface: "voice" (concise) or "chat" (rich)

    Returns:
        (agent_response_text, updated_conversation_history)
    """
    if DEMO_MODE:
        return _run_agent_demo(user_message, conversation_history, surface, session_id)

    try:
        tokens = get_all_access_tokens()
    except Exception as e:
        logging.warning("Swiggy not authed, falling back to demo: %s", e)
        return _run_agent_demo(user_message, conversation_history, surface, session_id)

    return _run_agent_live(user_message, conversation_history, surface, session_id, tokens)


# ─────────────────────────────────────────────
# Session management (in-memory, demo-grade)
# ─────────────────────────────────────────────

# Sessions keyed by session_id (call SID for voice, phone number for WA)
_sessions: dict[str, list[dict]] = {}


def get_session(session_id: str) -> list[dict]:
    """Get or create conversation history for a session."""
    return _sessions.get(session_id, [])


def update_session(session_id: str, history: list[dict]) -> None:
    """Update conversation history. Keep last 20 turns to avoid token overflow."""
    # Keep only last 20 messages (10 turns)
    _sessions[session_id] = history[-20:]


def clear_session(session_id: str) -> None:
    """Clear session after order placed or call ended."""
    _sessions.pop(session_id, None)


def process_message(
    session_id: str,
    user_message: str,
    surface: str = "voice"
) -> str:
    """
    High-level entry point.
    Takes a session ID + user message, returns agent response string.
    """
    history = get_session(session_id)
    response_text, updated_history = run_agent(user_message, history, surface, session_id=session_id)
    update_session(session_id, updated_history)
    return response_text
