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
from collections import OrderedDict
from threading import Lock

import anthropic

from recipe_engine import get_recipe_ingredients as _get_recipe_ingredients
from order_history import save_order, get_recent_orders
import store
import swiggy_address
from swiggy_search import get_cart_summary, remove_from_cart, search_and_add_to_cart
from swiggy_auth import get_access_tokens
from swiggy_scope import (
    demo_mode,
    ACTIVE_SWIGGY_SERVERS,
    ACTIVE_TOKEN_KEYS,
    SERVER_AUTH_KEYS as MCP_AUTH_KEYS,
    SWIGGY_SERVER_URLS as SWIGGY_SERVERS,
)
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

# max_retries=1 (default 2): when a model is overloaded, burning ~12s of SDK
# retries inside the voice call budget starves the tier fallback below — one
# quick retry, then we switch models instead.
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), max_retries=1)
# Chat/WhatsApp brain: Claude Sonnet 5 — best value/quality for grocery ordering
# with fridge reasoning and recipe carts. Voice stays on Haiku for the 7s deadline.
AGENT_MODEL = os.getenv("AGENT_MODEL", "claude-sonnet-5")
VOICE_MODEL = os.getenv("VOICE_MODEL", "claude-haiku-4-5")
# Sonnet 5 runs adaptive thinking ON when the param is omitted — that burns
# tokens and, on voice, latency: a measured "get milk" turn took 18.7s with
# thinking vs 6.3s without. Grocery ordering needs neither. Override to
# "adaptive" if you want reasoning back (slower, pricier).
CHAT_THINKING = os.getenv("CHAT_THINKING", "disabled")
VOICE_THINKING = os.getenv("VOICE_THINKING", "disabled")
# This timeout covers the WHOLE agent loop, not one model turn: Swiggy runs as a
# server-side MCP connector, so every search/lookup round trip happens inside a
# single create() call. A measured live turn is ~6s, so keep real headroom here.
VOICE_API_TIMEOUT_SECS = float(os.getenv("VOICE_API_TIMEOUT_SECS", "20.0"))
VOICE_CHECKOUT_API_TIMEOUT_SECS = float(
    os.getenv("VOICE_CHECKOUT_API_TIMEOUT_SECS", "55.0")
)
CHAT_API_TIMEOUT_SECS = float(os.getenv("CHAT_API_TIMEOUT_SECS", "30.0"))
CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", "1024"))
# When the primary model is overloaded/rate-limited (429/5xx), retry the same
# request once on the other tier instead of failing the conversation. Haiku 4.5
# returned 529s for a stretch on 2026-07-16, which killed every voice call.
# Empty string disables.
VOICE_OVERLOAD_FALLBACK = os.getenv("VOICE_OVERLOAD_FALLBACK", "claude-sonnet-5")
CHAT_OVERLOAD_FALLBACK = os.getenv("CHAT_OVERLOAD_FALLBACK", "claude-haiku-4-5")
# Composing "milk, bread and eggs, 319 rupees. Confirm?" costs a second model
# round trip — measured at 4.4s of a 13.2s turn to emit 26 tokens whose shape
# the voice prompt already dictates and whose contents the cart tool already
# returned. When enabled, voice turns say that line directly and skip the call.
# Enabled by default after live A/B testing; disable without a deploy if a
# response needs the model's judgement.
VOICE_FAST_CONFIRM = os.getenv("VOICE_FAST_CONFIRM", "true").lower() == "true"

DEMO_MODE = demo_mode()
DEFAULT_ADDRESS_ID = os.getenv("DEFAULT_ADDRESS_ID", "")
DEFAULT_ADDRESS_LABEL = os.getenv("DEFAULT_ADDRESS_LABEL", "Home")
DEFAULT_ADDRESS_AREA = os.getenv("DEFAULT_ADDRESS_AREA", "")
CONFIRM_RE = re.compile(
    r"\b(yes|yeah|yep|yup|haan|haa|ha|confirm(ed)?|theek hai|thik hai|place (it|the order)|order it|book it|go ahead|pakka|kar do|kardo|do it|sure|okay|ok|proceed)\b",
    re.I,
)

SPEND_TOOLS = {
    "swiggy-food": ["place_food_order"],
    "swiggy-instamart": ["checkout"],
    "swiggy-dineout": ["book_table"],
}
ORDER_PLACEMENT_RULES = (
    "\n\nPLACING THE ORDER:\n"
    "NEVER say an order is placed, and never give an order number or an "
    "arrival time, unless a checkout tool call in THIS turn returned success. "
    "If checkout failed or you did not call it, say plainly that the order did "
    "NOT go through and offer to try again. Telling someone their groceries "
    "are coming when no order exists is the worst thing you can do on this "
    "call.\n"
    "BEFORE checkout, read back EVERYTHING in the cart and the cart total, not "
    "just what was added this turn. The cart persists between calls, so it can "
    "hold items the caller never mentioned today — and checkout charges for all "
    "of them. The add tool returns cart_contents and cart_total for exactly "
    "this. If the cart holds anything the caller has not asked for in this "
    "call, name those items and ask whether to keep or drop them before "
    "placing the order.\n"
    "When checkout succeeds, use the message from the tool response as it is "
    "written — Swiggy requires the wording \"Instamart order placed "
    "successfully\" rather than a plain \"order placed\". Then close warmly: "
    "thank them, state the arrival time, and say goodbye.\n"
    "CANCELLING AN ORDER: if the caller wants to cancel an order that has "
    "already been placed, do NOT call any tool — no tool can do it. Tell them "
    "to call Swiggy customer care on 080 67466729. Removing items from the "
    "cart is different and is still handled by remove_from_cart."
)

VOICE_PAYMENT_RULES = (
    "\nVOICE PAYMENT: UPI needs an app or QR surface outside this phone call. "
    "Use cash on delivery: explicitly ask the caller to pay by cash, then only "
    "after they agree call checkout with paymentMethod=\"Cash\"."
)

CHAT_PAYMENT_RULES = (
    "\nCHAT PAYMENT: when the user is ready to pay, call get_payment_options. "
    "Let them choose a live UPI app, QR, or cash option; never force cash. Echo "
    "the selected UPI method id byte-for-byte into checkout, or use "
    "paymentMethod=\"Cash\" only when they explicitly choose cash."
)

CART_EDIT_RULES = (
    "\n\nCART EDITS: if the user removes, drops, cancels or changes their mind "
    "about anything already in the cart, you MUST call remove_from_cart before "
    "replying. Prefer its keep_only argument — name the items to KEEP. NEVER "
    "state that the cart changed, and never quote a new total, unless a tool "
    "call in THIS turn returned it: the cart lives on Swiggy's servers, not in "
    "this conversation, and repeating an earlier total is how you tell the user "
    "their groceries are gone when they are still there. After the tool returns, "
    "read back what is left and the new subtotal it reports."
)

ADDRESS_SELECTION_RULES = (
    "IF the user names ANY delivery destination other than the default — an "
    "office, a label, a person's name, a locality — you MUST call get_addresses "
    "before responding. Saved address tags are often personal names or "
    "nicknames, so you cannot tell from the word alone whether it is a saved "
    "address. NEVER say an address is not saved unless you have just called "
    "get_addresses and read the list. Match the entry whose addressTag or "
    "addressLine matches what they said, take its `id`, and use THAT id as the "
    "addressId for EVERY subsequent search_products, update_cart and checkout "
    "call — all tools in one order must use the same chosen addressId, never a "
    "mix. get_addresses returns 10 addresses per page. If the named address is "
    "not on page 1 and the response says there are more pages, call it again "
    "with page=2, then page=3, until you find the match or exhaust the pages. "
    "The server accepts page even though its published input schema is empty. "
    "If no page matches, name two or three saved addresses and ask which one "
    "they mean. Never "
    "silently fall back to the default address when the user asked for a "
    "different one."
)

LIVE_AUTH_NOT_READY_MESSAGE = (
    "Swiggy login is not ready for Instamart yet. "
    "Please refresh the Instamart login and try again."
)
LIVE_CHECKOUT_UNCERTAIN_MESSAGE = (
    "I couldn't confirm the checkout status. "
    "Please check your Swiggy app order history before trying again."
)
LIVE_GENERIC_FAILURE_MESSAGE = (
    "Sorry, I hit a problem reaching Swiggy. Please try again in a moment."
)
REFUSAL_MESSAGE = (
    "Sorry, I can't help with that request. "
    "Want groceries, snacks, drinks, or essentials instead?"
)


def _route_servers(user_message: str, surface: str) -> list[str]:
    """Return which Swiggy MCP servers to attach.

    Product mode is temporarily Instamart-only, so do not attach Food or Dineout.
    """
    return list(ACTIVE_SWIGGY_SERVERS)


def _is_confirmation(text):
    return bool(CONFIRM_RE.search(text or ""))


def _payment_rules_for_surface(surface: str) -> str:
    return VOICE_PAYMENT_RULES if surface == "voice" else CHAT_PAYMENT_RULES


def normalize_user_id(identity: str) -> str:
    return str(identity or "").removeprefix("whatsapp:")

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
    },
    {
        "name": "remove_from_cart",
        "description": (
            "Remove items from the Instamart cart in ONE call, keeping the rest. "
            "Use this whenever the user drops, cancels or changes their mind about "
            "part of the cart. STRONGLY PREFER keep_only: name the few items to "
            "KEEP rather than the many to drop, because catalogue names are "
            "misleading — dropping by the word 'sugar' also removes 'Monster "
            "Energy Zero Sugar'. Do NOT call get_cart and update_cart yourself to "
            "do this; this tool reads the cart, rewrites it and returns what is "
            "left with the new subtotal."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "address_id": {"type": "string", "description": "The delivery addressId for this order."},
                "keep_only": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Items to KEEP, e.g. ['monster', 'diet coke']. Everything else is removed.",
                },
                "remove": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Items to remove. Only use when keeping is harder to express than dropping.",
                },
            },
            "required": ["address_id"],
        },
    },
    {
        "name": "search_and_add_to_cart",
        "description": (
            "Search MANY Instamart items in parallel AND add the best in-stock "
            "match of each to the cart, in ONE call. ALWAYS use this for two or "
            "more items — a grocery list, or all ingredients from "
            "get_recipe_ingredients — instead of searching one at a time. Pass "
            "the delivery addressId for this order. It returns what was added "
            "(names, prices, subtotal) and anything not found. After it returns, "
            "just read the summary and ask the user to confirm; the cart is "
            "already built, so do NOT call search_products or update_cart again."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Item names, e.g. ['paneer', 'basmati rice', 'yogurt']",
                },
                "address_id": {
                    "type": "string",
                    "description": "The delivery addressId (default, or the user's chosen saved address).",
                },
                "quantity": {
                    "type": "integer",
                    "description": "Quantity per item (default 1).",
                    "default": 1,
                },
            },
            "required": ["queries", "address_id"],
        },
    }
]

LIVE_LOCAL_TOOLS = [
    tool for tool in TOOLS
    if tool["name"] in {"get_recipe_ingredients", "search_and_add_to_cart", "remove_from_cart"}
]
LOCAL_NAMES = {tool["name"] for tool in LIVE_LOCAL_TOOLS}


# ─────────────────────────────────────────────
# Tool executor — maps tool names → functions
# ─────────────────────────────────────────────

def execute_tool(
    tool_name: str, tool_input: dict, session_id: str = "", user_id: str = ""
) -> str:
    """Execute a tool call and return JSON string result."""
    user_id = user_id or session_id
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

        elif tool_name == "remove_from_cart":
            result = remove_from_cart(
                tool_input["address_id"],
                remove=tool_input.get("remove"),
                keep_only=tool_input.get("keep_only"),
            )

        elif tool_name == "search_and_add_to_cart":
            result = search_and_add_to_cart(
                tool_input["queries"],
                tool_input["address_id"],
                tool_input.get("quantity", 1),
            )

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
            if user_id and result.get("success"):
                items = tool_input.get("items", [])
                summary = (
                    f"{', '.join(i.get('name','') for i in items[:3])}"
                    f" from {tool_input.get('restaurant_name', '')}"
                )
                save_order(
                    session_id=user_id,
                    order_type="food",
                    summary=summary,
                    items=items,
                    restaurant_name=tool_input.get("restaurant_name", ""),
                    total_amount=tool_input.get("total_amount", 0),
                )

        elif tool_name == "place_grocery_order":
            result = place_instamart_order_mock(tool_input["items"])
            # Auto-save to order history on success
            if user_id and result.get("success"):
                items = tool_input.get("items", [])
                summary = (
                    f"Groceries: {', '.join(i.get('name','') for i in items[:4])}"
                    + (" & more" if len(items) > 4 else "")
                )
                save_order(
                    session_id=user_id,
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
            if user_id and result.get("success"):
                save_order(
                    session_id=user_id,
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
            orders = get_recent_orders(user_id, limit=limit) if user_id else []
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
2. Let the user accept or edit those selections.
3. When they accept the selection, call get_cart, not search again. Give the
   final cart, address, and payment summary in one
   short question. Only a yes to this final summary enables checkout.
4. After checkout succeeds, say done in one sentence and hang up.
- Use the verified default address unless the user names another saved address.
- A yes to a product, variant, cart edit, or address is not permission to place.
- MULTIPLE items or a recipe: call search_and_add_to_cart ONCE with EVERY item
  (and the addressId). It searches AND fills the cart in one step — never search
  items one at a time, and do NOT call update_cart yourself afterward.
- After search_and_add_to_cart returns, speak ONE line: the main items, the
  subtotal, mention anything not_found, and ask "Confirm?". Do not narrate "let
  me add these" — the cart is already built when the tool returns.
- RECIPE (e.g. "ingredients for paneer biryani"): get_recipe_ingredients, then
  pass that whole ingredient list to search_and_add_to_cart in ONE call.
- NEVER offer to make an ingredient from scratch. Paneer, ghee, curd, pastes and
  masalas are all sold ready on Instamart — always buy the ready product. Do not
  ask "should I get paneer or the milk to make it" or similar; just buy it.

CONFIRMATION:
- Selection: "[Brand/item and pack], [cart total] rupees. Keep these?"
- Final spend: "Full cart is [total] rupees, delivered to [address], cash on
  delivery. Place the order?"
- If there are more than 3 grocery items, summarize the count and name the most important 2-3 items.
- Do not read individual prices unless the user asks.

AFTER ORDER (only once checkout has actually succeeded):
"Instamart order placed successfully. Thank you! Arriving in [ETA]. Bye!"
Never say this unless a checkout tool call in this turn confirmed the order.

CANCELLING A PLACED ORDER: no tool can cancel one. Say: "I can't cancel it from
here, but Swiggy customer care can, on 080 67466729."

REPEAT ORDER:
- "order my usual" / "phir se" → use the available history tool: get_orders in
  live mode, get_order_history in demo mode. Repeat only Instamart orders.

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
- Before placing, show the full cart total, delivery address, and selected live
  payment method, then ask: "Ready to place? Reply **yes** to confirm."
- After the user accepts product selections, call get_cart rather than repeating
  the search, then prepare that final summary.
- After confirmation → place order → send confirmation with order ID

## Intent detection
- "get me milk, eggs, bread" → search each on Instamart → show cart total → confirm → place
- "items for alfredo pasta" → get_recipe_ingredients → search each on Instamart → show full cart → confirm → place
- Cooked meal or reservation requests → explain Instamart-only scope and offer ingredients, snacks, drinks, or essentials instead
- Mixed requests → handle only the Instamart items and clearly say which non-Instamart parts cannot be handled right now

## Repeat orders
- Triggers: "order my usual", "same as last time", "repeat my order", "what did I order last", "order again"
- Use the available history tool: get_orders in live mode, get_order_history in
  demo mode.
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

_LIVE_SYSTEM_SUFFIX_BASE = """

## LIVE Swiggy mode
You now have LIVE Swiggy Instamart tools. These tools use the user's real Swiggy account and can spend real money.

Hard safety rule: before checkout, state the full cart total, delivery address,
and payment method in one final summary. Ask whether to place the order. Only a
confirmation to THAT summary authorizes checkout; a "yes" to a product or
address question does not.

When fetching addresses, default to the address tagged Home or the most recently used; confirm the delivery address in one short line before placing.

ADDRESS & SPEED RULES: search_products requires a real addressId. Use the
verified default addressId injected below for ordinary searches. If none was
injected, call get_addresses first. If the user names another saved address,
follow the pagination and matching rules below. Before checkout, always state
the selected delivery address as part of the final confirmation. On voice,
keep that confirmation to one short question.
"""

LIVE_SYSTEM_SUFFIX = _LIVE_SYSTEM_SUFFIX_BASE + CART_EDIT_RULES + ORDER_PLACEMENT_RULES



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


_PAYMENT_TERMS = re.compile(
    r"\b(cash on delivery|pay by cash|upi|gpay|google pay|phonepe|paytm|bhim|"
    r"cred|scan[- ]?qr|super\.money)\b",
    re.I,
)
_ORDER_CHANGE_RE = re.compile(
    r"\b(but|add|remove|drop|change|instead|without|except|cancel|address|"
    r"office|home|workplace)\b",
    re.I,
)


def _checkout_ready(user_message: str, history: list[dict], surface: str) -> bool:
    """True only after the required final order summary was confirmed."""
    if not _is_confirmation(user_message):
        return False
    if _ORDER_CHANGE_RE.search(user_message or ""):
        return False

    previous = ""
    for message in reversed(history):
        if message.get("role") == "assistant":
            previous = _content_text(message.get("content"))
            break

    lowered = previous.lower()
    has_total = bool(re.search(r"(?:₹|\b\d+(?:\.\d+)?\s+rupees?\b)", previous, re.I))
    has_address = any(term in lowered for term in ("deliver to", "delivered to", "delivery to", "address"))
    has_payment = bool(_PAYMENT_TERMS.search(previous))
    asks_to_place = any(term in lowered for term in ("place the order", "proceed", "confirm"))
    if surface == "voice":
        has_payment = "cash on delivery" in lowered or "pay by cash" in lowered
    return has_total and has_address and has_payment and asks_to_place


def _selection_accepted(user_message: str, history: list[dict]) -> bool:
    if not _is_confirmation(user_message) or _ORDER_CHANGE_RE.search(user_message or ""):
        return False
    for message in reversed(history):
        if message.get("role") == "assistant":
            return _content_text(message.get("content")).rstrip().endswith("Keep these?")
    return False


def _final_checkout_line(summary: dict) -> str:
    total = str(summary.get("cart_total") or "")
    total = re.sub(r"[^\d.]", "", total)
    address = str(summary.get("address") or "").strip()
    if not total or not address or not summary.get("items"):
        return ""
    stores = int(summary.get("store_count") or 0)
    multi_store = (
        f" across {stores} stores; Instamart will place separate orders"
        if stores > 1 else ""
    )
    return (
        f"Full cart is {total} rupees{multi_store}, delivered to {address}, "
        "cash on delivery. Place the order?"
    )


def checkout_confirmation_pending(
    session_id: str, user_message: str, surface: str = "voice"
) -> bool:
    return _checkout_ready(user_message, get_session(session_id), surface)


def _mcp_tool_configs(checkout_enabled: bool) -> dict[str, dict[str, bool]]:
    """Model-facing gates; widget-owned payment tools stay server-owned."""
    return {
        "checkout": {"enabled": checkout_enabled},
        "get_delivery_status": {"enabled": False},
        "check_payment_status": {"enabled": False},
        "confirm_order": {"enabled": False},
    }


def _text_after_last_tool(content) -> str:
    """The reply the caller should hear, not the narration before it.

    A server-side MCP turn returns one assistant message holding text, the
    tool_use/tool_result blocks, and then more text. Joining all of it made the
    agent ask and answer itself in one breath — "Add this one? Perfect! It is
    in your cart" — which on a phone call sounds like two people talking. Keep
    only what comes after the last tool block; fall back to everything if the
    model spoke before calling tools and never after.
    """
    if isinstance(content, str):
        return content.strip()

    tail: list[str] = []
    for block in content or []:
        block_type = _block_value(block, "type")
        if block_type in ("tool_use", "mcp_tool_use", "tool_result", "mcp_tool_result"):
            tail = []
            continue
        text = _block_value(block, "text")
        if text:
            tail.append(text)
    return " ".join(tail).strip() or _content_text(content)


def _fast_confirm_line(tool_name: str, raw_result: str, user_message: str) -> str:
    """A spoken confirmation built from the cart tool's own output, or "".

    Only for the clean case: the voice surface, every requested item found,
    the cart actually updated, and an ASCII request — a Hindi or Hinglish
    caller must still get a reply composed in their language, so those fall
    through to the model.
    """
    if not VOICE_FAST_CONFIRM or tool_name != "search_and_add_to_cart":
        return ""
    if not user_message.isascii():
        return ""

    try:
        result = json.loads(raw_result)
    except (json.JSONDecodeError, TypeError):
        return ""

    added = result.get("added") or []
    if not added or result.get("not_found") or not result.get("cart_updated"):
        return ""
    # The cart may already hold items from earlier in the call or a previous
    # one. Reading back only what was just added understates what checkout
    # will charge for, so let the model describe the whole cart instead.
    if result.get("cart_has_other_items"):
        return ""
    # The line names items but not counts, so "two packets of butter" would be
    # understated. Anything other than one of each goes back to the model.
    if any(entry.get("quantity", 1) != 1 for entry in added):
        return ""

    # "item" is what the caller asked for ("milk"); "name" is the full
    # catalogue title ("Mother Dairy Pasteurised Homogenised Cow Milk"), which
    # is unbearable read aloud. Prefer the caller's own words.
    names = []
    for entry in added:
        item = str(entry.get("item") or entry.get("name") or "").strip()
        brand = str(entry.get("brand") or "").strip()
        variant = str(entry.get("variant") or "").strip()
        if brand and brand.lower() not in item.lower():
            item = f"{brand} {item}"
        if variant:
            item = f"{item} {variant}"
        names.append(item)
    names = [n for n in names if n]
    if not names:
        return ""

    if len(names) == 1:
        listed = names[0]
    else:
        listed = ", ".join(names[:-1]) + " and " + names[-1]

    # cart_total arrives as a formatted string like "\u20b91072".
    subtotal = result.get("cart_total") or result.get("subtotal")
    if isinstance(subtotal, str):
        subtotal = re.sub(r"[^\d.]", "", subtotal)
    try:
        rupees = int(round(float(subtotal)))
    except (TypeError, ValueError):
        return ""

    return f"{listed}, {rupees} rupees. Keep these?"


def _spend_server_for_tool(tool_name: str) -> str:
    for server_name, tool_names in SPEND_TOOLS.items():
        if tool_name in tool_names:
            return server_name
    return ""


def _spend_result_succeeded(tool_name: str, result_block) -> bool:
    """Business success, not merely a successful MCP transport call."""
    if _block_value(result_block, "is_error", False):
        return False
    text = _content_text(_block_value(result_block, "content", "")).lower()
    if tool_name == "checkout":
        return (
            "instamart order placed successfully" in text
            or bool(re.search(r'"status"\s*:\s*"placed"', text))
            or bool(re.search(r'"success"\s*:\s*true', text))
        )
    return bool(text)


def _agent_result(response_text, messages, order_placed=False, return_meta=False):
    if return_meta:
        return response_text, messages, {"order_placed": order_placed}
    return response_text, messages


def _model_for(surface):
    return VOICE_MODEL if surface == "voice" else AGENT_MODEL


# Models that think unless told not to. Only these need the disable switch —
# Haiku and the Opus family already skip thinking when the param is omitted, and
# sending an unsupported param to them would fail the request outright.
_THINKS_BY_DEFAULT = ("claude-sonnet-5",)
# Thinking is always on for the Fable family and any override is rejected (400).
_THINKING_ALWAYS_ON = ("claude-fable", "claude-mythos")


def _thinking_for_model(model: str, surface: str) -> dict:
    """Request extras controlling thinking for a specific model.

    Disabled by default on both surfaces: it cuts token spend on chat and is
    worth ~12s per turn on voice. Only sent to models that would otherwise
    think, so pointing a surface at Haiku stays a no-op. Takes the model as an
    argument because an overload fallback can swap tiers mid-request.
    """
    if model.startswith(_THINKING_ALWAYS_ON):
        return {}

    want = VOICE_THINKING if surface == "voice" else CHAT_THINKING
    if want == "adaptive":
        return {"thinking": {"type": "adaptive"}}
    if want == "disabled" and model.startswith(_THINKS_BY_DEFAULT):
        return {"thinking": {"type": "disabled"}}
    return {}


def _thinking_kwargs(surface: str) -> dict:
    return _thinking_for_model(_model_for(surface), surface)


def _overload_fallback_for(surface: str) -> str:
    return VOICE_OVERLOAD_FALLBACK if surface == "voice" else CHAT_OVERLOAD_FALLBACK


def _is_capacity_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    return status == 429 or (isinstance(status, int) and status >= 500)


# Anthropic reports a failure to reach Swiggy's MCP server as a 400
# invalid_request_error, which looks like a bad request but is transient — it
# arrives in bursts and the identical call succeeds moments later. Retrying
# turned an on-call "Sorry, I hit a problem reaching Swiggy" into a normal
# answer.
_MCP_TRANSIENT_MARKERS = (
    "error while communicating with mcp server",
    "mcp server returned an error while listing tools",
    "connection closed",
)


def _is_transient_mcp_error(exc: Exception) -> bool:
    if getattr(exc, "status_code", None) != 400:
        return False
    text = str(exc).lower()
    return any(marker in text for marker in _MCP_TRANSIENT_MARKERS)


def _create_message(surface: str, live: bool, **kwargs):
    """Create a message on the surface's model, falling back on capacity errors.

    When the primary tier is overloaded or rate-limited, the other tier is the
    retry: same request, same thinking policy resolved for the new model.
    """
    def attempt(model: str):
        endpoint = client.beta.messages if live else client.messages
        return endpoint.create(model=model, **_thinking_for_model(model, surface), **kwargs)

    primary = _model_for(surface)
    try:
        return attempt(primary)
    except anthropic.APIStatusError as exc:
        if _is_transient_mcp_error(exc):
            # Same model, same request: the connector, not the model, blinked.
            logging.warning("transient Swiggy MCP connector error; retrying once")
            try:
                return attempt(primary)
            except anthropic.APIStatusError:
                raise
        fallback = _overload_fallback_for(surface)
        if not fallback or fallback == primary or not _is_capacity_error(exc):
            raise
        logging.warning(
            "model %s capacity error (%s); retrying on %s",
            primary, getattr(exc, "status_code", "?"), fallback,
        )
        return attempt(fallback)


def _system_blocks(system_prompt: str) -> list[dict]:
    """System prompt as a cacheable block.

    The prompt prefix renders tools -> system -> messages, so a cache
    breakpoint on the last system block caches the tool schemas (including the
    ~30k tokens of Swiggy MCP toolsets) together with the prompt. Cache reads
    bill at ~0.1x, and that prefix is nearly all of our per-turn input.

    Note: the confirmation turn flips the checkout tool's enabled flag, which
    changes the tools tier and writes a second cache variant. Both variants
    stay warm within the 5-minute TTL, so an order flow still hits cache on
    every turn except the first of each variant.
    """
    return [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]


def _with_history_breakpoint(messages: list[dict]) -> list[dict]:
    """Mark the newest message as a cache breakpoint (non-destructively).

    The system-block breakpoint caches tools+system, but conversation history
    sits after it and was replaying at full price every call — measured 11.5k
    full-price tokens on turn two of a chat. A breakpoint on the last message
    block lets each request reuse the previous request's entire prefix.
    """
    if not messages:
        return messages

    last = messages[-1]
    if not isinstance(last, dict):
        return messages

    content = last.get("content")
    if isinstance(content, str):
        marked = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
    elif isinstance(content, list) and content and isinstance(content[-1], dict):
        marked = content[:-1] + [{**content[-1], "cache_control": {"type": "ephemeral"}}]
    else:
        return messages

    return messages[:-1] + [{**last, "content": marked}]


def _max_tokens_for(surface):
    return 400 if surface == "voice" else CHAT_MAX_TOKENS


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


def _save_live_order_if_any(
    content_blocks: list, session_id: str, user_id: str = ""
) -> bool:
    user_id = user_id or session_id
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
            if not tool_use or not _spend_result_succeeded(
                _block_value(tool_use, "name", ""), block
            ):
                continue

            try:
                order_type, summary, items, restaurant_name, total_amount = _live_order_summary(
                    _block_value(tool_use, "name", ""),
                    _block_value(tool_use, "input", {}),
                )
                if user_id:
                    save_order(
                        session_id=user_id,
                        order_type=order_type,
                        summary=summary,
                        items=items,
                        restaurant_name=restaurant_name,
                        total_amount=total_amount,
                    )
            except Exception:
                logging.exception("Failed to save live Swiggy order history")
            return True
    return False


# ─────────────────────────────────────────────
# Main agent runner
# ─────────────────────────────────────────────

def _run_agent_demo(
    user_message: str,
    conversation_history: list[dict],
    surface: str = "voice",
    session_id: str = "",
    return_meta: bool = False,
    user_id: str = "",
) -> tuple[str, list[dict]]:
    system_prompt = VOICE_SYSTEM_PROMPT if surface == "voice" else CHAT_SYSTEM_PROMPT
    messages = conversation_history + [{"role": "user", "content": user_message}]

    for _ in range(8):
        response = _create_message(
            surface,
            live=False,
            max_tokens=_max_tokens_for(surface),
            system=_system_blocks(system_prompt),
            tools=TOOLS,
            messages=messages,
            timeout=_api_timeout_for(surface),
        )
        if response.stop_reason == "refusal":
            messages.append({"role": "assistant", "content": REFUSAL_MESSAGE})
            return _agent_result(REFUSAL_MESSAGE, messages, return_meta=return_meta)
        text_blocks = [b.text for b in response.content if hasattr(b, "text")]
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn" or not tool_use_blocks:
            return _agent_result(" ".join(text_blocks).strip(), messages, return_meta=return_meta)

        tool_results = [
            {
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": execute_tool(
                    tu.name, tu.input, session_id=session_id, user_id=user_id
                ),
            }
            for tu in tool_use_blocks
        ]
        messages.append({"role": "user", "content": tool_results})

    return _agent_result(
        "Sorry, something went wrong. Please try again.", messages, return_meta=return_meta
    )


def _run_agent_live(
    user_message: str,
    conversation_history: list[dict],
    surface: str,
    session_id: str,
    tokens: dict[str, str],
    return_meta: bool = False,
    user_id: str = "",
) -> tuple[str, list[dict]]:
    messages = conversation_history + [{"role": "user", "content": user_message}]
    order_placed = False
    user_id = user_id or session_id
    checkout_enabled = _checkout_ready(user_message, conversation_history, surface)
    selection_accepted = _selection_accepted(user_message, conversation_history)

    try:
        system_prompt = (
            VOICE_SYSTEM_PROMPT if surface == "voice" else CHAT_SYSTEM_PROMPT
        ) + LIVE_SYSTEM_SUFFIX + _payment_rules_for_surface(surface)
        default_address = swiggy_address.get_cached_default()
        if default_address:
            # warm cache: refresh in the background if the TTL lapsed
            swiggy_address.maybe_background_refresh()
        else:
            # cold cache (first call after deploy): fetch now, or the model
            # burns the caller's first turn asking which address to use
            default_address = swiggy_address.get_default_blocking()
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

        if selection_accepted and surface == "voice":
            line = _final_checkout_line(get_cart_summary())
            if line:
                messages.append({"role": "assistant", "content": line})
                return _agent_result(line, messages, False, return_meta)

        if addr_id:
            system_prompt += (
                f"\n\nDEFAULT DELIVERY ADDRESS: {addr_label} ({addr_area}), "
                f"addressId {addr_id}. Use this addressId by default for "
                "search_products, update_cart (as selectedAddressId), and checkout. "
                "Do not call get_addresses merely to start searching because this "
                "verified default addressId is already available.\n"
                + ADDRESS_SELECTION_RULES
            )
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
                "configs": _mcp_tool_configs(checkout_enabled),
            }
            for name in active
        ]
        tools.extend(LIVE_LOCAL_TOOLS)

        response = None
        for _ in range(8):
            response = _create_message(
                surface,
                live=True,
                max_tokens=_max_tokens_for(surface),
                system=_system_blocks(system_prompt),
                tools=tools,
                mcp_servers=mcp_servers,
                betas=["mcp-client-2025-11-20"],
                messages=_with_history_breakpoint(messages),
                timeout=(
                    VOICE_CHECKOUT_API_TIMEOUT_SECS
                    if checkout_enabled and surface == "voice"
                    else _api_timeout_for(surface)
                ),
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

            tool_results = []
            fast_line = ""
            for b in local_tool_uses:
                raw = execute_tool(
                    b.name, b.input, session_id=session_id, user_id=user_id
                )
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": b.id, "content": raw}
                )
                if surface == "voice" and not fast_line:
                    fast_line = _fast_confirm_line(b.name, raw, user_message)
            messages.append({"role": "user", "content": tool_results})

            if fast_line:
                # The cart is built and the sentence is fully determined by its
                # contents; a second model round trip would only re-say it.
                messages.append({"role": "assistant", "content": fast_line})
                _save_live_order_if_any([], session_id or user_id)
                return _agent_result(fast_line, messages, False, return_meta)

        assistant_blocks = []
        for message in messages:
            if message.get("role") != "assistant":
                continue
            content = message.get("content")
            if isinstance(content, str):
                continue
            assistant_blocks.extend(content or [])
        order_placed = _save_live_order_if_any(assistant_blocks, session_id, user_id)

        # Whole-chain refusal (Fable declined and fallback declined too, or no
        # fallback configured) — content is empty or partial, don't read it.
        if response is not None and response.stop_reason == "refusal":
            messages.append({"role": "assistant", "content": REFUSAL_MESSAGE})
            return _agent_result(REFUSAL_MESSAGE, messages, order_placed, return_meta)

        final_text = ""
        for message in reversed(messages):
            if message.get("role") == "assistant":
                final_text = _text_after_last_tool(message.get("content"))
                break

        return _agent_result(
            final_text or "Done. Please check Swiggy for the latest status.",
            messages,
            order_placed,
            return_meta,
        )

    except Exception:
        logging.exception("Swiggy live agent failed")
        if checkout_enabled:
            return _agent_result(
                LIVE_CHECKOUT_UNCERTAIN_MESSAGE, messages, order_placed, return_meta
            )
        return _agent_result(
            LIVE_GENERIC_FAILURE_MESSAGE, messages, order_placed, return_meta
        )


def run_agent(
    user_message: str,
    conversation_history: list[dict],
    surface: str = "voice",  # "voice" or "chat"
    session_id: str = "",
    return_meta: bool = False,
    user_id: str = "",
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
        return _run_agent_demo(
            user_message,
            conversation_history,
            surface,
            session_id,
            return_meta,
            user_id=user_id,
        )

    try:
        tokens = get_access_tokens(ACTIVE_TOKEN_KEYS, user_id=user_id)
    except Exception as e:
        logging.warning("Swiggy live auth not ready: %s", e)
        messages = conversation_history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": LIVE_AUTH_NOT_READY_MESSAGE},
        ]
        return _agent_result(LIVE_AUTH_NOT_READY_MESSAGE, messages, return_meta=return_meta)

    return _run_agent_live(
        user_message,
        conversation_history,
        surface,
        session_id,
        tokens,
        return_meta,
        user_id=user_id,
    )


# ─────────────────────────────────────────────
# Session management
# ─────────────────────────────────────────────

# Hot write-through cache keyed by session_id (call SID for voice, phone number for WA).
_SESSION_CACHE_MAX = 1000
_sessions: OrderedDict[str, list[dict]] = OrderedDict()
_sessions_lock = Lock()


def _cache_session(session_id: str, history: list[dict]) -> None:
    _sessions.pop(session_id, None)
    _sessions[session_id] = history
    while len(_sessions) > _SESSION_CACHE_MAX:
        _sessions.popitem(last=False)


def get_session(session_id: str) -> list[dict]:
    """Get conversation history from the cache, then durable storage."""
    with _sessions_lock:
        if session_id in _sessions:
            history = _sessions.pop(session_id)
            _sessions[session_id] = history
            return history

    history = store.get_session(session_id)
    with _sessions_lock:
        _cache_session(session_id, history)
    return history


def _sanitize_history(history: list[dict]) -> list[dict]:
    """Collapse each turn to plain text before persisting.

    Server-side MCP turns embed `mcp_tool_use`/`mcp_tool_result` blocks (and
    local `tool_use`/`tool_result` pairs) that span several assistant messages
    across `pause_turn` iterations. Storing those raw and later slicing with
    history[-20:] can drop a result while keeping its use, which makes the next
    request 400 with "mcp_tool_use ... without a corresponding mcp_tool_result".
    Cross-turn context only needs the spoken text, so store text-only and let
    each new turn rebuild its own tool plumbing (also keeps replay cheap).
    """
    clean: list[dict] = []
    for message in history:
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue
        text = _content_text(message.get("content"))
        if not text:
            continue
        if clean and clean[-1]["role"] == role:
            clean[-1]["content"] = f"{clean[-1]['content']} {text}".strip()
        else:
            clean.append({"role": role, "content": text})
    # The API requires the first message to be from the user.
    while clean and clean[0]["role"] != "user":
        clean.pop(0)
    return clean


def update_session(session_id: str, history: list[dict]) -> None:
    """Persist conversation history as plain text, last 20 messages.

    Text-only storage avoids dangling MCP tool blocks after truncation and
    keeps replayed history cheap; each turn rebuilds its own tool calls.
    """
    sanitized = _sanitize_history(history)[-20:]
    store.update_session(session_id, sanitized)
    with _sessions_lock:
        _cache_session(session_id, sanitized)


def clear_session(session_id: str) -> None:
    """Clear session after order placed or call ended."""
    store.clear_session(session_id)
    with _sessions_lock:
        _sessions.pop(session_id, None)


def process_message(
    session_id: str,
    user_message: str,
    surface: str = "voice",
    *,
    return_meta: bool = False,
    user_id: str = "",
):
    """
    High-level entry point.
    Takes a session ID + user message, returns agent response string.
    """
    history = get_session(session_id)
    user_id = user_id or session_id
    result = run_agent(
        user_message,
        history,
        surface,
        session_id=session_id,
        return_meta=return_meta,
        user_id=user_id,
    )
    response_text, updated_history = result[:2]
    update_session(session_id, updated_history)
    return (response_text, result[2]) if return_meta else response_text
