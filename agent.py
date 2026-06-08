"""
Swiggy Voice Agent — Core Claude Agent

Handles multi-turn conversations across voice and chat surfaces.
Uses Claude claude-sonnet-4-6 with tool use for all Swiggy operations.

Surface modes:
  - "voice": concise responses, max 3 items, spoken naturally
  - "chat": rich markdown, tables, full lists
"""

import json
import os
from typing import Optional
import anthropic
from dotenv import load_dotenv

from recipe_engine import get_recipe_ingredients as _get_recipe_ingredients
from order_history import save_order, get_recent_orders
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

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

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

VOICE_SYSTEM_PROMPT = """You are Swiggy's voice ordering assistant on a live phone call.

VOICE RULES — NON-NEGOTIABLE:
- Plain spoken English only. Zero emojis, markdown, bullets, or symbols.
- MAX 20 words per response. Short = good. Silence after speaking is fine.
- Never narrate what you're doing ("Let me search...", "I found 3 options..."). Just do it and give the answer.
- Never read order IDs, tracking codes, or technical fields aloud.
- Prices: "469 rupees" not "₹469". Times: "30 minutes" not "30 mins".

ORDER FLOW:
1. User says what they want → search → immediately give ONE top result.
2. Confirm with a SINGLE short sentence → wait for yes/no.
3. On yes → place order → say done in one sentence → hang up.
- Always use saved address. Never ask for it.
- Only place after: yes, haan, okay, confirm, theek hai.

CONFIRMATION — ULTRA SHORT:
- Food: "[Restaurant name], [total] rupees, [ETA]. Shall I place it?"
- Grocery: "[N] items, [total] rupees, [ETA] on Instamart. Confirm?"
- Recipe cart: "Got all [N] ingredients, [total] rupees. Order them?"
- DO NOT list items. DO NOT say individual prices. One sentence only.

AFTER ORDER (one sentence max):
"Done! Arriving in [ETA]."

REPEAT ORDER:
- "order my usual" / "phir se" → call get_order_history → "[Item] from [place], [price] rupees again?"

LANGUAGE: Match whatever the user speaks — Hindi, Hinglish, English all fine.
"""

CHAT_SYSTEM_PROMPT = """You are Swiggy's AI ordering assistant for chat/WhatsApp.

## Your mission
Help users order food or groceries conversationally. Be helpful, clear, and efficient.

## Chat response rules
- Use markdown formatting (bold, tables, bullet points)
- Show up to 5 restaurant options with ratings, delivery time, distance
- Show cart as a table with item, qty, price
- Always confirm before placing order: "Ready to place? Reply **yes** to confirm."
- After confirmation → place order → send confirmation with order ID

## Intent detection
- "order biryani" / "I want pizza" → search_food_restaurants → show options → user picks → confirm → place
- "get me milk, eggs, bread" → search each on Instamart → show cart total → confirm → place
- "items for alfredo pasta" → get_recipe_ingredients → search each on Instamart → show full cart → confirm → place
- "book a table" / "dinner tonight" / "going out" → search_dineout_restaurants → show top 3 with deals → get slots for chosen restaurant → confirm → book
- Mixed orders → handle food + grocery separately, confirm both

## Repeat orders
- Triggers: "order my usual", "same as last time", "repeat my order", "what did I order last", "order again"
- Call get_order_history to look up past orders.
- If orders exist: show a summary of the last 1–3 orders and ask which one to repeat (or confirm the latest).
- If no history: say so and ask what they'd like instead.
- On confirmation, re-place the exact same order (same restaurant_id, items, quantities).

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


# ─────────────────────────────────────────────
# Main agent runner
# ─────────────────────────────────────────────

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
    system_prompt = VOICE_SYSTEM_PROMPT if surface == "voice" else CHAT_SYSTEM_PROMPT

    # Add user message to history
    messages = conversation_history + [
        {"role": "user", "content": user_message}
    ]

    max_tool_rounds = 8  # prevent infinite loops
    round_count = 0

    while round_count < max_tool_rounds:
        round_count += 1

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=system_prompt,
            tools=TOOLS,
            messages=messages
        )

        # Collect text from response
        text_blocks = [b.text for b in response.content if hasattr(b, "text")]
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

        # Add assistant turn to history
        messages.append({"role": "assistant", "content": response.content})

        # If no tool calls, we have a final answer
        if response.stop_reason == "end_turn" or not tool_use_blocks:
            final_text = " ".join(text_blocks).strip()
            return final_text, messages

        # Execute all tool calls in parallel-ish (sequential here, fast enough)
        tool_results = []
        for tool_use in tool_use_blocks:
            result_str = execute_tool(tool_use.name, tool_use.input, session_id=session_id)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": result_str
            })

        # Add tool results as user turn
        messages.append({"role": "user", "content": tool_results})

    # Fallback if max rounds hit
    return "Sorry, something went wrong. Please try again.", messages


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
