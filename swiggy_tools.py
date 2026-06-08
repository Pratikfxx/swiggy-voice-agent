"""
Swiggy Tools — Mock data layer for demo + real MCP connector when approved.

To switch to real MCP:
  1. Set DEMO_MODE=false in .env
  2. Add SWIGGY_CLIENT_ID, SWIGGY_CLIENT_SECRET
  3. Run OAuth flow once (see authenticate() below)
  4. Real calls replace mock calls transparently.

Mock data is hyper-realistic — real restaurant names, real prices,
real delivery times from Mumbai/Bangalore/Delhi.
"""

import os
import random
import time
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"


# ─────────────────────────────────────────────
# MOCK DATA — realistic Swiggy responses
# ─────────────────────────────────────────────

MOCK_RESTAURANTS = {
    "biryani": [
        {
            "id": "rest_001",
            "name": "Biryani Blues",
            "cuisine": "Biryani, North Indian",
            "rating": 4.5,
            "rating_count": "10K+",
            "delivery_time_mins": 30,
            "delivery_time_spoken": "about 30 minutes",
            "distance_km": 1.8,
            "discount": "₹125 off above ₹299",
            "price_for_two": 450,
        },
        {
            "id": "rest_002",
            "name": "Behrouz Biryani",
            "cuisine": "Biryani, Mughlai",
            "rating": 4.3,
            "rating_count": "50K+",
            "delivery_time_mins": 40,
            "delivery_time_spoken": "about 40 minutes",
            "distance_km": 3.2,
            "discount": "20% off up to ₹100",
            "price_for_two": 650,
        },
        {
            "id": "rest_003",
            "name": "Paradise Biryani",
            "cuisine": "Biryani, Andhra",
            "rating": 4.6,
            "rating_count": "20K+",
            "delivery_time_mins": 35,
            "delivery_time_spoken": "about 35 minutes",
            "distance_km": 2.5,
            "discount": None,
            "price_for_two": 400,
        },
    ],
    "pizza": [
        {
            "id": "rest_010",
            "name": "Domino's Pizza",
            "cuisine": "Pizza, Fast Food",
            "rating": 4.1,
            "rating_count": "100K+",
            "delivery_time_mins": 25,
            "delivery_time_spoken": "about 25 minutes",
            "distance_km": 1.1,
            "discount": "2 Pizzas at ₹99 each",
            "price_for_two": 500,
        },
        {
            "id": "rest_011",
            "name": "Pizza Hut",
            "cuisine": "Pizza, Italian",
            "rating": 4.0,
            "rating_count": "80K+",
            "delivery_time_mins": 35,
            "delivery_time_spoken": "about 35 minutes",
            "distance_km": 2.0,
            "discount": "30% off on medium pizzas",
            "price_for_two": 600,
        },
    ],
    "burger": [
        {
            "id": "rest_020",
            "name": "McDonald's",
            "cuisine": "Burgers, Fast Food",
            "rating": 4.2,
            "rating_count": "200K+",
            "delivery_time_mins": 20,
            "delivery_time_spoken": "about 20 minutes",
            "distance_km": 0.8,
            "discount": "Free McFlurry above ₹300",
            "price_for_two": 350,
        },
        {
            "id": "rest_021",
            "name": "Burger King",
            "cuisine": "Burgers, American",
            "rating": 4.0,
            "rating_count": "90K+",
            "delivery_time_mins": 25,
            "delivery_time_spoken": "about 25 minutes",
            "distance_km": 1.5,
            "discount": "Buy 1 Get 1 Whopper",
            "price_for_two": 400,
        },
    ],
    "chinese": [
        {
            "id": "rest_030",
            "name": "Mainland China",
            "cuisine": "Chinese, Asian",
            "rating": 4.4,
            "rating_count": "15K+",
            "delivery_time_mins": 45,
            "delivery_time_spoken": "about 45 minutes",
            "distance_km": 4.0,
            "discount": "₹150 off above ₹500",
            "price_for_two": 800,
        },
        {
            "id": "rest_031",
            "name": "Wow! China",
            "cuisine": "Chinese, Indo-Chinese",
            "rating": 4.2,
            "rating_count": "25K+",
            "delivery_time_mins": 30,
            "delivery_time_spoken": "about 30 minutes",
            "distance_km": 2.0,
            "discount": None,
            "price_for_two": 400,
        },
    ],
    "default": [
        {
            "id": "rest_100",
            "name": "Swiggy Pop Kitchen",
            "cuisine": "Multi-cuisine",
            "rating": 4.3,
            "rating_count": "5K+",
            "delivery_time_mins": 25,
            "delivery_time_spoken": "about 25 minutes",
            "distance_km": 1.5,
            "discount": "₹50 off on first order",
            "price_for_two": 300,
        },
    ]
}

MOCK_MENU = {
    "rest_001": [
        {"id": "item_001", "name": "Chicken Biryani", "price": 299, "description": "Hyderabadi dum biryani with tender chicken", "is_veg": False},
        {"id": "item_002", "name": "Mutton Biryani", "price": 399, "description": "Slow-cooked mutton biryani", "is_veg": False},
        {"id": "item_003", "name": "Veg Biryani", "price": 249, "description": "Fragrant vegetable dum biryani", "is_veg": True},
        {"id": "item_004", "name": "Raita", "price": 49, "description": "Cooling yogurt side", "is_veg": True},
    ],
    "rest_002": [
        {"id": "item_010", "name": "Dum Chicken Biryani", "price": 349, "description": "Persian-inspired dum biryani", "is_veg": False},
        {"id": "item_011", "name": "Shahi Mutton Biryani", "price": 449, "description": "Royal mutton biryani", "is_veg": False},
    ],
    "rest_010": [
        {"id": "item_020", "name": "Margherita Pizza (Regular)", "price": 199, "description": "Classic tomato and mozzarella", "is_veg": True},
        {"id": "item_021", "name": "Pepperoni Pizza (Medium)", "price": 399, "description": "Loaded with pepperoni", "is_veg": False},
        {"id": "item_022", "name": "Garlic Breadsticks", "price": 99, "description": "Crispy garlic bread", "is_veg": True},
    ],
    "default": [
        {"id": "item_default", "name": "Chef's Special", "price": 249, "description": "Today's special dish", "is_veg": True},
    ]
}

MOCK_INSTAMART_PRODUCTS = {
    "milk": {"id": "im_001", "name": "Amul Full Cream Milk 1L", "price": 68, "brand": "Amul", "unit": "1 L", "delivery_mins": 15},
    "eggs": {"id": "im_002", "name": "Farm Fresh Eggs (Pack of 6)", "price": 62, "brand": "Fresho", "unit": "6 pcs", "delivery_mins": 15},
    "bread": {"id": "im_003", "name": "Britannia Whole Wheat Bread", "price": 45, "brand": "Britannia", "unit": "400 g", "delivery_mins": 15},
    "butter": {"id": "im_004", "name": "Amul Butter 100g", "price": 55, "brand": "Amul", "unit": "100 g", "delivery_mins": 15},
    "cream": {"id": "im_005", "name": "Amul Fresh Cream 200ml", "price": 35, "brand": "Amul", "unit": "200 ml", "delivery_mins": 15},
    "heavy cream": {"id": "im_005", "name": "Amul Fresh Cream 200ml", "price": 35, "brand": "Amul", "unit": "200 ml", "delivery_mins": 15},
    "parmesan": {"id": "im_006", "name": "Gowardhan Parmesan Cheese 100g", "price": 180, "brand": "Gowardhan", "unit": "100 g", "delivery_mins": 18},
    "parmesan cheese": {"id": "im_006", "name": "Gowardhan Parmesan Cheese 100g", "price": 180, "brand": "Gowardhan", "unit": "100 g", "delivery_mins": 18},
    "pasta": {"id": "im_007", "name": "Borges Fettuccine Pasta 400g", "price": 120, "brand": "Borges", "unit": "400 g", "delivery_mins": 18},
    "fettuccine pasta": {"id": "im_007", "name": "Borges Fettuccine Pasta 400g", "price": 120, "brand": "Borges", "unit": "400 g", "delivery_mins": 18},
    "garlic": {"id": "im_008", "name": "Fresh Garlic 100g", "price": 25, "brand": "Fresh", "unit": "100 g", "delivery_mins": 12},
    "onion": {"id": "im_009", "name": "Fresh Onions 500g", "price": 30, "brand": "Fresh", "unit": "500 g", "delivery_mins": 12},
    "tomato": {"id": "im_010", "name": "Fresh Tomatoes 500g", "price": 35, "brand": "Fresh", "unit": "500 g", "delivery_mins": 12},
    "rice": {"id": "im_011", "name": "India Gate Basmati Rice 1kg", "price": 125, "brand": "India Gate", "unit": "1 kg", "delivery_mins": 18},
    "basmati rice": {"id": "im_011", "name": "India Gate Basmati Rice 1kg", "price": 125, "brand": "India Gate", "unit": "1 kg", "delivery_mins": 18},
    "chicken": {"id": "im_012", "name": "Licious Fresh Chicken Curry Cut 500g", "price": 199, "brand": "Licious", "unit": "500 g", "delivery_mins": 20},
    "yogurt": {"id": "im_013", "name": "Amul Dahi 400g", "price": 55, "brand": "Amul", "unit": "400 g", "delivery_mins": 15},
    "paneer": {"id": "im_014", "name": "Amul Paneer 200g", "price": 90, "brand": "Amul", "unit": "200 g", "delivery_mins": 15},
    "salt": {"id": "im_015", "name": "Tata Salt 1kg", "price": 25, "brand": "Tata", "unit": "1 kg", "delivery_mins": 15},
    "black pepper": {"id": "im_016", "name": "Everest Black Pepper Powder 50g", "price": 45, "brand": "Everest", "unit": "50 g", "delivery_mins": 15},
    "cooking oil": {"id": "im_017", "name": "Fortune Sunflower Oil 1L", "price": 150, "brand": "Fortune", "unit": "1 L", "delivery_mins": 15},
    "sugar": {"id": "im_018", "name": "Tata Sugar 1kg", "price": 48, "brand": "Tata", "unit": "1 kg", "delivery_mins": 15},
    "flour": {"id": "im_019", "name": "Aashirvaad Atta 1kg", "price": 62, "brand": "Aashirvaad", "unit": "1 kg", "delivery_mins": 15},
    "all-purpose flour": {"id": "im_019b", "name": "Pillsbury Maida 500g", "price": 45, "brand": "Pillsbury", "unit": "500 g", "delivery_mins": 15},
    "maida": {"id": "im_019b", "name": "Pillsbury Maida 500g", "price": 45, "brand": "Pillsbury", "unit": "500 g", "delivery_mins": 15},
    "ghee": {"id": "im_020", "name": "Amul Pure Ghee 500ml", "price": 295, "brand": "Amul", "unit": "500 ml", "delivery_mins": 18},
    "soy sauce": {"id": "im_021", "name": "Ching's Soy Sauce 200ml", "price": 55, "brand": "Ching's", "unit": "200 ml", "delivery_mins": 15},
    "sesame oil": {"id": "im_022", "name": "Borges Sesame Oil 250ml", "price": 145, "brand": "Borges", "unit": "250 ml", "delivery_mins": 18},
    "mozzarella cheese": {"id": "im_023", "name": "Amul Mozzarella Cheese 200g", "price": 120, "brand": "Amul", "unit": "200 g", "delivery_mins": 15},
    "biryani masala": {"id": "im_024", "name": "Shan Biryani Masala 60g", "price": 65, "brand": "Shan", "unit": "60 g", "delivery_mins": 15},
    "ginger garlic paste": {"id": "im_025", "name": "Priya Ginger Garlic Paste 200g", "price": 48, "brand": "Priya", "unit": "200 g", "delivery_mins": 15},
    "mint": {"id": "im_026", "name": "Fresh Mint Leaves 50g", "price": 20, "brand": "Fresh", "unit": "50 g", "delivery_mins": 12},
    "fresh mint": {"id": "im_026", "name": "Fresh Mint Leaves 50g", "price": 20, "brand": "Fresh", "unit": "50 g", "delivery_mins": 12},
    "saffron": {"id": "im_027", "name": "Lion Saffron 1g", "price": 85, "brand": "Lion", "unit": "1 g", "delivery_mins": 18},
    "cardamom": {"id": "im_028", "name": "Everest Cardamom Powder 50g", "price": 95, "brand": "Everest", "unit": "50 g", "delivery_mins": 15},
    "baking powder": {"id": "im_029", "name": "Weikfield Baking Powder 100g", "price": 38, "brand": "Weikfield", "unit": "100 g", "delivery_mins": 15},
    "vanilla essence": {"id": "im_030", "name": "Weikfield Vanilla Essence 20ml", "price": 32, "brand": "Weikfield", "unit": "20 ml", "delivery_mins": 15},
}

MOCK_SAVED_ADDRESS = {
    "id": "addr_home",
    "label": "Home",
    "full_address": "Flat 4B, Sunshine Apartments, Bandra West, Mumbai 400050",
    "short": "Bandra West, Mumbai"
}

MOCK_DINEOUT_RESTAURANTS = {
    "italian": [
        {
            "id": "dine_001",
            "name": "Trattoria",
            "cuisine": "Italian, Continental",
            "rating": 4.6,
            "rating_count": "3.2K",
            "area": "Bandra West",
            "distance_km": 1.4,
            "price_for_two": 1800,
            "deals": "20% off on total bill",
            "timings": "12:00 PM – 11:30 PM",
            "short_description": "Trattoria in Bandra West, 4.6 stars, 20% off"
        },
        {
            "id": "dine_002",
            "name": "Prego – Westin",
            "cuisine": "Italian, Fine Dining",
            "rating": 4.8,
            "rating_count": "1.8K",
            "area": "Powai",
            "distance_km": 8.2,
            "price_for_two": 4000,
            "deals": "Complimentary dessert for 2",
            "timings": "7:00 PM – 11:00 PM",
            "short_description": "Prego at Westin, fine dining, complimentary dessert"
        },
    ],
    "chinese": [
        {
            "id": "dine_010",
            "name": "Yauatcha",
            "cuisine": "Chinese, Dim Sum",
            "rating": 4.7,
            "rating_count": "5.1K",
            "area": "BKC",
            "distance_km": 4.5,
            "price_for_two": 3500,
            "deals": "Flat 15% off on weekdays",
            "timings": "12:00 PM – 11:00 PM",
            "short_description": "Yauatcha BKC, dim sum specialists, 4.7 stars"
        },
    ],
    "indian": [
        {
            "id": "dine_020",
            "name": "Bombay Canteen",
            "cuisine": "Modern Indian",
            "rating": 4.5,
            "rating_count": "8.3K",
            "area": "Lower Parel",
            "distance_km": 5.1,
            "price_for_two": 2200,
            "deals": "Free welcome drink per person",
            "timings": "12:30 PM – 11:30 PM",
            "short_description": "Bombay Canteen, modern Indian, free welcome drink"
        },
        {
            "id": "dine_021",
            "name": "Indian Accent",
            "cuisine": "Contemporary Indian",
            "rating": 4.9,
            "rating_count": "2.1K",
            "area": "Lodhi Road",
            "distance_km": 7.3,
            "price_for_two": 5000,
            "deals": "Tasting menu at ₹4500 per person",
            "timings": "7:00 PM – 11:00 PM",
            "short_description": "Indian Accent, 4.9 stars, premium dining experience"
        },
    ],
    "rooftop": [
        {
            "id": "dine_030",
            "name": "Aer – Four Seasons",
            "cuisine": "Continental, Bar",
            "rating": 4.6,
            "rating_count": "4.7K",
            "area": "Worli",
            "distance_km": 6.0,
            "price_for_two": 3000,
            "deals": "Happy hours 6–8 PM",
            "timings": "6:00 PM – 1:00 AM",
            "short_description": "Aer rooftop bar at Four Seasons, stunning views"
        },
    ],
    "default": [
        {
            "id": "dine_100",
            "name": "The Table",
            "cuisine": "Continental, European",
            "rating": 4.5,
            "rating_count": "6.2K",
            "area": "Colaba",
            "distance_km": 3.8,
            "price_for_two": 2500,
            "deals": "20% off for Swiggy One members",
            "timings": "12:00 PM – 11:30 PM",
            "short_description": "The Table, Colaba, 4.5 stars, 20% off"
        },
    ]
}

# Slots: next 3 available slots per restaurant
MOCK_DINEOUT_SLOTS = {
    "dine_001": [
        {"slot_id": "slot_001_1", "time": "7:30 PM", "date": "Today", "available_seats": 4},
        {"slot_id": "slot_001_2", "time": "8:00 PM", "date": "Today", "available_seats": 2},
        {"slot_id": "slot_001_3", "time": "8:30 PM", "date": "Today", "available_seats": 6},
    ],
    "dine_010": [
        {"slot_id": "slot_010_1", "time": "7:00 PM", "date": "Today", "available_seats": 4},
        {"slot_id": "slot_010_2", "time": "8:00 PM", "date": "Today", "available_seats": 2},
    ],
    "dine_020": [
        {"slot_id": "slot_020_1", "time": "8:00 PM", "date": "Today", "available_seats": 4},
        {"slot_id": "slot_020_2", "time": "9:00 PM", "date": "Today", "available_seats": 2},
    ],
    "default": [
        {"slot_id": "slot_def_1", "time": "7:30 PM", "date": "Today", "available_seats": 4},
        {"slot_id": "slot_def_2", "time": "8:30 PM", "date": "Today", "available_seats": 2},
    ]
}


# ─────────────────────────────────────────────
# TOOL FUNCTIONS — called by agent
# ─────────────────────────────────────────────

def get_saved_address() -> dict:
    """Returns user's default saved address."""
    if DEMO_MODE:
        return {"success": True, "addresses": [MOCK_SAVED_ADDRESS], "default": MOCK_SAVED_ADDRESS}
    # TODO: real MCP call → get_addresses
    return {"success": True, "addresses": [MOCK_SAVED_ADDRESS], "default": MOCK_SAVED_ADDRESS}


def search_food_restaurants(query: str, location: str = "home") -> dict:
    """Search restaurants by cuisine or dish name."""
    if DEMO_MODE:
        query_lower = query.lower()
        results = None
        for key in MOCK_RESTAURANTS:
            if key in query_lower or query_lower in key:
                results = MOCK_RESTAURANTS[key]
                break
        if not results:
            results = MOCK_RESTAURANTS["default"]
        return {
            "success": True,
            "results": results[:3],  # top 3 for voice
            "location": MOCK_SAVED_ADDRESS["short"],
            "count": len(results)
        }
    # TODO: real MCP → search_restaurants(query, addressId)
    raise NotImplementedError("Set DEMO_MODE=true until Swiggy MCP access is granted")


def get_restaurant_menu(restaurant_id: str, dish_query: str = "") -> dict:
    """Get menu items from a specific restaurant."""
    if DEMO_MODE:
        menu = MOCK_MENU.get(restaurant_id, MOCK_MENU["default"])
        if dish_query:
            filtered = [item for item in menu if dish_query.lower() in item["name"].lower()]
            if filtered:
                menu = filtered
        return {"success": True, "items": menu, "restaurant_id": restaurant_id}
    # TODO: real MCP → get_restaurant_menu(restaurantId)
    raise NotImplementedError("Set DEMO_MODE=true until Swiggy MCP access is granted")


def search_instamart_products(product_name: str, quantity_hint: str = "1") -> dict:
    """Search Instamart for a grocery product."""
    if DEMO_MODE:
        key = product_name.lower().strip()
        product = MOCK_INSTAMART_PRODUCTS.get(key)
        if not product:
            # fuzzy match
            for k, v in MOCK_INSTAMART_PRODUCTS.items():
                if key in k or k in key:
                    product = v
                    break
        if product:
            return {
                "success": True,
                "found": True,
                "product": product,
                "query": product_name
            }
        return {
            "success": True,
            "found": False,
            "product": {"id": f"im_generic_{hash(key)}", "name": product_name, "price": 50, "brand": "Local", "unit": quantity_hint, "delivery_mins": 15},
            "query": product_name
        }
    # TODO: real MCP → search_products(query)
    raise NotImplementedError("Set DEMO_MODE=true until Swiggy MCP access is granted")


def place_food_order_mock(restaurant_id: str, items: list[dict], address_id: str = "addr_home") -> dict:
    """Place a food order (mock — returns realistic confirmation)."""
    if DEMO_MODE:
        order_id = f"ORD{random.randint(10000000, 99999999)}"
        total = sum(item.get("price", 0) * item.get("quantity", 1) for item in items)
        delivery_fee = 30 if total < 200 else 0
        return {
            "success": True,
            "order_id": order_id,
            "status": "confirmed",
            "total": total + delivery_fee,
            "delivery_fee": delivery_fee,
            "eta_mins": random.randint(25, 45),
            "eta_spoken": "about 30 to 40 minutes",
            "payment": "Cash on Delivery",
            "tracking_url": f"https://swiggy.com/track/{order_id}"
        }
    raise NotImplementedError("Set DEMO_MODE=true until Swiggy MCP access is granted")


def search_dineout_restaurants(query: str, guests: int = 2, time_pref: str = "tonight") -> dict:
    """Search Dineout for restaurants to dine in."""
    if DEMO_MODE:
        query_lower = query.lower()
        results = None
        for key in MOCK_DINEOUT_RESTAURANTS:
            if key in query_lower or query_lower in key:
                results = MOCK_DINEOUT_RESTAURANTS[key]
                break
        if not results:
            results = MOCK_DINEOUT_RESTAURANTS["default"]
        return {
            "success": True,
            "results": results[:3],
            "guests": guests,
            "time_preference": time_pref
        }
    raise NotImplementedError("Set DEMO_MODE=true until Swiggy MCP access is granted")


def get_dineout_slots(restaurant_id: str, date: str = "today") -> dict:
    """Get available table slots for a restaurant."""
    if DEMO_MODE:
        slots = MOCK_DINEOUT_SLOTS.get(restaurant_id, MOCK_DINEOUT_SLOTS["default"])
        return {"success": True, "slots": slots, "restaurant_id": restaurant_id}
    raise NotImplementedError("Set DEMO_MODE=true until Swiggy MCP access is granted")


def book_dineout_table_mock(restaurant_id: str, restaurant_name: str, slot_id: str, slot_time: str, guests: int) -> dict:
    """Book a table at a Dineout restaurant (mock)."""
    if DEMO_MODE:
        booking_id = f"DT{random.randint(10000000, 99999999)}"
        return {
            "success": True,
            "booking_id": booking_id,
            "status": "confirmed",
            "restaurant": restaurant_name,
            "guests": guests,
            "time": slot_time,
            "date": "Today",
            "confirmation_note": "Show this booking ID at the restaurant entrance."
        }
    raise NotImplementedError("Set DEMO_MODE=true until Swiggy MCP access is granted")


def place_instamart_order_mock(items: list[dict], address_id: str = "addr_home") -> dict:
    """Place an Instamart grocery order (mock)."""
    if DEMO_MODE:
        order_id = f"IM{random.randint(10000000, 99999999)}"
        total = sum(item.get("price", 0) * item.get("quantity", 1) for item in items)
        return {
            "success": True,
            "order_id": order_id,
            "status": "confirmed",
            "total": total,
            "eta_mins": random.randint(12, 20),
            "eta_spoken": "about 15 to 18 minutes",
            "payment": "Cash on Delivery",
        }
    raise NotImplementedError("Set DEMO_MODE=true until Swiggy MCP access is granted")
