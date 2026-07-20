"""
Recipe Engine — maps dish names to grocery ingredients.
Used when user says "items for X" or "ingredients for X".
Swap in a real recipe API (Spoonacular, Edamam) when needed.
"""

from typing import Optional

# Ingredient: (name, quantity, unit)
RECIPE_DB: dict[str, list[dict]] = {
    # Italian
    "alfredo pasta": [
        {"name": "fettuccine pasta", "qty": "400", "unit": "g"},
        {"name": "heavy cream", "qty": "200", "unit": "ml"},
        {"name": "parmesan cheese", "qty": "100", "unit": "g"},
        {"name": "butter", "qty": "50", "unit": "g"},
        {"name": "garlic", "qty": "4", "unit": "cloves"},
        {"name": "black pepper", "qty": "1", "unit": "tsp"},
        {"name": "salt", "qty": "1", "unit": "tsp"},
    ],
    "pasta": [
        {"name": "pasta", "qty": "400", "unit": "g"},
        {"name": "tomato puree", "qty": "400", "unit": "g"},
        {"name": "olive oil", "qty": "3", "unit": "tbsp"},
        {"name": "garlic", "qty": "3", "unit": "cloves"},
        {"name": "basil", "qty": "1", "unit": "bunch"},
        {"name": "parmesan cheese", "qty": "50", "unit": "g"},
        {"name": "onion", "qty": "1", "unit": "piece"},
    ],
    "pizza": [
        {"name": "pizza base", "qty": "2", "unit": "pieces"},
        {"name": "pizza sauce", "qty": "1", "unit": "bottle"},
        {"name": "mozzarella cheese", "qty": "200", "unit": "g"},
        {"name": "bell peppers", "qty": "2", "unit": "pieces"},
        {"name": "onion", "qty": "1", "unit": "piece"},
        {"name": "olives", "qty": "50", "unit": "g"},
    ],

    # Indian
    "chicken biryani": [
        {"name": "basmati rice", "qty": "500", "unit": "g"},
        {"name": "chicken", "qty": "750", "unit": "g"},
        {"name": "onion", "qty": "3", "unit": "pieces"},
        {"name": "tomato", "qty": "2", "unit": "pieces"},
        {"name": "yogurt", "qty": "200", "unit": "g"},
        {"name": "biryani masala", "qty": "3", "unit": "tbsp"},
        {"name": "ginger garlic paste", "qty": "2", "unit": "tbsp"},
        {"name": "fresh mint", "qty": "1", "unit": "bunch"},
        {"name": "saffron", "qty": "1", "unit": "pinch"},
        {"name": "ghee", "qty": "3", "unit": "tbsp"},
        {"name": "cooking oil", "qty": "4", "unit": "tbsp"},
    ],
    "paneer biryani": [
        {"name": "basmati rice", "qty": "500", "unit": "g"},
        {"name": "paneer", "qty": "400", "unit": "g"},
        {"name": "onion", "qty": "3", "unit": "pieces"},
        {"name": "tomato", "qty": "2", "unit": "pieces"},
        {"name": "yogurt", "qty": "200", "unit": "g"},
        {"name": "biryani masala", "qty": "3", "unit": "tbsp"},
        {"name": "ginger garlic paste", "qty": "2", "unit": "tbsp"},
        {"name": "fresh mint", "qty": "1", "unit": "bunch"},
        {"name": "ghee", "qty": "3", "unit": "tbsp"},
    ],
    "mutton biryani": [
        {"name": "basmati rice", "qty": "500", "unit": "g"},
        {"name": "mutton", "qty": "750", "unit": "g"},
        {"name": "onion", "qty": "3", "unit": "pieces"},
        {"name": "tomato", "qty": "2", "unit": "pieces"},
        {"name": "yogurt", "qty": "200", "unit": "g"},
        {"name": "biryani masala", "qty": "3", "unit": "tbsp"},
        {"name": "ginger garlic paste", "qty": "2", "unit": "tbsp"},
        {"name": "fresh mint", "qty": "1", "unit": "bunch"},
        {"name": "ghee", "qty": "3", "unit": "tbsp"},
    ],
    "biryani": [
        {"name": "basmati rice", "qty": "500", "unit": "g"},
        {"name": "onion", "qty": "3", "unit": "pieces"},
        {"name": "tomato", "qty": "2", "unit": "pieces"},
        {"name": "yogurt", "qty": "200", "unit": "g"},
        {"name": "biryani masala", "qty": "3", "unit": "tbsp"},
        {"name": "ginger garlic paste", "qty": "2", "unit": "tbsp"},
        {"name": "fresh mint", "qty": "1", "unit": "bunch"},
        {"name": "ghee", "qty": "3", "unit": "tbsp"},
    ],
    "dal makhani": [
        {"name": "black lentils (urad dal)", "qty": "250", "unit": "g"},
        {"name": "kidney beans (rajma)", "qty": "50", "unit": "g"},
        {"name": "butter", "qty": "50", "unit": "g"},
        {"name": "cream", "qty": "100", "unit": "ml"},
        {"name": "tomato puree", "qty": "200", "unit": "g"},
        {"name": "onion", "qty": "2", "unit": "pieces"},
        {"name": "ginger garlic paste", "qty": "2", "unit": "tbsp"},
        {"name": "dal makhani masala", "qty": "2", "unit": "tbsp"},
    ],
    "butter chicken": [
        {"name": "chicken", "qty": "750", "unit": "g"},
        {"name": "tomato puree", "qty": "400", "unit": "g"},
        {"name": "butter", "qty": "50", "unit": "g"},
        {"name": "cream", "qty": "100", "unit": "ml"},
        {"name": "onion", "qty": "2", "unit": "pieces"},
        {"name": "ginger garlic paste", "qty": "2", "unit": "tbsp"},
        {"name": "butter chicken masala", "qty": "3", "unit": "tbsp"},
        {"name": "yogurt", "qty": "100", "unit": "g"},
        {"name": "kasuri methi", "qty": "1", "unit": "tbsp"},
    ],
    "paneer tikka": [
        {"name": "paneer", "qty": "300", "unit": "g"},
        {"name": "yogurt", "qty": "150", "unit": "g"},
        {"name": "bell peppers", "qty": "2", "unit": "pieces"},
        {"name": "onion", "qty": "2", "unit": "pieces"},
        {"name": "tikka masala", "qty": "2", "unit": "tbsp"},
        {"name": "ginger garlic paste", "qty": "1", "unit": "tbsp"},
        {"name": "lemon", "qty": "1", "unit": "piece"},
        {"name": "cooking oil", "qty": "2", "unit": "tbsp"},
    ],
    "chole bhature": [
        {"name": "chickpeas (chole)", "qty": "250", "unit": "g"},
        {"name": "maida (all-purpose flour)", "qty": "300", "unit": "g"},
        {"name": "onion", "qty": "3", "unit": "pieces"},
        {"name": "tomato", "qty": "3", "unit": "pieces"},
        {"name": "chole masala", "qty": "3", "unit": "tbsp"},
        {"name": "yogurt", "qty": "50", "unit": "g"},
        {"name": "cooking oil", "qty": "500", "unit": "ml"},
    ],
    "pav bhaji": [
        {"name": "pav (bread rolls)", "qty": "8", "unit": "pieces"},
        {"name": "potato", "qty": "4", "unit": "pieces"},
        {"name": "cauliflower", "qty": "200", "unit": "g"},
        {"name": "peas", "qty": "100", "unit": "g"},
        {"name": "tomato", "qty": "3", "unit": "pieces"},
        {"name": "onion", "qty": "2", "unit": "pieces"},
        {"name": "pav bhaji masala", "qty": "3", "unit": "tbsp"},
        {"name": "butter", "qty": "75", "unit": "g"},
    ],
    "samosa": [
        {"name": "maida (all-purpose flour)", "qty": "300", "unit": "g"},
        {"name": "potato", "qty": "5", "unit": "pieces"},
        {"name": "peas", "qty": "100", "unit": "g"},
        {"name": "cumin seeds", "qty": "1", "unit": "tsp"},
        {"name": "coriander powder", "qty": "1", "unit": "tsp"},
        {"name": "garam masala", "qty": "1", "unit": "tsp"},
        {"name": "cooking oil", "qty": "500", "unit": "ml"},
    ],

    # Chinese
    "fried rice": [
        {"name": "basmati rice", "qty": "400", "unit": "g"},
        {"name": "eggs", "qty": "3", "unit": "pieces"},
        {"name": "spring onion", "qty": "4", "unit": "stalks"},
        {"name": "carrot", "qty": "1", "unit": "piece"},
        {"name": "peas", "qty": "100", "unit": "g"},
        {"name": "soy sauce", "qty": "3", "unit": "tbsp"},
        {"name": "sesame oil", "qty": "1", "unit": "tbsp"},
        {"name": "garlic", "qty": "4", "unit": "cloves"},
    ],
    "noodles": [
        {"name": "hakka noodles", "qty": "300", "unit": "g"},
        {"name": "cabbage", "qty": "200", "unit": "g"},
        {"name": "carrot", "qty": "2", "unit": "pieces"},
        {"name": "spring onion", "qty": "4", "unit": "stalks"},
        {"name": "soy sauce", "qty": "3", "unit": "tbsp"},
        {"name": "chili sauce", "qty": "2", "unit": "tbsp"},
        {"name": "sesame oil", "qty": "1", "unit": "tbsp"},
        {"name": "garlic", "qty": "3", "unit": "cloves"},
        {"name": "cooking oil", "qty": "3", "unit": "tbsp"},
    ],

    # Breakfast
    "pancakes": [
        {"name": "all-purpose flour", "qty": "200", "unit": "g"},
        {"name": "eggs", "qty": "2", "unit": "pieces"},
        {"name": "milk", "qty": "300", "unit": "ml"},
        {"name": "butter", "qty": "50", "unit": "g"},
        {"name": "sugar", "qty": "2", "unit": "tbsp"},
        {"name": "baking powder", "qty": "2", "unit": "tsp"},
        {"name": "vanilla essence", "qty": "1", "unit": "tsp"},
    ],
    "omelette": [
        {"name": "eggs", "qty": "3", "unit": "pieces"},
        {"name": "onion", "qty": "1", "unit": "piece"},
        {"name": "tomato", "qty": "1", "unit": "piece"},
        {"name": "green chili", "qty": "2", "unit": "pieces"},
        {"name": "coriander", "qty": "1", "unit": "bunch"},
        {"name": "butter", "qty": "1", "unit": "tbsp"},
        {"name": "salt", "qty": "1", "unit": "tsp"},
    ],
    "poha": [
        {"name": "flattened rice (poha)", "qty": "300", "unit": "g"},
        {"name": "onion", "qty": "2", "unit": "pieces"},
        {"name": "potato", "qty": "2", "unit": "pieces"},
        {"name": "green chili", "qty": "3", "unit": "pieces"},
        {"name": "mustard seeds", "qty": "1", "unit": "tsp"},
        {"name": "curry leaves", "qty": "1", "unit": "sprig"},
        {"name": "turmeric", "qty": "0.5", "unit": "tsp"},
        {"name": "lemon", "qty": "1", "unit": "piece"},
        {"name": "cooking oil", "qty": "2", "unit": "tbsp"},
    ],

    # Desserts
    "kheer": [
        {"name": "full-fat milk", "qty": "1", "unit": "liter"},
        {"name": "basmati rice", "qty": "50", "unit": "g"},
        {"name": "sugar", "qty": "100", "unit": "g"},
        {"name": "cardamom powder", "qty": "1", "unit": "tsp"},
        {"name": "cashews", "qty": "30", "unit": "g"},
        {"name": "raisins", "qty": "30", "unit": "g"},
        {"name": "saffron", "qty": "1", "unit": "pinch"},
    ],
    "gulab jamun": [
        {"name": "khoya (mawa)", "qty": "200", "unit": "g"},
        {"name": "maida", "qty": "3", "unit": "tbsp"},
        {"name": "sugar", "qty": "300", "unit": "g"},
        {"name": "cardamom", "qty": "4", "unit": "pods"},
        {"name": "rose water", "qty": "1", "unit": "tbsp"},
        {"name": "cooking oil", "qty": "500", "unit": "ml"},
    ],
}

# Aliases — common shorthand → canonical key
ALIASES: dict[str, str] = {
    "chicken biryani": "chicken biryani",
    "veg biryani": "biryani",
    "paneer dum biryani": "paneer biryani",
    "chicken dum biryani": "chicken biryani",
    "pasta alfredo": "alfredo pasta",
    "white pasta": "alfredo pasta",
    "red pasta": "pasta",
    "tomato pasta": "pasta",
    "butter chicken": "butter chicken",
    "murgh makhani": "butter chicken",
    "tikka masala": "butter chicken",
    "paneer tikka": "paneer tikka",
    "dal": "dal makhani",
    "daal": "dal makhani",
    "chole": "chole bhature",
    "chana": "chole bhature",
}


def get_recipe_ingredients(dish_name: str) -> dict:
    """
    Returns ingredient list for a dish.
    Returns closest match with confidence score.
    """
    key = dish_name.lower().strip()

    # Direct match
    if key in RECIPE_DB:
        return {
            "found": True,
            "dish": key.title(),
            "ingredients": RECIPE_DB[key],
            "serves": 2,
            "note": "Quantities for 2 servings"
        }

    # Alias match
    if key in ALIASES:
        canonical = ALIASES[key]
        return {
            "found": True,
            "dish": canonical.title(),
            "ingredients": RECIPE_DB[canonical],
            "serves": 2,
            "note": "Quantities for 2 servings"
        }

    # Fuzzy: check if key is a substring of any recipe
    for recipe_name, ingredients in RECIPE_DB.items():
        if key in recipe_name or recipe_name in key:
            return {
                "found": True,
                "dish": recipe_name.title(),
                "ingredients": ingredients,
                "serves": 2,
                "note": f"Showing ingredients for {recipe_name.title()}"
            }

    # Partial word match
    key_words = set(key.split())
    best_match = None
    best_score = 0
    for recipe_name in RECIPE_DB:
        recipe_words = set(recipe_name.split())
        overlap = len(key_words & recipe_words)
        if overlap > best_score:
            best_score = overlap
            best_match = recipe_name

    if best_match and best_score > 0:
        return {
            "found": True,
            "dish": best_match.title(),
            "ingredients": RECIPE_DB[best_match],
            "serves": 2,
            "note": f"Closest match found: {best_match.title()}"
        }

    return {
        "found": False,
        "dish": dish_name,
        "ingredients": [],
        "note": f"Recipe not found for '{dish_name}'. Try a common dish name."
    }


def list_supported_recipes() -> list[str]:
    return sorted(RECIPE_DB.keys())
