"""Recipe matching must refuse to guess.

A single shared word is not a dish. "paneer butter masala" matched
"paneer biryani" on the word "paneer" and confidently returned rice, mint
and biryani masala for a curry — the agent then read that out as fact.
"""

from recipe_engine import get_recipe_ingredients


def _names(dish):
    return [i["name"] for i in get_recipe_ingredients(dish)["ingredients"]]


def test_one_shared_word_is_not_a_match():
    result = get_recipe_ingredients("paneer tandoori surprise")
    if result["found"]:
        # If it matched, it must share more than a single word with the key.
        matched = set(result["dish"].lower().split())
        assert len(matched & {"paneer", "tandoori", "surprise"}) >= 2


def test_unknown_dish_reports_not_found_rather_than_guessing():
    result = get_recipe_ingredients("completely invented dish name")
    assert result["found"] is False
    assert result["ingredients"] == []


def test_paneer_butter_masala_is_a_curry_not_a_biryani():
    names = _names("paneer butter masala")
    assert "paneer" in names
    for biryani_only in ("basmati rice", "biryani masala", "fresh mint"):
        assert biryani_only not in names


def test_known_dishes_still_resolve():
    assert "chicken" in _names("chicken biryani")
    assert "fettuccine pasta" in _names("alfredo pasta")


def test_unknown_dish_hands_the_job_back_to_the_model():
    """A 21-dish database cannot cover what callers ask for.

    Telling the model to "try a common dish name" made it give up on french
    toast, maggi and dosa — dishes it knows perfectly well.
    """
    result = get_recipe_ingredients("french toast")
    assert result["found"] is False
    note = result["note"].lower()
    assert "cannot help" in note or "do not tell" in note
    assert "search_and_add_to_cart" in result["note"]
    assert "try a common dish name" not in note


def test_curated_dishes_do_not_take_the_fallback_path():
    result = get_recipe_ingredients("alfredo pasta")
    assert result["found"] is True
    assert "search_and_add_to_cart" not in result["note"]
