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
