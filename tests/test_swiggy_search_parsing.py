"""Search/cart response parsing.

Swiggy returns TWO content blocks — human-readable display instructions
first, then the JSON — and moved products from {"data": {"products": []}} to
a top-level {"products": []}. Reading only block zero silently emptied every
multi-item cart in production while reporting no error at all.
"""

import json
import types

import swiggy_search


def _block(text):
    return types.SimpleNamespace(text=text)


def _result(*texts, is_error=False):
    return types.SimpleNamespace(content=[_block(t) for t in texts], is_error=is_error)


PROSE = 'Found 20 product(s) matching "milk".\n\nDISPLAY INSTRUCTIONS:\n- Show each product'

VARIATION = {
    "skuId": "UP6SXNFACP",
    "spinId": "SPIN1",
    "displayName": "Mother Dairy Cow Milk",
    "brandName": "Mother Dairy",
    "quantityDescription": "500 ml",
    "price": {"mrp": 30, "offerPrice": 30},
    "isInStockAndAvailable": True,
}
PRODUCT = {"displayName": "Milk", "brand": "Mother Dairy", "variations": [VARIATION]}


def test_products_found_behind_the_prose_block():
    result = _result(PROSE, json.dumps({"products": [PRODUCT], "nextOffset": 12}))
    match = swiggy_search._top_match("milk", result)
    assert match["found"] is True
    assert match["skuId"] == "UP6SXNFACP"
    assert match["price"] == 30


def test_legacy_data_products_envelope_still_parses():
    result = _result(PROSE, json.dumps({"data": {"products": [PRODUCT]}}))
    assert swiggy_search._top_match("milk", result)["found"] is True


def test_no_json_block_reports_not_found_without_raising():
    assert swiggy_search._top_match("milk", _result(PROSE))["found"] is False


def test_out_of_stock_variation_is_skipped():
    dead = dict(VARIATION, isInStockAndAvailable=False)
    result = _result(PROSE, json.dumps({"products": [{"variations": [dead]}]}))
    assert swiggy_search._top_match("milk", result)["found"] is False


def test_cart_accepted_reads_the_cart_body_not_the_prose():
    cart = json.dumps({"cartId": "C1", "items": [{"skuId": "X"}], "cartTotalAmount": 319})
    assert swiggy_search._cart_accepted(_result("Cart updated successfully.", cart)) is True


def test_cart_rejected_when_the_tool_errors():
    cart = json.dumps({"cartId": "C1"})
    assert swiggy_search._cart_accepted(_result("boom", cart, is_error=True)) is False


def test_cart_rejected_when_no_cart_body_comes_back():
    assert swiggy_search._cart_accepted(_result("Cart updated successfully.")) is False


def test_merge_keeps_what_is_already_in_the_cart():
    """update_cart REPLACES the cart, so adding item two without resending
    item one silently deleted it. Confirmed live: adding Diet Coke removed
    Monster Energy."""
    existing = [{"spinId": "S1", "skuId": "MONSTER", "quantity": 1}]
    additions = [{"spinId": "S2", "skuId": "COKE", "quantity": 1}]
    merged = swiggy_search._merge_cart_items(existing, additions)
    assert [i["skuId"] for i in merged] == ["MONSTER", "COKE"]


def test_merge_does_not_duplicate_an_item_already_present():
    existing = [{"spinId": "S1", "skuId": "MONSTER", "quantity": 2}]
    additions = [{"spinId": "S1", "skuId": "MONSTER", "quantity": 1}]
    merged = swiggy_search._merge_cart_items(existing, additions)
    assert len(merged) == 1
    assert merged[0]["quantity"] == 2


def test_merge_into_an_empty_cart_is_just_the_additions():
    additions = [{"spinId": "S2", "skuId": "COKE", "quantity": 1}]
    assert swiggy_search._merge_cart_items([], additions) == additions


def test_existing_items_are_read_from_the_cart_json_block():
    cart = json.dumps({
        "cartId": "C1",
        "items": [
            {"spinId": "S1", "skuId": "MONSTER", "quantity": 2, "itemName": "Monster"},
            {"spinId": "S2", "skuId": "COKE", "quantity": 1, "itemName": "Diet Coke"},
            {"itemName": "broken row without ids"},
        ],
    })

    class FakeSession:
        async def call_tool(self, name, args):
            assert name == "get_cart"
            return _result("Cart retrieved successfully.", cart)

    import asyncio

    items = asyncio.run(swiggy_search._existing_cart_items(FakeSession()))
    assert items == [
        {"spinId": "S1", "skuId": "MONSTER", "quantity": 2},
        {"spinId": "S2", "skuId": "COKE", "quantity": 1},
    ]


def test_unreadable_cart_does_not_wipe_the_additions():
    class FailingSession:
        async def call_tool(self, name, args):
            raise RuntimeError("get_cart down")

    import asyncio

    assert asyncio.run(swiggy_search._existing_cart_items(FailingSession())) == []


import pytest


@pytest.mark.parametrize("phrase,term,count", [
    ("six pack of diet coke", "diet coke", 6),
    ("diet coke 6 pack", "diet coke", 6),
    ("one can of diet coke", "diet coke", 1),
    ("2 litres of milk", "milk", 2),
    ("two packets of butter", "butter", 2),
    ("a can of coke", "coke", 1),
    ("milk", "milk", None),
])
def test_pack_size_is_split_off_the_search_term(phrase, term, count):
    """Sent verbatim, "six pack of diet coke" came back as Red Bull."""
    assert swiggy_search._split_pack_size(phrase) == (term, count)


@pytest.mark.parametrize("phrase", ["6 seed bread", "7up", "dozen eggs"])
def test_numbers_that_belong_to_the_product_name_survive(phrase):
    term, count = swiggy_search._split_pack_size(phrase)
    assert term == phrase
    assert count is None


def _coke_product():
    def variation(desc, price, sku):
        return {
            "skuId": sku, "spinId": "SP" + sku, "displayName": "Diet Coke Can",
            "brandName": "Coca-Cola", "quantityDescription": desc,
            "price": {"offerPrice": price}, "isInStockAndAvailable": True,
        }
    return {"displayName": "Diet Coke", "variations": [
        variation("330 ml x 6", 259, "SIX"),
        variation("330 ml x 4", 186, "FOUR"),
        variation("330 ml", 50, "ONE"),
    ]}


def test_bare_request_takes_the_cheapest_pack():
    result = _result(PROSE, json.dumps({"products": [_coke_product()]}))
    assert swiggy_search._top_match("diet coke", result)["skuId"] == "ONE"


def test_named_pack_size_wins_over_cheapest():
    result = _result(PROSE, json.dumps({"products": [_coke_product()]}))
    assert swiggy_search._top_match("six pack of diet coke", result, 6)["skuId"] == "SIX"


def test_unavailable_pack_size_falls_back_to_cheapest():
    result = _result(PROSE, json.dumps({"products": [_coke_product()]}))
    assert swiggy_search._top_match("99 pack of diet coke", result, 99)["skuId"] == "ONE"


def _product(name, sku):
    return {
        "displayName": name,
        "variations": [{
            "skuId": sku,
            "spinId": "SP" + sku,
            "displayName": name,
            "quantityDescription": "350 ml",
            "price": {"offerPrice": 100},
            "isInStockAndAvailable": True,
        }],
    }


def test_exact_single_product_beats_a_combo_ranked_first():
    products = [
        _product("Monster Energy Ultra Zero Sugar, Coca-Cola Zero Combo", "COMBO"),
        _product("Monster Energy Ultra Zero Sugar", "SINGLE"),
    ]
    result = _result(PROSE, json.dumps({"products": products}))
    assert swiggy_search._top_match("monster energy zero", result)["skuId"] == "SINGLE"


def test_a_relevant_product_beats_an_unrelated_first_result():
    products = [
        _product("Supreme Harvest Cassia Taj Roll", "CASSIA"),
        _product("Organic Cinnamon Whole", "CINNAMON"),
    ]
    result = _result(PROSE, json.dumps({"products": products}))
    assert swiggy_search._top_match("cinnamon", result)["skuId"] == "CINNAMON"


def test_equal_relevance_preserves_swiggys_ranking():
    products = [_product("NOICE Multigrain Bread", "FIRST"), _product("White Bread", "SECOND")]
    result = _result(PROSE, json.dumps({"products": products}))
    assert swiggy_search._top_match("bread", result)["skuId"] == "FIRST"


def test_unavailable_best_match_does_not_corrupt_the_fallback_order():
    unavailable_combo = _product(
        "Monster Energy Ultra Zero Sugar, Coca-Cola Zero Combo", "COMBO"
    )
    unavailable_combo["variations"][0]["isInStockAndAvailable"] = False
    exact_single = _product("Monster Energy Ultra Zero Sugar", "SINGLE")
    mango = _product("Monster Ultra Fiesta Mango Energy Drink Zero Sugar", "MANGO")
    result = _result(
        PROSE,
        json.dumps({"products": [unavailable_combo, mango, exact_single]}),
    )
    assert swiggy_search._top_match("monster energy zero", result)["skuId"] == "SINGLE"


def test_query_word_does_not_match_inside_an_unrelated_word():
    products = [_product("Toilet Cleaner", "TOILET"), _product("Sunflower Oil", "OIL")]
    result = _result(PROSE, json.dumps({"products": products}))
    assert swiggy_search._top_match("oil", result)["skuId"] == "OIL"


def test_keep_list_matches_the_right_lines():
    """Dropping by "sugar" also removed "Monster Energy Zero Sugar", which is
    why the tool prefers naming what to keep."""
    monster = "Monster Energy Ultra Zero Sugar350ml, Coca-Cola Zero Can300ml"
    assert swiggy_search._matches(monster, "monster") is True
    assert swiggy_search._matches("Coca-Cola Diet Coke Can 330ml", "diet coke") is True
    assert swiggy_search._matches("NOICE 5 Seed Multigrain Bread", "diet coke") is False


def test_word_boundaries_stop_partial_word_hits():
    assert swiggy_search._matches("Amul Butter", "but") is True      # prefix is intended
    assert swiggy_search._matches("Amul Butter", "rebutter") is False


def test_removal_requires_something_to_go_on():
    import asyncio

    out = asyncio.run(swiggy_search._remove_from_cart([], [], "ADDR1"))
    assert "error" in out


CART_LINES = [
    {"itemName": "Monster Energy Ultra Zero Sugar", "skuId": "M", "spinId": "SM", "quantity": 1},
    {"itemName": "Diet Coke Can", "skuId": "D", "spinId": "SD", "quantity": 1},
    {"itemName": "NOICE 5 Seed Multigrain Bread", "skuId": "B", "spinId": "SB", "quantity": 1},
]


def test_keep_list_retains_only_the_named_products():
    keep, dropped = swiggy_search._partition_cart(CART_LINES, [], ["monster", "diet coke"])
    assert [k["skuId"] for k in keep] == ["M", "D"]
    assert [d["skuId"] for d in dropped] == ["B"]


def test_a_category_word_matches_nothing_rather_than_everything():
    """"keep only the drinks" matched no product name — they are "Monster
    Energy Ultra Zero Sugar" and "Diet Coke Can" — and wiped the cart. The
    caller of this function must refuse to write an empty cart back."""
    keep, dropped = swiggy_search._partition_cart(CART_LINES, [], ["drinks"])
    assert keep == []
    assert len(dropped) == 3


def test_remove_list_drops_only_the_named_products():
    keep, dropped = swiggy_search._partition_cart(CART_LINES, ["bread"], [])
    assert [d["skuId"] for d in dropped] == ["B"]
    assert [k["skuId"] for k in keep] == ["M", "D"]


def test_refusal_payload_says_the_cart_is_untouched():
    """The model read the old refusal and told the caller their cart was
    empty, which was the opposite of the truth."""
    import inspect

    source = inspect.getsource(swiggy_search._remove_from_cart)
    assert "cart_unchanged" in source
    assert "Do NOT say the cart is empty" in source


def test_address_parser_finds_json_behind_a_prose_block():
    """get_addresses returns "Found 29 saved addresses (page 1 of 3)" first.
    Taking block zero is the bug that silently emptied product searches."""
    import types

    import swiggy_address

    prose = types.SimpleNamespace(text="Found 29 saved addresses (page 1 of 3):")
    data = types.SimpleNamespace(text=json.dumps({"addresses": [
        {"id": "A1", "addressTag": "Ghar", "addressLine": "A403 Satellite Gardens"},
    ]}))
    result = types.SimpleNamespace(content=[prose, data], structuredContent=None)

    parsed = swiggy_address._parse_addresses_result(result)
    assert parsed["id"] == "A1"
    assert parsed["label"] == "Ghar"


def test_address_parser_returns_none_when_no_block_is_json():
    import types

    import swiggy_address

    result = types.SimpleNamespace(
        content=[types.SimpleNamespace(text="no addresses here")], structuredContent=None
    )
    assert swiggy_address._parse_addresses_result(result) is None
