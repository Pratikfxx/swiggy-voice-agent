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
