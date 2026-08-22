import asyncio
from unittest.mock import AsyncMock, patch

import agent
import swiggy_address
import swiggy_search


def test_local_cart_helpers_receive_the_normalized_user_id():
    with patch.object(agent, "search_and_add_to_cart", return_value={}) as add:
        agent.execute_tool(
            "search_and_add_to_cart",
            {"queries": ["milk"], "address_id": "A1", "quantity": 1},
            user_id="+91111",
        )
        assert add.call_args.kwargs["user_id"] == "+91111"

    with patch.object(agent, "remove_from_cart", return_value={}) as remove:
        agent.execute_tool(
            "remove_from_cart",
            {"address_id": "A1", "keep_only": ["milk"]},
            user_id="+92222",
        )
        assert remove.call_args.kwargs["user_id"] == "+92222"


def test_cart_summary_resolves_the_callers_token():
    with patch.object(swiggy_search, "get_access_token", return_value="token") as token:
        with patch.object(
            swiggy_search, "open_authenticated_mcp", side_effect=RuntimeError("stop after token")
        ):
            try:
                swiggy_search.get_cart_summary(user_id="+91111")
            except RuntimeError:
                pass
    token.assert_called_once_with("im", user_id="+91111")


def test_address_cache_is_isolated_per_user():
    async def fake_fetch(user_id=""):
        return {"id": "A-" + user_id, "label": user_id, "area": "X"}

    with patch.object(swiggy_address, "fetch_default_address", side_effect=fake_fetch):
        asyncio.run(swiggy_address.refresh_default_address("user-a"))
        asyncio.run(swiggy_address.refresh_default_address("user-b"))

    assert swiggy_address.get_cached_default("user-a")["id"] == "A-user-a"
    assert swiggy_address.get_cached_default("user-b")["id"] == "A-user-b"
