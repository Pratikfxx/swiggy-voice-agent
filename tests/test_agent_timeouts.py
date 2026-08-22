import importlib
import json
import os
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch

import swiggy_auth


def _fresh_agent():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ResourceWarning)
        sys.modules.pop("agent", None)
        return importlib.import_module("agent")


class AgentTimeoutTests(unittest.TestCase):
    def test_live_spend_result_reports_order_metadata_without_prose_completion(self):
        agent = _fresh_agent()

        class FakeResponse:
            content = [
                {
                    "type": "mcp_tool_use",
                    "name": "checkout",
                    "id": "checkout-1",
                    "input": {"items": [{"name": "Milk"}]},
                },
                {
                    "type": "mcp_tool_result",
                    "tool_use_id": "checkout-1",
                    "is_error": False,
                    "content": [{
                        "type": "text",
                        "text": "Instamart order placed successfully. Arriving in 15 minutes.",
                    }],
                },
                {"type": "text", "text": "Your groceries are on the way."},
            ]
            stop_reason = "end_turn"

        with (
            patch.object(agent.client.beta.messages, "create", return_value=FakeResponse()),
            patch.object(agent.swiggy_address, "maybe_background_refresh"),
            patch.object(agent.swiggy_address, "get_cached_default", return_value=None),
            patch.object(agent, "save_order"),
        ):
            _, _, meta = agent._run_agent_live(
                "yes", [], "voice", "call-test", {"im": "im-token"}, return_meta=True
            )

        self.assertEqual(meta, {"order_placed": True})

    def test_live_spend_error_does_not_report_order_metadata(self):
        agent = _fresh_agent()

        class FakeResponse:
            content = [
                {
                    "type": "mcp_tool_use",
                    "name": "checkout",
                    "id": "checkout-1",
                    "input": {"items": [{"name": "Milk"}]},
                },
                {
                    "type": "mcp_tool_result",
                    "tool_use_id": "checkout-1",
                    "is_error": True,
                },
                {"type": "text", "text": "I couldn't complete that request."},
            ]
            stop_reason = "end_turn"

        with (
            patch.object(agent.client.beta.messages, "create", return_value=FakeResponse()),
            patch.object(agent.swiggy_address, "maybe_background_refresh"),
            patch.object(agent.swiggy_address, "get_cached_default", return_value=None),
            patch.object(agent, "save_order"),
        ):
            _, _, meta = agent._run_agent_live(
                "yes", [], "voice", "call-test", {"im": "im-token"}, return_meta=True
            )

        self.assertEqual(meta, {"order_placed": False})

    def test_business_failure_is_not_treated_as_an_order(self):
        agent = _fresh_agent()

        blocks = [
            {"type": "mcp_tool_use", "name": "checkout", "id": "checkout-1", "input": {}},
            {
                "type": "mcp_tool_result",
                "tool_use_id": "checkout-1",
                "is_error": False,
                "content": [{
                    "type": "text",
                    "text": "Checkout failed: one or more stores could not place the order.",
                }],
            },
        ]

        with patch.object(agent, "save_order") as save:
            placed = agent._save_live_order_if_any(blocks, "call-1", "user-1")

        self.assertFalse(placed)
        save.assert_not_called()

    def test_unknown_empty_checkout_result_is_uncertain_not_success(self):
        agent = _fresh_agent()
        blocks = [
            {"type": "mcp_tool_use", "name": "checkout", "id": "checkout-1", "input": {}},
            {"type": "mcp_tool_result", "tool_use_id": "checkout-1", "is_error": False},
        ]
        assert agent._save_live_order_if_any(blocks, "call-1", "user-1") is False

    def test_process_message_default_returns_bare_string(self):
        agent = _fresh_agent()

        with (
            patch.object(agent, "get_session", return_value=[]),
            patch.object(agent, "run_agent", return_value=("hello", [])),
            patch.object(agent, "update_session"),
        ):
            result = agent.process_message("call-test", "hi")

        self.assertEqual(result, "hello")

    def test_process_message_can_return_order_metadata(self):
        agent = _fresh_agent()

        with (
            patch.object(agent, "get_session", return_value=[]),
            patch.object(
                agent,
                "run_agent",
                return_value=("accepted", [], {"order_placed": True}),
            ),
            patch.object(agent, "update_session"),
        ):
            result = agent.process_message("call-test", "yes", return_meta=True)

        self.assertEqual(result, ("accepted", {"order_placed": True}))

    def test_live_mode_uses_only_active_instamart_token(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            token_store = os.path.join(tmpdir, ".swiggy_tokens.json")
            with patch.dict(
                os.environ,
                {
                    "DEMO_MODE": "false",
                    "ANTHROPIC_API_KEY": "test-key",
                    "SWIGGY_IM_TOKEN": "im-token",
                },
                clear=True,
            ):
                agent = _fresh_agent()

                with (
                    patch.object(swiggy_auth, "TOKEN_STORE", token_store),
                    patch.object(agent, "_run_agent_live", return_value=("live", [])) as live,
                    patch.object(agent, "_run_agent_demo", return_value=("demo", [])) as demo,
                ):
                    response, _ = agent.run_agent(
                        "milk",
                        [],
                        surface="voice",
                        session_id="call-test",
                    )

        self.assertEqual(response, "live")
        demo.assert_not_called()
        live.assert_called_once()
        self.assertEqual(live.call_args.args[4], {"im": "im-token"})

    def test_run_agent_passes_user_id_to_token_resolution(self):
        agent = _fresh_agent()
        with (
            patch.object(agent, "get_access_tokens", return_value={"im": "im-token"}) as get_tokens,
            patch.object(agent, "_run_agent_live", return_value=("live", [])),
        ):
            agent.run_agent("milk", [], session_id="call-test", user_id="user-test")

        get_tokens.assert_called_once_with(agent.ACTIVE_TOKEN_KEYS, user_id="user-test")

    def test_live_mode_fails_closed_when_active_instamart_token_is_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            token_store = os.path.join(tmpdir, ".swiggy_tokens.json")
            with patch.dict(
                os.environ,
                {
                    "DEMO_MODE": "false",
                    "ANTHROPIC_API_KEY": "test-key",
                },
                clear=True,
            ):
                agent = _fresh_agent()

                with (
                    patch.object(swiggy_auth, "TOKEN_STORE", token_store),
                    patch.object(agent, "_run_agent_live", return_value=("live", [])) as live,
                    patch.object(agent, "_run_agent_demo", return_value=("demo", [])) as demo,
                ):
                    response, _ = agent.run_agent(
                        "milk",
                        [],
                        surface="voice",
                        session_id="call-test",
                    )

        self.assertIn("Swiggy login is not ready", response)
        demo.assert_not_called()
        live.assert_not_called()

    def test_demo_voice_calls_use_short_api_timeout(self):
        agent = _fresh_agent()
        captured = {}

        class FakeTextBlock:
            type = "text"
            text = "ok"

        class FakeResponse:
            content = [FakeTextBlock()]
            stop_reason = "end_turn"

        def fake_create(**kwargs):
            captured.update(kwargs)
            return FakeResponse()

        with patch.object(agent.client.messages, "create", side_effect=fake_create):
            response, _ = agent._run_agent_demo("milk", [], surface="voice", session_id="call-test")

        self.assertEqual(response, "ok")
        self.assertEqual(captured["timeout"], agent._api_timeout_for("voice"))
        self.assertLess(captured["timeout"], agent._api_timeout_for("chat"))

    def test_live_voice_calls_use_short_api_timeout(self):
        agent = _fresh_agent()
        captured = {}

        class FakeTextBlock:
            type = "text"
            text = "ok"

        class FakeResponse:
            content = [FakeTextBlock()]
            stop_reason = "end_turn"

        def fake_create(**kwargs):
            captured.update(kwargs)
            return FakeResponse()

        tokens = {"food": "food-token", "im": "im-token", "dineout": "dineout-token"}
        with (
            patch.object(agent.client.beta.messages, "create", side_effect=fake_create),
            patch.object(agent.swiggy_address, "maybe_background_refresh"),
            patch.object(agent.swiggy_address, "get_cached_default", return_value=None),
        ):
            response, _ = agent._run_agent_live("milk", [], "voice", "call-test", tokens)

        self.assertEqual(response, "ok")
        self.assertEqual(captured["timeout"], agent._api_timeout_for("voice"))
        self.assertLess(captured["timeout"], agent._api_timeout_for("chat"))
        self.assertNotIn("speed", captured)

    def test_final_checkout_turn_uses_a_longer_api_timeout(self):
        agent = _fresh_agent()
        captured = {}
        history = [{
            "role": "assistant",
            "content": (
                "Full cart is 217 rupees, delivered to Ghar address. "
                "Payment is cash on delivery. Place the order?"
            ),
        }]

        class FakeResponse:
            content = [type("B", (), {"type": "text", "text": "not placed"})()]
            stop_reason = "end_turn"

        def fake_create(**kwargs):
            captured.update(kwargs)
            return FakeResponse()

        with (
            patch.object(agent.client.beta.messages, "create", side_effect=fake_create),
            patch.object(agent.swiggy_address, "maybe_background_refresh"),
            patch.object(agent.swiggy_address, "get_cached_default", return_value=None),
        ):
            agent._run_agent_live("yes", history, "voice", "call-test", {"im": "token"})

        self.assertEqual(captured["timeout"], agent.VOICE_CHECKOUT_API_TIMEOUT_SECS)
        self.assertGreater(captured["timeout"], agent._api_timeout_for("voice"))

    def test_live_confirmed_checkout_failure_does_not_invite_retry(self):
        agent = _fresh_agent()
        tokens = {"im": "im-token"}
        history = [{
            "role": "assistant",
            "content": (
                "Your full cart is 217 rupees, delivered to Ghar. "
                "Payment is cash on delivery. Place the order?"
            ),
        }]

        def fail_create(**kwargs):
            raise RuntimeError("network dropped after checkout")

        with (
            patch.object(agent.client.beta.messages, "create", side_effect=fail_create),
            patch.object(agent.swiggy_address, "maybe_background_refresh"),
            patch.object(agent.swiggy_address, "get_cached_default", return_value=None),
            patch.object(agent.logging, "exception"),
        ):
            response, _ = agent._run_agent_live("yes", history, "voice", "call-test", tokens)

        self.assertIn("check your Swiggy app", response)
        self.assertIn("before trying again", response)
        self.assertNotIn("Please try again in a moment", response)

    def test_failure_after_a_product_confirmation_is_not_called_checkout_uncertainty(self):
        agent = _fresh_agent()
        tokens = {"im": "im-token"}
        history = [{"role": "assistant", "content": "Milk, 59 rupees. Add it?"}]

        with (
            patch.object(agent.client.beta.messages, "create", side_effect=RuntimeError("down")),
            patch.object(agent.swiggy_address, "maybe_background_refresh"),
            patch.object(agent.swiggy_address, "get_cached_default", return_value=None),
            patch.object(agent.logging, "exception"),
        ):
            response, _ = agent._run_agent_live("yes", history, "voice", "call-test", tokens)

        self.assertEqual(response, agent.LIVE_GENERIC_FAILURE_MESSAGE)


if __name__ == "__main__":
    unittest.main()


def test_reply_uses_only_text_after_the_last_tool_call():
    """A server-side MCP turn holds narration, tool blocks, then the answer.

    Joining all of it made the agent ask and answer itself in one breath:
    "Add this one? Perfect! It is in your cart." — two voices on one call.
    """
    import agent

    content = [
        {"type": "text", "text": "Add this one?"},
        {"type": "mcp_tool_use", "name": "update_cart", "id": "t1", "input": {}},
        {"type": "mcp_tool_result", "tool_use_id": "t1", "is_error": False},
        {"type": "text", "text": "Amul Taaza is in your cart, 59 rupees. Confirm?"},
    ]
    spoken = agent._text_after_last_tool(content)
    assert spoken == "Amul Taaza is in your cart, 59 rupees. Confirm?"
    assert "Add this one?" not in spoken


def test_reply_falls_back_when_model_only_spoke_before_tools():
    import agent

    content = [
        {"type": "text", "text": "Checking that now."},
        {"type": "mcp_tool_use", "name": "search_products", "id": "t1", "input": {}},
    ]
    assert agent._text_after_last_tool(content) == "Checking that now."


def test_plain_string_content_passes_through():
    import agent

    assert agent._text_after_last_tool("Done. Arriving in 12 minutes.") == "Done. Arriving in 12 minutes."


def test_checkout_requires_the_final_cart_address_and_payment_confirmation():
    """A generic "yes" may confirm a variant or address, not spending."""
    import agent

    product_confirmation = [
        {"role": "assistant", "content": "Milk, 59 rupees. Add it?"},
    ]
    final_confirmation = [
        {
            "role": "assistant",
            "content": (
                "Your full cart is 217 rupees, delivered to Ghar. "
                "Payment is cash on delivery. Place the order?"
            ),
        },
    ]

    assert agent._checkout_ready("yes", product_confirmation, "voice") is False
    assert agent._checkout_ready("yes", final_confirmation, "voice") is True
    assert agent._checkout_ready("get me milk", final_confirmation, "voice") is False
    assert agent._checkout_ready("yes, but remove bread", final_confirmation, "voice") is False
    assert agent._checkout_ready("haan, deliver to office instead", final_confirmation, "voice") is False
    assert agent._checkout_ready("yes, add a coke too", final_confirmation, "voice") is False


def test_chat_checkout_accepts_a_named_upi_method_after_final_summary():
    import agent

    history = [{
        "role": "assistant",
        "content": (
            "Your full cart is 217 rupees, delivered to Ghar. "
            "Payment is Google Pay UPI. Proceed?"
        ),
    }]
    assert agent._checkout_ready("proceed", history, "chat") is True


def test_internal_widget_tools_are_not_exposed_to_the_model():
    import agent

    configs = agent._mcp_tool_configs(checkout_enabled=False)
    assert configs["checkout"] == {"enabled": False}
    for internal in ("get_delivery_status", "check_payment_status", "confirm_order"):
        assert configs[internal] == {"enabled": False}


class FastConfirmTests(unittest.TestCase):
    """Skipping the second model round trip is worth ~4.4s of a 13.2s turn,
    but only when the sentence is fully determined by the cart tool's output."""

    def _line(self, raw, user_message="milk and bread", enabled=True):
        import agent

        with patch.object(agent, "VOICE_FAST_CONFIRM", enabled):
            return agent._fast_confirm_line("search_and_add_to_cart", json.dumps(raw), user_message)

    def test_clean_cart_produces_a_spoken_confirmation(self):
        line = self._line({
            "added": [{"item": "milk"}, {"item": "bread"}, {"item": "eggs"}],
            "not_found": [], "cart_updated": True, "subtotal": 319,
        })
        self.assertEqual(line, "milk, bread and eggs, 319 rupees. Keep these?")

    def test_single_item_reads_naturally(self):
        line = self._line({"added": [{"item": "milk"}], "not_found": [], "cart_updated": True, "subtotal": 59})
        self.assertEqual(line, "milk, 59 rupees. Keep these?")

    def test_missing_items_fall_through_to_the_model(self):
        self.assertEqual(self._line({
            "added": [{"item": "milk"}], "not_found": ["caviar"],
            "cart_updated": True, "subtotal": 59,
        }), "")

    def test_unsaved_cart_falls_through_to_the_model(self):
        self.assertEqual(self._line({
            "added": [{"query": "milk"}], "not_found": [], "cart_updated": False, "subtotal": 59,
        }), "")

    def test_non_ascii_request_falls_through_so_hindi_is_answered_in_hindi(self):
        self.assertEqual(self._line(
            {"added": [{"item": "doodh"}], "not_found": [], "cart_updated": True, "subtotal": 59},
            user_message="mujhe doodh chahiye है",
        ), "")

    def test_disabled_by_default_flag_falls_through(self):
        self.assertEqual(self._line(
            {"added": [{"item": "milk"}], "not_found": [], "cart_updated": True, "subtotal": 59},
            enabled=False,
        ), "")

    def test_a_cart_with_other_items_falls_through_to_the_model(self):
        """Merging means the cart can hold items from a previous call. Reading
        back only this turn's additions hid them until checkout charged for
        them — the "random orders" a caller received."""
        line = self._line({
            "added": [{"item": "milk"}],
            "not_found": [], "cart_updated": True,
            "subtotal": 28, "cart_total": 405,
            "cart_has_other_items": True,
        })
        self.assertEqual(line, "")

    def test_a_clean_cart_quotes_the_cart_total(self):
        line = self._line({
            "added": [{"item": "milk"}, {"item": "bread"}],
            "not_found": [], "cart_updated": True,
            "subtotal": 77, "cart_total": 79,
            "cart_has_other_items": False,
        })
        self.assertEqual(line, "milk and bread, 79 rupees. Keep these?")

    def test_a_formatted_rupee_total_is_still_spoken(self):
        """cart_total comes back as "\u20b979", not 79, and the float() on it
        failed silently — disabling the shortcut entirely."""
        line = self._line({
            "added": [{"item": "milk"}, {"item": "bread"}],
            "not_found": [], "cart_updated": True,
            "subtotal": 77, "cart_total": "\u20b979",
            "cart_has_other_items": False,
        })
        self.assertEqual(line, "milk and bread, 79 rupees. Keep these?")

    def test_other_tools_are_never_short_circuited(self):
        import agent

        with patch.object(agent, "VOICE_FAST_CONFIRM", True):
            self.assertEqual(agent._fast_confirm_line("get_order_history", "{}", "usual"), "")

    def test_speaks_the_callers_words_not_the_catalogue_title(self):
        line = self._line({
            "added": [
                {
                    "item": "milk",
                    "name": "Mother Dairy Pasteurised Homogenised Cow Milk",
                    "brand": "Mother Dairy",
                    "variant": "500 ml",
                },
                {
                    "item": "bread",
                    "name": "NOICE 5 Seed Multigrain Bread (Zero Maida)",
                    "brand": "NOICE",
                    "variant": "250 g",
                },
            ],
            "not_found": [], "cart_updated": True, "subtotal": 79,
        })
        self.assertEqual(
            line,
            "Mother Dairy milk 500 ml and NOICE bread 250 g, 79 rupees. Keep these?",
        )
        self.assertNotIn("Pasteurised", line)

    def test_multiple_quantities_fall_through_so_counts_are_not_understated(self):
        line = self._line({
            "added": [{"item": "butter", "quantity": 2}],
            "not_found": [], "cart_updated": True, "subtotal": 504,
        })
        self.assertEqual(line, "")


def test_fast_selection_yes_is_a_distinct_non_spending_transition():
    import agent

    history = [{
        "role": "assistant",
        "content": "Diet Coke 330 ml and Monster Zero 350 ml, 171 rupees. Keep these?",
    }]
    assert agent._selection_accepted("yes", history) is True
    assert agent._selection_accepted("yes, remove diet coke", history) is False
    assert agent._selection_accepted("yes", [{"role": "assistant", "content": "Confirm?"}]) is False


def test_final_checkout_line_uses_real_cart_address_and_total():
    import agent

    line = agent._final_checkout_line({
        "cart_total": "₹171",
        "address": "Ghar",
        "items": ["Diet Coke", "Monster Energy Zero"],
    })
    assert line == (
        "Full cart is 171 rupees, delivered to Ghar, cash on delivery. "
        "Place the order?"
    )

class TransientMcpRetryTests(unittest.TestCase):
    """Anthropic reports "cannot reach Swiggy" as a 400, which is not retried
    by the capacity path. It arrives in bursts and succeeds moments later."""

    def _error(self, message, status=400):
        import anthropic

        response = type("R", (), {"status_code": status, "headers": {}, "request": None})()
        exc = anthropic.APIStatusError(message, response=response, body=None)
        exc.status_code = status
        return exc

    def test_connector_error_is_recognised_as_transient(self):
        import agent

        self.assertTrue(agent._is_transient_mcp_error(
            self._error("Error code: 400 - Error while communicating with MCP server.")))
        self.assertTrue(agent._is_transient_mcp_error(
            self._error("MCP server returned an error while listing tools: Connection closed")))

    def test_ordinary_bad_request_is_not_retried(self):
        import agent

        self.assertFalse(agent._is_transient_mcp_error(
            self._error("Error code: 400 - messages.0: invalid role")))

    def test_capacity_errors_stay_on_the_fallback_path(self):
        import agent

        self.assertFalse(agent._is_transient_mcp_error(self._error("overloaded", status=529)))
        self.assertTrue(agent._is_capacity_error(self._error("overloaded", status=529)))

    def test_a_transient_error_is_retried_once_on_the_same_model(self):
        import agent

        attempts = []

        def flaky(model, **kwargs):
            attempts.append(model)
            if len(attempts) == 1:
                raise self._error("Error while communicating with MCP server.")
            return "ok"

        with patch.object(agent, "_model_for", return_value="claude-haiku-4-5"), \
             patch.object(agent.client.beta, "messages") as beta:
            beta.create.side_effect = lambda model, **kw: flaky(model, **kw)
            result = agent._create_message("voice", True, messages=[], max_tokens=10)

        self.assertEqual(result, "ok")
        self.assertEqual(attempts, ["claude-haiku-4-5", "claude-haiku-4-5"])
