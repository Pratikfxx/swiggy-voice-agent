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

    def test_live_confirmed_checkout_failure_does_not_invite_retry(self):
        agent = _fresh_agent()
        tokens = {"im": "im-token"}

        def fail_create(**kwargs):
            raise RuntimeError("network dropped after checkout")

        with (
            patch.object(agent.client.beta.messages, "create", side_effect=fail_create),
            patch.object(agent.swiggy_address, "maybe_background_refresh"),
            patch.object(agent.swiggy_address, "get_cached_default", return_value=None),
            patch.object(agent.logging, "exception"),
        ):
            response, _ = agent._run_agent_live("yes", [], "voice", "call-test", tokens)

        self.assertIn("check your Swiggy app", response)
        self.assertIn("before trying again", response)
        self.assertNotIn("Please try again in a moment", response)


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


def test_checkout_is_disabled_until_the_caller_confirms(monkeypatch):
    """The spend gate is what stands between a demo and a real order.

    checkout is only offered to the model when the caller's latest message
    reads as a confirmation. Without that, an over-eager model cannot spend
    money even if it decides to.
    """
    import agent

    captured = {}

    class FakeResponse:
        stop_reason = "end_turn"
        content = [type("B", (), {"type": "text", "text": "Milk, 59 rupees. Confirm?"})()]

    def fake_create(surface, live, **kwargs):
        captured["tools"] = kwargs.get("tools")
        return FakeResponse()

    monkeypatch.setattr(agent, "_create_message", fake_create)
    monkeypatch.setattr(agent, "get_access_tokens", lambda keys, user_id=None: {"im": "tok"})
    monkeypatch.setattr(agent.swiggy_address, "get_cached_default", lambda: {"id": "A1", "label": "Home", "area": ""})
    monkeypatch.setattr(agent.swiggy_address, "maybe_background_refresh", lambda: None)

    def spend_enabled_for(message):
        agent.run_agent(message, [], surface="voice", session_id="s", user_id="u")
        for tool in captured["tools"]:
            configs = tool.get("configs") if isinstance(tool, dict) else None
            if configs and "checkout" in configs:
                return configs["checkout"]["enabled"]
        raise AssertionError("checkout config not found")

    assert spend_enabled_for("get me milk") is False
    assert spend_enabled_for("haan") is True


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
        self.assertEqual(line, "milk, bread and eggs, 319 rupees. Confirm?")

    def test_single_item_reads_naturally(self):
        line = self._line({"added": [{"item": "milk"}], "not_found": [], "cart_updated": True, "subtotal": 59})
        self.assertEqual(line, "milk, 59 rupees. Confirm?")

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

    def test_other_tools_are_never_short_circuited(self):
        import agent

        with patch.object(agent, "VOICE_FAST_CONFIRM", True):
            self.assertEqual(agent._fast_confirm_line("get_order_history", "{}", "usual"), "")

    def test_speaks_the_callers_words_not_the_catalogue_title(self):
        line = self._line({
            "added": [
                {"item": "milk", "name": "Mother Dairy Pasteurised Homogenised Cow Milk"},
                {"item": "bread", "name": "NOICE 5 Seed Multigrain Bread (Zero Maida)"},
            ],
            "not_found": [], "cart_updated": True, "subtotal": 79,
        })
        self.assertEqual(line, "milk and bread, 79 rupees. Confirm?")
        self.assertNotIn("Pasteurised", line)

    def test_multiple_quantities_fall_through_so_counts_are_not_understated(self):
        line = self._line({
            "added": [{"item": "butter", "quantity": 2}],
            "not_found": [], "cart_updated": True, "subtotal": 504,
        })
        self.assertEqual(line, "")
