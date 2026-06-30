import importlib
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
