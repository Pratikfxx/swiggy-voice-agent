import importlib
import os
import sys
import unittest
import warnings
from unittest.mock import patch


def _fresh_agent(env=None):
    base = {"ANTHROPIC_API_KEY": "test-key", "DEMO_MODE": "true"}
    base.update(env or {})
    with patch.dict(os.environ, base, clear=True):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ResourceWarning)
            sys.modules.pop("agent", None)
            return importlib.import_module("agent")


class FakeTextBlock:
    type = "text"
    text = "ok"


class FakeResponse:
    content = [FakeTextBlock()]
    stop_reason = "end_turn"


class FakeRefusalResponse:
    content = []
    stop_reason = "refusal"


class FableModelConfigTests(unittest.TestCase):
    def test_chat_defaults_to_fable_voice_stays_haiku(self):
        agent = _fresh_agent()
        self.assertEqual(agent._model_for("chat"), "claude-fable-5")
        self.assertEqual(agent._model_for("voice"), "claude-haiku-4-5")

    def test_chat_gets_more_output_headroom_than_voice(self):
        agent = _fresh_agent()
        self.assertEqual(agent._max_tokens_for("voice"), 400)
        self.assertGreaterEqual(agent._max_tokens_for("chat"), 4096)

    def test_demo_chat_sets_effort_but_no_fallbacks(self):
        agent = _fresh_agent()
        captured = {}

        def fake_create(**kwargs):
            captured.update(kwargs)
            return FakeResponse()

        with patch.object(agent.client.messages, "create", side_effect=fake_create):
            response, _ = agent._run_agent_demo("milk", [], surface="chat", session_id="wa-test")

        self.assertEqual(response, "ok")
        self.assertEqual(captured["model"], "claude-fable-5")
        self.assertEqual(captured["output_config"], {"effort": agent.CHAT_EFFORT})
        self.assertNotIn("fallbacks", captured)
        self.assertNotIn("betas", captured)

    def test_demo_voice_has_no_fable_extras(self):
        agent = _fresh_agent()
        captured = {}

        def fake_create(**kwargs):
            captured.update(kwargs)
            return FakeResponse()

        with patch.object(agent.client.messages, "create", side_effect=fake_create):
            response, _ = agent._run_agent_demo("milk", [], surface="voice", session_id="call-test")

        self.assertEqual(response, "ok")
        self.assertEqual(captured["model"], "claude-haiku-4-5")
        self.assertNotIn("output_config", captured)
        self.assertNotIn("fallbacks", captured)

    def test_live_chat_includes_fallbacks_and_effort(self):
        agent = _fresh_agent({"DEMO_MODE": "false"})
        captured = {}

        def fake_create(**kwargs):
            captured.update(kwargs)
            return FakeResponse()

        with (
            patch.object(agent.client.beta.messages, "create", side_effect=fake_create),
            patch.object(agent.swiggy_address, "maybe_background_refresh"),
            patch.object(agent.swiggy_address, "get_cached_default", return_value=None),
        ):
            response, _ = agent._run_agent_live("milk", [], "chat", "wa-test", {"im": "t"})

        self.assertEqual(response, "ok")
        self.assertEqual(captured["model"], "claude-fable-5")
        self.assertEqual(captured["output_config"], {"effort": agent.CHAT_EFFORT})
        self.assertEqual(captured["fallbacks"], [{"model": "claude-opus-4-8"}])
        self.assertIn("mcp-client-2025-11-20", captured["betas"])
        self.assertIn("server-side-fallback-2026-06-01", captured["betas"])

    def test_live_voice_excludes_fable_extras(self):
        agent = _fresh_agent({"DEMO_MODE": "false"})
        captured = {}

        def fake_create(**kwargs):
            captured.update(kwargs)
            return FakeResponse()

        with (
            patch.object(agent.client.beta.messages, "create", side_effect=fake_create),
            patch.object(agent.swiggy_address, "maybe_background_refresh"),
            patch.object(agent.swiggy_address, "get_cached_default", return_value=None),
        ):
            response, _ = agent._run_agent_live("milk", [], "voice", "call-test", {"im": "t"})

        self.assertEqual(response, "ok")
        self.assertEqual(captured["model"], "claude-haiku-4-5")
        self.assertNotIn("output_config", captured)
        self.assertNotIn("fallbacks", captured)
        self.assertEqual(captured["betas"], ["mcp-client-2025-11-20"])

    def test_fallback_can_be_disabled_via_env(self):
        agent = _fresh_agent({"DEMO_MODE": "false", "AGENT_FALLBACK_MODEL": ""})
        extras, betas = agent._fable_chat_kwargs("chat", live=True)
        self.assertNotIn("fallbacks", extras)
        self.assertEqual(betas, [])
        self.assertEqual(extras["output_config"], {"effort": agent.CHAT_EFFORT})

    def test_demo_refusal_returns_safe_message(self):
        agent = _fresh_agent()

        with patch.object(agent.client.messages, "create", return_value=FakeRefusalResponse()):
            response, messages = agent._run_agent_demo("bad", [], surface="chat", session_id="wa-test")

        self.assertEqual(response, agent.REFUSAL_MESSAGE)
        self.assertEqual(messages[-1]["content"], agent.REFUSAL_MESSAGE)

    def test_live_refusal_returns_safe_message(self):
        agent = _fresh_agent({"DEMO_MODE": "false"})

        with (
            patch.object(agent.client.beta.messages, "create", return_value=FakeRefusalResponse()),
            patch.object(agent.swiggy_address, "maybe_background_refresh"),
            patch.object(agent.swiggy_address, "get_cached_default", return_value=None),
        ):
            response, _ = agent._run_agent_live("bad", [], "chat", "wa-test", {"im": "t"})

        self.assertEqual(response, agent.REFUSAL_MESSAGE)


if __name__ == "__main__":
    unittest.main()
