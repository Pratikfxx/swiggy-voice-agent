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


class ChatModelConfigTests(unittest.TestCase):
    def test_chat_defaults_to_sonnet_voice_stays_haiku(self):
        agent = _fresh_agent()
        self.assertEqual(agent._model_for("chat"), "claude-sonnet-5")
        self.assertEqual(agent._model_for("voice"), "claude-haiku-4-5")

    def test_chat_max_tokens_modest_voice_smaller(self):
        agent = _fresh_agent()
        self.assertEqual(agent._max_tokens_for("voice"), 400)
        self.assertEqual(agent._max_tokens_for("chat"), 1024)

    def test_demo_chat_disables_thinking_to_save_tokens(self):
        agent = _fresh_agent()
        captured = {}

        def fake_create(**kwargs):
            captured.update(kwargs)
            return FakeResponse()

        with patch.object(agent.client.messages, "create", side_effect=fake_create):
            response, _ = agent._run_agent_demo("milk", [], surface="chat", session_id="wa-test")

        self.assertEqual(response, "ok")
        self.assertEqual(captured["model"], "claude-sonnet-5")
        self.assertEqual(captured["thinking"], {"type": "disabled"})
        self.assertNotIn("fallbacks", captured)
        self.assertNotIn("output_config", captured)

    def test_demo_voice_has_no_thinking_param(self):
        agent = _fresh_agent()
        captured = {}

        def fake_create(**kwargs):
            captured.update(kwargs)
            return FakeResponse()

        with patch.object(agent.client.messages, "create", side_effect=fake_create):
            response, _ = agent._run_agent_demo("milk", [], surface="voice", session_id="call-test")

        self.assertEqual(response, "ok")
        self.assertEqual(captured["model"], "claude-haiku-4-5")
        self.assertNotIn("thinking", captured)

    def test_live_chat_disables_thinking_no_fallback(self):
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
        self.assertEqual(captured["model"], "claude-sonnet-5")
        self.assertEqual(captured["thinking"], {"type": "disabled"})
        self.assertEqual(captured["betas"], ["mcp-client-2025-11-20"])
        self.assertNotIn("fallbacks", captured)

    def test_live_voice_no_thinking_param(self):
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
        self.assertNotIn("thinking", captured)
        self.assertEqual(captured["betas"], ["mcp-client-2025-11-20"])

    def test_thinking_can_be_re_enabled_via_env(self):
        agent = _fresh_agent({"CHAT_THINKING": "adaptive"})
        self.assertEqual(
            agent._thinking_kwargs("chat"), {"thinking": {"type": "adaptive"}}
        )

    def test_invalid_thinking_value_is_ignored(self):
        agent = _fresh_agent({"CHAT_THINKING": "garbage"})
        self.assertEqual(agent._thinking_kwargs("chat"), {})

    def test_voice_on_sonnet_disables_thinking(self):
        """Worth ~12s a turn on a live call — and voice may run Sonnet when Haiku is overloaded."""
        agent = _fresh_agent({"VOICE_MODEL": "claude-sonnet-5"})
        self.assertEqual(
            agent._thinking_kwargs("voice"), {"thinking": {"type": "disabled"}}
        )

    def test_haiku_never_gets_a_thinking_param(self):
        """Haiku does not think when the param is omitted; sending an unsupported switch would fail the call."""
        agent = _fresh_agent({"VOICE_MODEL": "claude-haiku-4-5"})
        self.assertEqual(agent._thinking_kwargs("voice"), {})

    def test_fable_never_gets_a_thinking_param(self):
        """Thinking is always on for Fable; any override is rejected with a 400."""
        agent = _fresh_agent({"AGENT_MODEL": "claude-fable-5"})
        self.assertEqual(agent._thinking_kwargs("chat"), {})

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
