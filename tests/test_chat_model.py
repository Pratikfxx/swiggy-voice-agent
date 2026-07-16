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

    def test_voice_falls_back_to_sonnet_when_haiku_overloaded(self):
        agent = _fresh_agent({"DEMO_MODE": "false"})
        import anthropic, httpx

        attempts = []

        def fake_create(**kwargs):
            attempts.append(kwargs)
            if kwargs["model"] == "claude-haiku-4-5":
                raise anthropic.APIStatusError(
                    "overloaded",
                    response=httpx.Response(529, request=httpx.Request("POST", "http://t")),
                    body={"error": {"type": "overloaded_error"}},
                )
            return FakeResponse()

        with (
            patch.object(agent.client.beta.messages, "create", side_effect=fake_create),
            patch.object(agent.swiggy_address, "maybe_background_refresh"),
            patch.object(agent.swiggy_address, "get_cached_default", return_value={"id": "a1", "label": "Home", "area": "X"}),
        ):
            response, _ = agent._run_agent_live("milk", [], "voice", "call-t", {"im": "t"})

        self.assertEqual(response, "ok")
        self.assertEqual([a["model"] for a in attempts], ["claude-haiku-4-5", "claude-sonnet-5"])
        # thinking policy must be re-resolved for the fallback model
        self.assertNotIn("thinking", attempts[0])
        self.assertEqual(attempts[1]["thinking"], {"type": "disabled"})

    def test_non_capacity_errors_do_not_fall_back(self):
        agent = _fresh_agent({"DEMO_MODE": "false"})
        import anthropic, httpx

        def fake_create(**kwargs):
            raise anthropic.APIStatusError(
                "bad request",
                response=httpx.Response(400, request=httpx.Request("POST", "http://t")),
                body={},
            )

        with (
            patch.object(agent.client.beta.messages, "create", side_effect=fake_create),
            patch.object(agent.swiggy_address, "maybe_background_refresh"),
            patch.object(agent.swiggy_address, "get_cached_default", return_value=None),
            patch.object(agent.swiggy_address, "get_default_blocking", return_value=None),
            patch.object(agent.logging, "exception"),
        ):
            response, _ = agent._run_agent_live("milk", [], "voice", "call-t", {"im": "t"})

        # surfaces as the generic failure message, not a silent cross-tier retry
        self.assertIn("problem reaching Swiggy", response)

    def test_cold_address_cache_fetches_blocking_before_prompting(self):
        agent = _fresh_agent({"DEMO_MODE": "false"})
        captured = {}

        def fake_create(**kwargs):
            captured.update(kwargs)
            return FakeResponse()

        with (
            patch.object(agent.client.beta.messages, "create", side_effect=fake_create),
            patch.object(agent.swiggy_address, "get_cached_default", return_value=None),
            patch.object(
                agent.swiggy_address,
                "get_default_blocking",
                return_value={"id": "addr9", "label": "Ghar", "area": "Nagpur"},
            ) as blocking,
        ):
            response, _ = agent._run_agent_live("milk", [], "voice", "call-t", {"im": "t"})

        blocking.assert_called_once()
        system_text = "".join(b["text"] for b in captured["system"])
        self.assertIn("DEFAULT DELIVERY ADDRESS: Ghar (Nagpur), addressId addr9", system_text)

    def test_system_prompt_is_cacheable_block(self):
        """One breakpoint on the system block caches tools+system (~30k tokens/turn)."""
        agent = _fresh_agent({"DEMO_MODE": "false"})
        captured = {}

        def fake_create(**kwargs):
            captured.update(kwargs)
            return FakeResponse()

        with (
            patch.object(agent.client.beta.messages, "create", side_effect=fake_create),
            patch.object(agent.swiggy_address, "maybe_background_refresh"),
            patch.object(agent.swiggy_address, "get_cached_default", return_value={"id": "a1", "label": "Home", "area": "X"}),
        ):
            agent._run_agent_live("milk", [], "chat", "wa-t", {"im": "t"})

        self.assertEqual(captured["system"][-1]["cache_control"], {"type": "ephemeral"})

    def test_demo_system_prompt_is_cacheable_block(self):
        agent = _fresh_agent()
        captured = {}

        def fake_create(**kwargs):
            captured.update(kwargs)
            return FakeResponse()

        with patch.object(agent.client.messages, "create", side_effect=fake_create):
            agent._run_agent_demo("milk", [], surface="chat", session_id="wa-t")

        self.assertEqual(captured["system"][-1]["cache_control"], {"type": "ephemeral"})

    def test_warm_address_cache_skips_blocking_fetch(self):
        agent = _fresh_agent({"DEMO_MODE": "false"})

        with (
            patch.object(agent.client.beta.messages, "create", return_value=FakeResponse()),
            patch.object(agent.swiggy_address, "maybe_background_refresh") as bg,
            patch.object(agent.swiggy_address, "get_cached_default", return_value={"id": "a1", "label": "Home", "area": "X"}),
            patch.object(agent.swiggy_address, "get_default_blocking") as blocking,
        ):
            agent._run_agent_live("milk", [], "voice", "call-t", {"im": "t"})

        blocking.assert_not_called()
        bg.assert_called_once()

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
