import importlib
import sys
import unittest
import warnings
from unittest.mock import patch


def _fresh_agent():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ResourceWarning)
        sys.modules.pop("agent", None)
        return importlib.import_module("agent")


class AgentTimeoutTests(unittest.TestCase):
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
        self.assertEqual(captured["speed"], "fast")


if __name__ == "__main__":
    unittest.main()
