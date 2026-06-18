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


def _fresh_voice_handler():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ResourceWarning)
        for name in ("voice_handler", "agent"):
            sys.modules.pop(name, None)
        return importlib.import_module("voice_handler")


class InstamartOnlyTests(unittest.IsolatedAsyncioTestCase):
    def test_live_routing_is_instamart_only_for_now(self):
        agent = _fresh_agent()

        for surface in ("voice", "chat"):
            for message in ("one masala dosa", "book a table tonight", "milk and eggs"):
                self.assertEqual(
                    agent._route_servers(message, surface),
                    ["swiggy-instamart"],
                )

    def test_voice_prompt_is_instamart_only(self):
        agent = _fresh_agent()
        prompt = agent.VOICE_SYSTEM_PROMPT

        self.assertIn("Instamart-only", prompt)
        self.assertNotIn("Food:", prompt)
        self.assertNotIn("restaurant", prompt.lower())
        self.assertNotIn("dineout", prompt.lower())
        self.assertNotIn("book a table", prompt.lower())

    def test_chat_prompt_is_instamart_only(self):
        agent = _fresh_agent()
        prompt = agent.CHAT_SYSTEM_PROMPT

        self.assertIn("Instamart-only", prompt)
        self.assertNotIn("book a table", prompt.lower())
        self.assertNotIn("dineout", prompt.lower())
        self.assertNotIn("restaurant options", prompt.lower())

    async def test_voice_greeting_names_instamart(self):
        voice_handler = _fresh_voice_handler()

        class FakeRequest:
            async def form(self):
                return {"CallSid": "call-test"}

        with patch.object(voice_handler, "generate_tts_audio", return_value=None):
            response = await voice_handler.voice_answer(FakeRequest())

        twiml = response.body.decode()
        self.assertIn("Instamart", twiml)
        self.assertIn("groceries", twiml.lower())
        self.assertNotIn("What would you like to order?", twiml)


if __name__ == "__main__":
    unittest.main()
