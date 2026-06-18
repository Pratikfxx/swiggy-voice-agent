import importlib
import sys
import unittest
import warnings
from unittest.mock import patch


def _fresh_voice_handler():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ResourceWarning)
        for name in ("voice_handler", "agent"):
            sys.modules.pop(name, None)
        return importlib.import_module("voice_handler")


def _fresh_agent():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ResourceWarning)
        sys.modules.pop("agent", None)
        return importlib.import_module("agent")


class VoiceHandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_gather_hints_cover_common_food_and_grocery_orders(self):
        voice_handler = _fresh_voice_handler()

        with patch.object(voice_handler, "generate_tts_audio", return_value=None):
            twiml = await voice_handler.make_twiml_response(
                "What would you like to order?",
                session_id="call-test",
            )

        for expected in ("dosa", "burger", "gatorade", "paneer", "diapers", "coke"):
            self.assertIn(expected, twiml)

    def test_voice_turn_logging_uses_visible_uvicorn_logger(self):
        voice_handler = _fresh_voice_handler()

        with patch.object(voice_handler.voice_logger, "info") as info:
            voice_handler.log_voice_input("call-test", "one masala dosa", 0.72)

        info.assert_called_once()
        message, call_sid, speech, confidence = info.call_args.args
        self.assertIn("VOICE in", message)
        self.assertEqual(call_sid, "call-test")
        self.assertEqual(speech, "one masala dosa")
        self.assertEqual(confidence, 0.72)

    async def test_non_final_response_reprompts_instead_of_hanging_up_after_silence(self):
        voice_handler = _fresh_voice_handler()

        with patch.object(voice_handler, "generate_tts_audio", return_value=None):
            twiml = await voice_handler.make_twiml_response(
                "Found Masala Dosa nearby. Should I add one?",
                session_id="call-test",
                is_final=False,
            )

        self.assertNotIn("Goodbye", twiml)
        self.assertNotIn("<Hangup", twiml)
        self.assertGreaterEqual(twiml.count("<Gather"), 2)
        self.assertIn("I didn't catch that", twiml)

    async def test_non_final_response_uses_more_patient_gather_timeout(self):
        voice_handler = _fresh_voice_handler()

        with patch.object(voice_handler, "generate_tts_audio", return_value=None):
            twiml = await voice_handler.make_twiml_response(
                "What would you like to order?",
                session_id="call-test",
                is_final=False,
            )

        self.assertIn('timeout="7"', twiml)

    def test_voice_prompt_allows_natural_context_instead_of_ultra_clipped_replies(self):
        agent = _fresh_agent()

        self.assertNotIn("MAX 20 words", agent.VOICE_SYSTEM_PROMPT)
        self.assertIn("short natural sentences", agent.VOICE_SYSTEM_PROMPT)
        self.assertIn("Be warm, not robotic", agent.VOICE_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
