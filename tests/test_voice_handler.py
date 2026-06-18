import importlib
import asyncio
import sys
import time
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
    async def test_gather_hints_cover_common_instamart_orders(self):
        voice_handler = _fresh_voice_handler()

        with patch.object(voice_handler, "generate_tts_audio", return_value=None):
            twiml = await voice_handler.make_twiml_response(
                "What would you like to order?",
                session_id="call-test",
            )

        for expected in ("gatorade", "paneer", "diapers", "coke", "milk", "detergent"):
            self.assertIn(expected, twiml)
        for stale in ("dosa", "burger", "biryani"):
            self.assertNotIn(stale, twiml)

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

    async def test_voice_process_returns_keepalive_when_agent_exceeds_deadline(self):
        voice_handler = _fresh_voice_handler()

        class FakeRequest:
            async def form(self):
                return {
                    "CallSid": "slow-call",
                    "SpeechResult": "get milk",
                    "Confidence": "0.94",
                }

        def slow_process_message(*args, **kwargs):
            time.sleep(0.2)
            return "Late agent response"

        with (
            patch.object(voice_handler, "process_message", side_effect=slow_process_message),
            patch.object(voice_handler, "generate_tts_audio", return_value=None),
            patch.object(voice_handler, "VOICE_AGENT_TIMEOUT_SECS", 0.05, create=True),
        ):
            start = time.monotonic()
            response = await voice_handler.voice_process(FakeRequest())
            elapsed = time.monotonic() - start

        twiml = response.body.decode()
        self.assertLess(elapsed, 0.15)
        self.assertIn("taking a bit longer", twiml)
        self.assertIn("<Gather", twiml)
        self.assertNotIn("Late agent response", twiml)
        self.assertNotIn("<Hangup", twiml)

    async def test_voice_process_does_not_block_event_loop_during_slow_agent(self):
        voice_handler = _fresh_voice_handler()

        class FakeRequest:
            async def form(self):
                return {
                    "CallSid": "nonblocking-call",
                    "SpeechResult": "get milk",
                    "Confidence": "0.94",
                }

        def slow_process_message(*args, **kwargs):
            time.sleep(0.2)
            return "Late agent response"

        with (
            patch.object(voice_handler, "process_message", side_effect=slow_process_message),
            patch.object(voice_handler, "generate_tts_audio", return_value=None),
            patch.object(voice_handler, "VOICE_AGENT_TIMEOUT_SECS", 0.05, create=True),
        ):
            start = time.monotonic()
            task = asyncio.create_task(voice_handler.voice_process(FakeRequest()))
            await asyncio.sleep(0.01)
            event_loop_delay = time.monotonic() - start
            await task

        self.assertLess(event_loop_delay, 0.08)

    async def test_voice_process_acknowledges_multi_item_requests_without_agent_gap(self):
        voice_handler = _fresh_voice_handler()

        class FakeRequest:
            async def form(self):
                return {
                    "CallSid": "multi-item-call",
                    "SpeechResult": "get milk and bread",
                    "Confidence": "0.94",
                }

        with (
            patch.object(voice_handler, "process_message") as process_message,
            patch.object(voice_handler, "generate_tts_audio", return_value=None),
        ):
            start = time.monotonic()
            response = await voice_handler.voice_process(FakeRequest())
            elapsed = time.monotonic() - start

        twiml = response.body.decode()
        process_message.assert_not_called()
        self.assertLess(elapsed, 0.05)
        self.assertIn("one item at a time", twiml)
        self.assertIn("milk", twiml)
        self.assertIn("bread", twiml)
        self.assertIn("<Gather", twiml)
        self.assertNotIn("<Hangup", twiml)

    async def test_voice_process_consumes_fast_pending_item_on_confirmation(self):
        voice_handler = _fresh_voice_handler()
        voice_handler._voice_fast_pending["pending-call"] = "milk"

        class FakeRequest:
            async def form(self):
                return {
                    "CallSid": "pending-call",
                    "SpeechResult": "yes",
                    "Confidence": "0.94",
                }

        def fake_process_message(*args, **kwargs):
            self.assertEqual(kwargs["user_message"], "get milk")
            return "I found Amul milk. Add this?"

        with (
            patch.object(voice_handler, "process_message", side_effect=fake_process_message),
            patch.object(voice_handler, "generate_tts_audio", return_value=None),
        ):
            response = await voice_handler.voice_process(FakeRequest())

        twiml = response.body.decode()
        self.assertNotIn("pending-call", voice_handler._voice_fast_pending)
        self.assertIn("I found Amul milk", twiml)

    def test_clean_for_voice_removes_search_narration_preamble(self):
        voice_handler = _fresh_voice_handler()

        cleaned = voice_handler.clean_for_voice(
            "I'll search for milk and bread for you. Got Amul Taaza milk and Modern bread."
        )

        self.assertNotIn("I'll search", cleaned)
        self.assertEqual(cleaned, "Got Amul Taaza milk and Modern bread.")

    async def test_unusual_activity_disables_elevenlabs_after_first_401(self):
        voice_handler = _fresh_voice_handler()
        voice_handler.ELEVENLABS_API_KEY = "test-elevenlabs-key"
        post_calls = 0

        class FakeResponse:
            status_code = 401
            text = '{"detail":{"status":"detected_unusual_activity","message":"Free Tier access disabled"}}'
            content = b""

        class FakeAsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def post(self, *args, **kwargs):
                nonlocal post_calls
                post_calls += 1
                return FakeResponse()

        with (
            patch.object(voice_handler.httpx, "AsyncClient", FakeAsyncClient),
            patch.object(voice_handler.voice_logger, "warning") as warning,
        ):
            first = await voice_handler.generate_tts_audio("Hello there")
            second = await voice_handler.generate_tts_audio("Hello again")

        self.assertIsNone(first)
        self.assertIsNone(second)
        self.assertEqual(post_calls, 1)
        warning.assert_called()

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
