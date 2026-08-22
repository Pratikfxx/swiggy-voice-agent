import importlib
import asyncio
import os
import sys
import time
import unittest
import warnings
from unittest.mock import patch


def _fresh_voice_handler():
    os.environ["TWILIO_VALIDATE_WEBHOOKS"] = "false"
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
    async def test_live_order_completion_uses_tool_metadata_not_reply_prose(self):
        voice_handler = _fresh_voice_handler()

        def fake_process_message(*args, **kwargs):
            self.assertTrue(kwargs["return_meta"])
            placed = kwargs["session_id"] == "placed-call"
            return "Your request was accepted.", {"order_placed": placed}

        with (
            patch.object(voice_handler, "process_message", side_effect=fake_process_message),
            patch.object(voice_handler, "DEMO_MODE", False, create=True),
        ):
            voice_handler._voice_agent_job_ids.update({"placed-call": 1, "failed-call": 1})
            await voice_handler._run_voice_agent_background("placed-call", "yes", 1)
            await voice_handler._run_voice_agent_background("failed-call", "yes", 1)

        self.assertTrue(voice_handler._voice_agent_results["placed-call"]["final"])
        self.assertFalse(voice_handler._voice_agent_results["failed-call"]["final"])

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

    def test_twilio_fallback_uses_neural_voice_by_default_and_respects_override(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TWILIO_TTS_VOICE", None)
            default_voice_handler = _fresh_voice_handler()
        default_twiml = default_voice_handler.make_voice_waiting_twiml(
            "default-call", "Checking Instamart."
        )
        self.assertIn('voice="Polly.Kajal-Neural"', default_twiml)

        with patch.dict(os.environ, {"TWILIO_TTS_VOICE": "Polly.Aditi"}, clear=False):
            override_voice_handler = _fresh_voice_handler()
        override_twiml = override_voice_handler.make_voice_waiting_twiml(
            "override-call", "Checking Instamart."
        )
        self.assertIn('voice="Polly.Aditi"', override_twiml)
        self.assertNotIn('voice="Polly.Kajal-Neural"', override_twiml)

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

    async def test_run_voice_agent_with_deadline_returns_keepalive_when_agent_exceeds_deadline(self):
        voice_handler = _fresh_voice_handler()

        def slow_process_message(*args, **kwargs):
            time.sleep(0.2)
            return "Late agent response"

        with (
            patch.object(voice_handler, "process_message", side_effect=slow_process_message),
            patch.object(voice_handler, "VOICE_AGENT_TIMEOUT_SECS", 0.05, create=True),
        ):
            start = time.monotonic()
            agent_response, elapsed, timed_out = await voice_handler.run_voice_agent_with_deadline(
                "slow-call",
                "get milk",
            )
            elapsed = time.monotonic() - start

        self.assertLess(elapsed, 0.15)
        self.assertTrue(timed_out)
        self.assertIn("taking a bit longer", agent_response)

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

    async def test_voice_process_starts_background_job_without_waiting_for_single_item(self):
        voice_handler = _fresh_voice_handler()

        class FakeRequest:
            async def form(self):
                return {
                    "CallSid": "background-call",
                    "SpeechResult": "get gatorade",
                    "Confidence": "0.94",
                }

        def slow_process_message(*args, **kwargs):
            time.sleep(0.2)
            return "I found Gatorade. Add this?"

        with (
            patch.object(voice_handler, "process_message", side_effect=slow_process_message),
            patch.object(voice_handler, "generate_tts_audio", return_value=None),
        ):
            start = time.monotonic()
            response = await voice_handler.voice_process(FakeRequest())
            elapsed = time.monotonic() - start
            await asyncio.sleep(0.25)

        twiml = response.body.decode()
        self.assertLess(elapsed, 0.08)
        self.assertIn("Checking Instamart", twiml)
        self.assertIn("/voice/result", twiml)
        self.assertIn("background-call", voice_handler._voice_agent_results)

    async def test_voice_result_returns_ready_background_response(self):
        voice_handler = _fresh_voice_handler()
        voice_handler._voice_agent_results["ready-call"] = {
            "response": "I found Gatorade. Add this?",
            "elapsed": 0.5,
            "final": False,
        }

        class FakeRequest:
            query_params = {"callSid": "ready-call", "poll": "1"}

            async def form(self):
                return {}

        with patch.object(voice_handler, "generate_tts_audio", return_value=None):
            response = await voice_handler.voice_result(FakeRequest())

        twiml = response.body.decode()
        self.assertIn("I found Gatorade", twiml)
        self.assertIn("<Gather", twiml)
        self.assertNotIn("ready-call", voice_handler._voice_agent_results)

    async def test_voice_result_keeps_call_alive_without_repeating_filler(self):
        voice_handler = _fresh_voice_handler()

        class FakeRequest:
            query_params = {"callSid": "pending-result-call", "poll": "1"}

            async def form(self):
                return {}

        response = await voice_handler.voice_result(FakeRequest())

        twiml = response.body.decode()
        self.assertNotIn("<Say", twiml)
        self.assertIn("/voice/result", twiml)

    async def test_voice_result_speaks_only_at_configured_wait_polls(self):
        voice_handler = _fresh_voice_handler()

        class FakeRequest:
            def __init__(self, poll):
                self.query_params = {"callSid": "sparse-poll-call", "poll": str(poll)}

            async def form(self):
                return {}

        # Follow the configured cadence rather than pinning poll numbers — the
        # cadence is switchable per deploy via VOICE_WAIT_CADENCE.
        configured = voice_handler._VOICE_WAIT_LINES
        self.assertTrue(configured, "a wait cadence must be configured")

        for poll, expected in configured.items():
            twiml = (await voice_handler.voice_result(FakeRequest(poll))).body.decode()
            self.assertIn("<Say", twiml)
            self.assertIn(expected, twiml)

        silent_polls = [p for p in range(1, 11) if p not in configured]
        self.assertTrue(silent_polls, "some polls should hold silently")
        for poll in silent_polls:
            twiml = (await voice_handler.voice_result(FakeRequest(poll))).body.decode()
            self.assertNotIn("<Say", twiml)
            self.assertIn("<Redirect", twiml)

    def test_voice_polling_budget_allows_live_search_to_finish_without_dead_air(self):
        voice_handler = _fresh_voice_handler()

        self.assertGreaterEqual(voice_handler.VOICE_RESULT_MAX_POLLS, 8)
        self.assertGreaterEqual(voice_handler.VOICE_AGENT_TIMEOUT_SECS, 15)

    async def test_voice_process_starts_background_job_for_multi_item_requests(self):
        voice_handler = _fresh_voice_handler()

        class FakeRequest:
            async def form(self):
                return {
                    "CallSid": "multi-item-call",
                    "SpeechResult": "get milk and bread",
                    "Confidence": "0.94",
                }

        with (
            patch.object(voice_handler, "process_message", return_value="I found milk and bread. Add these?"),
            patch.object(voice_handler, "generate_tts_audio", return_value=None),
        ):
            start = time.monotonic()
            response = await voice_handler.voice_process(FakeRequest())
            elapsed = time.monotonic() - start
            await asyncio.sleep(0.05)

        twiml = response.body.decode()
        self.assertLess(elapsed, 0.05)
        self.assertIn("Checking Instamart", twiml)
        self.assertIn("/voice/result", twiml)
        self.assertIn("multi-item-call", voice_handler._voice_agent_results)

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
            await asyncio.sleep(0.05)

        twiml = response.body.decode()
        self.assertNotIn("pending-call", voice_handler._voice_fast_pending)
        self.assertIn("Checking Instamart", twiml)
        self.assertEqual(
            voice_handler._voice_agent_results["pending-call"]["response"],
            "I found Amul milk. Add this?",
        )

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


class CleanForVoiceTests(unittest.TestCase):
    """TTS text hygiene — a live call said "oddlooks" because an em dash was
    deleted rather than replaced, fusing the words on either side."""

    def setUp(self):
        self.voice_handler = _fresh_voice_handler()

    def test_em_dash_never_fuses_neighbouring_words(self):
        clean = self.voice_handler.clean_for_voice("That's odd—looks like it is gone.")
        self.assertNotIn("oddlooks", clean)
        self.assertIn("odd, looks", clean)

    def test_curly_apostrophe_survives_as_ascii(self):
        clean = self.voice_handler.clean_for_voice("It’s ready. Don’t worry.")
        self.assertIn("It's ready", clean)
        self.assertIn("Don't worry", clean)

    def test_rupee_sign_is_spoken_after_the_number(self):
        self.assertEqual(self.voice_handler.clean_for_voice("Total: ₹128"), "Total: 128 rupees")

    def test_emoji_removal_does_not_fuse_words(self):
        self.assertEqual(
            self.voice_handler.clean_for_voice("Done\U0001F389Arriving soon."), "Done Arriving soon."
        )

    def test_devanagari_is_preserved(self):
        hindi = "ठीक है"
        self.assertIn(hindi, self.voice_handler.clean_for_voice(hindi))

    def test_no_space_left_before_punctuation(self):
        clean = self.voice_handler.clean_for_voice("Amul Taaza — 59 rupees — done.")
        self.assertNotIn(" ,", clean)


class FarewellTests(unittest.TestCase):
    """"Cancel my last order" used to match the bare word "cancel" and hang up
    on the caller, so Swiggy's cancellation guidance never got a chance to
    run."""

    def setUp(self):
        self.voice_handler = _fresh_voice_handler()

    def test_a_bare_cancel_still_ends_the_call(self):
        for spoken in ("cancel", "stop", "  cancel.", "bye", "hang up", "band karo"):
            self.assertTrue(self.voice_handler.is_farewell(spoken), spoken)

    def test_cancelling_an_order_is_not_a_farewell(self):
        for spoken in (
            "cancel my last order",
            "I want to cancel the order I just placed",
            "cancel that item",
            "stop the order",
        ):
            self.assertFalse(self.voice_handler.is_farewell(spoken), spoken)
