import importlib
import os
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


class FakeRequest:
    """Minimal stand-in for a Starlette Request (headers + url.scheme only)."""

    class _URL:
        def __init__(self, scheme):
            self.scheme = scheme

    def __init__(self, headers=None, scheme="http"):
        self.headers = headers or {}
        self.url = self._URL(scheme)


class ResolveBaseUrlTests(unittest.TestCase):
    """Twilio must always be handed a URL it can actually reach.

    A stale/unset BASE_URL pointing at localhost is what produces
    "an application error occurred" mid-call.
    """

    def test_public_configured_base_url_wins_over_request_host(self):
        vh = _fresh_voice_handler()
        req = FakeRequest({"host": "attacker.example.com", "x-forwarded-proto": "https"})
        with patch.dict(os.environ, {"BASE_URL": "https://myapp.up.railway.app"}):
            self.assertEqual(vh.resolve_base_url(req), "https://myapp.up.railway.app")

    def test_localhost_base_url_falls_back_to_request_host(self):
        vh = _fresh_voice_handler()
        req = FakeRequest({"host": "abc-123.trycloudflare.com", "x-forwarded-proto": "https"})
        with patch.dict(os.environ, {"BASE_URL": "http://localhost:8000"}):
            self.assertEqual(vh.resolve_base_url(req), "https://abc-123.trycloudflare.com")

    def test_unset_base_url_falls_back_to_request_host(self):
        vh = _fresh_voice_handler()
        req = FakeRequest({"host": "abc-123.ngrok.io", "x-forwarded-proto": "https"})
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BASE_URL", None)
            self.assertEqual(vh.resolve_base_url(req), "https://abc-123.ngrok.io")

    def test_x_forwarded_host_preferred_over_host(self):
        vh = _fresh_voice_handler()
        req = FakeRequest(
            {
                "host": "internal-service:8080",
                "x-forwarded-host": "public.example.com",
                "x-forwarded-proto": "https",
            }
        )
        with patch.dict(os.environ, {"BASE_URL": "http://localhost:8000"}):
            self.assertEqual(vh.resolve_base_url(req), "https://public.example.com")

    def test_comma_separated_forwarded_headers_take_first_hop(self):
        vh = _fresh_voice_handler()
        req = FakeRequest(
            {"x-forwarded-host": "public.example.com, internal", "x-forwarded-proto": "https, http"}
        )
        with patch.dict(os.environ, {"BASE_URL": "http://localhost:8000"}):
            self.assertEqual(vh.resolve_base_url(req), "https://public.example.com")

    def test_localhost_request_host_does_not_produce_unreachable_url(self):
        vh = _fresh_voice_handler()
        req = FakeRequest({"host": "localhost:8000"})
        with patch.dict(os.environ, {"BASE_URL": "http://localhost:8000"}):
            # Nothing better available — keep the configured value rather than invent one
            self.assertEqual(vh.resolve_base_url(req), "http://localhost:8000")

    def test_no_request_returns_configured_value(self):
        vh = _fresh_voice_handler()
        with patch.dict(os.environ, {"BASE_URL": "https://myapp.up.railway.app"}):
            self.assertEqual(vh.resolve_base_url(None), "https://myapp.up.railway.app")

    def test_trailing_slash_is_normalised(self):
        vh = _fresh_voice_handler()
        with patch.dict(os.environ, {"BASE_URL": "https://myapp.up.railway.app/"}):
            self.assertEqual(vh.resolve_base_url(None), "https://myapp.up.railway.app")


class TtsCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_cached_audio_url_follows_current_base_url(self):
        """A cached phrase must not pin the URL to a dead tunnel host."""
        vh = _fresh_voice_handler()
        vh._tts_cache.add("deadbeef")

        with (
            patch.object(vh.hashlib, "md5") as md5,
            patch.object(vh, "ELEVENLABS_API_KEY", "k"),
        ):
            md5.return_value.hexdigest.return_value = "deadbeef"
            first = await vh.generate_tts_audio("hi", base_url="https://old.trycloudflare.com")
            second = await vh.generate_tts_audio("hi", base_url="https://new.trycloudflare.com")

        self.assertEqual(first, "https://old.trycloudflare.com/audio/deadbeef")
        self.assertEqual(second, "https://new.trycloudflare.com/audio/deadbeef")


if __name__ == "__main__":
    unittest.main()
