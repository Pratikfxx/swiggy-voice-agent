import base64
import importlib
import json
import os
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch

import swiggy_auth


def _jwt(payload):
    def encode(part):
        raw = json.dumps(part, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{encode({'alg': 'none', 'typ': 'JWT'})}.{encode(payload)}.sig"


def _fresh_main():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=ResourceWarning)
        for name in (
            "main",
            "voice_handler",
            "whatsapp_handler",
            "agent",
        ):
            sys.modules.pop(name, None)
        return importlib.import_module("main")


class HealthTests(unittest.TestCase):
    def test_health_requires_only_active_instamart_token_for_readiness(self):
        env = {
            "DEMO_MODE": "false",
            "ANTHROPIC_API_KEY": "test-key",
            "TWILIO_ACCOUNT_SID": "AC" + "1" * 32,
            "TWILIO_AUTH_TOKEN": "test-token",
            "ELEVENLABS_API_KEY": "test-eleven",
            "SWIGGY_IM_TOKEN": _jwt({"exp": 1300}),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            token_store = os.path.join(tmpdir, ".swiggy_tokens.json")
            with patch.dict(os.environ, env, clear=True):
                main = _fresh_main()
                with patch.object(swiggy_auth, "TOKEN_STORE", token_store):
                    with patch.object(swiggy_auth.time, "time", return_value=1000):
                        result = main.health()

        self.assertTrue(result["swiggy"])
        self.assertEqual(result["swiggy_required_tokens"], ["im"])
        self.assertFalse(result["swiggy_tokens"]["im"]["expired"])
        self.assertFalse(result["swiggy_tokens"]["food"]["logged_in"])
        self.assertFalse(result["swiggy_tokens"]["dineout"]["logged_in"])

    def test_health_reports_swiggy_ready_from_safe_token_status(self):
        env = {
            "DEMO_MODE": "false",
            "ANTHROPIC_API_KEY": "test-key",
            "TWILIO_ACCOUNT_SID": "AC" + "1" * 32,
            "TWILIO_AUTH_TOKEN": "test-token",
            "ELEVENLABS_API_KEY": "test-eleven",
            "SWIGGY_FOOD_TOKEN": _jwt({"exp": 1300}),
            "SWIGGY_IM_TOKEN": _jwt({"exp": 1300}),
            "SWIGGY_DINEOUT_TOKEN": _jwt({"exp": 1300}),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            token_store = os.path.join(tmpdir, ".swiggy_tokens.json")
            with patch.dict(os.environ, env, clear=True):
                main = _fresh_main()
                with patch.object(swiggy_auth, "TOKEN_STORE", token_store):
                    with patch.object(swiggy_auth.time, "time", return_value=1000):
                        result = main.health()

        self.assertTrue(result["swiggy"])
        self.assertEqual(result["swiggy_tokens"]["food"]["expires_in_s"], 300)
        self.assertFalse(result["swiggy_tokens"]["im"]["expired"])
        self.assertNotIn("access_token", result["swiggy_tokens"]["dineout"])
        self.assertNotIn("token", result["swiggy_tokens"]["dineout"])

    def test_health_marks_swiggy_unready_when_a_token_is_expired(self):
        env = {
            "DEMO_MODE": "false",
            "ANTHROPIC_API_KEY": "test-key",
            "TWILIO_ACCOUNT_SID": "AC" + "1" * 32,
            "TWILIO_AUTH_TOKEN": "test-token",
            "SWIGGY_FOOD_TOKEN": _jwt({"exp": 1300}),
            "SWIGGY_IM_TOKEN": _jwt({"exp": 990}),
            "SWIGGY_DINEOUT_TOKEN": _jwt({"exp": 1300}),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            token_store = os.path.join(tmpdir, ".swiggy_tokens.json")
            with patch.dict(os.environ, env, clear=True):
                main = _fresh_main()
                with patch.object(swiggy_auth, "TOKEN_STORE", token_store):
                    with patch.object(swiggy_auth.time, "time", return_value=1000):
                        result = main.health()

        self.assertFalse(result["swiggy"])
        self.assertTrue(result["swiggy_tokens"]["im"]["expired"])


if __name__ == "__main__":
    unittest.main()
