import base64
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import swiggy_auth


def _jwt(payload):
    def encode(part):
        raw = json.dumps(part, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{encode({'alg': 'none', 'typ': 'JWT'})}.{encode(payload)}.sig"


class SwiggyAuthStatusTests(unittest.TestCase):
    def test_env_jwt_status_reports_expiry_without_token_value(self):
        token = _jwt({"exp": 1120})

        with tempfile.TemporaryDirectory() as tmpdir:
            token_store = os.path.join(tmpdir, ".swiggy_tokens.json")
            with patch.dict(os.environ, {"SWIGGY_FOOD_TOKEN": token}, clear=True):
                with patch.object(swiggy_auth, "TOKEN_STORE", token_store):
                    with patch.object(swiggy_auth.time, "time", return_value=1000):
                        result = swiggy_auth.status()

        self.assertTrue(result["food"]["logged_in"])
        self.assertEqual(result["food"]["source"], "env")
        self.assertEqual(result["food"]["expires_in_s"], 120)
        self.assertFalse(result["food"]["expired"])
        self.assertNotIn("access_token", result["food"])
        self.assertNotIn("token", result["food"])

    def test_env_jwt_status_marks_expired_tokens(self):
        token = _jwt({"exp": 995})

        with tempfile.TemporaryDirectory() as tmpdir:
            token_store = os.path.join(tmpdir, ".swiggy_tokens.json")
            with patch.dict(os.environ, {"SWIGGY_FOOD_TOKEN": token}, clear=True):
                with patch.object(swiggy_auth, "TOKEN_STORE", token_store):
                    with patch.object(swiggy_auth.time, "time", return_value=1000):
                        result = swiggy_auth.status()

        self.assertTrue(result["food"]["logged_in"])
        self.assertEqual(result["food"]["expires_in_s"], 0)
        self.assertTrue(result["food"]["expired"])


if __name__ == "__main__":
    unittest.main()
