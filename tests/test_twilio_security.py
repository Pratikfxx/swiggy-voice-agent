import importlib
import os
import sys
import unittest
import warnings
from unittest.mock import patch

from twilio.request_validator import RequestValidator


def _fresh(module="twilio_security"):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ResourceWarning)
        sys.modules.pop(module, None)
        return importlib.import_module(module)


class FakeURL:
    def __init__(self, path="/whatsapp/webhook", query="", scheme="https", netloc="app.example.com"):
        self.path = path
        self.query = query
        self.scheme = scheme
        self.netloc = netloc


class FakeRequest:
    def __init__(self, headers=None, url=None):
        self.headers = headers or {}
        self.url = url or FakeURL()


AUTH_TOKEN = "test-auth-token"


def _signed_request(form, path="/whatsapp/webhook", query="", host="app.example.com", scheme="https"):
    url = f"{scheme}://{host}{path}" + (f"?{query}" if query else "")
    sig = RequestValidator(AUTH_TOKEN).compute_signature(url, form)
    return FakeRequest(
        headers={"host": host, "x-forwarded-proto": scheme, "x-twilio-signature": sig},
        url=FakeURL(path=path, query=query),
    )


class TwilioSignatureTests(unittest.TestCase):
    """Webhooks drive live Swiggy spend on one account — forged POSTs must bounce."""

    def setUp(self):
        self.env = patch.dict(os.environ, {"TWILIO_AUTH_TOKEN": AUTH_TOKEN}, clear=False)
        self.env.start()
        os.environ.pop("TWILIO_VALIDATE_WEBHOOKS", None)
        self.ts = _fresh()

    def tearDown(self):
        self.env.stop()

    def test_correctly_signed_request_passes(self):
        form = {"From": "whatsapp:+911234567890", "Body": "get milk"}
        self.assertTrue(self.ts.verify_twilio_request(_signed_request(form), form))

    def test_unsigned_request_rejected(self):
        form = {"From": "whatsapp:+911234567890", "Body": "yes order it"}
        req = FakeRequest(headers={"host": "app.example.com", "x-forwarded-proto": "https"})
        self.assertFalse(self.ts.verify_twilio_request(req, form))

    def test_tampered_body_rejected(self):
        form = {"From": "whatsapp:+911234567890", "Body": "get milk"}
        req = _signed_request(form)
        tampered = {**form, "Body": "yes order 50 packets"}
        self.assertFalse(self.ts.verify_twilio_request(req, tampered))

    def test_query_string_participates_in_signature(self):
        form = {"CallSid": "CA123"}
        req = _signed_request(form, path="/voice/result", query="callSid=CA123&poll=2")
        self.assertTrue(self.ts.verify_twilio_request(req, form))

    def test_forwarded_host_used_for_signature_url(self):
        """Behind Railway the internal host differs; the forwarded host is what Twilio signed."""
        form = {"From": "whatsapp:+911234567890", "Body": "hi"}
        url = "https://public.example.com/whatsapp/webhook"
        sig = RequestValidator(AUTH_TOKEN).compute_signature(url, form)
        req = FakeRequest(
            headers={
                "host": "internal:8080",
                "x-forwarded-host": "public.example.com",
                "x-forwarded-proto": "https",
                "x-twilio-signature": sig,
            },
            url=FakeURL(),
        )
        self.assertTrue(self.ts.verify_twilio_request(req, form))

    def test_no_auth_token_skips_validation(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TWILIO_AUTH_TOKEN", None)
            ts = _fresh()
            req = FakeRequest()
            self.assertTrue(ts.verify_twilio_request(req, {"Body": "hi"}))

    def test_env_kill_switch_disables_validation(self):
        with patch.dict(os.environ, {"TWILIO_VALIDATE_WEBHOOKS": "false"}):
            req = FakeRequest()  # unsigned
            self.assertTrue(self.ts.verify_twilio_request(req, {"Body": "hi"}))


if __name__ == "__main__":
    unittest.main()
