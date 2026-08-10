import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import main
import voice_stream


AUTH_TOKEN = "stream-test-auth-token"
CALL_SID = "CA-test-call"
STREAM_SID = "MZ-test-stream"


class VoiceStreamTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(
            os.environ,
            {
                "TWILIO_AUTH_TOKEN": AUTH_TOKEN,
                "TWILIO_VALIDATE_WEBHOOKS": "false",
                "BASE_URL": "https://stream.example.com",
            },
            clear=False,
        )
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_token_round_trip(self):
        token = voice_stream.mint_stream_token(CALL_SID)

        self.assertTrue(voice_stream.verify_stream_token(token, CALL_SID))

    def test_expired_token_rejected(self):
        with patch.object(voice_stream.time, "time", return_value=1000):
            token = voice_stream.mint_stream_token(CALL_SID)
        with patch.object(voice_stream.time, "time", return_value=1301):
            self.assertFalse(voice_stream.verify_stream_token(token, CALL_SID))

    def test_tampered_token_rejected(self):
        token = voice_stream.mint_stream_token(CALL_SID)
        tampered = token[:-1] + ("A" if token[-1] != "A" else "B")

        self.assertFalse(voice_stream.verify_stream_token(tampered, CALL_SID))

    def test_wrong_call_sid_rejected(self):
        token = voice_stream.mint_stream_token(CALL_SID)

        self.assertFalse(voice_stream.verify_stream_token(token, "CA-other-call"))

    def test_fixed_dev_token_works_without_auth_secret(self):
        with patch.dict(os.environ, {"TWILIO_AUTH_TOKEN": ""}):
            token = voice_stream.mint_stream_token(CALL_SID)

            self.assertEqual(token, voice_stream.DEV_STREAM_TOKEN)
            self.assertTrue(voice_stream.verify_stream_token(token, CALL_SID))

    def test_websocket_refuses_connection_without_valid_token(self):
        with TestClient(main.app) as client:
            with self.assertRaises(WebSocketDisconnect):
                with client.websocket_connect("/voice/stream?token=invalid"):
                    pass

    def test_full_stream_sequence_echoes_media(self):
        token = voice_stream.mint_stream_token(CALL_SID)
        payload = "//8A"

        with TestClient(main.app) as client:
            with client.websocket_connect(f"/voice/stream?token={token}") as websocket:
                websocket.send_json({"event": "connected"})
                websocket.send_json(
                    {
                        "event": "start",
                        "start": {"streamSid": STREAM_SID, "callSid": CALL_SID},
                    }
                )
                websocket.send_json(
                    {"event": "media", "media": {"payload": payload}}
                )

                self.assertEqual(
                    websocket.receive_json(),
                    {
                        "event": "media",
                        "streamSid": STREAM_SID,
                        "media": {"payload": payload},
                    },
                )
                websocket.send_json({"event": "stop"})

    def test_malformed_json_does_not_kill_stream(self):
        token = voice_stream.mint_stream_token(CALL_SID)

        with TestClient(main.app) as client:
            with client.websocket_connect(f"/voice/stream?token={token}") as websocket:
                websocket.send_text("not-json")
                websocket.send_json(
                    {
                        "event": "start",
                        "start": {"streamSid": STREAM_SID, "callSid": CALL_SID},
                    }
                )
                websocket.send_json(
                    {"event": "media", "media": {"payload": "//8A"}}
                )

                self.assertEqual(websocket.receive_json()["media"]["payload"], "//8A")
                websocket.send_json({"event": "stop"})

    def test_stream_test_returns_signed_wss_twiml(self):
        with TestClient(main.app) as client:
            response = client.post(
                "/voice/stream-test",
                data={"CallSid": CALL_SID},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("<Connect>", response.text)
        self.assertIn("<Stream", response.text)
        self.assertIn("wss://stream.example.com/voice/stream?token=", response.text)


if __name__ == "__main__":
    unittest.main()
