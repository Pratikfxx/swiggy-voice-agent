import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, urlsplit

from fastapi.testclient import TestClient

import main
import swiggy_link
import whatsapp_handler


SECRET = "link-test-secret"
USER_ID = "+919876543210"


def _clear_states():
    with swiggy_link._state_lock:
        swiggy_link._pending_states.clear()


def _client(monkeypatch):
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", SECRET)
    monkeypatch.setenv("BASE_URL", "https://app.example.com")
    monkeypatch.setattr(main, "prewarm_tts", AsyncMock())
    monkeypatch.setattr(main.swiggy_address, "refresh_default_address", AsyncMock())
    return TestClient(main.app)


def setup_function():
    _clear_states()


def teardown_function():
    _clear_states()


def test_signed_link_token_round_trip(monkeypatch):
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", SECRET)
    token = swiggy_link.mint_link_token(USER_ID, "im")

    assert swiggy_link.verify_link_token(token) == (USER_ID, "im")


def test_expired_link_token_rejected(monkeypatch):
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", SECRET)
    with patch.object(swiggy_link.time, "time", return_value=1000):
        token = swiggy_link.mint_link_token(USER_ID, "im")
    with patch.object(swiggy_link.time, "time", return_value=1900):
        assert swiggy_link.verify_link_token(token) is None


def test_tampered_link_token_rejected(monkeypatch):
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", SECRET)
    token = swiggy_link.mint_link_token(USER_ID, "im")
    tampered = token[:-1] + ("A" if token[-1] != "A" else "B")

    assert swiggy_link.verify_link_token(tampered) is None


def test_start_redirects_and_remembers_oauth_state(monkeypatch):
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", SECRET)
    authorize_url = "https://mcp.swiggy.com/auth/authorize?state=oauth-state"

    def build_url(key, redirect_uri):
        assert key == "im"
        assert redirect_uri == "https://app.example.com/link/swiggy/callback"
        return authorize_url, "oauth-state", "pkce-verifier"

    monkeypatch.setattr(swiggy_link.swiggy_auth, "build_authorize_url", build_url)
    token = swiggy_link.mint_link_token(USER_ID, "im")
    with _client(monkeypatch) as client:
        response = client.get(
            "/link/swiggy/start", params={"token": token}, follow_redirects=False
        )

    assert response.status_code == 302
    assert response.headers["location"] == authorize_url
    assert swiggy_link._pending_states["oauth-state"] == (
        USER_ID,
        "im",
        "pkce-verifier",
        swiggy_link._pending_states["oauth-state"][3],
    )


def test_unknown_callback_state_fails_without_storage(monkeypatch):
    save = MagicMock()
    exchange = MagicMock()
    monkeypatch.setattr(swiggy_link.store, "save_user_token", save)
    monkeypatch.setattr(swiggy_link.swiggy_auth, "exchange_code", exchange)
    with _client(monkeypatch) as client:
        response = client.get(
            "/link/swiggy/callback", params={"code": "auth-code", "state": "unknown"}
        )

    assert response.status_code == 400
    assert "invalid or expired" in response.text
    exchange.assert_not_called()
    save.assert_not_called()


def test_successful_callback_persists_under_remembered_user(monkeypatch):
    swiggy_link._remember_state(
        "oauth-state", USER_ID, "im", "pkce-verifier", swiggy_link.time.time()
    )
    record = {
        "access_token": "access-secret",
        "refresh_token": "refresh-secret",
        "expires_at": 2000,
    }
    exchange = MagicMock(return_value=record)
    save = MagicMock()
    monkeypatch.setattr(swiggy_link.swiggy_auth, "exchange_code", exchange)
    monkeypatch.setattr(swiggy_link.store, "save_user_token", save)
    with _client(monkeypatch) as client:
        response = client.get(
            "/link/swiggy/callback", params={"code": "auth-code", "state": "oauth-state"}
        )

    assert response.status_code == 200
    assert "Go back to WhatsApp" in response.text
    exchange.assert_called_once_with(
        "im", "auth-code", "pkce-verifier", "https://app.example.com/link/swiggy/callback"
    )
    save.assert_called_once_with(USER_ID, "im", record)
    assert "oauth-state" not in swiggy_link._pending_states


def test_exchange_failure_is_diagnosable_without_secrets(monkeypatch, caplog):
    swiggy_link._remember_state(
        "oauth-state", USER_ID, "im", "pkce-verifier", swiggy_link.time.time()
    )
    monkeypatch.setattr(
        swiggy_link.swiggy_auth,
        "exchange_code",
        MagicMock(side_effect=RuntimeError("access-secret auth-code pkce-verifier")),
    )
    with _client(monkeypatch) as client, caplog.at_level(
        logging.WARNING, logger="uvicorn.error"
    ):
        response = client.get(
            "/link/swiggy/callback",
            params={"code": "auth-code", "state": "oauth-state"},
        )

    assert response.status_code == 502
    assert "redirect URI may not be accepted by Swiggy" in response.text
    assert "redirect URI may not be accepted by Swiggy" in caplog.text
    for secret in ("access-secret", "auth-code", "pkce-verifier"):
        assert secret not in caplog.text


def test_whatsapp_link_triggers_skip_agent_for_all_phrases(monkeypatch):
    async def run():
        with (
            patch.dict("os.environ", {"TWILIO_AUTH_TOKEN": SECRET, "BASE_URL": "https://app.example.com"}),
            patch.object(whatsapp_handler, "_send", new=AsyncMock()) as send,
            patch.object(whatsapp_handler, "process_message") as process,
        ):
            for phrase in ("link", "login", "link swiggy", "connect swiggy"):
                send.reset_mock()
                await whatsapp_handler._handle_incoming_inner(
                    f"whatsapp:{USER_ID}", phrase, 0, "", ""
                )
                message = send.await_args.args[1]
                assert "/link/swiggy/start?token=" in message
                token = parse_qs(urlsplit(message).query)["token"][0]
                assert swiggy_link.verify_link_token(token) == (USER_ID, "im")
                send.assert_awaited_once()
            process.assert_not_called()

    asyncio.run(run())


def test_whatsapp_ordinary_message_still_reaches_agent(monkeypatch):
    async def run():
        with (
            patch.dict("os.environ", {"TWILIO_AUTH_TOKEN": SECRET}),
            patch.object(whatsapp_handler, "_send", new=AsyncMock()) as send,
            patch.object(whatsapp_handler, "process_message", return_value="agent reply") as process,
        ):
            await whatsapp_handler._handle_incoming_inner(
                f"whatsapp:{USER_ID}", "order milk", 0, "", ""
            )
            process.assert_called_once_with(
                session_id=f"whatsapp:{USER_ID}",
                user_message="order milk",
                surface="chat",
                user_id=USER_ID,
            )
            send.assert_awaited_once_with(f"whatsapp:{USER_ID}", "agent reply")

    asyncio.run(run())
