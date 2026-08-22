import base64
import json
import logging
import os
import tempfile
import unittest
from unittest.mock import patch

import pytest

import swiggy_auth


def _jwt(payload):
    def encode(part):
        raw = json.dumps(part, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{encode({'alg': 'none', 'typ': 'JWT'})}.{encode(payload)}.sig"


def _prepare_user_store(monkeypatch, tmp_path):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("ORDER_HISTORY_DB", str(tmp_path / "swiggy.db"))
    swiggy_auth.store._init()
    monkeypatch.setattr(
        swiggy_auth,
        "TOKEN_STORE",
        str(tmp_path / ".swiggy_tokens.json"),
    )
    for env_var in swiggy_auth.ENV_TOKEN_VARS.values():
        monkeypatch.delenv(env_var, raising=False)
    return swiggy_auth.store


def test_per_user_tokens_resolve_independently(monkeypatch, tmp_path):
    store = _prepare_user_store(monkeypatch, tmp_path)
    expires_at = 9_999_999_999
    store.save_user_token(
        "user-a", "im", {"access_token": "access-a", "refresh_token": "refresh-a", "expires_at": expires_at}
    )
    store.save_user_token(
        "user-b", "im", {"access_token": "access-b", "refresh_token": "refresh-b", "expires_at": expires_at}
    )

    assert swiggy_auth.get_access_token("im", user_id="user-a") == "access-a"
    assert swiggy_auth.get_access_token("im", user_id="user-b") == "access-b"


def test_missing_user_token_falls_back_to_env_and_no_user_behavior_is_unchanged(
    monkeypatch, tmp_path
):
    _prepare_user_store(monkeypatch, tmp_path)
    monkeypatch.setenv("SWIGGY_IM_TOKEN", "env-token")

    assert swiggy_auth.get_access_token("im", user_id="missing") == "env-token"
    assert swiggy_auth.get_access_token("im") == "env-token"
    assert swiggy_auth.get_access_token("im", user_id=None) == "env-token"
    assert swiggy_auth.get_access_tokens(("im",)) == {"im": "env-token"}
    assert swiggy_auth.get_access_tokens(("im",), user_id=None) == {"im": "env-token"}


def test_env_token_is_only_available_to_the_configured_owner(monkeypatch, tmp_path):
    _prepare_user_store(monkeypatch, tmp_path)
    monkeypatch.setenv("SWIGGY_IM_TOKEN", "env-token")
    monkeypatch.setenv("SWIGGY_ENV_TOKEN_USER_ID", "+918459710806")

    assert swiggy_auth.get_access_token("im", user_id="whatsapp:+918459710806") == "env-token"
    assert swiggy_auth.get_access_token("im", user_id=None) == "env-token"
    with pytest.raises(RuntimeError, match="link.*Swiggy account"):
        swiggy_auth.get_access_token("im", user_id="+919999999999")


def test_user_token_store_failure_falls_back_to_env(monkeypatch, tmp_path):
    _prepare_user_store(monkeypatch, tmp_path)
    monkeypatch.setenv("SWIGGY_IM_TOKEN", "env-token")

    def fail_get_user_token(user_id, server_key):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(
        swiggy_auth.store,
        "get_user_token",
        fail_get_user_token,
    )

    assert swiggy_auth.get_access_token("im", user_id="user-a") == "env-token"


def test_expired_user_token_refreshes_and_persists_under_same_user(monkeypatch, tmp_path):
    store = _prepare_user_store(monkeypatch, tmp_path)
    store.save_user_token(
        "user-a",
        "im",
        {"access_token": "old-access", "refresh_token": "old-refresh", "expires_at": 0},
    )
    with (
        patch.object(swiggy_auth.time, "time", side_effect=[1000, 1001]),
        patch.object(
            swiggy_auth,
            "_post_token",
            return_value={"access_token": "new-access", "refresh_token": "new-refresh", "expires_in": 3600},
        ) as post_token,
    ):
        assert swiggy_auth.get_access_token("im", user_id="user-a") == "new-access"

    assert post_token.call_args.args[0]["refresh_token"] == "old-refresh"
    assert store.get_user_token("user-a", "im")["access_token"] == "new-access"
    assert store.get_user_token("user-a", "im")["refresh_token"] == "new-refresh"


def test_token_values_are_absent_from_success_refresh_and_failure_logs(
    monkeypatch, tmp_path, caplog
):
    store = _prepare_user_store(monkeypatch, tmp_path)
    access_token = "access-secret-value"
    refresh_token = "refresh-secret-value"
    authorization_code = "authorization-code-secret"
    store.save_user_token(
        "user-a",
        "im",
        {"access_token": access_token, "refresh_token": refresh_token, "expires_at": 0},
    )

    with caplog.at_level(logging.INFO):
        with patch.object(
            swiggy_auth,
            "_post_token",
            return_value={"access_token": "refreshed-secret", "expires_in": 3600},
        ):
            assert swiggy_auth.get_access_token("im", user_id="user-a") == "refreshed-secret"

        with patch.object(
            swiggy_auth,
            "_post_token",
            side_effect=RuntimeError(f"remote rejected {refresh_token}"),
        ):
            store.save_user_token(
                "user-a",
                "im",
                {"access_token": access_token, "refresh_token": refresh_token, "expires_at": 0},
            )
            with pytest.raises(RuntimeError) as refresh_error:
                swiggy_auth.get_access_token("im", user_id="user-a")

        with patch.object(
            swiggy_auth,
            "_post_token",
            side_effect=RuntimeError(f"remote rejected {authorization_code}"),
        ):
            with pytest.raises(RuntimeError) as exchange_error:
                swiggy_auth.exchange_code("im", authorization_code, "verifier", "http://localhost/callback")

    output = caplog.text
    assert access_token not in output
    assert refresh_token not in output
    assert authorization_code not in output
    assert refresh_token not in str(refresh_error.value)
    assert authorization_code not in str(exchange_error.value)


class SwiggyAuthStatusTests(unittest.TestCase):
    def test_get_access_tokens_can_require_only_instamart(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            token_store = os.path.join(tmpdir, ".swiggy_tokens.json")
            with patch.dict(os.environ, {"SWIGGY_IM_TOKEN": "im-token"}, clear=True):
                with patch.object(swiggy_auth, "TOKEN_STORE", token_store):
                    result = swiggy_auth.get_access_tokens(("im",))

        self.assertEqual(result, {"im": "im-token"})

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
