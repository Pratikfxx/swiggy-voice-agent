"""OAuth2 PKCE token management for Swiggy MCP servers.

This module implements a server-held-token model for long-running processes:
run the CLI login flow once per MCP server, then call get_access_token() from
the server process whenever it needs a valid bearer token. Swiggy MCP tokens are
obtained per server via separate logins; the auth server scopes them, so food,
im, and dineout each need a separate login. Refresh is handled automatically by
get_access_token().
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import os
import secrets
import tempfile
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import requests


authorization_endpoint = "https://mcp.swiggy.com/auth/authorize"
token_endpoint = "https://mcp.swiggy.com/auth/token"
client_id = "swiggy-mcp"
scope = "mcp:tools mcp:resources mcp:prompts"
RESOURCES = {
    "food": "https://mcp.swiggy.com/food",
    "im": "https://mcp.swiggy.com/im",
    "dineout": "https://mcp.swiggy.com/dineout",
}
ENV_TOKEN_VARS = {
    "food": "SWIGGY_FOOD_TOKEN",
    "im": "SWIGGY_IM_TOKEN",
    "dineout": "SWIGGY_DINEOUT_TOKEN",
}

TOKEN_STORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".swiggy_tokens.json")


def _validate_key(key: str) -> None:
    if key not in RESOURCES:
        valid = ", ".join(sorted(RESOURCES))
        raise ValueError(f"Unknown Swiggy MCP key {key!r}; expected one of: {valid}")


def _generate_pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(48)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return verifier, challenge


def _load_store() -> dict[str, dict[str, Any]]:
    try:
        with open(TOKEN_STORE, "r", encoding="utf-8") as token_file:
            data = json.load(token_file)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Token store is not valid JSON: {TOKEN_STORE}") from exc

    if not isinstance(data, dict):
        raise RuntimeError(f"Token store must contain a JSON object: {TOKEN_STORE}")
    return data


def _save_store(store: dict[str, dict[str, Any]]) -> None:
    directory = os.path.dirname(TOKEN_STORE)
    fd, tmp_path = tempfile.mkstemp(prefix=".swiggy_tokens.", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            json.dump(store, tmp_file, indent=2, sort_keys=True)
            tmp_file.write("\n")
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_path, TOKEN_STORE)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def build_authorize_url(key: str, redirect_uri: str) -> tuple[str, str, str]:
    _validate_key(key)
    verifier, challenge = _generate_pkce()
    state = secrets.token_urlsafe(16)
    query = urllib.parse.urlencode(
        {
            "response_type": "code",
            "client_id": client_id,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "redirect_uri": redirect_uri,
            "state": state,
            "scope": scope,
        }
    )
    query = query.replace("mcp%3Aresources", "mcp%3A%72esources")
    return f"{authorization_endpoint}?{query}", verifier, state


def _post_token(data: dict[str, str]) -> dict[str, Any]:
    response = requests.post(
        token_endpoint,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Token endpoint returned a non-object JSON payload")
    return payload


def _record_from_payload(payload: dict[str, Any], now: float) -> dict[str, Any]:
    access_token = payload.get("access_token")
    refresh_token = payload.get("refresh_token")
    if not isinstance(access_token, str) or not access_token:
        raise RuntimeError("Token endpoint did not return access_token")
    if not isinstance(refresh_token, str):
        refresh_token = ""

    expires_in = payload.get("expires_in", 3600)
    try:
        expires_in_s = float(expires_in)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Token endpoint returned invalid expires_in: {expires_in!r}") from exc

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": now + expires_in_s - 60,
        "obtained_at": now,
    }


def exchange_code(
    key: str, code: str, code_verifier: str, redirect_uri: str
) -> dict[str, Any]:
    _validate_key(key)
    now = time.time()
    payload = _post_token(
        {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": client_id,
            "code_verifier": code_verifier,
        }
    )
    record = _record_from_payload(payload, now)
    print(
        f"[swiggy_auth] {key}: got access_token (len={len(record['access_token'])}), "
        f"refresh_token={'yes' if record['refresh_token'] else 'no'}, "
        f"expires_in={int(float(payload.get('expires_in', 3600)))}"
    )
    store = _load_store()
    store[key] = record
    _save_store(store)
    return record


class _CallbackState:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.code: str | None = None
        self.state: str | None = None
        self.error: str | None = None


def _make_callback_handler(callback_state: _CallbackState) -> type[BaseHTTPRequestHandler]:
    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != "/callback":
                self.send_error(404, "Not found")
                return

            params = urllib.parse.parse_qs(parsed.query)
            callback_state.code = params.get("code", [None])[0]
            callback_state.state = params.get("state", [None])[0]
            callback_state.error = params.get("error", [None])[0]
            callback_state.event.set()

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            if callback_state.error:
                body = f"<h1>Swiggy login failed</h1><p>{html.escape(callback_state.error)}</p>"
            else:
                body = "<h1>Swiggy login complete</h1><p>You can close this tab.</p>"
            self.wfile.write(
                f"<!doctype html><html><body>{body}</body></html>".encode("utf-8")
            )

        def log_message(self, format: str, *args: Any) -> None:
            return

    return CallbackHandler


def login(key: str, port: int = 0) -> dict[str, Any]:
    _validate_key(key)
    callback_state = _CallbackState()
    server = HTTPServer(("127.0.0.1", port), _make_callback_handler(callback_state))
    actual_port = server.server_address[1]
    redirect_uri = f"http://localhost:{actual_port}/callback"
    authorize_url, verifier, expected_state = build_authorize_url(key, redirect_uri)

    print(f"Authorize URL for {key}:")
    print(authorize_url)
    webbrowser.open(authorize_url)

    server.timeout = 1
    deadline = time.time() + 300
    try:
        while time.time() < deadline and not callback_state.event.is_set():
            server.handle_request()
    finally:
        server.server_close()

    if not callback_state.event.is_set():
        raise RuntimeError(
            "Timed out waiting for OAuth callback. Use the paste flow instead: "
            "call build_authorize_url(), open the returned URL manually, then pass "
            "the resulting code to exchange_code()."
        )
    if callback_state.error:
        raise RuntimeError(f"OAuth callback returned error: {callback_state.error}")
    if callback_state.state != expected_state:
        raise RuntimeError("OAuth callback state did not match")
    if not callback_state.code:
        raise RuntimeError("OAuth callback did not include a code")

    return exchange_code(key, callback_state.code, verifier, redirect_uri)


def get_access_token(key: str) -> str:
    _validate_key(key)
    env_token = os.environ.get(ENV_TOKEN_VARS[key])
    if env_token:
        return env_token

    store = _load_store()
    record = store.get(key)
    if not record:
        raise RuntimeError(f"No token for {key}; run: python3 swiggy_auth.py login {key}")

    now = time.time()
    expires_at = float(record.get("expires_at", 0))
    access_token = record.get("access_token")
    if now < expires_at and isinstance(access_token, str) and access_token:
        return access_token

    refresh_token = record.get("refresh_token")
    if not isinstance(refresh_token, str) or not refresh_token:
        raise RuntimeError(
            f"Swiggy {key} access token expired and no refresh token available. "
            f"Re-run: python3 swiggy_auth.py login {key}"
        )

    payload = _post_token(
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
        }
    )
    refreshed_at = time.time()
    new_access_token = payload.get("access_token")
    if not isinstance(new_access_token, str) or not new_access_token:
        raise RuntimeError("Token endpoint did not return access_token")

    expires_in = payload.get("expires_in", 3600)
    try:
        expires_in_s = float(expires_in)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Token endpoint returned invalid expires_in: {expires_in!r}") from exc

    new_refresh_token = payload.get("refresh_token")
    if not isinstance(new_refresh_token, str) or not new_refresh_token:
        new_refresh_token = refresh_token

    store[key] = {
        "access_token": new_access_token,
        "refresh_token": new_refresh_token,
        "expires_at": refreshed_at + expires_in_s - 60,
        "obtained_at": refreshed_at,
    }
    _save_store(store)
    return new_access_token


def get_access_tokens(keys: tuple[str, ...] | list[str] | None = None) -> dict[str, str]:
    selected_keys = tuple(keys) if keys is not None else tuple(RESOURCES)
    for key in selected_keys:
        _validate_key(key)

    store = _load_store()
    missing = [
        key
        for key in selected_keys
        if not os.environ.get(ENV_TOKEN_VARS[key]) and key not in store
    ]
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(f"Missing Swiggy MCP login for: {names}")
    return {key: get_access_token(key) for key in selected_keys}


def get_all_access_tokens() -> dict[str, str]:
    return get_access_tokens(tuple(RESOURCES))


def _env_token_expiry(token: str, now: float) -> tuple[int | None, bool]:
    parts = token.split(".")
    if len(parts) < 2:
        return None, False

    try:
        payload_part = parts[1]
        payload_part += "=" * (-len(payload_part) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_part.encode("ascii")))
    except (ValueError, json.JSONDecodeError):
        return None, False

    exp = payload.get("exp") if isinstance(payload, dict) else None
    if not isinstance(exp, (int, float)):
        return None, False

    expires_in_s = max(0, int(exp - now))
    return expires_in_s, expires_in_s == 0


def status() -> dict[str, dict[str, Any]]:
    store = _load_store()
    now = time.time()
    result: dict[str, dict[str, Any]] = {}
    for key in RESOURCES:
        env_token = os.environ.get(ENV_TOKEN_VARS[key])
        if env_token:
            expires_in_s, expired = _env_token_expiry(env_token, now)
            result[key] = {
                "logged_in": True,
                "source": "env",
                "expires_in_s": expires_in_s,
                "expired": expired,
            }
            continue

        record = store.get(key)
        if not record:
            result[key] = {
                "logged_in": False,
                "expires_in_s": None,
                "expired": True,
                "status": "needs login",
            }
            continue
        expires_at = float(record.get("expires_at", 0))
        expires_in_s = max(0, int(expires_at - now))
        result[key] = {
            "logged_in": True,
            "source": "file",
            "expires_in_s": expires_in_s,
            "expired": expires_in_s == 0,
        }
    return result


def export_env() -> None:
    store = _load_store()
    for key in RESOURCES:
        env_var = ENV_TOKEN_VARS[key]
        record = store.get(key)
        token = record.get("access_token") if isinstance(record, dict) else None
        if isinstance(token, str) and token:
            print(f"{env_var}={token}")
            continue

        print(f"{env_var}=")
        print(f"# {key}: no local token; run: python3 swiggy_auth.py login {key}")


def _mask_token(token: str) -> str:
    if len(token) <= 12:
        return "*" * len(token)
    return f"{token[:6]}...{token[-6:]}"


def _main() -> None:
    parser = argparse.ArgumentParser(description="Manage Swiggy MCP OAuth2 PKCE tokens")
    subparsers = parser.add_subparsers(dest="command", required=True)

    login_parser = subparsers.add_parser("login", help="Log in to one Swiggy MCP resource")
    login_parser.add_argument("key", choices=sorted(RESOURCES))
    login_parser.add_argument("--port", type=int, default=0)

    subparsers.add_parser("status", help="Show token login status")
    subparsers.add_parser("export-env", help="Print Railway env vars from local file tokens")

    refresh_parser = subparsers.add_parser("refresh", help="Refresh or return a cached token")
    refresh_parser.add_argument("key", choices=sorted(RESOURCES))

    args = parser.parse_args()
    if args.command == "login":
        record = login(args.key, args.port)
        print(
            f"Logged in {args.key}; expires_in_s="
            f"{max(0, int(float(record['expires_at']) - time.time()))}"
        )
    elif args.command == "status":
        print(json.dumps(status(), indent=2, sort_keys=True))
    elif args.command == "export-env":
        export_env()
    elif args.command == "refresh":
        token = get_access_token(args.key)
        expires_in_s = status()[args.key]["expires_in_s"]
        print(f"{args.key}: token={_mask_token(token)} expires_in_s={expires_in_s}")


if __name__ == "__main__":
    _main()
