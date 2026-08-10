"""Browser-based Swiggy account linking for WhatsApp users."""

from __future__ import annotations

import base64
import binascii
import hashlib
import html
import hmac
import logging
import os
import threading
import time
from urllib.parse import parse_qs, urlencode, urlsplit

import store
import swiggy_auth
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from swiggy_scope import ACTIVE_SWIGGY_SERVERS, SERVER_AUTH_KEYS
from voice_handler import resolve_base_url


router = APIRouter(prefix="/link/swiggy", tags=["swiggy-link"])
link_logger = logging.getLogger("uvicorn.error")

LINK_TOKEN_TTL_SECONDS = 15 * 60
MAX_PENDING_STATES = 500

_pending_states: dict[str, tuple[str, str, str, float]] = {}
_state_lock = threading.Lock()


def _b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64url_decode(value: str) -> bytes | None:
    try:
        padded = value + "=" * (-len(value) % 4)
        decoded = base64.b64decode(padded, altchars=b"-_", validate=True)
        return decoded if _b64url_encode(decoded) == value else None
    except (binascii.Error, ValueError, TypeError):
        return None


def mint_link_token(user_id: str, server_key: str) -> str:
    secret = os.getenv("TWILIO_AUTH_TOKEN", "")
    if not secret:
        raise RuntimeError("Swiggy link signing secret is not configured")

    expires_at = int(time.time()) + LINK_TOKEN_TTL_SECONDS
    payload = f"{user_id}:{server_key}:{expires_at}"
    signature = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return _b64url_encode(f"{payload}:{signature}".encode("ascii"))


def verify_link_token(token: str) -> tuple[str, str] | None:
    secret = os.getenv("TWILIO_AUTH_TOKEN", "")
    if not secret:
        return None

    decoded = _b64url_decode(token)
    if decoded is None:
        return None
    try:
        signed, provided_signature = decoded.decode("ascii").rsplit(":", 1)
        user_id, server_key, expires_text = signed.rsplit(":", 2)
        expires_at = int(expires_text)
    except (UnicodeDecodeError, ValueError):
        return None

    if not user_id or not server_key or expires_at <= int(time.time()):
        return None

    expected_signature = hmac.new(secret.encode(), signed.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(provided_signature, expected_signature):
        return None
    return user_id, server_key


def active_server_key() -> str:
    active_servers = tuple(ACTIVE_SWIGGY_SERVERS)
    if len(active_servers) != 1:
        raise RuntimeError("Swiggy linking requires exactly one active server")
    return SERVER_AUTH_KEYS[active_servers[0]]


def _remember_state(
    state: str, user_id: str, server_key: str, verifier: str, created_at: float
) -> None:
    with _state_lock:
        cutoff = created_at - LINK_TOKEN_TTL_SECONDS
        for pending_state, entry in list(_pending_states.items()):
            if entry[3] <= cutoff:
                _pending_states.pop(pending_state, None)
        if len(_pending_states) >= MAX_PENDING_STATES:
            oldest_state = min(_pending_states, key=lambda key: _pending_states[key][3])
            _pending_states.pop(oldest_state, None)
        _pending_states[state] = (user_id, server_key, verifier, created_at)


def _take_state(state: str) -> tuple[str, str, str, float] | None:
    now = time.time()
    with _state_lock:
        entry = _pending_states.pop(state, None)
    if entry is None or entry[3] + LINK_TOKEN_TTL_SECONDS <= now:
        return None
    return entry


def _page(title: str, message: str, status_code: int = 200) -> HTMLResponse:
    body = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title></head><body>"
        f"<h1>{html.escape(title)}</h1><p>{html.escape(message)}</p>"
        "</body></html>"
    )
    return HTMLResponse(body, status_code=status_code)


def _exchange_failure(exc: Exception) -> HTMLResponse:
    link_logger.warning(
        "Swiggy link exchange failed (%s); the redirect URI may not be accepted by Swiggy",
        exc.__class__.__name__,
    )
    return _page(
        "Swiggy linking failed",
        "Swiggy could not complete account linking. The redirect URI may not be accepted by Swiggy. Please try again.",
        status_code=502,
    )


@router.get("/start")
def start_link(request: Request, token: str = ""):
    claims = verify_link_token(token)
    if claims is None:
        link_logger.warning("Swiggy link start rejected")
        return PlainTextResponse("Unable to start Swiggy linking.", status_code=403)

    user_id, server_key = claims
    try:
        callback_uri = f"{resolve_base_url(request).rstrip('/')}/link/swiggy/callback"
        authorize_url, first, second = swiggy_auth.build_authorize_url(
            server_key, callback_uri
        )
        url_state = parse_qs(urlsplit(authorize_url).query).get("state", [None])[0]
        if url_state == first:
            state, verifier = first, second
        elif url_state == second:
            state, verifier = second, first
        else:
            state, verifier = first, second
        _remember_state(state, user_id, server_key, verifier, time.time())
        return RedirectResponse(authorize_url, status_code=302)
    except Exception as exc:
        link_logger.warning("Swiggy link start failed (%s)", exc.__class__.__name__)
        return PlainTextResponse("Unable to start Swiggy linking.", status_code=500)


@router.get("/callback")
def link_callback(request: Request, code: str | None = None, state: str | None = None):
    pending = _take_state(state or "")
    if pending is None:
        link_logger.warning("Swiggy link callback rejected: unknown or expired state")
        return _page(
            "Swiggy linking failed",
            "This Swiggy linking request is invalid or expired.",
            status_code=400,
        )
    if not code:
        return _exchange_failure(RuntimeError("authorization code missing"))

    user_id, server_key, verifier, _created_at = pending
    try:
        callback_uri = f"{resolve_base_url(request).rstrip('/')}/link/swiggy/callback"
        record = swiggy_auth.exchange_code(server_key, code, verifier, callback_uri)
        store.save_user_token(user_id, server_key, record)
    except Exception as exc:
        return _exchange_failure(exc)

    return _page(
        "Swiggy linking complete",
        "Your Swiggy account is linked. Go back to WhatsApp to continue.",
    )


def build_start_url(user_id: str, server_key: str, base_url: str) -> str:
    token = mint_link_token(user_id, server_key)
    return f"{base_url.rstrip('/')}/link/swiggy/start?{urlencode({'token': token})}"
