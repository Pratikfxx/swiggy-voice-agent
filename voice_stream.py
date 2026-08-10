"""Throwaway Twilio Media Streams echo probe."""

import base64
import binascii
import hashlib
import hmac
import json
import logging
import os
import time
from urllib.parse import quote, urlparse

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from twilio.twiml.voice_response import Connect, VoiceResponse

from twilio_security import verify_twilio_request
from voice_handler import resolve_base_url


router = APIRouter(prefix="/voice", tags=["voice-stream"])
stream_logger = logging.getLogger("uvicorn.error")

STREAM_TOKEN_TTL_SECONDS = 300
DEV_STREAM_TOKEN = "dev-stream-token"


def _b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64url_decode(value: str) -> bytes | None:
    try:
        padded = value + "=" * (-len(value) % 4)
        decoded = base64.b64decode(padded, altchars=b"-_", validate=True)
        return decoded if _b64url_encode(decoded) == value else None
    except (binascii.Error, ValueError, TypeError):
        return None


def mint_stream_token(call_sid: str) -> str:
    """Mint a short-lived token bound to one Twilio call."""
    secret = os.getenv("TWILIO_AUTH_TOKEN", "")
    if not secret:
        return DEV_STREAM_TOKEN

    expires_at = int(time.time()) + STREAM_TOKEN_TTL_SECONDS
    payload = f"{call_sid}:{expires_at}"
    signature = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return _b64url_encode(f"{payload}:{signature}".encode("ascii"))


def _verified_call_sid(token: str, expected_call_sid: str | None = None) -> str | None:
    secret = os.getenv("TWILIO_AUTH_TOKEN", "")
    if not secret:
        return "" if token == DEV_STREAM_TOKEN else None

    decoded = _b64url_decode(token)
    if decoded is None:
        return None
    try:
        signed, provided_signature = decoded.decode("ascii").rsplit(":", 1)
        token_call_sid, expires_text = signed.rsplit(":", 1)
        expires_at = int(expires_text)
    except (UnicodeDecodeError, ValueError):
        return None

    if expires_at <= int(time.time()) or (
        expected_call_sid is not None and token_call_sid != expected_call_sid
    ):
        return None

    expected_signature = hmac.new(secret.encode(), signed.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(provided_signature, expected_signature):
        return None
    return token_call_sid


def verify_stream_token(token: str, call_sid: str | None = None) -> bool:
    return _verified_call_sid(token, call_sid) is not None


def _stream_url(request: Request, token: str) -> str:
    base_url = urlparse(resolve_base_url(request))._replace(scheme="wss").geturl().rstrip("/")
    return f"{base_url}/voice/stream?token={quote(token, safe='')}"


@router.post("/stream-test")
async def stream_test(request: Request):
    form = await request.form()
    if not verify_twilio_request(request, form):
        return Response(status_code=403)

    call_sid = str(form.get("CallSid", ""))
    response = VoiceResponse()
    connect = Connect()
    connect.stream(url=_stream_url(request, mint_stream_token(call_sid)))
    response.append(connect)
    return Response(content=str(response), media_type="application/xml")


@router.websocket("/stream")
async def stream(websocket: WebSocket):
    token = websocket.query_params.get("token", "")
    token_call_sid = _verified_call_sid(token)
    if token_call_sid is None:
        stream_logger.warning("VOICE stream auth rejected")
        await websocket.close(code=1008)
        return

    await websocket.accept()
    started_at = time.monotonic()
    call_sid = token_call_sid or "unknown"
    stream_sid = "unknown"
    frames_in = frames_out = 0

    try:
        while True:
            try:
                message = json.loads(await websocket.receive_text())
            except (json.JSONDecodeError, TypeError):
                continue

            if not isinstance(message, dict):
                continue

            event = message.get("event")
            if event == "start":
                start = message.get("start")
                if not isinstance(start, dict):
                    continue
                incoming_call_sid = str(start.get("callSid", ""))
                if token_call_sid and incoming_call_sid != token_call_sid:
                    stream_logger.warning("VOICE stream call mismatch")
                    await websocket.close(code=1008)
                    return
                call_sid = incoming_call_sid or call_sid
                stream_sid = str(start.get("streamSid", "")) or stream_sid
            elif event == "media":
                media = message.get("media")
                if not isinstance(media, dict) or not isinstance(media.get("payload"), str):
                    continue
                frames_in += 1
                await websocket.send_json(
                    {
                        "event": "media",
                        "streamSid": stream_sid,
                        "media": {"payload": media["payload"]},
                    }
                )
                frames_out += 1
            elif event == "stop":
                return
    except WebSocketDisconnect:
        pass
    except Exception:
        stream_logger.exception("VOICE stream endpoint error")
    finally:
        stream_logger.info(
            "VOICE stream call=%s stream=%s frames_in=%d frames_out=%d duration=%.1fs",
            call_sid,
            stream_sid,
            frames_in,
            frames_out,
            time.monotonic() - started_at,
        )
