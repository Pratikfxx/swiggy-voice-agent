"""
Twilio Voice Handler — Phone call flow

Flow:
  1. Incoming call → /voice/answer → greet + start Gather
  2. User speaks → Twilio sends SpeechResult to /voice/process
  3. Agent processes → returns TwiML with spoken response + next Gather
  4. On order placed or "bye/cancel" → hang up

ElevenLabs TTS generates natural-sounding audio for each agent response.
Falls back to Twilio <Say> if ElevenLabs key not set.
"""

import logging
import os
import re
import time
import hashlib
import json
import asyncio
import httpx
from urllib.parse import quote, urlparse
from typing import Optional
from fastapi import APIRouter, Request, Form
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse, Gather, Say, Play, Hangup
from dotenv import load_dotenv

from agent import process_message, clear_session
from twilio_security import verify_twilio_request


_FAREWELL_RE = re.compile(r"\b(bye|goodbye|good bye|cancel|hang up|end call|band karo|band kar do|stop)\b", re.I)
_VOICE_CONFIRM_RE = re.compile(r"\b(yes|yeah|yep|haan|haa|ha|okay|ok|confirm|theek hai|thik hai|sure|go ahead)\b", re.I)
_VOICE_ITEM_COMMAND_RE = re.compile(
    r"\b(get|bring|add|order|need|want|me|please|some|a|an|the|from|on|swiggy|instamart|grocery|groceries|items?)\b",
    re.I,
)
_VOICE_ITEM_SPLIT_RE = re.compile(r"\s*(?:,|&|\+|\band\b|\baur\b)\s*", re.I)
# A split fragment carrying any of these is a question/command about the order
# ("which address", "what is the total", "use my office"), not a grocery item.
# Guards the multi-item fast path from mis-splitting natural sentences on "and".
_VOICE_NON_ITEM_RE = re.compile(
    r"\b(which|what|where|when|why|how|who|address|addresses|total|price|cost|"
    r"deliver|delivery|use|using|change|switch|office|home|work|is|are|was|will|"
    r"would|can|could|should|do|does|my|your|this|that|confirm|cancel|checkout|"
    r"place|cart|pay|payment|for)\b",
    re.I,
)


def clean_for_voice(text: str) -> str:
    """Strip emojis, markdown, and symbols that TTS reads literally."""
    # Remove emojis
    text = re.sub(r'[^\x00-\x7Fऀ-ॿÀ-ɏ]+', '', text)
    # Remove model narration that sounds robotic on a live call.
    text = re.sub(
        r"^\s*(?:i(?:'ll| will)|let me|lemme)\s+(?:search|look|check)(?:\s+for)?[^.?!]*[.?!]\s*",
        "",
        text,
        flags=re.I,
    )
    # Remove markdown bold/italic
    text = re.sub(r'\*+([^*]+)\*+', r'\1', text)
    text = re.sub(r'_+([^_]+)_+', r'\1', text)
    # Remove markdown headers
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    # Remove bullet points
    text = re.sub(r'^\s*[-•]\s+', '', text, flags=re.MULTILINE)
    # Collapse extra whitespace/newlines
    text = re.sub(r'\n+', '. ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

load_dotenv()

router = APIRouter(prefix="/voice", tags=["voice"])
voice_logger = logging.getLogger("uvicorn.error")

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
SPEECH_HINTS = ",".join(
    [
        "yes",
        "haan",
        "no",
        "nahi",
        "confirm",
        "cancel",
        "order",
        "instamart",
        "grocery",
        "groceries",
        "theek hai",
        "noodles",
        "paneer",
        "chicken",
        "milk",
        "eggs",
        "bread",
        "gatorade",
        "coke",
        "water",
        "chips",
        "curd",
        "cheese",
        "maggi",
        "atta",
        "rice",
        "oil",
        "sugar",
        "salt",
        "soap",
        "shampoo",
        "chocolate",
        "ice cream",
        "coffee",
        "tea",
        "diapers",
        "detergent",
        "toothpaste",
    ]
)

# Circuit breaker — skip ElevenLabs after repeated 4xx failures
_el_failures = 0
_el_disabled_until = 0.0
_el_disabled_reason = ""
_EL_MAX_FAILURES = 3
_EL_BACKOFF_SECS = 300
DEFAULT_GATHER_TIMEOUT = 7
# Outer deadline for the whole background turn. Must stay above agent.py's
# VOICE_API_TIMEOUT_SECS (20s) or it kills the API call before it can answer,
# and below the ~24s the /voice/result poll loop can keep the caller held.
VOICE_AGENT_TIMEOUT_SECS = float(os.getenv("VOICE_AGENT_TIMEOUT_SECS", "22.0"))
# Must cover VOICE_AGENT_TIMEOUT_SECS even if each poll is only ~2s of audio,
# otherwise the caller hears "taking longer" while the answer is still coming.
# Extra polls cost nothing: the loop exits as soon as the result lands.
VOICE_RESULT_MAX_POLLS = int(os.getenv("VOICE_RESULT_MAX_POLLS", "12"))
# The agent now batch-searches multiple items in parallel (~one search's time),
# so a normal grocery list no longer risks the voice deadline. Only trip the
# one-at-a-time guard for long lists; smaller ones go straight to the agent.
_VOICE_MULTI_ITEM_GUARD = int(os.getenv("VOICE_MULTI_ITEM_GUARD", "6"))
SILENCE_REPROMPT = "I didn't catch that. Say the item again, or say cancel."
VOICE_AGENT_TIMEOUT_MESSAGE = (
    "Swiggy is taking a bit longer. I'm still here. "
    "Say the item again, or try one item at a time."
)


_LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1", ""}


def get_base_url() -> str:
    """Return BASE_URL env var — set by Railway in prod, or by start.sh locally."""
    # NOTE: Do NOT call load_dotenv(override=True) here — that would let a local
    # .env file override Railway's env vars, breaking voice callbacks in production.
    return os.getenv("BASE_URL", "http://localhost:8000")


def _is_publicly_reachable(url: str) -> bool:
    if not url:
        return False
    try:
        return (urlparse(url).hostname or "") not in _LOCAL_HOSTS
    except ValueError:
        return False


def _base_url_from_request(request: Request) -> str:
    """Rebuild our public origin from the request Twilio actually reached us on.

    Tunnels (ngrok/cloudflared) rotate their hostname on every restart and
    Railway injects its own, so a stale or unset BASE_URL points Twilio at
    localhost and the call dies with "an application error occurred". The
    forwarded headers describe the host Twilio really used, so trust those.
    """
    # Never raise: this runs inside the Twilio webhook, and an exception here is
    # itself an "application error" on the call.
    try:
        headers = getattr(request, "headers", None) or {}
        host = (headers.get("x-forwarded-host") or headers.get("host") or "").split(",")[0].strip()
        if not host or host.split(":")[0] in _LOCAL_HOSTS:
            return ""

        scheme = (
            headers.get("x-forwarded-proto")
            or getattr(getattr(request, "url", None), "scheme", "")
            or "https"
        ).split(",")[0].strip()
        return f"{scheme}://{host}"
    except Exception:
        voice_logger.warning("could not derive base URL from request", exc_info=True)
        return ""


def resolve_base_url(request: Optional[Request] = None) -> str:
    """Public origin for callback URLs.

    An explicitly configured public BASE_URL always wins — it is the operator's
    stated intent and is not attacker-controllable. Only when it is unset or
    still pointing at localhost do we fall back to the request's own host.
    """
    configured = get_base_url().strip().rstrip("/")
    if _is_publicly_reachable(configured):
        return configured

    if request is not None:
        derived = _base_url_from_request(request)
        if derived:
            return derived

    return configured or "http://localhost:8000"


# TTS cache — remembers which phrases we already rendered to /tmp. Stores the
# cache key, not a full URL: the base URL can change between calls (tunnel
# restart), which would make a cached absolute URL point at a dead host.
_tts_cache: set[str] = set()
_voice_fast_pending: dict[str, str] = {}
_voice_agent_tasks: dict[str, asyncio.Task] = {}
_voice_agent_results: dict[str, dict] = {}
_voice_agent_job_ids: dict[str, int] = {}
_voice_agent_next_job_id = 0


def log_voice_input(call_sid: str, speech_result: str, confidence: float) -> None:
    voice_logger.info("VOICE in call=%s speech=%r confidence=%.2f", call_sid, speech_result, confidence)


def log_voice_output(call_sid: str, elapsed: float, agent_response: str) -> None:
    voice_logger.info("VOICE out call=%s elapsed=%.1fs reply=%r", call_sid, elapsed, agent_response)


def _extract_fast_instamart_items(text: str) -> list[str]:
    parts = _VOICE_ITEM_SPLIT_RE.split(text or "")
    items = []
    for part in parts:
        # Skip fragments that are questions/commands, not grocery items.
        if _VOICE_NON_ITEM_RE.search(part):
            continue
        item = _VOICE_ITEM_COMMAND_RE.sub(" ", part)
        item = re.sub(r"\s{2,}", " ", item).strip(" .,!?:;")
        if not item:
            continue
        # Grocery items are short noun phrases; anything longer is a sentence.
        if len(item.split()) > 4:
            continue
        items.append(item)
    return items


def _fast_voice_reply_or_message(call_sid: str, speech_result: str) -> tuple[str, str]:
    pending_item = _voice_fast_pending.get(call_sid)
    if pending_item:
        _voice_fast_pending.pop(call_sid, None)
        if _VOICE_CONFIRM_RE.search(speech_result or ""):
            voice_logger.info("VOICE fast pending call=%s item=%r", call_sid, pending_item)
            return "", f"get {pending_item}"

    items = _extract_fast_instamart_items(speech_result)
    if len(items) <= _VOICE_MULTI_ITEM_GUARD:
        return "", speech_result

    first, second = items[0], items[1]
    _voice_fast_pending[call_sid] = first
    voice_logger.info("VOICE fast multi-item call=%s first=%r remaining=%r", call_sid, first, items[1:])
    reply = (
        "Let's keep the call fast and do one item at a time. "
        f"Starting with {first}; {second} can come next. "
        f"Say yes to find {first}, or say another item."
    )
    return reply, speech_result


async def run_voice_agent_with_deadline(call_sid: str, speech_result: str) -> tuple[str, float, bool]:
    start = time.monotonic()
    try:
        agent_response = await asyncio.wait_for(
            asyncio.to_thread(
                process_message,
                session_id=call_sid,
                user_message=speech_result,
                surface="voice",
            ),
            timeout=VOICE_AGENT_TIMEOUT_SECS,
        )
    except asyncio.TimeoutError:
        elapsed = time.monotonic() - start
        voice_logger.warning(
            "VOICE timeout call=%s elapsed=%.1fs speech=%r",
            call_sid,
            elapsed,
            speech_result,
        )
        return VOICE_AGENT_TIMEOUT_MESSAGE, elapsed, True

    elapsed = time.monotonic() - start
    return agent_response, elapsed, False


async def _run_voice_agent_background(call_sid: str, speech_result: str, job_id: int) -> None:
    try:
        agent_response, elapsed, timed_out = await run_voice_agent_with_deadline(call_sid, speech_result)
        final = False if timed_out else is_order_complete(agent_response)
        result = {
            "response": agent_response,
            "elapsed": elapsed,
            "final": final,
        }
    except Exception:
        voice_logger.exception("VOICE background failed call=%s speech=%r", call_sid, speech_result)
        result = {
            "response": "Sorry, I hit a problem reaching Swiggy. Please try again in a moment.",
            "elapsed": 0.0,
            "final": False,
        }

    if _voice_agent_job_ids.get(call_sid) == job_id:
        _voice_agent_results[call_sid] = result
        _voice_agent_tasks.pop(call_sid, None)


def start_voice_agent_job(call_sid: str, speech_result: str) -> None:
    global _voice_agent_next_job_id
    _voice_agent_next_job_id += 1
    job_id = _voice_agent_next_job_id
    _voice_agent_job_ids[call_sid] = job_id
    _voice_agent_results.pop(call_sid, None)
    _voice_agent_tasks[call_sid] = asyncio.create_task(
        _run_voice_agent_background(call_sid, speech_result, job_id)
    )


def make_voice_waiting_twiml(
    call_sid: str, message: str, poll: int = 1, base_url: Optional[str] = None
) -> str:
    vr = VoiceResponse()
    vr.say(message, voice="Polly.Aditi", language="en-IN")
    vr.pause(length=1)
    base = (base_url or get_base_url()).rstrip("/")
    vr.redirect(
        f"{base}/voice/result?callSid={quote(call_sid)}&poll={poll}",
        method="POST",
    )
    return str(vr)


def _elevenlabs_error_status(response_text: str) -> str:
    try:
        payload = json.loads(response_text or "{}")
    except json.JSONDecodeError:
        return ""
    detail = payload.get("detail", {})
    return detail.get("status", "") if isinstance(detail, dict) else ""


async def generate_tts_audio(text: str, base_url: Optional[str] = None) -> Optional[str]:
    """
    Generate speech audio via ElevenLabs (async to avoid blocking event loop).
    Returns public URL to audio file, or None to fall back to Twilio <Say>.

    Fixes:
    - Uses async httpx so Twilio's 15s timeout isn't eaten by a blocking call.
    - Removed invalid 'speed' from voice_settings (not a valid field).
    - Updated to eleven_turbo_v2_5 for lower latency.
    """
    global _el_failures, _el_disabled_until, _el_disabled_reason
    if not ELEVENLABS_API_KEY:
        return None
    if _el_failures >= _EL_MAX_FAILURES and time.time() < _el_disabled_until:
        return None

    base = (base_url or get_base_url()).rstrip("/")
    cache_key = hashlib.md5(text.encode()).hexdigest()
    if cache_key in _tts_cache:
        return f"{base}/audio/{cache_key}"

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}",
                headers={
                    "xi-api-key": ELEVENLABS_API_KEY,
                    "Content-Type": "application/json"
                },
                json={
                    "text": text,
                    "model_id": "eleven_turbo_v2_5",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75
                    }
                }
            )
        if response.status_code == 200:
            _el_failures = 0
            audio_path = f"/tmp/tts_{cache_key}.mp3"
            with open(audio_path, "wb") as f:
                f.write(response.content)
            _tts_cache.add(cache_key)
            return f"{base}/audio/{cache_key}"
        else:
            status = _elevenlabs_error_status(response.text)
            if response.status_code == 401 and status == "detected_unusual_activity":
                _el_failures = _EL_MAX_FAILURES
                _el_disabled_until = float("inf")
                _el_disabled_reason = status
                voice_logger.warning(
                    "ElevenLabs disabled for this container: detected unusual activity; using Twilio Polly fallback"
                )
                return None
            voice_logger.warning("ElevenLabs TTS error status=%s body=%s", response.status_code, response.text[:200])
            _el_failures += 1
            if _el_failures >= _EL_MAX_FAILURES:
                _el_disabled_until = time.time() + _EL_BACKOFF_SECS
                _el_disabled_reason = f"http_{response.status_code}"
                voice_logger.warning("ElevenLabs circuit open for %ds (failure #%d)", _EL_BACKOFF_SECS, _el_failures)
    except Exception as e:
        voice_logger.warning("ElevenLabs TTS exception: %s", e)
    return None


async def make_twiml_response(
    agent_text: str,
    session_id: str,
    is_final: bool = False,
    gather_timeout: int = DEFAULT_GATHER_TIMEOUT,
    base_url: Optional[str] = None,
) -> str:
    """
    Build TwiML response that speaks agent_text then either:
    - Gathers more input (is_final=False)
    - Hangs up (is_final=True)
    Async because generate_tts_audio is now async.
    """
    vr = VoiceResponse()

    # Clean text before any TTS — strip emojis and markdown
    spoken_text = clean_for_voice(agent_text)

    base = (base_url or get_base_url()).rstrip("/")

    # Try ElevenLabs TTS first, fall back to Twilio <Say>
    audio_url = await generate_tts_audio(spoken_text, base_url=base)

    if is_final:
        if audio_url:
            vr.play(audio_url)
        else:
            vr.say(spoken_text, voice="Polly.Aditi", language="en-IN")
        vr.hangup()
    else:
        gather = Gather(
            input="speech",
            action=f"{base}/voice/process",
            method="POST",
            timeout=gather_timeout,
            speech_timeout="auto",
            language="en-IN",
            hints=SPEECH_HINTS,
        )
        if audio_url:
            gather.play(audio_url)
        else:
            gather.say(spoken_text, voice="Polly.Aditi", language="en-IN")
        vr.append(gather)

        retry_gather = Gather(
            input="speech",
            action=f"{base}/voice/process",
            method="POST",
            timeout=gather_timeout,
            speech_timeout="auto",
            language="en-IN",
            hints=SPEECH_HINTS,
        )
        retry_gather.say(SILENCE_REPROMPT, voice="Polly.Aditi", language="en-IN")
        vr.append(retry_gather)

    return str(vr)


GREETING = "Hi, this is Swiggy Instamart. What groceries or essentials should I get for you?"


async def prewarm_tts() -> None:
    """Render the canned lines once at startup.

    First caller otherwise pays ElevenLabs latency on the greeting; the rest
    are the reprompt/timeout lines every call can hit. No-op without a key,
    and failures just mean the per-call fallback path runs as before.
    """
    if not ELEVENLABS_API_KEY:
        return
    for line in (
        GREETING,
        "I'm here. What Instamart items should I get for you?",
        SILENCE_REPROMPT,
        VOICE_AGENT_TIMEOUT_MESSAGE,
    ):
        try:
            await generate_tts_audio(clean_for_voice(line))
        except Exception:
            voice_logger.warning("TTS pre-warm failed for %r", line[:40], exc_info=True)


def is_farewell(text: str) -> bool:
    """Detect if user wants to end the call."""
    return bool(_FAREWELL_RE.search(text or ""))


def is_order_complete(response_text: str) -> bool:
    """
    Detect if agent has actually placed an order (call can end).
    IMPORTANT: Be very specific — do NOT use words like 'confirmed' alone
    because the agent uses them mid-conversation ('let me confirm your order').
    Only trigger on phrases that unambiguously mean the order was placed.
    """
    completion_phrases = [
        "order is confirmed",
        "order placed",
        "enjoy your meal",
        "delivery updates on",
        "order id",
        "will be delivered",
        "your order is on its way",
        "arriving in",          # e.g. "arriving in 30 minutes. Goodbye!"
        "table is booked",
        "reservation is confirmed",
    ]
    text = response_text.lower()
    return any(p in text for p in completion_phrases)


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@router.post("/answer")
async def voice_answer(request: Request):
    """
    Entry point when someone calls the Twilio number.
    Greet and start gathering input.
    """
    form = await request.form()
    if not verify_twilio_request(request, form):
        return Response(status_code=403)
    call_sid = form.get("CallSid", "unknown")

    twiml = await make_twiml_response(
        GREETING, session_id=call_sid, base_url=resolve_base_url(request)
    )
    return Response(content=twiml, media_type="application/xml")


@router.post("/process")
async def voice_process(request: Request):
    """
    Processes speech input from Twilio Gather.
    Runs agent, returns spoken TwiML response.
    """
    form = await request.form()
    if not verify_twilio_request(request, form):
        return Response(status_code=403)
    call_sid = form.get("CallSid", "unknown")
    speech_result = form.get("SpeechResult", "")
    confidence = float(form.get("Confidence", 0))
    log_voice_input(call_sid, speech_result, confidence)

    # NOTE: Confidence check removed — Twilio returns 0.0 for short but valid
    # utterances like "yes", "haan", "okay", which breaks the confirmation flow.
    # We trust the SpeechResult text and let the agent handle ambiguity.

    # Farewell check
    if is_farewell(speech_result):
        vr = VoiceResponse()
        vr.say("Alright, no problem. Call back anytime. Goodbye!", voice="Polly.Aditi", language="en-IN")
        vr.hangup()
        clear_session(call_sid)
        return Response(content=str(vr), media_type="application/xml")

    base_url = resolve_base_url(request)

    # Empty input
    if not speech_result.strip():
        twiml = await make_twiml_response(
            "I'm here. What Instamart items should I get for you?",
            session_id=call_sid,
            base_url=base_url,
        )
        return Response(content=twiml, media_type="application/xml")

    fast_reply, speech_result = _fast_voice_reply_or_message(call_sid, speech_result)
    if fast_reply:
        twiml = await make_twiml_response(
            fast_reply,
            session_id=call_sid,
            gather_timeout=DEFAULT_GATHER_TIMEOUT,
            base_url=base_url,
        )
        return Response(content=twiml, media_type="application/xml")

    start_voice_agent_job(call_sid, speech_result)
    twiml = make_voice_waiting_twiml(
        call_sid, "Checking Instamart now. One moment.", base_url=base_url
    )
    return Response(content=twiml, media_type="application/xml")


@router.post("/result")
async def voice_result(request: Request):
    """Poll for a background voice agent result while keeping the call audible."""
    form = await request.form()
    if not verify_twilio_request(request, form):
        return Response(status_code=403)
    call_sid = request.query_params.get("callSid") or form.get("CallSid", "")
    try:
        poll = int(request.query_params.get("poll", "1"))
    except ValueError:
        poll = 1

    base_url = resolve_base_url(request)

    result = _voice_agent_results.pop(call_sid, None)
    if result:
        _voice_agent_job_ids.pop(call_sid, None)
        _voice_agent_tasks.pop(call_sid, None)
        agent_response = result["response"]
        elapsed = float(result.get("elapsed", 0.0))
        final = bool(result.get("final", False))
        log_voice_output(call_sid, elapsed, agent_response)
        if final:
            clear_session(call_sid)
        twiml = await make_twiml_response(
            agent_text=agent_response,
            session_id=call_sid,
            is_final=final,
            gather_timeout=DEFAULT_GATHER_TIMEOUT,
            base_url=base_url,
        )
        return Response(content=twiml, media_type="application/xml")

    if poll < VOICE_RESULT_MAX_POLLS:
        twiml = make_voice_waiting_twiml(
            call_sid,
            "Still checking Instamart.",
            poll=poll + 1,
            base_url=base_url,
        )
        return Response(content=twiml, media_type="application/xml")

    _voice_agent_job_ids.pop(call_sid, None)
    _voice_agent_tasks.pop(call_sid, None)
    twiml = await make_twiml_response(
        VOICE_AGENT_TIMEOUT_MESSAGE,
        session_id=call_sid,
        gather_timeout=DEFAULT_GATHER_TIMEOUT,
        base_url=base_url,
    )
    return Response(content=twiml, media_type="application/xml")


@router.post("/status")
async def voice_status(request: Request):
    """Twilio call status webhook — cleanup on call end."""
    form = await request.form()
    if not verify_twilio_request(request, form):
        return Response(status_code=403)
    call_sid = form.get("CallSid", "")
    call_status = form.get("CallStatus", "")
    if call_status in ("completed", "failed", "busy", "no-answer"):
        clear_session(call_sid)
    return Response(status_code=204)
