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

import os
import re
import hashlib
import httpx
from typing import Optional
from fastapi import APIRouter, Request, Form
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse, Gather, Say, Play, Hangup
from dotenv import load_dotenv

from agent import process_message, clear_session


def clean_for_voice(text: str) -> str:
    """Strip emojis, markdown, and symbols that TTS reads literally."""
    # Remove emojis
    text = re.sub(r'[^\x00-\x7Fऀ-ॿÀ-ɏ]+', '', text)
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

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")


def get_base_url() -> str:
    """Return BASE_URL env var — set by Railway in prod, or by start.sh locally."""
    # NOTE: Do NOT call load_dotenv(override=True) here — that would let a local
    # .env file override Railway's env vars, breaking voice callbacks in production.
    return os.getenv("BASE_URL", "http://localhost:8000")

# TTS cache — avoid re-generating same phrases
_tts_cache: dict[str, str] = {}


async def generate_tts_audio(text: str) -> Optional[str]:
    """
    Generate speech audio via ElevenLabs (async to avoid blocking event loop).
    Returns public URL to audio file, or None to fall back to Twilio <Say>.

    Fixes:
    - Uses async httpx so Twilio's 15s timeout isn't eaten by a blocking call.
    - Removed invalid 'speed' from voice_settings (not a valid field).
    - Updated to eleven_turbo_v2_5 for lower latency.
    """
    if not ELEVENLABS_API_KEY:
        return None

    cache_key = hashlib.md5(text.encode()).hexdigest()
    if cache_key in _tts_cache:
        return _tts_cache[cache_key]

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
            audio_path = f"/tmp/tts_{cache_key}.mp3"
            with open(audio_path, "wb") as f:
                f.write(response.content)
            url = f"{get_base_url()}/audio/{cache_key}"
            _tts_cache[cache_key] = url
            return url
        else:
            print(f"ElevenLabs TTS error {response.status_code}: {response.text[:200]}")
    except Exception as e:
        print(f"ElevenLabs TTS exception: {e}")
    return None


async def make_twiml_response(
    agent_text: str,
    session_id: str,
    is_final: bool = False,
    gather_timeout: int = 4
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

    # Try ElevenLabs TTS first, fall back to Twilio <Say>
    audio_url = await generate_tts_audio(spoken_text)

    if is_final:
        if audio_url:
            vr.play(audio_url)
        else:
            vr.say(spoken_text, voice="Polly.Aditi", language="en-IN")
        vr.hangup()
    else:
        base = get_base_url()
        gather = Gather(
            input="speech",
            action=f"{base}/voice/process",
            method="POST",
            timeout=gather_timeout,
            speech_timeout="auto",
            language="en-IN",
            hints="yes,haan,no,nahi,confirm,cancel,biryani,pizza,milk,eggs,bread,order,theek hai"
        )
        if audio_url:
            gather.play(audio_url)
        else:
            gather.say(spoken_text, voice="Polly.Aditi", language="en-IN")
        vr.append(gather)

        # If user doesn't speak within timeout — hang up cleanly (no repeat loop)
        vr.say("No problem, call back anytime you're hungry. Goodbye!", voice="Polly.Aditi", language="en-IN")
        vr.hangup()

    return str(vr)


def is_farewell(text: str) -> bool:
    """Detect if user wants to end the call."""
    farewells = ["bye", "goodbye", "cancel", "nevermind", "nothing", "hang up", "end call", "band kar"]
    return any(f in text.lower() for f in farewells)


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
    call_sid = form.get("CallSid", "unknown")

    greeting = "Hi, Swiggy here! What would you like to order?"

    twiml = await make_twiml_response(greeting, session_id=call_sid)
    return Response(content=twiml, media_type="application/xml")


@router.post("/process")
async def voice_process(request: Request):
    """
    Processes speech input from Twilio Gather.
    Runs agent, returns spoken TwiML response.
    """
    form = await request.form()
    call_sid = form.get("CallSid", "unknown")
    speech_result = form.get("SpeechResult", "")
    confidence = float(form.get("Confidence", 0))

    # NOTE: Confidence check removed — Twilio returns 0.0 for short but valid
    # utterances like "yes", "haan", "okay", which breaks the confirmation flow.
    # We trust the SpeechResult text and let the agent handle ambiguity.

    # Farewell check
    if is_farewell(speech_result):
        vr = VoiceResponse()
        vr.say("Alright, no problem! Call back anytime you're hungry. Goodbye!", voice="Polly.Aditi", language="en-IN")
        vr.hangup()
        clear_session(call_sid)
        return Response(content=str(vr), media_type="application/xml")

    # Empty input
    if not speech_result.strip():
        twiml = await make_twiml_response(
            "I'm here! What would you like to order?",
            session_id=call_sid
        )
        return Response(content=twiml, media_type="application/xml")

    # Run agent
    agent_response = process_message(
        session_id=call_sid,
        user_message=speech_result,
        surface="voice"
    )

    # Check if order is complete → hang up after speaking
    final = is_order_complete(agent_response)
    if final:
        clear_session(call_sid)

    twiml = await make_twiml_response(
        agent_text=agent_response,
        session_id=call_sid,
        is_final=final,
        gather_timeout=4
    )
    return Response(content=twiml, media_type="application/xml")


@router.post("/status")
async def voice_status(request: Request):
    """Twilio call status webhook — cleanup on call end."""
    form = await request.form()
    call_sid = form.get("CallSid", "")
    call_status = form.get("CallStatus", "")
    if call_status in ("completed", "failed", "busy", "no-answer"):
        clear_session(call_sid)
    return Response(status_code=204)
