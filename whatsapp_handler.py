"""
WhatsApp Handler — Twilio WA Webhook

Handles incoming WhatsApp messages (text + voice notes).
Same agent as voice, runs in "chat" mode for richer responses.

Setup:
  - Sandbox: text "join <code>" to whatsapp:+14155238886
  - Production: WhatsApp Business number via Twilio
"""

import asyncio
import logging
import os
import base64
import httpx
from fastapi import APIRouter, Request, Form
from fastapi.responses import Response
from twilio.rest import Client as TwilioClient
from dotenv import load_dotenv

from agent import process_message, get_session, update_session

load_dotenv()

router = APIRouter(prefix="/whatsapp", tags=["whatsapp"])

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
wa_logger = logging.getLogger("uvicorn.error")

# Keep strong refs to in-flight background tasks so they aren't GC'd mid-run
_background_tasks: set[asyncio.Task] = set()


async def transcribe_voice_note(media_url: str, media_content_type: str) -> str:
    """
    Download a WhatsApp voice note and transcribe it.

    Priority:
      1. ElevenLabs Scribe v1  — already-configured key, great Hindi/Hinglish
      2. Deepgram nova-2        — best accuracy, needs DEEPGRAM_API_KEY
      3. OpenAI Whisper-1       — reliable fallback, needs OPENAI_API_KEY
    """
    if not media_url:
        return ""

    try:
        # ── Download audio from Twilio (requires Basic auth) ────────────────
        async with httpx.AsyncClient() as client:
            dl = await client.get(
                media_url,
                auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
                timeout=15.0,
                follow_redirects=True,
            )
            if dl.status_code != 200:
                print(f"Audio download failed: {dl.status_code}")
                return ""
            audio_bytes = dl.content

        # Derive a safe filename/mime for uploads
        ext = "ogg"
        mime = media_content_type or "audio/ogg"
        if "mp4" in mime or "m4a" in mime:
            ext = "m4a"
        elif "mpeg" in mime or "mp3" in mime:
            ext = "mp3"
        filename = f"voice.{ext}"

        # ── 1. ElevenLabs Scribe (uses existing ELEVENLABS_API_KEY) ─────────
        if ELEVENLABS_API_KEY:
            try:
                async with httpx.AsyncClient() as client:
                    r = await client.post(
                        "https://api.elevenlabs.io/v1/speech-to-text",
                        headers={"xi-api-key": ELEVENLABS_API_KEY},
                        files={"file": (filename, audio_bytes, mime)},
                        data={
                            "model_id": "scribe_v1",
                            "language_code": "mul",  # auto-detect (Hindi, English, etc.)
                        },
                        timeout=20.0,
                    )
                if r.status_code == 200:
                    text = r.json().get("text", "").strip()
                    if text:
                        print(f"ElevenLabs STT: {text!r}")
                        return text
                else:
                    print(f"ElevenLabs STT error {r.status_code}: {r.text[:200]}")
            except Exception as e:
                print(f"ElevenLabs STT exception: {e}")

        # ── 2. Deepgram nova-2 (best for Indian accents) ─────────────────────
        deepgram_key = os.getenv("DEEPGRAM_API_KEY", "")
        if deepgram_key:
            try:
                async with httpx.AsyncClient() as client:
                    r = await client.post(
                        "https://api.deepgram.com/v1/listen"
                        "?model=nova-2&detect_language=true&smart_format=true",
                        headers={
                            "Authorization": f"Token {deepgram_key}",
                            "Content-Type": mime,
                        },
                        content=audio_bytes,
                        timeout=15.0,
                    )
                if r.status_code == 200:
                    text = (
                        r.json()
                        .get("results", {})
                        .get("channels", [{}])[0]
                        .get("alternatives", [{}])[0]
                        .get("transcript", "")
                        .strip()
                    )
                    if text:
                        print(f"Deepgram STT: {text!r}")
                        return text
            except Exception as e:
                print(f"Deepgram STT exception: {e}")

        # ── 3. OpenAI Whisper-1 ───────────────────────────────────────────────
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if openai_key:
            try:
                import io
                async with httpx.AsyncClient() as client:
                    r = await client.post(
                        "https://api.openai.com/v1/audio/transcriptions",
                        headers={"Authorization": f"Bearer {openai_key}"},
                        files={"file": (filename, io.BytesIO(audio_bytes), mime)},
                        data={"model": "whisper-1"},
                        timeout=20.0,
                    )
                if r.status_code == 200:
                    text = r.json().get("text", "").strip()
                    if text:
                        print(f"Whisper STT: {text!r}")
                        return text
            except Exception as e:
                print(f"Whisper STT exception: {e}")

    except Exception as e:
        print(f"Voice note download error: {e}")

    return ""


async def analyze_fridge_image(media_url: str) -> list[str]:
    """
    Download a fridge/pantry photo from Twilio and pass it to Claude Vision.
    Returns a list of food items identified (e.g. ['milk', 'eggs', 'spinach']).
    """
    import anthropic as _anthropic

    try:
        # Download image (Twilio requires auth)
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                media_url,
                auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
                timeout=15.0
            )
            if resp.status_code != 200:
                return []
            image_bytes = resp.content
            content_type = resp.headers.get("content-type", "image/jpeg")

        # Base64-encode for Claude Vision
        image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

        _client = _anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        message = await asyncio.to_thread(
            _client.messages.create,
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": content_type,
                                "data": image_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Look at this fridge or pantry photo. "
                                "List ONLY the food and drink items you can clearly see. "
                                "One item per line, plain English, no quantities, no formatting. "
                                "Example:\nmilk\neggs\nspinach\nbutton mushrooms\n"
                                "If this is not a fridge/pantry photo, reply with exactly: NOT_FRIDGE"
                            )
                        }
                    ],
                }
            ],
        )

        raw = message.content[0].text.strip()
        if raw == "NOT_FRIDGE" or not raw:
            return []

        items = [line.strip().lower() for line in raw.splitlines() if line.strip()]
        return items

    except Exception as e:
        print(f"Fridge analysis error: {e}")
        return []


def send_whatsapp_message(to: str, body: str) -> None:
    """Send a WhatsApp message via Twilio."""
    twilio_client.messages.create(
        from_=TWILIO_WHATSAPP_NUMBER,
        to=to,
        body=body
    )


async def _send(to: str, body: str) -> None:
    """Async wrapper — Twilio REST call is blocking, keep it off the event loop."""
    await asyncio.to_thread(send_whatsapp_message, to, body)


@router.post("/webhook")
async def whatsapp_webhook(request: Request):
    """
    Main WhatsApp webhook — parse the form, ack Twilio immediately, and handle
    the message in a background task. Agent turns can outlive Twilio's webhook
    timeout, and a timed-out webhook gets retried (double processing).
    """
    form = await request.form()

    from_number = form.get("From", "")       # e.g. "whatsapp:+919876543210"
    body = form.get("Body", "").strip()
    num_media = int(form.get("NumMedia", 0))
    media_url = form.get("MediaUrl0", "")
    media_content_type = form.get("MediaContentType0", "")

    task = asyncio.create_task(
        _handle_incoming(from_number, body, num_media, media_url, media_content_type)
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return Response(status_code=200)


async def _handle_incoming(
    from_number: str,
    body: str,
    num_media: int,
    media_url: str,
    media_content_type: str,
) -> None:
    try:
        await _handle_incoming_inner(from_number, body, num_media, media_url, media_content_type)
    except Exception:
        wa_logger.exception("WhatsApp handling failed from=%s", from_number)
        try:
            await _send(from_number, "Sorry, something went wrong. Please try again.")
        except Exception:
            wa_logger.exception("WhatsApp error reply failed from=%s", from_number)


async def _handle_incoming_inner(
    from_number: str,
    body: str,
    num_media: int,
    media_url: str,
    media_content_type: str,
) -> None:
    session_id = from_number  # phone number as session key

    # ── Handle fridge/pantry photo ───────────────────────────────────────────
    if num_media > 0 and "image" in media_content_type:
        await _send(from_number, "📸 Scanning your fridge... give me a sec!")
        items = await analyze_fridge_image(media_url)

        if not items:
            await _send(
                from_number,
                "Hmm, I couldn't make out food items in that photo. "
                "Try a clearer shot of your fridge or pantry shelves!"
            )
            return

        # Store inventory in session so agent can reference it
        history = get_session(session_id)
        # Inject as a system-style user message so it persists in conversation memory
        fridge_note = (
            f"[FRIDGE SCAN] I can see the following items already in the user's fridge/pantry: "
            f"{', '.join(items)}. "
            f"When the user asks for recipe ingredients or a grocery list, "
            f"DO NOT order items already in the fridge. Only order what is missing."
        )
        history.append({"role": "user", "content": fridge_note})
        history.append({
            "role": "assistant",
            "content": (
                f"Got it! I've scanned your fridge and noted: *{', '.join(items)}*. "
                f"I won't order these again — just the missing bits next time you ask for a recipe or grocery run."
            )
        })
        update_session(session_id, history)

        item_list = "\n".join(f"  • {i}" for i in items)
        await _send(
            from_number,
            f"✅ *Fridge scanned!* I can see:\n{item_list}\n\n"
            f"Next time you ask for recipe ingredients, I'll only order what's *missing* from this list. 🛒"
        )
        return

    # ── Handle voice notes ────────────────────────────────────────────────────
    if num_media > 0 and "audio" in media_content_type:
        # Immediate ack so user knows we received it (transcription takes 1–3s)
        await _send(from_number, "🎤 Got your voice note, transcribing...")

        body = await transcribe_voice_note(media_url, media_content_type)

        if not body:
            await _send(
                from_number,
                "Sorry, I couldn't make that out. Could you send it again or type your order?"
            )
            return

        # Echo back what was understood so the user can catch mis-transcriptions
        await _send(from_number, f'_Heard: "{body}"_')

    # Handle empty messages
    if not body:
        await _send(
            from_number,
            "👋 Hi! I'm Swiggy's ordering assistant.\n\n"
            "Just tell me what you want:\n"
            "• *Order biryani* — I'll find the best restaurant\n"
            "• *Get milk, eggs, bread* — Instamart delivery in ~15 mins\n"
            "• *Items for alfredo pasta* — I'll get all the ingredients\n\n"
            "What would you like today?"
        )
        return

    # Special commands
    if body.lower() in ["hi", "hello", "hey", "start", "help"]:
        await _send(
            from_number,
            "👋 *Hey! I'm Swiggy.* Ready to order?\n\n"
            "Try:\n"
            "• _Order chicken biryani_\n"
            "• _Get milk, eggs, and bread_\n"
            "• _Ingredients for butter chicken_\n"
            "• _Order pizza under ₹300_\n\n"
            "Just speak naturally — I'll handle the rest! 🍛"
        )
        return

    # Run agent — blocking call, keep it off the event loop
    agent_response = await asyncio.to_thread(
        process_message,
        session_id=session_id,
        user_message=body,
        surface="chat",
    )

    # Send response
    await _send(from_number, agent_response)
