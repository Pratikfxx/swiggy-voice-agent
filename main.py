"""
Swiggy Voice Agent — FastAPI Server

Routes:
  POST /voice/answer      — Twilio: incoming call
  POST /voice/process     — Twilio: speech result
  POST /voice/status      — Twilio: call status updates
  POST /whatsapp/webhook  — Twilio: WhatsApp messages
  GET  /audio/{hash}      — Serve ElevenLabs TTS audio
  GET  /health            — Health check
  GET  /                  — Demo info page
"""

import logging
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from dotenv import load_dotenv

import swiggy_auth
import swiggy_address
from voice_handler import router as voice_router
from whatsapp_handler import router as whatsapp_router

load_dotenv()

app = FastAPI(
    title="Swiggy Voice Agent",
    description="Order food and groceries via phone call or WhatsApp",
    version="1.0.0"
)

app.include_router(voice_router)
app.include_router(whatsapp_router)


@app.on_event("startup")
async def _warm():
    try:
        await swiggy_address.refresh_default_address()
    except Exception:
        logging.exception("startup address warm failed")


@app.get("/health")
def health():
    try:
        swiggy_tokens = swiggy_auth.status()
        swiggy_ready = all(
            info.get("logged_in") and not info.get("expired", False)
            for info in swiggy_tokens.values()
        )
    except Exception as exc:
        logging.exception("swiggy auth health failed")
        swiggy_tokens = {"error": exc.__class__.__name__}
        swiggy_ready = False

    return {
        "status": "ok",
        "demo_mode": os.getenv("DEMO_MODE", "true"),
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "twilio": bool(os.getenv("TWILIO_ACCOUNT_SID")),
        "elevenlabs": bool(os.getenv("ELEVENLABS_API_KEY")),
        "swiggy": swiggy_ready,
        "swiggy_tokens": swiggy_tokens,
    }


@app.get("/audio/{cache_key}")
def serve_audio(cache_key: str):
    """Serve generated TTS audio files."""
    audio_path = Path(f"/tmp/tts_{cache_key}.mp3")
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(audio_path, media_type="audio/mpeg")


@app.get("/", response_class=HTMLResponse)
def demo_page():
    phone = os.getenv("TWILIO_PHONE_NUMBER", "Not configured")
    wa = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886").replace("whatsapp:", "")
    base_url = os.getenv("BASE_URL", "http://localhost:8000")
    demo_mode = os.getenv("DEMO_MODE", "true")

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Swiggy Voice Agent</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * {{ box-sizing: border-box; margin: 0; padding: 0; }}
            body {{ font-family: -apple-system, sans-serif; background: #FFF8F3; color: #111; padding: 24px; }}
            .header {{ background: #FC8019; color: white; border-radius: 12px; padding: 24px; margin-bottom: 24px; }}
            .header h1 {{ font-size: 28px; margin-bottom: 4px; }}
            .header p {{ opacity: 0.9; font-size: 14px; }}
            .card {{ background: white; border-radius: 10px; padding: 20px; margin-bottom: 16px; border: 1px solid #f0e6dc; }}
            .card h2 {{ font-size: 16px; font-weight: 700; margin-bottom: 12px; color: #FC8019; }}
            .stat {{ display: inline-block; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; margin: 3px; }}
            .ok {{ background: #EDFAF2; color: #16a34a; }}
            .warn {{ background: #FFF3E8; color: #FC8019; }}
            .channel {{ display: flex; align-items: center; gap: 12px; padding: 12px; background: #FFF8F3; border-radius: 8px; margin-bottom: 8px; }}
            .channel .icon {{ font-size: 28px; }}
            .channel h3 {{ font-size: 14px; font-weight: 700; }}
            .channel p {{ font-size: 12px; color: #888; }}
            code {{ background: #f5f5f5; padding: 2px 6px; border-radius: 4px; font-size: 12px; }}
            .demo-box {{ background: #1a1a1a; color: #eee; border-radius: 8px; padding: 16px; font-size: 12px; line-height: 1.8; font-family: monospace; }}
            .demo-box .user {{ color: #FC8019; }}
            .demo-box .bot {{ color: #22C55E; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🎙️ Swiggy Voice Agent</h1>
            <p>Order food &amp; groceries via phone call or WhatsApp · Powered by Claude + Swiggy MCP</p>
        </div>

        <div class="card">
            <h2>System Status</h2>
            <span class="stat {'ok' if os.getenv('ANTHROPIC_API_KEY') else 'warn'}">
                Claude {'✓' if os.getenv('ANTHROPIC_API_KEY') else '✗ Missing key'}
            </span>
            <span class="stat {'ok' if os.getenv('TWILIO_ACCOUNT_SID') else 'warn'}">
                Twilio {'✓' if os.getenv('TWILIO_ACCOUNT_SID') else '✗ Missing key'}
            </span>
            <span class="stat {'ok' if os.getenv('ELEVENLABS_API_KEY') else 'warn'}">
                ElevenLabs {'✓' if os.getenv('ELEVENLABS_API_KEY') else '⚠ Optional (using Twilio TTS)'}
            </span>
            <span class="stat {'warn' if demo_mode == 'true' else 'ok'}">
                Mode: {'🔶 DEMO (mock data)' if demo_mode == 'true' else '🟢 LIVE (Swiggy MCP)'}
            </span>
        </div>

        <div class="card">
            <h2>Active Channels</h2>
            <div class="channel">
                <div class="icon">📞</div>
                <div>
                    <h3>Phone Line</h3>
                    <p>Call <strong>{phone}</strong> → speak your order → done</p>
                    <p>Webhook: <code>{base_url}/voice/answer</code></p>
                </div>
            </div>
            <div class="channel">
                <div class="icon">💬</div>
                <div>
                    <h3>WhatsApp</h3>
                    <p>Message <strong>{wa}</strong> → text or voice note → done</p>
                    <p>Webhook: <code>{base_url}/whatsapp/webhook</code></p>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Demo Conversations</h2>
            <div class="demo-box">
<span class="user">User: "Ek chicken biryani chahiye"</span>
<span class="bot">Bot:  "Biryani Blues is nearby — 4.5 stars, 30 mins, 299 rupees. Want that?"</span>
<span class="user">User: "Haan"</span>
<span class="bot">Bot:  "Placed! Arriving in 30 minutes. Order confirmed!"</span>
            </div>
            <br>
            <div class="demo-box">
<span class="user">User: "Order milk, eggs, bread and items for Alfredo pasta"</span>
<span class="bot">Bot:  "Adding to cart: milk, eggs, bread, pasta, heavy cream, parmesan, butter, garlic.</span>
<span class="bot">       Total 480 rupees via Instamart, delivery in 15 mins. Confirm?"</span>
<span class="user">User: "Yes"</span>
<span class="bot">Bot:  "✅ Order placed! #IM29485. Arriving in ~15 minutes."</span>
            </div>
        </div>

        <div class="card">
            <h2>API Endpoints</h2>
            <p><code>GET /health</code> — System health check</p><br>
            <p><code>POST /voice/answer</code> — Twilio voice webhook</p><br>
            <p><code>POST /whatsapp/webhook</code> — Twilio WA webhook</p><br>
            <p><code>GET /docs</code> — Full API documentation</p>
        </div>
    </body>
    </html>
    """


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
