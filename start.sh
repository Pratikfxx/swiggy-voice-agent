#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Swiggy Voice Agent — One-command local start
# Usage: bash start.sh
# ─────────────────────────────────────────────────────────────

cd "$(dirname "$0")"

echo ""
echo "🚀 Swiggy Voice Agent — Starting up..."
echo ""

# ── 1. Check .env ───────────────────────────────────────────
if [ ! -f .env ]; then
  echo "❌ .env file not found."
  exit 1
fi

source .env

if [[ "$ANTHROPIC_API_KEY" == *"PASTE"* ]] || [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "❌ ANTHROPIC_API_KEY not set in .env"
  exit 1
fi

echo "✓ Keys loaded"

# ── 1b. Pick the best Python (Homebrew > /usr/local > system) ─
if [ -x "/opt/homebrew/bin/python3" ]; then
  PYTHON="/opt/homebrew/bin/python3"
elif [ -x "/usr/local/bin/python3" ]; then
  PYTHON="/usr/local/bin/python3"
elif command -v brew &>/dev/null; then
  echo "Installing Python via Homebrew (one-time, ~1 min)..."
  brew install python3 -q 2>&1 | tail -3
  PYTHON="$(brew --prefix)/bin/python3"
else
  PYTHON="python3"
fi

# Verify pip is modern enough (need >=22 for current packages)
PIP_VER=$($PYTHON -m pip --version 2>/dev/null | awk '{print $2}' | cut -d. -f1)
if [ -z "$PIP_VER" ] || [ "$PIP_VER" -lt 22 ] 2>/dev/null; then
  echo "Upgrading pip..."
  $PYTHON -m pip install --upgrade pip -q 2>/dev/null || true
fi

echo "✓ Python: $($PYTHON --version 2>&1)  ($PYTHON)"

# ── 2. Install Python dependencies ──────────────────────────
echo "Installing dependencies..."
$PYTHON -m pip install -r requirements.txt --user -q \
  || $PYTHON -m pip install -r requirements.txt -q \
  || $PYTHON -m pip install -r requirements.txt --break-system-packages -q
echo "✓ Dependencies ready"

# Make sure user-installed scripts are on PATH
export PATH="$HOME/Library/Python/$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/bin:$PATH"

# ── 3. Start FastAPI server ──────────────────────────────────
export PORT=8000
export PYTHONUNBUFFERED=1

# Kill anything on port 8000
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
sleep 1

$PYTHON -m uvicorn main:app --host 0.0.0.0 --port $PORT > /tmp/uvicorn.log 2>&1 &
SERVER_PID=$!
echo "✓ Server starting (PID $SERVER_PID)..."

# Wait for server (up to 10s)
for i in {1..10}; do
  if curl -s http://localhost:8000/ > /dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! curl -s http://localhost:8000/ > /dev/null 2>&1; then
  echo "❌ Server failed to start. Log:"
  cat /tmp/uvicorn.log
  kill $SERVER_PID 2>/dev/null
  exit 1
fi
echo "✓ Server is up at http://localhost:8000"

# ── 4. Tunnel: try ngrok first, then cloudflared ────────────
echo ""
echo "Starting tunnel..."

PUBLIC_URL=""

# ── 4a. ngrok ───────────────────────────────────────────────
start_ngrok() {
  # Install if missing
  if ! command -v ngrok &> /dev/null; then
    if command -v brew &> /dev/null; then
      brew install ngrok/ngrok/ngrok -q 2>/dev/null || true
    fi
  fi
  export PATH="$HOME/.local/bin:$PATH"

  if ! command -v ngrok &> /dev/null; then
    return 1
  fi

  # Auth token (optional but required for v3 free)
  if [ -n "$NGROK_AUTH_TOKEN" ]; then
    ngrok config add-authtoken "$NGROK_AUTH_TOKEN" > /dev/null 2>&1
  fi

  pkill -f "ngrok http" 2>/dev/null || true
  sleep 1
  ngrok http $PORT --log=stdout --log-level=info > /tmp/ngrok.log 2>&1 &
  NGROK_PID=$!
  sleep 5

  PUBLIC_URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | $PYTHON -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for t in data.get('tunnels', []):
        if t.get('proto') == 'https':
            print(t['public_url'])
            break
except:
    pass
" 2>/dev/null)

  if [ -n "$PUBLIC_URL" ]; then
    TUNNEL_PID=$NGROK_PID
    TUNNEL_NAME="ngrok"
    return 0
  fi
  kill $NGROK_PID 2>/dev/null
  return 1
}

# ── 4b. cloudflared (zero-auth, always free) ────────────────
start_cloudflared() {
  # Install if missing
  if ! command -v cloudflared &> /dev/null; then
    echo "  Installing cloudflared..."
    if command -v brew &> /dev/null; then
      brew install cloudflared -q 2>/dev/null || true
    fi
    if ! command -v cloudflared &> /dev/null; then
      # Direct download for macOS
      ARCH=$(uname -m)
      if [[ "$ARCH" == "arm64" ]]; then
        CF_URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-arm64.tgz"
      else
        CF_URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-amd64.tgz"
      fi
      curl -sL "$CF_URL" -o /tmp/cloudflared.tgz
      tar -xzf /tmp/cloudflared.tgz -C /usr/local/bin/ 2>/dev/null || \
      tar -xzf /tmp/cloudflared.tgz -C "$HOME/.local/bin/"
      chmod +x /usr/local/bin/cloudflared 2>/dev/null || \
      chmod +x "$HOME/.local/bin/cloudflared" 2>/dev/null
      rm -f /tmp/cloudflared.tgz
    fi
  fi

  if ! command -v cloudflared &> /dev/null; then
    return 1
  fi

  pkill -f "cloudflared tunnel" 2>/dev/null || true
  sleep 1
  cloudflared tunnel --url http://localhost:$PORT --no-autoupdate > /tmp/cloudflared.log 2>&1 &
  CLOUDFLARED_PID=$!

  # Wait up to 15s for URL
  for i in {1..15}; do
    PUBLIC_URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' /tmp/cloudflared.log 2>/dev/null | head -1)
    [ -n "$PUBLIC_URL" ] && break
    sleep 1
  done

  if [ -n "$PUBLIC_URL" ]; then
    TUNNEL_PID=$CLOUDFLARED_PID
    TUNNEL_NAME="cloudflared"
    return 0
  fi
  kill $CLOUDFLARED_PID 2>/dev/null
  return 1
}

# Try ngrok first, fall back to cloudflared
if start_ngrok; then
  echo "✓ Tunnel via ngrok"
elif start_cloudflared; then
  echo "✓ Tunnel via cloudflared"
else
  echo "⚠️  Could not start tunnel automatically."
  echo "   Install ngrok: https://ngrok.com/download"
  echo "   Or cloudflared: https://developers.cloudflare.com/cloudflared"
fi

# ── 5. Update BASE_URL in server memory ─────────────────────
if [ -n "$PUBLIC_URL" ]; then
  # Hot-patch .env BASE_URL (server reads it via dotenv on each request)
  if grep -q "^BASE_URL=" .env; then
    sed -i '' "s|^BASE_URL=.*|BASE_URL=$PUBLIC_URL|" .env
  fi
fi

# ── 6. Print summary ─────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅  SWIGGY VOICE AGENT IS LIVE"
echo ""
if [ -n "$PUBLIC_URL" ]; then
  echo "  Public URL : $PUBLIC_URL"
  echo ""
  echo "  📞 TWILIO VOICE WEBHOOK:"
  echo "     $PUBLIC_URL/voice/answer"
  echo ""
  echo "  💬 TWILIO WHATSAPP WEBHOOK:"
  echo "     $PUBLIC_URL/whatsapp/webhook"
  echo ""
  echo "  🌐 Dashboard: $PUBLIC_URL"
else
  echo "  Local only: http://localhost:8000"
  echo ""
  echo "  → Set webhook manually via http://localhost:4040 (ngrok)"
  echo "    or check /tmp/cloudflared.log for your URL."
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Press Ctrl+C to stop all services."
echo ""

# ── 7. Cleanup on exit ───────────────────────────────────────
trap "echo ''; echo 'Stopping...'; kill $SERVER_PID $TUNNEL_PID 2>/dev/null; echo 'Done.'" EXIT
wait $SERVER_PID
