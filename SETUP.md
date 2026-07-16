# Swiggy Voice Agent — Setup Guide
## Zero to live demo in ~2 hours

---

## DAY 1 — Get it running (3–4 hours)

### Step 1: Get your API keys (30 min)

Open these in parallel — all have free tiers or instant signups:

| Service | Where | What you need | Time |
|---------|-------|--------------|------|
| **Anthropic** | console.anthropic.com | API key | 5 min |
| **Twilio** | twilio.com/try-twilio | Account SID, Auth Token, phone number | 15 min |
| **ElevenLabs** | elevenlabs.io | API key (free: 10K chars/mo) | 5 min |
| **Railway** | railway.app | Account (deploy in 1 click) | 5 min |

**Twilio phone number:** After signup → Console → Phone Numbers → Buy a Number → India (+91) or US (+1) → pick one with Voice capability (~$1/month).

**Twilio WhatsApp sandbox:** Console → Messaging → Try it Out → WhatsApp. Note the sandbox number (+1 415 523 8886) and the join code.

---

### Step 2: Local setup (15 min)

```bash
# Clone / navigate to project
cd swiggy-voice-agent

# Create virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env from template
cp .env.example .env
```

Edit `.env` — fill in your keys:
```env
ANTHROPIC_API_KEY=sk-ant-...
TWILIO_ACCOUNT_SID=AC...
TWILIO_AUTH_TOKEN=...
TWILIO_PHONE_NUMBER=+1...
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886
ELEVENLABS_API_KEY=...
DEMO_MODE=true
BASE_URL=http://localhost:8000   # update after deploy

# Models (optional — these are the defaults)
AGENT_MODEL=claude-sonnet-5            # chat/WhatsApp brain — best value/quality for ordering
VOICE_MODEL=claude-haiku-4-5           # live phone calls — cheapest + fastest
CHAT_THINKING=disabled                 # Sonnet 5 thinks by default; disabled saves tokens. "adaptive" re-enables
VOICE_THINKING=disabled                # same for voice — worth ~12s per turn on a live call
CHAT_MAX_TOKENS=1024                   # per-reply output cap for chat
```

---

### Step 3: Test locally (10 min)

```bash
# Start the server
python main.py

# In another terminal — test the agent directly
python -c "
from agent import process_message
print(process_message('test_session', 'Order chicken biryani', surface='chat'))
"

# Should print a response with restaurant options
```

Open http://localhost:8000 — you should see the status page with ✓ for your services.

---

### Step 4: Deploy to Railway (10 min)

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

OR use Railway web dashboard:
1. Go to railway.app → New Project → Deploy from GitHub
2. Connect your repo
3. Add environment variables (copy from .env)
4. Deploy → get your public URL (e.g. https://swiggy-agent.railway.app)

Update `.env` and Railway env vars:
```env
BASE_URL=https://swiggy-agent.railway.app
```

Re-deploy after updating BASE_URL.

---

### Step 5: Configure Twilio webhooks (10 min)

**Voice:**
1. Twilio Console → Phone Numbers → Your number
2. Under "Voice & Fax" → "A call comes in"
3. Set to: `https://swiggy-agent.railway.app/voice/answer`
4. Method: HTTP POST
5. Status callback: `https://swiggy-agent.railway.app/voice/status`
6. Save

**WhatsApp:**
1. Twilio Console → Messaging → Settings → WhatsApp sandbox settings
2. "When a message comes in": `https://swiggy-agent.railway.app/whatsapp/webhook`
3. Method: HTTP POST
4. Save

---

### Step 6: Test the full flow (20 min)

**Phone test:**
1. Call your Twilio number from your phone
2. Say "Order chicken biryani"
3. Agent should respond with a restaurant option
4. Say "yes" or "haan"
5. Agent places order and hangs up ✓

**WhatsApp test:**
1. Send "join <your-sandbox-code>" to WhatsApp +1 415 523 8886
2. Send "Order milk and eggs"
3. Agent should respond with cart + confirmation ask
4. Reply "yes"
5. Order confirmed ✓

**Recipe test:**
- WhatsApp: "Items for alfredo pasta"
- Should get: pasta, heavy cream, parmesan, butter, garlic → Instamart cart ✓

---

## DAY 2 — Polish for demo (2–3 hours)

### Morning: Voice quality

1. Test ElevenLabs voice — go to elevenlabs.io, pick a voice that sounds natural for Indian English
2. Update `ELEVENLABS_VOICE_ID` in .env (try "Aria" or "Brian" for neutral accents)
3. Test call again — voice should sound natural, not robotic

### Afternoon: Demo script prep

Record a Loom video showing:
1. **30-second demo**: Phone call → "biryani" → confirmed in 1 turn
2. **60-second demo**: WhatsApp → "milk, eggs, bread, and items for alfredo pasta" → full ingredient cart
3. **The car demo**: Use iPhone Siri Shortcut (see below)

### Siri Shortcut setup (15 min — wow factor):

1. iPhone → Shortcuts app → New Shortcut
2. Add action: "Get Contents of URL"
   - URL: `https://swiggy-agent.railway.app/whatsapp/webhook`
   - Method: POST
   - Body (Form): `From=whatsapp:+91YOUR_NUMBER&Body=<Dictate Text>`
3. Add "Dictate Text" before the URL step
4. Name it "Order on Swiggy"
5. "Hey Siri, Order on Swiggy" → speak order → done

---

## Apply to Swiggy Builders Club (do this NOW, in parallel)

Go to: https://mcp.swiggy.com/builders/access/

Fill in:
- **Integration name**: "VoiceSwiggy — Voice & WhatsApp Ordering Agent"
- **Servers needed**: food, instamart
- **Use case**: "Phone and WhatsApp agent that lets users order food and groceries by speaking naturally, with recipe-to-ingredients expansion. Targeting India's 500M feature phone and WhatsApp users."
- **Demo video**: Record your Loom and attach it

When approved, update `.env`:
```env
DEMO_MODE=false
SWIGGY_CLIENT_ID=your_client_id
SWIGGY_CLIENT_SECRET=your_secret
```

The code will automatically use real Swiggy data. No other changes needed.

---

## Demo day checklist

- [ ] Server running on Railway (green health check)
- [ ] Phone number configured and tested
- [ ] WhatsApp sandbox working
- [ ] ElevenLabs voice sounds natural
- [ ] Demo script rehearsed (3 scenarios)
- [ ] Loom recording ready
- [ ] Builders Club application submitted
- [ ] Backup: laptop with ngrok running (in case Railway has issues)

---

## Quick troubleshooting

**"I didn't catch that" on every call**
→ Check Twilio logs (Console → Monitor → Logs → Voice). Usually a webhook URL issue.

**Agent responds but voice is robotic**
→ ElevenLabs key not set. Add it to .env and redeploy.

**WhatsApp not responding**
→ Confirm you joined the sandbox ("join <code>" to +14155238886 first).

**Order confirmation not triggering hangup**
→ Check agent logs at `/health` — likely an Anthropic API key issue.

---

## Cost estimate for demo

| Service | Cost |
|---------|------|
| Anthropic (Claude claude-sonnet-4-6) | ~$0.50 for 50 demo calls |
| Twilio phone number | $1/month |
| Twilio voice calls | ~$0.013/min (~$0.50 for demo) |
| Twilio WhatsApp | Free sandbox |
| ElevenLabs | Free tier (10K chars) |
| Railway | Free hobby tier |
| **Total for demo** | **~$2–3** |
