"""
GLM4Free API server — FastAPI wrapper with OpenAI-compatible endpoints.

Features:
  - Session auto-recovery (re-initializes if Z.AI token expires)
  - System prompt support (prepended to first user message)
  - Web UI at /
  - OpenAI-compatible /v1/chat/completions + /v1/models
  - Simple /chat endpoint
"""

import uuid
import time
import json
import asyncio
import threading

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime

from glm4free.client import ZChat, BASE_URL, AVAILABLE_MODELS, DEFAULT_MODEL, generate_za_signature


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="GLM4Free API",
    description=(
        "Unofficial free API for Z.AI's GLM-4/5 model.\n\n"
        "Supports a simple `/chat` endpoint and a fully **OpenAI-compatible** "
        "`/v1/chat/completions` endpoint — drop-in replacement for the OpenAI SDK."
    ),
    version="1.0.0",
)

_bot: Optional[ZChat] = None
_lock = threading.Lock()


# ── Session management (auto-recovery) ────────────────────────────────────────

def _init_bot() -> bool:
    global _bot
    _bot = ZChat()
    return _bot.initialize()


@app.on_event("startup")
async def startup():
    ok = await asyncio.get_event_loop().run_in_executor(None, _init_bot)
    if ok:
        print(f"[+] Z.AI session ready  |  user: {_bot.user_name}  |  model: {_bot.model}")
    else:
        print("[!] Warning: session init failed — will retry on first request.")


def get_bot() -> ZChat:
    """Returns active bot, auto-recovering if token is missing."""
    global _bot
    if not _bot or not _bot.token:
        with _lock:
            if not _bot or not _bot.token:
                print("[*] No active session — re-initializing...")
                ok = _init_bot()
                if not ok:
                    raise HTTPException(status_code=503, detail="Z.AI session unavailable.")
    return _bot


def recover_session():
    """Force a fresh session. Called when Z.AI returns 401."""
    global _bot
    with _lock:
        print("[*] Token expired — recovering session...")
        ok = _init_bot()
        if not ok:
            raise HTTPException(status_code=503, detail="Z.AI session recovery failed.")
        print(f"[+] Session recovered  |  user: {_bot.user_name}")


# ── System prompt helper ───────────────────────────────────────────────────────

def apply_system_prompt(messages: list) -> list:
    """
    GLM may silently ignore role=system messages.
    This extracts all system messages and prepends them to the first user
    message so the instructions are always seen by the model.

    [system: "You are a pirate", user: "Hello"]
    → [user: "[System instructions]\\nYou are a pirate\\n\\nHello"]
    """
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    non_system   = [m for m in messages if m["role"] != "system"]

    if not system_parts:
        return non_system

    system_block = "\n\n".join(system_parts)
    patched      = []
    injected     = False

    for m in non_system:
        if m["role"] == "user" and not injected:
            patched.append({
                "role": "user",
                "content": f"[System instructions]\n{system_block}\n\n{m['content']}"
            })
            injected = True
        else:
            patched.append(m)

    if not injected:
        patched.insert(0, {"role": "user", "content": f"[System instructions]\n{system_block}"})

    return patched


# ── Core streaming helper ──────────────────────────────────────────────────────

def stream_chunks(bot: ZChat, messages: list, web_search: bool, thinking: bool):
    """
    Yields text chunks from Z.AI SSE stream.
    On 401, auto-recovers the session and retries once.
    """
    def _do_stream(b: ZChat):
        prompt = messages[-1]["content"] if messages else ""
        sig, ts, suffix = generate_za_signature(prompt, b.token, b.user_id, b.salt_key)
        url  = f"{BASE_URL}/api/v2/chat/completions?{suffix}"
        now  = datetime.now()
        hdrs = {
            "Origin":        BASE_URL,
            "Referer":       f"{BASE_URL}/",
            "Authorization": f"Bearer {b.token}",
            "X-Signature":   sig,
            "X-FE-Version":  b.fe_version,
            "Content-Type":  "application/json",
        }
        payload = {
            "model":            b.model,
            "chat_id":          str(uuid.uuid4()),
            "messages":         messages,
            "signature_prompt": prompt,
            "stream":           True,
            "params":           {},
            "extra":            {},
            "features": {
                "image_generation": False,
                "web_search":       web_search,
                "auto_web_search":  web_search,
                "preview_mode":     False,
                "flags":            [],
                "enable_thinking":  thinking,
            },
            "variables": {
                "{{USER_NAME}}":        b.user_name,
                "{{USER_LOCATION}}":    "Unknown",
                "{{CURRENT_DATETIME}}": now.strftime("%Y-%m-%d %H:%M:%S"),
                "{{CURRENT_DATE}}":     now.strftime("%Y-%m-%d"),
                "{{CURRENT_TIME}}":     now.strftime("%H:%M:%S"),
                "{{CURRENT_WEEKDAY}}":  now.strftime("%A"),
                "{{CURRENT_TIMEZONE}}": "Europe/Paris",
                "{{USER_LANGUAGE}}":    "en-US",
            },
            "background_tasks": {"title_generation": True, "tags_generation": True},
        }

        with b.session.post(url, headers=hdrs, json=payload, stream=True, timeout=60) as r:
            if r.status_code == 401:
                return "AUTH_EXPIRED"
            if r.status_code != 200:
                raise HTTPException(status_code=r.status_code, detail=f"Z.AI error: {r.text}")
            for line in r.iter_lines():
                if line:
                    decoded = line.decode("utf-8")
                    if decoded.startswith("data: "):
                        data_str = decoded[6:]
                        if data_str == "[DONE]":
                            return "DONE"
                        try:
                            d = json.loads(data_str)
                            if "data" in d and "delta_content" in d["data"]:
                                yield d["data"]["delta_content"]
                            elif "choices" in d:
                                yield d["choices"][0]["delta"].get("content", "")
                        except Exception:
                            pass
        return "DONE"

    result = yield from _do_stream(bot)

    if result == "AUTH_EXPIRED":
        recover_session()
        yield from _do_stream(_bot)


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class SimpleChatRequest(BaseModel):
    message:    str           = Field(...,  description="Your message")
    system:     Optional[str] = Field(None, description="Optional system prompt")
    web_search: bool          = Field(False, description="Enable web search")
    thinking:   bool          = Field(True,  description="Enable chain-of-thought thinking")
    stream:     bool          = Field(False, description="Stream tokens")


class Message(BaseModel):
    role:    Literal["system", "user", "assistant"]
    content: str


class OpenAIChatRequest(BaseModel):
    model:      str          = Field("glm-5", description="Model: glm-5, glm-4.7, glm-4.5")
    messages:   List[Message]
    stream:     bool         = Field(False)
    web_search: bool         = Field(False, description="GLM4Free: enable web search")
    thinking:   bool         = Field(True,  description="GLM4Free: enable thinking mode")


# ── Web UI ─────────────────────────────────────────────────────────────────────

WEB_UI = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GLM4Free Chat</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#0f0f0f;color:#e8e8e8;height:100dvh;display:flex;flex-direction:column}
  header{padding:14px 20px;background:#1a1a1a;border-bottom:1px solid #2a2a2a;display:flex;align-items:center;gap:12px;flex-shrink:0}
  header h1{font-size:1.1rem;font-weight:600;color:#fff}
  #statusBadge{font-size:.75rem;color:#666;margin-left:auto}
  .model-select{background:#2a2a2a;border:1px solid #3a3a3a;color:#e8e8e8;padding:4px 8px;border-radius:6px;font-size:.8rem;cursor:pointer}
  #system-bar{padding:8px 20px;background:#141414;border-bottom:1px solid #2a2a2a;display:flex;gap:8px;align-items:center;flex-shrink:0}
  #system-bar input{flex:1;background:#1e1e1e;border:1px solid #2a2a2a;color:#aaa;padding:6px 10px;border-radius:8px;font-size:.8rem;outline:none}
  #system-bar input:focus{border-color:#2563eb}
  #system-bar label{font-size:.75rem;color:#555;white-space:nowrap}
  #messages{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:16px}
  .msg{display:flex;gap:10px;max-width:820px;width:100%}
  .msg.user{margin-left:auto;flex-direction:row-reverse}
  .bubble{padding:10px 14px;border-radius:14px;line-height:1.55;font-size:.92rem;white-space:pre-wrap;word-break:break-word;max-width:75vw}
  .user .bubble{background:#2563eb;color:#fff;border-bottom-right-radius:4px}
  .bot .bubble{background:#1e1e1e;border:1px solid #2a2a2a;border-bottom-left-radius:4px}
  .avatar{width:30px;height:30px;border-radius:50%;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:.8rem;font-weight:700;margin-top:2px}
  .user .avatar{background:#2563eb;color:#fff}
  .bot .avatar{background:#2a2a2a;color:#aaa}
  .toolbar{padding:8px 20px;background:#141414;border-top:1px solid #2a2a2a;display:flex;gap:10px;align-items:center;flex-shrink:0}
  .toggle{background:#1e1e1e;border:1px solid #2a2a2a;color:#888;padding:5px 12px;border-radius:20px;font-size:.75rem;cursor:pointer;transition:all .15s}
  .toggle.on{background:#1d3a6e;border-color:#2563eb;color:#93c5fd}
  #input-row{padding:12px 20px;background:#1a1a1a;border-top:1px solid #2a2a2a;display:flex;gap:10px;flex-shrink:0}
  #input{flex:1;background:#242424;border:1px solid #333;color:#e8e8e8;padding:10px 14px;border-radius:10px;font-size:.92rem;outline:none;resize:none;max-height:140px;line-height:1.4}
  #input:focus{border-color:#2563eb}
  #send{background:#2563eb;color:#fff;border:none;padding:10px 18px;border-radius:10px;font-size:.9rem;cursor:pointer;font-weight:600;transition:background .15s;flex-shrink:0;align-self:flex-end}
  #send:hover{background:#1d4ed8}
  #send:disabled{background:#1e3a6e;cursor:not-allowed}
  ::-webkit-scrollbar{width:4px}
  ::-webkit-scrollbar-track{background:transparent}
  ::-webkit-scrollbar-thumb{background:#333;border-radius:4px}
</style>
</head>
<body>
<header>
  <h1>⚡ GLM4Free</h1>
  <select class="model-select" id="modelSelect">
    <option value="glm-5">glm-5</option>
    <option value="glm-4.7">glm-4.7</option>
    <option value="glm-4.5">glm-4.5</option>
  </select>
  <span id="statusBadge">connecting…</span>
</header>

<div id="system-bar">
  <label>System:</label>
  <input id="systemPrompt" type="text" placeholder="Optional system prompt (e.g. You are a helpful assistant that speaks like a pirate)" />
</div>

<div id="messages"></div>

<div class="toolbar">
  <button class="toggle"    id="toggleSearch" onclick="toggle(this,'search')">🔍 Web Search</button>
  <button class="toggle on" id="toggleThink"  onclick="toggle(this,'thinking')">🧠 Thinking</button>
</div>

<div id="input-row">
  <textarea id="input" rows="1" placeholder="Type a message… (Enter to send, Shift+Enter for newline)"></textarea>
  <button id="send" onclick="sendMessage()">Send</button>
</div>

<script>
const state = { search: false, thinking: true };

function toggle(btn, key) {
  state[key] = !state[key];
  btn.classList.toggle('on', state[key]);
}

const input = document.getElementById('input');
input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 140) + 'px';
});
input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

fetch('/health')
  .then(r => r.json())
  .then(d => { document.getElementById('statusBadge').textContent = '● ' + d.user + ' · ' + d.model; })
  .catch(() => { document.getElementById('statusBadge').textContent = '● offline'; });

function addMessage(role, content) {
  const wrap = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'user' ? 'U' : 'AI';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = content;
  div.appendChild(avatar);
  div.appendChild(bubble);
  wrap.appendChild(div);
  wrap.scrollTop = wrap.scrollHeight;
  return bubble;
}

async function sendMessage() {
  const text = input.value.trim();
  if (!text) return;

  const sendBtn = document.getElementById('send');
  sendBtn.disabled = true;
  input.value = '';
  input.style.height = 'auto';

  addMessage('user', text);
  const botBubble = addMessage('bot', '');
  botBubble.textContent = '…';

  const model      = document.getElementById('modelSelect').value;
  const systemText = document.getElementById('systemPrompt').value.trim();
  const messages   = [];
  if (systemText) messages.push({ role: 'system', content: systemText });
  messages.push({ role: 'user', content: text });

  try {
    const res = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, messages, stream: true, web_search: state.search, thinking: state.thinking }),
    });

    botBubble.textContent = '';
    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') break;
          try {
            const chunk = JSON.parse(data);
            botBubble.textContent += chunk.choices?.[0]?.delta?.content || '';
            document.getElementById('messages').scrollTop = 999999;
          } catch {}
        }
      }
    }
  } catch (err) {
    botBubble.textContent = 'Error: ' + err.message;
  }

  sendBtn.disabled = false;
  input.focus();
}
</script>
</body>
</html>"""


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["Web UI"])
def web_ui():
    """Chat UI — open this in your phone browser."""
    return WEB_UI


@app.get("/health", tags=["Meta"])
def health():
    bot = get_bot()
    return {"status": "ok", "user": bot.user_name, "model": bot.model}


@app.get("/v1/models", tags=["OpenAI Compatible"])
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "created": 1700000000, "owned_by": "z-ai"}
            for m in AVAILABLE_MODELS
        ],
    }


@app.post("/chat", tags=["Simple Chat"])
def simple_chat(req: SimpleChatRequest):
    """Single-turn chat with optional system prompt."""
    bot = get_bot()
    messages = []
    if req.system:
        messages.append({"role": "system", "content": req.system})
    messages.append({"role": "user", "content": req.message})
    messages = apply_system_prompt(messages)

    if req.stream:
        return StreamingResponse(
            stream_chunks(bot, messages, req.web_search, req.thinking),
            media_type="text/plain",
        )

    full = "".join(stream_chunks(bot, messages, req.web_search, req.thinking))
    return {"reply": full, "model": bot.model}


@app.post("/v1/chat/completions", tags=["OpenAI Compatible"])
def openai_chat(req: OpenAIChatRequest):
    """OpenAI-compatible endpoint. System prompts are fully supported."""
    bot = get_bot()
    model_to_use   = req.model if req.model in AVAILABLE_MODELS else DEFAULT_MODEL
    original_model = bot.model
    bot.model      = model_to_use

    raw_messages = [{"role": m.role, "content": m.content} for m in req.messages]
    messages     = apply_system_prompt(raw_messages)

    created = int(time.time())
    cid     = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    if req.stream:
        def generate():
            for chunk in stream_chunks(bot, messages, req.web_search, req.thinking):
                yield f"data: {json.dumps({'id':cid,'object':'chat.completion.chunk','created':created,'model':model_to_use,'choices':[{'index':0,'delta':{'content':chunk},'finish_reason':None}]})}\n\n"
            yield f"data: {json.dumps({'id':cid,'object':'chat.completion.chunk','created':created,'model':model_to_use,'choices':[{'index':0,'delta':{},'finish_reason':'stop'}]})}\n\ndata: [DONE]\n\n"
            bot.model = original_model
        return StreamingResponse(generate(), media_type="text/event-stream")

    full = "".join(stream_chunks(bot, messages, req.web_search, req.thinking))
    bot.model = original_model
    return {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": model_to_use,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": full}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens":     sum(len(m["content"].split()) for m in messages),
            "completion_tokens": len(full.split()),
            "total_tokens":      sum(len(m["content"].split()) for m in messages) + len(full.split()),
        },
    }
