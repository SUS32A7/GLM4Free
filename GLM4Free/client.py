import requests
import json
import time
import uuid
import base64
import urllib.parse
import sys
import hmac
import hashlib
import re
from datetime import datetime

BASE_URL = "https://chat.z.ai"

AVAILABLE_MODELS = [
    "glm-5",       # Latest flagship — 744B MoE, best reasoning & coding
    "glm-4.7",     # Previous flagship — solid general purpose
    "glm-4.5",     # Lighter — 355B MoE, faster responses
]
DEFAULT_MODEL = "glm-5"

def scrape_config():

    print("[*] Scraping configuration from Z.AI...")
    salt_key = "key-@@@@)))()((9))-xxxx&&&%%%%%"
    fe_version = "prod-fe-1.0.185"
    
    try:
        r = requests.get(BASE_URL, timeout=10)
        if r.status_code == 200:
            version_match = re.search(r'prod-fe-\d+\.\d+\.\d+', r.text)
            if version_match:
                fe_version = version_match.group(0)
        
        return salt_key, fe_version
        
    except Exception as e:
        print(f"[!] Scraping error: {e}")
        return salt_key, fe_version

def generate_za_signature(prompt, token, user_id, salt_key):
    """
    Generates X-Signature and URL parameters for Z.AI API.
    """
    timestamp = str(int(time.time() * 1000))
    request_id = str(uuid.uuid4())
    
    bucket = int(int(timestamp) / 300000)
    w_key = hmac.new(salt_key.encode(), str(bucket).encode(), hashlib.sha256).hexdigest()
    
    payload_dict = {
        "timestamp": timestamp,
        "requestId": request_id, 
        "user_id": user_id
    }
    sorted_items = sorted(payload_dict.items(), key=lambda x: x[0])
    sorted_payload = ",".join([f"{k},{v}" for k, v in sorted_items])
    
    prompt_b64 = base64.b64encode(prompt.strip().encode()).decode()
    
    data_to_sign = f"{sorted_payload}|{prompt_b64}|{timestamp}"
    signature = hmac.new(w_key.encode(), data_to_sign.encode(), hashlib.sha256).hexdigest()
    
    browser_info = {
        "version": "0.0.1", "platform": "web", "token": token,
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
        "language": "en-US", "screen_resolution": "1920x1080",
        "viewport_size": "1920x1080", "timezone": "Europe/Paris", "timezone_offset": "-60"
    }
    params = {**payload_dict, **browser_info}
    url_params = urllib.parse.urlencode(params)
    
    return signature, timestamp, f"{url_params}&signature_timestamp={timestamp}"

class ZChat:
    def __init__(self):
        self.session = requests.Session()
        self.token = ""
        self.user_id = ""
        self.chat_id = str(uuid.uuid4())
        self.messages = []
        self.model = DEFAULT_MODEL
        self.use_web_search = False
        self.use_thinking = True
        self.use_image_gen = False
        self.use_preview_mode = False
        self.user_name = "Guest"
        self.salt_key = None
        self.fe_version = None
        self.headers = {"Origin": BASE_URL, "Referer": f"{BASE_URL}/"}

    def initialize(self):
        self.salt_key, self.fe_version = scrape_config()
        
        print("[*] Initializing Z.AI Session...")
        try:
            r = self.session.post(f"{BASE_URL}/api/v1/auths/guest", headers=self.headers, json={}, timeout=15)

            r = self.session.get(f"{BASE_URL}/api/v1/auths/", headers=self.headers, timeout=15)
            
            if r.status_code == 200:
                data = r.json()
                self.token = data.get("token", "")

                if not self.token:
                    r2 = self.session.post(f"{BASE_URL}/api/v1/auths/guest", headers=self.headers, json={})
                    if r2.status_code == 200:
                        self.token = r2.json().get("token")
                
                if self.token:
                    try:
                        padded = self.token.split(".")[1] + "=="
                        payload = json.loads(base64.b64decode(padded).decode())
                        self.user_id = payload.get("id", "")
                        self.user_name = payload.get("email", "Guest").split("@")[0]
                        print(f"[+] Connected. UserID: {self.user_id[:8]}... (Name: {self.user_name})")
                    except:
                         print("[!] Token decode failed, but connected.")
                else:
                    print("[!] No token in auth response.")
            else:
                print(f"[!] Auth Failed: {r.status_code}")
        except Exception as e:
            print(f"[!] Initialization Error: {e}")

    def _get_context_vars(self):
        now = datetime.now()
        return {
            "{{USER_NAME}}": self.user_name,
            "{{USER_LOCATION}}": "Unknown",
            "{{CURRENT_DATETIME}}": now.strftime("%Y-%m-%d %H:%M:%S"),
            "{{CURRENT_DATE}}": now.strftime("%Y-%m-%d"),
            "{{CURRENT_TIME}}": now.strftime("%H:%M:%S"),
            "{{CURRENT_WEEKDAY}}": now.strftime("%A"),
            "{{CURRENT_TIMEZONE}}": "Europe/Paris",
            "{{USER_LANGUAGE}}": "en-US"
        }

    def chat(self, prompt):
        self.messages.append({"role": "user", "content": prompt})

        sig, ts, suffix = generate_za_signature(prompt, self.token, self.user_id, self.salt_key)
        
        url = f"{BASE_URL}/api/v2/chat/completions?{suffix}"
        
        headers = {
            **self.headers,
            "Authorization": f"Bearer {self.token}",
            "X-Signature": sig,
            "X-FE-Version": self.fe_version,
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "chat_id": self.chat_id,
            "messages": self.messages,
            "signature_prompt": prompt,
            "stream": True,
            "params": {},
            "extra": {},
            "features": {
                "image_generation": self.use_image_gen,
                "web_search": self.use_web_search,
                "auto_web_search": self.use_web_search,
                "preview_mode": self.use_preview_mode,
                "flags": [],
                "enable_thinking": self.use_thinking
            },
            "variables": self._get_context_vars(),
            "background_tasks": {
                "title_generation": True,
                "tags_generation": True
            }
        }

        full_assistant_content = ""
        
        print("\nAI: ", end="", flush=True)
        try:
            with self.session.post(url, headers=headers, json=payload, stream=True, timeout=60) as r:
                if r.status_code != 200:
                    print(f"\n[!] Error {r.status_code}: {r.text}")
                    self.messages.pop()
                    return

                for line in r.iter_lines():
                    if line:
                        decoded = line.decode('utf-8')
                        if decoded.startswith("data: "):
                            data_str = decoded[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                data_json = json.loads(data_str)
                                
                                if "data" in data_json and "delta_content" in data_json["data"]:
                                    chunk = data_json["data"]["delta_content"]
                                    full_assistant_content += chunk
                                    print(chunk, end="", flush=True)
                                    
                                elif "choices" in data_json:
                                     chunk = data_json["choices"][0]["delta"].get("content", "")
                                     full_assistant_content += chunk
                                     print(chunk, end="", flush=True)
                                     
                            except: pass
            
            if full_assistant_content:
                self.messages.append({"role": "assistant", "content": full_assistant_content})
            print("\n")
            
        except Exception as e:
            print(f"\n[!] Stream Connection Error: {e}")
            if self.messages: self.messages.pop()

def main():
    bot = ZChat()
    bot.initialize()
    
    print("\n--- Z.AI Auto-Config Chat Console ---")
    print(f"Current Settings: Model={bot.model}, WebSearch={bot.use_web_search}, Thinking={bot.use_thinking}")
    print("Commands: /search, /thinking, /image, /preview, /model, /new, /history, /exit\n")
    
    while True:
        try:
            user_input = input(f"[{len(bot.messages)}] {bot.user_name} > ").strip()
            if not user_input: continue
            
            if user_input.startswith("/"):
                cmd = user_input.split(" ")
                if cmd[0] == "/exit": break
                elif cmd[0] == "/new":
                    bot.messages = []
                    bot.chat_id = str(uuid.uuid4())
                    print("[*] History cleared. New conversation started.")
                elif cmd[0] == "/history":
                    print(f"\n--- Conversation ({len(bot.messages)} messages) ---")
                    for idx, m in enumerate(bot.messages):
                        content_peek = m['content'].replace("\n", " ")[:60]
                        print(f"[{idx}] {m['role'].upper()}: {content_peek}...")
                    print("-----------------------------\n")
                elif cmd[0] == "/search":
                    bot.use_web_search = not bot.use_web_search
                    print(f"[*] Web Search: {'ON' if bot.use_web_search else 'OFF'}")
                elif cmd[0] == "/thinking":
                    bot.use_thinking = not bot.use_thinking
                    print(f"[*] Thinking Mode: {'ON' if bot.use_thinking else 'OFF'}")
                elif cmd[0] == "/image":
                    bot.use_image_gen = not bot.use_image_gen
                    print(f"[*] Image Gen: {'ON' if bot.use_image_gen else 'OFF'}, Does not work, only returning a prompt")
                elif cmd[0] == "/preview":
                    bot.use_preview_mode = not bot.use_preview_mode
                    print(f"[*] Preview Mode: {'ON' if bot.use_preview_mode else 'OFF'}")
                elif cmd[0] == "/model":
                    if len(cmd) < 2:
                        print(f"[*] Current model: {bot.model}")
                        print(f"[*] Available: {', '.join(AVAILABLE_MODELS)}")
                        print(f"[*] Usage: /model glm-5")
                    elif cmd[1] in AVAILABLE_MODELS:
                        bot.model = cmd[1]
                        bot.messages = []
                        bot.chat_id = str(uuid.uuid4())
                        print(f"[*] Model switched to: {bot.model} (history cleared)")
                    else:
                        print(f"[!] Unknown model '{cmd[1]}'. Available: {', '.join(AVAILABLE_MODELS)}")
                else:
                    print("[!] Unknown command.")
                continue
                
            bot.chat(user_input)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n[!] Critical Error: {e}")

if __name__ == "__main__":
    main()
