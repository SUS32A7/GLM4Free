# example.py

from GLM4Free.client import ZChat
import time

def main():
    bot = ZChat()
    bot.initialize()
    
    if not bot.user_id:
        print("[!] Failed to initialize.")
        return

    print(f"Client Ready. Model: {bot.model}")
    print(f"User: {bot.user_name} ({bot.user_id})")
    
    user_input = input("Enter your message: ")

    bot.chat(user_input)
    
    print("--- Changing Settings: Thinking=False, WebSearch=True ---")
    bot.use_thinking = False
    bot.use_web_search = True
    
    print(f"[2] Sending 'Python version?'... (Search: {bot.use_web_search})")
    bot.chat("What is the latest version of Python?")
    
    print("\n--- Done ---")

if __name__ == "__main__":
    main()
