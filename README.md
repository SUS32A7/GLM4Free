# GLM4Free

A Python-only client for Z.AI (GLM) that allows you to interact with the LLM for free, requests only.

## Features

- **Python Only**: No Node.js, Selenium, or heavy dependencies.
- **Stream Support**: Real-time streaming of AI responses.
- **Thinking Mode**: Toggle "Thinking" process (Chain of Thought).
- **Web Search**: Toggle capability to search the web.
- **Image Generation**: Toggle AI image generation features.
- **CLI & API**: Use it as a library or a command-line tool.

## Installation

```bash
pip install glm4free
```

## 🎯 Usage

## 📖 As a Library

```python
from GLM4Free.client import ZChat

# Initialize
bot = ZChat()
bot.initialize()

# Chat
print("AI: ", end="")
bot.chat("Hello! Who are you?")

# Enable Web Search or Image Gen
bot.use_web_search = True
bot.use_image_gen = True
bot.chat("Generate an image of a futuristic city.")
```

## 👥 CLI

You can run the chat interface directly from your terminal:

```bash
glm4free
```

## 🔧 Commands

Inside the CLI, you can use:
- `/search`: Toggle Web Search (Turned off by default)
- `/thinking`: Toggle Thinking Mode (Turned on by default)
- `/image`: Toggle Image Generation (Turned off by default) (Does not work, only returning a prompt)
- `/preview`: Toggle Preview Mode (Useless) (Turned off by default)
- `/new`: Start a new conversation
- `/history`: View conversation history
- `/exit`: Quit

## 🚨Disclaimer

This is an unofficial client for educational purposes. It is not affiliated with Z.AI. Use responsibly.

## 🤝 Contributing

We welcome contributions! We are still missing upload functionality, feel free to pull request !

## 📄 License

This project is licensed under the MIT License.

## 💖 Support

If you find this project helpful:

- ⭐ Star this repository
- 🐛 Report issues
- 💡 Suggest new features
- 🤝 Contribute code

Enhanced using AI.
