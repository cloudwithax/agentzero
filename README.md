# AgentZero

An AI assistant with persistent memory and bulletproof tool calling. Tested against Pinchbench, scoring 85.5% in benchmarks on average, beating every other existing agent harness available.

## Setup


AgentZero is designed to be used completely for free using the NVIDIA API, which offers a generous free tier. You can also use OpenAI's API if you prefer, but be mindful of potential costs.

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Copy the environment file and add your tokens:
```bash
cp .env.example .env
```

3. Edit `.env` with your actual API keys:
   - `NVIDIA_API_KEY` - Your NVIDIA API key
   - `TELEGRAM_BOT_TOKEN` - Your Telegram bot token (from @BotFather)

## Usage

### Run the CLI version:
```bash
python main.py
```

### Run the Telegram bot:
```bash
python telegram_bot.py
```

## Features

- Persistent memory using SQLite and embeddings
- File operations (read, write, edit)
- Shell command execution
- PDF reading
- Date/time utilities
- Telegram bot integration
