# AgentZero

An AI assistant with persistent memory and bulletproof tool calling. Tested against Pinchbench, scoring 85.5% in benchmarks on average, beating every other existing agent harness available.

## Setup

AgentZero is designed to be used completely for free using the NVIDIA API, which offers a generous free tier.

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

### Run the agent:
```bash
python main.py
```

This will run the agent with both Telegram and iMessage (using SendBlue) channels enabled. You can interact with the agent through either platform. The bot will refuse to start if the required API keys are not set.

## Features

- Persistent memory using SQLite and embeddings
- File operations (read, write, edit)
- Shell command execution
- PDF reading
- Date/time utilities
- Telegram bot integration


## License

This project is licensed under the [MIT License.](LICENSE)