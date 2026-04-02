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
   - `MODEL_ID` - Your model ID. Image attachments are passed as native multimodal input only for these models:
     - `black-forest-labs/flux.1-kontext-dev`
     - `google/paligemma`
     - `meta/llama-3.2-11b-vision-instruct`
     - `meta/llama-3.2-90b-vision-instruct`
     - `meta/llama-4-maverick-17b-128e-instruct`
     - `meta/llama-4-scout-17b-16e-instruct`
     - `microsoft/phi-3.5-vision-instruct`
     - `moonshotai/kimi-k2-5`
     - `nvidia/llama-3.1-nemotron-nano-vl-8b-v1`
     - `qwen/qwen3.5-397b-a17b`

Optional Sendblue reliability settings:
- `SENDBLUE_WEBHOOK_PORT` - Run local webhook server instead of polling.
- `SENDBLUE_RECEIVE_WEBHOOK_URL` - If set, periodically verifies your Sendblue `receive` webhook and re-adds it if missing using append-only `POST /api/account/webhooks`.
- `SENDBLUE_TYPING_WEBHOOK_URL` - Optional dedicated URL for Sendblue `typing_indicator` webhook registration. If unset, the bot reuses `SENDBLUE_RECEIVE_WEBHOOK_URL`.
- `SENDBLUE_WEBHOOK_CHECK_INTERVAL` - Seconds between webhook checks (default: `60`, min enforced: `10`).
- `SENDBLUE_ATTACHMENT_DEBOUNCE_SECONDS` - Debounce window for attachment-first inbound webhook sequences before sending to the agent (default: `2.0`).
- `SENDBLUE_TYPING_DEBOUNCE_SECONDS` - Optional debounce extension window for typing webhook events when a sender already has pending queued content (default: attachment debounce value).
- `SENDBLUE_VOICE_MEMO_TRANSCRIPTION_ENABLED` - Enable/disable voice memo transcription for inbound iMessage audio attachments (default: `1`).
- `SENDBLUE_VOICE_MEMO_GRPC_SERVER` - NVIDIA Riva gRPC server for hosted Parakeet transcription (default: `grpc.nvcf.nvidia.com:443`).
- `SENDBLUE_VOICE_MEMO_FUNCTION_ID` - Function ID for `nvidia/parakeet-ctc-0_6b-asr` (default: `d8dd4e9b-fbf5-4fb0-9dba-8cf436c8d965`).
- `SENDBLUE_VOICE_MEMO_MODEL` - Optional Riva model name override. Usually not required when function ID is set.
- `SENDBLUE_VOICE_MEMO_LANGUAGE` - Language hint (`en-US` default for Parakeet, override if you route to a different ASR model).
- `SENDBLUE_VOICE_MEMO_FFMPEG_BIN` - Optional `ffmpeg` binary/path used to convert iMessage `.m4a`/`.caf` voice memos when direct transcription fails (default: `ffmpeg`).
- `SENDBLUE_VOICE_MEMO_MAX_BYTES` - Maximum voice memo download size in bytes for transcription (default: `26214400`, 25MB).

For iMessage native `.m4a`/`.caf` voice memo fallback conversion, install `ffmpeg` on the host machine.

Multimodal/NVIDIA asset handling:
- `NVCF_ASSET_UPLOAD_ENABLED` - Upload inbound multimodal image URLs to NVIDIA NVCF Assets before inference and pass `asset_id` references (default: `1`, set `0` to disable).
- This is enabled by default for supported multimodal models and helps with large/private channel attachment URLs that cannot be fetched reliably by the model endpoint.

## Usage

### Run the agent:
```bash
python main.py
```

### Run the agent as a daemon:
```bash
python main.py --daemon
```

In daemon mode, output is written to `agentzero.out.log` and `agentzero.err.log`.

### Stop the daemon:
```bash
python main.py --stop
```

Daemon mode uses `agentzero.pid` to track the running background process.
If the PID file is missing or stale, `--stop` will also try to find and stop orphaned `main.py --daemon` processes.

This will run the agent with both Telegram and iMessage (using SendBlue) channels enabled. You can interact with the agent through either platform. The bot will refuse to start if the required API keys are not set.

## Features

- Persistent memory using SQLite and embeddings
- File operations (read, write, edit)
- Shell command execution
- PDF reading
- Date/time utilities
- Telegram bot integration
- Telegram image uploads (single + media group)
- Sendblue image attachments (single + multiple)
- Sendblue voice memo transcription with NVIDIA Parakeet CTC 0.6B ASR


## License

This project is licensed under the [MIT License.](LICENSE)