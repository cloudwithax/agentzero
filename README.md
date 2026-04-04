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
- `SENDBLUE_STARTUP_REPLAY_ENABLED` - On startup, fetch and process recent inbound messages sent while the bot was offline (default: `1`).
- `SENDBLUE_STARTUP_LOOKBACK_SECONDS` - Startup lookback window for offline message replay (default: `21600`, i.e. 6 hours).
- `SENDBLUE_STARTUP_UNREAD_ONLY` - When Sendblue read-state fields are present, replay only unread inbound messages (default: `1`).
- `SENDBLUE_STARTUP_REPLAY_UNKNOWN_READ_STATE` - If read-state is missing, still replay recent inbound messages within lookback window (default: `1`).
- `SENDBLUE_DEDUP_TTL_SECONDS` - TTL for in-memory Sendblue message-handle dedupe across startup replay + webhook/polling handoff (default: `60`, min enforced: `10`).
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

Optional Telegram reliability settings:

- `TELEGRAM_REPLAY_PENDING_UPDATES_ON_STARTUP` - Process queued Telegram updates immediately after reconnect and before normal polling starts (default: `1`).
- `TELEGRAM_PENDING_UPDATES_BATCH_SIZE` - Batch size used when draining startup backlog (default: `100`, max: `100`).
- `TELEGRAM_PENDING_UPDATES_MAX_BATCHES` - Max startup batches to drain per boot (default: `5`).

Optional memory cadence and consolidation settings:

- `AUTO_MEMORY_ENABLED` - Enable automatic memory extraction from completed turns (default: `1`).
- `AUTO_MEMORY_MIN_MESSAGES_PER_MEMORY` - Lower bound for cadence gating (default: `10`).
- `AUTO_MEMORY_TARGET_MESSAGES_PER_MEMORY` - Preferred cadence interval (default: `15`).
- `AUTO_MEMORY_MAX_MESSAGES_PER_MEMORY` - Upper bound before forced capture (default: `20`).
- `AUTO_MEMORY_DEDUPE_THRESHOLD` - Similarity threshold for skipping near-duplicate memory candidates (default: `0.90`).
- `MEMORY_DREAM_ENABLED` - Enable background-style dream consolidation (default: `1`).
- `MEMORY_DREAM_LOOKBACK_DAYS` - Activity lookback used to infer off-peak hours (default: `21`).
- `MEMORY_DREAM_MIN_DAYS` - Minimum distinct active days before learned scheduling is considered reliable (default: `14`).
- `MEMORY_DREAM_OFFPEAK_WINDOW_HOURS` - Width of the inferred off-peak window used for consolidation (default: `6`).
- `MEMORY_DREAM_MIN_INTERVAL_HOURS` - Minimum spacing between dream runs (default: `24`).
- `MEMORY_DREAM_MIN_CANDIDATES` - Minimum eligible short-term memories required to run consolidation (default: `4`).
- `MEMORY_DREAM_CANDIDATE_LIMIT` - Max short-term candidate memories passed into a dream cycle (default: `24`).
- `MEMORY_DREAM_MIN_AGE_HOURS` - Minimum source-memory age for dream consolidation eligibility (default: `24`).

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
- Cross-channel slash commands (`/start`, `/setprompt`, `/clear`, `/memorystats`; plus `/memorycadence` alias on Telegram and `/memory_stats` + `/memorycadence` aliases on iMessage)
- File operations (read, write, edit)
- Shell command execution
- PDF reading
- Date/time utilities
- Telegram bot integration
- Telegram image uploads (single + media group)
- Sendblue image attachments (single + multiple)
- Sendblue voice memo transcription with NVIDIA Parakeet CTC 0.6B ASR
- Auto memory extraction with cadence control (default target: ~1 memory per 15 messages)
- Dream-style long-term memory consolidation during inferred off-peak hours

## License

This project is licensed under the [MIT License.](LICENSE)
