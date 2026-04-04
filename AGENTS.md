# AgentZero Codebase Guide for AI Agents

This document helps AI agents work effectively in the AgentZero codebase, a modular async Python agent framework that interacts with NVIDIA-hosted chat models.

Maintenance reminder: Always append new lessons to the "Session Pitfalls + Fixes" section (or a new date-stamped section), and do not replace prior entries.

## Project Overview

This is a modular **async Python agent framework** that interacts with NVIDIA-hosted chat models.
It combines tool calling, persistent memory, adaptive planning/few-shot behavior, and multi-channel integrations (Telegram + Sendblue iMessage), including multimodal image handling and Sendblue voice memo transcription.

## Project Structure

```
../agentzero/
├── main.py             # Bootstrap, daemon mode, integration startup
├── handler.py          # Request orchestration, prompt building, memory context
├── api.py              # API retry logic, tool-call loop, tool-leak guards
├── tools.py            # Tool implementations, registry, argument validation
├── integrations.py     # Telegram + Sendblue integrations, attachments, webhooks
├── memory.py           # SQLite-backed memory store + embeddings flow
├── planning.py         # TaskAnalyzer/TaskPlanner
├── capabilities.py     # Capability profile + adaptive formatter
├── examples.py         # Few-shot example bank/manager
├── validation.py       # Output parsing/validation helpers
├── run_all_tests.py    # Convenience test runner (subset)
└── tests/              # Standalone async test scripts
```

## Essential Commands

### Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

### Running the Agent

```bash
# Foreground
python3 main.py

# Daemon mode
python3 main.py --daemon

# Stop daemon
python3 main.py --stop
```

### Running Tests

```bash
# Convenience subset runner
PYTHONPATH=. python3 run_all_tests.py

# Run a specific test script
PYTHONPATH=. python3 tests/test_sendblue_debounce.py
```

### Dependencies

`requirements.txt` is present. Core dependencies:

- `aiohttp` - HTTP client for API calls
- `python-dotenv` - .env loading
- `python-telegram-bot` - Telegram integration
- `nvidia-riva-client` - Hosted ASR for Sendblue voice memo transcription
- `Pillow` + `pillow-heif` - Image decoding/conversion
- `strip-markdown` - Final plain-text normalization
- `numpy` - Supporting numeric utilities

External binaries used by integration/tool paths:

- `pdftotext` - required by `read_pdf` tool
- `ffmpeg` - used for iMessage voice memo conversion and image conversion fallback
- `ImageMagick` (`magick`/`convert`) - preferred converter for some non-native image formats

## Code Patterns & Conventions

### Async/Await Pattern

All tool functions and API calls are async:

```python
async def tool_function(param):
    result = await some_async_operation()
    return {"success": True, "data": result}
```

### Tool Registry Pattern

Tools are registered in a global `TOOLS` dictionary with aliases, and args are validated via `validate_tool_args()`:

```python
TOOLS = {
    "read": read_file_tool,
    "read_file": read_file_tool,  # Alias
    "readFile": read_file_tool,   # Another alias
    # ...
}
```

### Tool Result Format

All tools return a consistent dictionary format:

```python
{"success": True, "content": "...", "...": "..."}  # Success
{"success": False, "error": "error message"}          # Failure
```

### API Payload Pattern

- `BASE_PAYLOAD` is the template for API requests
- Always copy it: `current_payload = BASE_PAYLOAD.copy()`
- Never mutate the global `BASE_PAYLOAD` directly

### Multimodal Content Pattern

- Message content can be plain text or a multimodal block list (`[{"type": "text"}, {"type": "image_url"}]`).
- `integrations.py` builds user content and normalizes attachment handling per model capability.

### Tool Call Execution Flow

1. `api.process_response()` reads the assistant message and checks for tool calls.
2. If tool calls are missing but leaked in content, strict inference guards attempt recovery.
3. `execute_tool_calls()` JSON-parses args, validates via `validate_tool_args()`, and executes from `TOOLS`.
4. Tool results are appended as `role: "tool"` messages.
5. A follow-up API call is made with a fresh payload copy.
6. Loop continues until no more tool calls remain.
7. Final assistant text is markdown-stripped before returning.

## Testing Approach

### Test Style

- Tests use `asyncio.run()` pattern
- Mock API responses using `unittest.mock.AsyncMock`
- Tests can be run standalone: `PYTHONPATH=. python3 tests/test_file.py`
- No pytest configuration - tests are self-contained scripts

### Test Categories

1. **Core API/tool flow**: `test_simple.py`, `test_tools.py`, `test_process_response.py`, `test_tool_calling_flow.py`
2. **Memory/learning/planning**: `test_memory.py`, `test_learning.py`, `test_learning_deterministic.py`, `test_learning_improvement.py`, `test_consortium_mode.py`
3. **Integrations**: `test_setprompt_command.py`, `test_multimodal_integrations.py`, `test_sendblue_debounce.py`, `test_sendblue_voice_memo.py`

## Important Gotchas

### API Key Handling

- `.env` is loaded in `main.py` via `python-dotenv`; prefer env-provided credentials.
- `handler.py` includes a fallback NVIDIA API key string; treat it as a development fallback and do not rely on it operationally.

### Payload Mutation Risk

- **Critical**: Always use `BASE_PAYLOAD.copy()` before modifying
- This applies in both `handler.py` and `api.py` follow-up calls.

### Tool Call Loop

- `process_response()` handles multiple rounds of tool calls via `while` loop
- Each iteration: execute tools → append results → API follow-up → check for more tool calls
- Loop exits when response has no `tool_calls`
- Tool-leak protection retries once with a guard message when internal tool-call content appears in user-visible output.

### PDF Tool Dependency

- `read_pdf` tool requires `pdftotext` binary (poppler-utils package)
- Uses subprocess to call: `pdftotext -layout <filepath> -`

### Sendblue Formatting + Test Invocation

- Outbound Sendblue text is normalized in `_format_sendblue_message_content()` before send.
- `SENDBLUE_FORCE_CARRIAGE_RETURNS=1` (default) converts `\n` to `\r` for iMessage formatting reliability.
- Running tests from `tests/` directly can fail imports; use `PYTHONPATH=.`.

### Error Handling

- API errors checked via `"error" in response_data`
- Tool errors return `{"success": False, "error": "..."}` format
- Network errors have retry logic with exponential backoff

### Rate Limiting

- `api_call_with_retry()` handles rate limits automatically
- Retries up to 3 times with exponential backoff
- Checks for `"rate limit"` in error messages

### Session Pitfalls + Fixes (2026-04-03)

- **Guideline for future sessions:** Keep appending newly discovered pitfalls and their fixes to this section (or a new date-stamped section), instead of replacing old entries. Include a concrete remediation and, when applicable, the exact validation command/test used.

- **Pitfall: iMessage formatting was inconsistent (sometimes single dense paragraph, sometimes line-broken).**
  **Fix:** Normalize outbound Sendblue text right before send in `integrations.py` via `_format_sendblue_message_content()` and route all outbound content through it from `send_imessage()`.
- **Pitfall: Newline variants arrived mixed (`\\n`, `\\r\\n`, and real newlines), causing unpredictable rendering.**
  **Fix:** Canonicalize all line endings to `\n`, collapse excessive blank lines, then convert to carriage returns for iMessage delivery.
- **Pitfall: Receipt-style key/value outputs were hard to read when model returned one long paragraph.**
  **Fix:** Add deterministic split rules for common labels (`name:`, `order #:`, `date:`, `items:`, `drinks:`, `sauces:`, `restaurant #:`), with sentence splitting fallback.
- **Pitfall: Behavior needed an operational toggle for rollback/troubleshooting.**
  **Fix:** Add env switch `SENDBLUE_FORCE_CARRIAGE_RETURNS` (default `1`). Set to `0` to keep LF newlines.
- **Pitfall: Running test scripts directly from `tests/` caused `ModuleNotFoundError: No module named 'integrations'`.**
  **Fix:** Run tests with project root on path, for example:
  `PYTHONPATH=. ../agentzero/.venv/bin/python tests/test_sendblue_debounce.py`
- **Pitfall: Formatting changes can regress silently if only manual QA is used.**
  **Fix:** Add explicit regression tests that assert Sendblue payload content uses carriage returns and that dense receipt text is split predictably.

## Key Functions Reference

| Function                   | Purpose                                    | Location               |
| -------------------------- | ------------------------------------------ | ---------------------- |
| `initialize_agent()`       | Build memory/planning/handler stack        | `main.py:243`          |
| `AgentHandler.handle()`    | Main request processing pipeline           | `handler.py:981`       |
| `api_call_with_retry()`    | API call with retry + asset header wiring  | `api.py:237`           |
| `execute_tool_calls()`     | Validate + execute model tool calls        | `api.py:298`           |
| `process_response()`       | Multi-round tool-call loop + leak handling | `api.py:342`           |
| `start_sendblue_bot()`     | Start Sendblue webhook/polling runtime     | `integrations.py:2266` |
| `run_telegram_bot_async()` | Start Telegram runtime                     | `integrations.py:2487` |

## Telegram Bot Commands

The Telegram bot supports the following slash commands:

| Command      | Description                                                                                                                                                                                |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `/start`     | Initialize interaction with the bot                                                                                                                                                        |
| `/setprompt` | Change the system prompt. After sending this command, the bot will ask you to provide the new prompt in your next message. The prompt is stored persistently and takes effect immediately. |
| `/clear`     | Clear conversation history for the current session/chat context.                                                                                                                           |

## Adding New Tools

1. Implement an async tool function in `tools.py` returning `{"success": ...}` format.
2. Register it in `TOOLS` (and aliases if needed) in `tools.py`.
3. Add/adjust required arg validation in `validate_tool_args()` in `tools.py`.
4. Add the tool schema entry to `BASE_PAYLOAD["tools"]` in `handler.py`.
5. Add coverage in `tests/test_simple.py` and extend integration tests when behavior is channel- or flow-specific.

Example tool signature:

```python
async def my_tool(param: str):
    try:
        result = await do_something(param)
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
```
