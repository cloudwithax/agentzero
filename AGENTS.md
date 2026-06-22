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
├── self_heal.py         # Self-healing subsystem (uses agent's own API)
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
- `strip-markdown` - Final plain-text normalization
- `numpy` - Supporting numeric utilities

External binaries used by integration/tool paths:

- `pdftotext` - required by `read_pdf` tool
- `ffmpeg` - used for iMessage voice memo conversion fallback
- `ImageMagick` (`magick`/`convert`) - required for inbound image-to-JPEG/base64 conversion

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
- `SENDBLUE_FORCE_CARRIAGE_RETURNS=1` (default) converts `\n` to `\r\r` for iMessage formatting reliability.
- Running tests from `tests/` directly can fail imports; use `PYTHONPATH=.`.

### Error Handling

- API errors checked via `"error" in response_data`
- Tool errors return `{"success": False, "error": "..."}` format
- Network errors have retry logic with exponential backoff

### Rate Limiting

- `api_call_with_retry()` handles rate limits automatically
- Retries up to 3 times with exponential backoff
- Checks for `"rate limit"` in error messages

### Session Pitfalls + Fixes

- **Guideline for future sessions:** Keep appending newly discovered pitfalls and their fixes to this section (or a new date-stamped section), instead of replacing old entries. Include a concrete remediation and, when applicable, the exact validation command/test used.

- **Pitfall: iMessage formatting was inconsistent (sometimes single dense paragraph, sometimes line-broken).**
  **Fix:** Normalize outbound Sendblue text right before send in `integrations.py` via `_format_sendblue_message_content()` and route all outbound content through it from `send_imessage()`.
- **Pitfall: Newline variants arrived mixed (`\\n`, `\\r\\n`, and real newlines), causing unpredictable rendering.**
  **Fix:** Canonicalize all line endings to `\n`, collapse excessive blank lines, then convert to double carriage returns for iMessage delivery.
- **Pitfall: Receipt-style key/value outputs were hard to read when model returned one long paragraph.**
  **Fix:** Add deterministic split rules for common labels (`name:`, `order #:`, `date:`, `items:`, `drinks:`, `sauces:`, `restaurant #:`), with sentence splitting fallback.
- **Pitfall: Behavior needed an operational toggle for rollback/troubleshooting.**
  **Fix:** Add env switch `SENDBLUE_FORCE_CARRIAGE_RETURNS` (default `1`). Set to `0` to keep LF newlines.
- **Pitfall: Running test scripts directly from `tests/` caused `ModuleNotFoundError: No module named 'integrations'`.**
  **Fix:** Run tests with project root on path, for example:
  `PYTHONPATH=. ../agentzero/.venv/bin/python tests/test_sendblue_debounce.py`
- **Pitfall: Formatting changes can regress silently if only manual QA is used.**
  **Fix:** Add explicit regression tests that assert Sendblue payload content uses carriage returns and that dense receipt text is split predictably.
- **Pitfall: Messages sent while the bot was offline were skipped in Sendblue webhook mode and on fresh polling startups.**
  **Fix:** Add startup replay in `integrations.py` (`_replay_sendblue_startup_backlog`) using configurable lookback + unread detection, then process immediately before entering webhook/polling loops.
- **Pitfall: Startup replay and live webhooks/polling can double-process the same Sendblue message during handoff.**
  **Fix:** Seed handle-based in-memory dedupe across replay + runtime (`SENDBLUE_DEDUP_TTL_SECONDS`) and cover behavior in `tests/test_sendblue_debounce.py`.
- **Pitfall: Telegram queued updates could remain delayed until regular polling stabilized after reconnect.**
  **Fix:** Drain pending updates first via `_replay_telegram_pending_updates()` before `start_polling()`, controlled by `TELEGRAM_REPLAY_PENDING_UPDATES_ON_STARTUP`.
- **Pitfall: Conversation logs could grow quickly while persistent memories remained sparse because memory writes depended on explicit `remember()` tool calls.**
  **Fix:** Add post-response auto-memory cadence capture in `handler.py` with bounded ratio controls (`AUTO_MEMORY_MIN_MESSAGES_PER_MEMORY=10`, `AUTO_MEMORY_TARGET_MESSAGES_PER_MEMORY=15`, `AUTO_MEMORY_MAX_MESSAGES_PER_MEMORY=20`) plus near-duplicate filtering (`AUTO_MEMORY_DEDUPE_THRESHOLD`).
- **Pitfall: Without consolidation, short-term memories accumulate and retrieval quality drifts over time.**
  **Fix:** Add dream-cycle consolidation in `handler.py` + `memory.py` that learns off-peak windows from 2-3 weeks of usage (`infer_offpeak_hours` with 21-day lookback / 14-day minimum), writes `long_term_memory` entries with model-assigned significance, and marks source memories as consolidated.
- **Pitfall: Conversational replies were delivered as one dense outbound text even when the model intended pacing/beat breaks.**
  **Fix:** Add agent-directed chunk extraction in `integrations.py` (`_split_outbound_message_chunks`) that honors `<message>...</message>` blocks and fan-outs each chunk as a separate outbound Sendblue/Telegram message while preserving attachment delivery on the final chunk.
- **Pitfall: New chunked-delivery behavior could regress and silently collapse back to single-message sends.**
  **Fix:** Add regression assertions in `tests/test_sendblue_debounce.py` for explicit chunk extraction and multi-call Sendblue dispatch (`test_split_outbound_message_chunks_prefers_explicit_blocks`, `test_send_imessage_sends_explicit_message_blocks_separately`). Validate with: `PYTHONPATH=. .venv/bin/python tests/test_sendblue_debounce.py`.
- **Pitfall: Multi-part replies still felt abrupt when chunk boundaries were respected but no pacing cue existed.**
  **Fix:** Extend outbound parsing in `integrations.py` to support `<typing seconds="..."/>` directives between `<message>` blocks, then emit channel-appropriate typing pauses before the next chunk.
- **Pitfall: Typing-directive behavior can silently break while basic chunk splitting still passes.**
  **Fix:** Add regression assertions in `tests/test_sendblue_debounce.py` (`test_split_outbound_message_chunks_ignores_typing_directives`, `test_send_imessage_typing_directive_triggers_indicator`) and validate with: `PYTHONPATH=. .venv/bin/python tests/test_sendblue_debounce.py`.
- **Pitfall: Internal consortium voting tool (`consortium_agree`) was exposed in the primary agent tool schema, allowing inappropriate top-level usage.**
  **Fix:** Remove `consortium_agree` from `BASE_PAYLOAD["tools"]`, expose explicit task controls (`consortium_start`, `consortium_stop`, `consortium_status`) for the main agent, and keep `consortium_agree` internal to consortium-member turns only.
- **Pitfall: Tool execution accepted any registered tool name, even when not declared in the active payload schema.**
  **Fix:** Enforce payload-scoped tool execution in `api.py` by filtering tool calls against the payload’s declared tool names before execution; return a structured tool error when unavailable.
- **Pitfall: Cross-channel continuity was fragmented because rolling context only read messages from the active `session_id` (`tg_*` vs `imessage_*`).**
  **Fix:** Add explicit cross-channel recall parsing in `handler.py` for prompts like "remember what we were talking about on telegram/imessage", fetch last-N channel messages via new `memory.py` prefix helpers (`get_recent_session_ids_by_prefix`, `get_recent_conversation_messages_for_prefix`), and inject the selected history into system context for that turn.
- **Pitfall: Natural-language trigger detection missed contractions/typos (for example, "we we're talking about"), causing silent non-injection of requested channel history.**
  **Fix:** Broaden recall regex token matching to tolerate apostrophes and short filler words, then add regression coverage in `tests/test_memory_maintenance.py` (`test_cross_channel_recall_injects_requested_history`, `test_cross_channel_recall_prefers_current_session_when_same_channel`). Validate with: `PYTHONPATH=. python3 tests/test_memory_maintenance.py`.
- **Pitfall: Early Sendblue voice memo rows could be stored with only `[Voice memo attachments not transcribed]` URL blocks, leaving conversation history without transcript text.**
  **Fix:** Add startup backfill in `integrations.py` (`_backfill_untranscribed_voice_memo_conversations`) that retries legacy URLs and updates `conversations.content` via new `memory.py` helpers (`get_conversation_messages_with_untranscribed_voice_memos`, `update_conversation_message_content`), plus regression coverage in `tests/test_sendblue_voice_memo.py` (`test_backfill_untranscribed_voice_memos_updates_conversation_content`). Validate with: `PYTHONPATH=. .venv/bin/python tests/test_sendblue_voice_memo.py`.
- **Pitfall: Auto-tapback reactions can fail silently if inbound `message_handle` metadata is dropped during debounce/replay/polling paths.**
  **Fix:** Add explicit Sendblue reactions support in `integrations.py` (`send_reaction`, `_maybe_send_random_sendblue_tapback`) and propagate `message_handle`/`part_index` through startup replay plus queued webhook flush paths before calling `process_imessage_and_reply`. Guard behavior with env controls (`SENDBLUE_AUTO_TAPBACK_ENABLED`, `SENDBLUE_TAPBACK_PROBABILITY`) and relevance heuristics. Validate with: `PYTHONPATH=. .venv/bin/python tests/test_sendblue_debounce.py`.
- **Pitfall: Implementing a capability in `integrations.py` is not enough for autonomous use if it is not also in the public tool schema.**
  **Fix:** Expose Sendblue tapbacks as a first-class tool by adding `send_tapback_tool` in `tools.py`, registering it in `TOOLS` + `validate_tool_args()`, and adding `send_tapback` to `BASE_PAYLOAD["tools"]` in `handler.py`. Validate with: `PYTHONPATH=. .venv/bin/python tests/test_simple.py`.
- **Pitfall: Exposing `send_tapback` in the tool schema was still insufficient because the model could not see the concrete inbound `message_handle` IDs required to call it.**
  **Fix:** Thread Sendblue `message_handle` / `part_index` metadata through `handle_imessage()` into `AgentHandler.handle()`, persist it on conversation rows, and inject an `[Available iMessage tapback handles ...]` context block into the system prompt for iMessage sessions. Validate with: `PYTHONPATH=. .venv/bin/python tests/test_memory_maintenance.py` and `PYTHONPATH=. .venv/bin/python tests/test_sendblue_debounce.py`.
- **Pitfall: Startup message replay and pending-update draining were brittle enough to replay stale or broken channel state on boot.**
  **Fix:** Remove Sendblue startup backlog replay and Telegram pending-update replay from `integrations.py`, keep only live webhook/polling handling, and drop the related env knobs/tests/docs. Validate with: `PYTHONPATH=. .venv/bin/python tests/test_sendblue_debounce.py`.
- **Pitfall: Repeated top-level user turns could arrive with nearly identical request bodies, making upstream cache reuse or repeated phrasing more likely.**
  **Fix:** Add per-request no-cache headers plus a unique `X-Request-Id` in `api.py`, and inject a one-time freshness token into the main visible-response system prompt in `handler.py` so repeated turns are structurally distinct. Validate with: `PYTHONPATH=. .venv/bin/python tests/test_process_response.py` and `PYTHONPATH=. .venv/bin/python tests/test_memory_maintenance.py`.
- **Pitfall: Cron-based scheduled tasks never fired on quiet periods if scheduler startup depended on a user message reaching `handle()`.**
  **Fix:** Start reminder scheduler during runtime bootstrap in `main.py` (`await handler.start_reminder_scheduler()`) and also guard with idempotent startup inside `AgentHandler.handle()`. Validate with: `PYTHONPATH=. .venv/bin/python tests/test_simple.py` and `PYTHONPATH=. .venv/bin/python tests/test_reminder_tasks.py`.
- **Pitfall: Scheduled tasks that required model output could accidentally invoke normal tool loops instead of direct inference.**
  **Fix:** Route reminder AI execution through a dedicated direct-inference path in `handler.py` (`_run_direct_ai_inference`) with `tools=[]`, then persist outputs via `ReminderScheduler` state and optional session message logging. Validate with: `PYTHONPATH=. .venv/bin/python tests/test_reminder_tasks.py`.
- **Pitfall: Reminder tasks could run successfully but never reach the user because execution only wrote to conversation history and did not route back through Telegram/iMessage delivery.**
  **Fix:** Add integration-side session delivery target registration in `integrations.py`, expose `deliver_scheduled_session_output(session_id, output)`, and wire `ReminderScheduler` to use a delivery callback from `handler.py` after each run. Validate with: `PYTHONPATH=. .venv/bin/python tests/test_reminder_tasks.py`.
- **Pitfall: The model could create reminder tasks without an explicit `session_id`, causing scheduled outputs to lose their return path even when delivery plumbing existed.**
  **Fix:** Default `reminder_create_tool()` in `tools.py` to the active tool runtime session (`_runtime_session_id`) when `session_id` is omitted, and add regression coverage in `tests/test_reminder_tasks.py`. Validate with: `PYTHONPATH=. .venv/bin/python tests/test_reminder_tasks.py`.
- **Pitfall: Reminder persistence stored all scheduled tasks inside one `agent_state` JSON blob, which made reminders less durable/inspectable and tied startup loading to one serialized state object.**
  **Fix:** Add a first-class `reminders` SQLite table in `memory.py`, persist scheduler state there by default, and have `ReminderScheduler` load from that table on startup with a fallback migration path from legacy `agent_state`. Validate with: `PYTHONPATH=. .venv/bin/python tests/test_reminder_tasks.py` and `PYTHONPATH=. .venv/bin/python tests/test_simple.py`.
- **Pitfall: One-off reminders that explicitly targeted today could silently roll into a later weekly/monthly cron match after the requested same-day time had already passed.**
  **Fix:** Add a simpler same-day resolution path in `reminder_tasks.py` for one-off tasks: prefer the next matching time within the current day first, and if the cron expression explicitly targets today but no same-day slot remains, fail with a clear error instead of rolling forward. Validate with: `PYTHONPATH=. .venv/bin/python tests/test_reminder_tasks.py` and `PYTHONPATH=. .venv/bin/python tests/test_simple.py`.
- **Pitfall: Exposing a local OpenAI-compatible endpoint without strict bearer auth and explicit opt-in could accidentally open unauthenticated access.**
  **Fix:** Add a dedicated `openai_compat_server.py` with required bearer-key validation for `/v1/models` and `/v1/chat/completions`, and gate startup in `main.py` behind `OPENAI_COMPAT_ENABLED=1` plus mandatory `OPENAI_COMPAT_API_KEY`. Validate with: `PYTHONPATH=. .venv/bin/python tests/test_openai_compat_server.py`.
- **Pitfall: Model-authored outbound `<message>`/`<typing .../>` directives were often malformed, causing visible tag leakage and fragmented multi-send behavior.**
  **Fix:** Remove outbound chunk fan-out and typing-directive pacing in `integrations.py`, collapse tagged outputs into one sanitized message, and strip delivery directives in `handler.py`/`openai_compat_server.py` before returning visible text. Validate with: `PYTHONPATH=. .venv/bin/python tests/test_sendblue_debounce.py`, `PYTHONPATH=. .venv/bin/python tests/test_memory_maintenance.py`, and `PYTHONPATH=. .venv/bin/python tests/test_openai_compat_server.py`.
- **Pitfall: ACP import and messaging reliability collapsed when `cryptography` was unavailable, because RSA type hints/identity generation assumed the module existed and discovery/send depended on non-existent socket listeners.**
  **Fix:** Make ACP crypto-optional in `acp.py` (safe fallback signatures, optional identity key, robust security plugin verification), switch ACP runtime to an in-process network registry with lazy agent initialization, and add a core tool-path integration test that discovers and messages a live peer app (`tests/test_acp_core_tool_flow.py`). Validate with: `PYTHONPATH=. .venv/bin/python tests/test_acp.py`, `PYTHONPATH=. .venv/bin/python tests/test_acp_core_tool_flow.py`, `PYTHONPATH=. .venv/bin/python tests/test_simple.py`, and `PYTHONPATH=. .venv/bin/python tests/test_process_response.py`.
- **Pitfall: ACP could only discover in-process agents, so a real external ACP server was invisible and unreachable from `acp_discover_peers` / `acp_send_message`.**
  **Fix:** Extend `acp.py` with remote ACP HTTP discovery + execution: load `ACP_REMOTE_ENDPOINTS`, query `/agents`, map `agent_name -> endpoint`, and send outbound ACP runs via `POST /runs` using `mode: "sync"` and ACP message-part input (`role: "user"` + `parts[].content`). Add regression coverage in `tests/test_acp_remote_http.py` and include it in `run_all_tests.py`. Validate with: `PYTHONPATH=. .venv/bin/python tests/test_acp_remote_http.py`, `PYTHONPATH=. .venv/bin/python tests/test_acp_core_tool_flow.py`, and a live tool-call probe using `execute_tool_calls` + `acp_send_message` against a remote ACP endpoint.
- **Pitfall: Unbounded inbound image attachments could overwhelm multimodal turns, making per-message behavior unpredictable when users sent many images.**
  **Fix:** Add a configurable cap in `integrations.py` (`MAX_IMAGE_ATTACHMENTS_PER_MESSAGE`, default `20`) and truncate extra attachments with an explicit context note to the model. Add regression coverage in `tests/test_multimodal_integrations.py` (`test_multimodal_message_blocks_respect_attachment_limit`). Validate with: `PYTHONPATH=. .venv/bin/python tests/test_multimodal_integrations.py`.
- **Pitfall: NVIDIA NVCF image-asset upload paths added API coupling and conflicted with strict provider-side per-message image limits.**
  **Fix:** Remove NVCF asset creation/upload + header injection paths (`integrations.py` + `api.py`), enforce `MAX_IMAGE_ATTACHMENTS_PER_MESSAGE=8` by default, and convert inbound images to JPEG base64 data URLs via ImageMagick before model calls. Add regression coverage in `tests/test_multimodal_integrations.py` and `tests/test_process_response.py`. Validate with: `PYTHONPATH=. .venv/bin/python tests/test_multimodal_integrations.py`, `PYTHONPATH=. .venv/bin/python tests/test_process_response.py`.
- **Pitfall: The core agent could narrate `here.now` publishes ("pushing/posting now") without making tool calls, because action-intent retries shared the same one-shot budget as generic tool-leak formatting retries and missed some publish phrasing.**
  **Fix:** Split action-intent retries into their own budget in `agentic_loop.py`, broaden narration detection to catch `push/post/upload/deploy` phrasing used in real logs, and add regression coverage in `tests/test_agentic_loop.py`. Validate with: `PYTHONPATH=. python3 tests/test_agentic_loop.py`.
- **Pitfall: `here.now` folder publishes could 504 because the publish script traversed and uploaded repository metadata (`.git`) plus local publish cache state (`.herenow`), inflating the manifest far beyond the actual site contents.**
  **Fix:** Update `/home/clxud/.agents/skills/here-now/scripts/publish.sh` to prune `.git`, `.hg`, `.svn`, and `.herenow` during directory walks and skip those paths in the manifest builder. Validate with: `cd workspace/libido-doomsday-clock && /home/clxud/.agents/skills/here-now/scripts/publish.sh . --client codex`.
- **Pitfall: Restoring the main agentic loop changed the real follow-up path from inline `api.process_response()` logic to `agentic_loop.run_agentic_loop()`, so end-to-end regressions that only mocked `api.api_call_with_retry` no longer exercised the live multi-round execution path.**
  **Fix:** Update the loop-facing regressions to patch `agentic_loop.api_call_with_retry` / `agentic_loop.execute_tool_calls` directly, add a complex multi-round agent-flow test in `tests/test_process_response.py`, and align OpenAI-compat mocks with the current `aiohttp` response contract. Validate with: `PYTHONPATH=. python3 tests/test_process_response.py`, `PYTHONPATH=. python3 tests/test_agentic_loop.py`, and `PYTHONPATH=. python3 tests/test_openai_compat_tool_calls.py`.
- **Pitfall: The iMessage typing-indicator teardown could raise after a successful response if the temporary typing session object did not expose an awaitable `.close()`, which could break reply delivery during cleanup.**
  **Fix:** Guard optional session shutdown in `integrations.py` (`process_imessage_and_reply`) and add channel-flow regression coverage for Telegram interim/final delivery plus Sendblue typing shutdown timing. Validate with: `PYTHONPATH=. python3 tests/test_sendblue_debounce.py` and `PYTHONPATH=. python3 tests/test_telegram_message_flow.py`.
- **Pitfall: The convenience smoke test `tests/test_simple.py` could hang for tens of seconds because its `grep` check recursively walked the entire repository instead of a bounded test fixture path.**
  **Fix:** Scope the smoke-test grep to `tests/` so `run_all_tests.py` remains usable as a fast baseline. Validate with: `PYTHONPATH=. python3 tests/test_simple.py`.
- **Pitfall: `run_all_tests.py` masked child-test failures and dropped the ambient environment by replacing `env` with only `PYTHONPATH`, which caused misleading green runs and broke tests that rely on `sys.executable` / standard env state (for example `tests/test_skills.py`).**
  **Fix:** Rebuild `run_all_tests.py` to preserve `os.environ`, prepend `PYTHONPATH`, run tests with `sys.executable`, collect non-zero exits, and return failure when any child test fails. Also fix `tests/test_setprompt_command.py` to use `AsyncMock.reset_mock()` so warning noise does not hide real failures. Validate with: `PYTHONPATH=. python3 run_all_tests.py`.
- **Pitfall: A live in-process Sendblue smoke uncovered that the model naturally called `read` with `limit`/`offset`, but `read_file_tool()` only accepted `filepath`, causing unnecessary failed tool rounds before recovery.**
  **Fix:** Extend `read_file_tool()` in `tools.py` to support model-friendly line slicing via optional `limit` and human-friendly `offset`, add fast coverage in `tests/test_simple.py`, and keep a reusable programmatic harness in `scripts/live_sendblue_smoke.py` that drives `initialize_agent()` + `process_imessage_and_reply()` while intercepting only outbound Sendblue HTTP. Validate with: `PYTHONPATH=. python3 tests/test_simple.py` and `PYTHONPATH=. python3 scripts/live_sendblue_smoke.py`.
- **Pitfall: Programmatic smoke harnesses can silently use the wrong model/config if `.env` is loaded after importing `handler`/`main`, because `MODEL_ID` is bound at import time.**
  **Fix:** Load `.env` before importing runtime modules in `scripts/live_sendblue_smoke.py` so the shim exercises the same configured model as the live app. Validate with: `PYTHONPATH=. python3 scripts/live_sendblue_smoke.py` and confirm the reported model matches the configured `.env` value.
- **Pitfall: Live startup against the current Sendblue API can reject `typing_indicator` webhook registration with a generic "Invalid webhook type. Must be one of ..." message that omits the literal `typing_indicator` string, causing the monitor to retry forever instead of disabling itself.**
  **Fix:** Broaden `_is_sendblue_invalid_typing_webhook_type_error()` in `integrations.py` to treat that provider message shape as unrecoverable and add regression coverage in `tests/test_sendblue_debounce.py`. Validate with: `PYTHONPATH=. python3 tests/test_sendblue_debounce.py`.
- **Pitfall: Live Sendblue-path prompts can generate ripgrep-style `grep` arguments (`include`, `max_matches`) that were previously unsupported by `grep_tool()`, causing avoidable tool errors during otherwise-correct multi-step runs.**
  **Fix:** Extend `grep_tool()` in `tools.py` to accept `include` and `max_matches` (plus tolerant extra kwargs), and add quick coverage in `tests/test_simple.py`/`tests/test_tools.py`. Validate with: `PYTHONPATH=. python3 tests/test_simple.py` and `PYTHONPATH=. python3 scripts/live_sendblue_smoke.py --prompt \"Live Sendblue smoke test scenario 3...\" --require-tool grep --require-tool bash`.
- **Pitfall: Even when users explicitly requested tool execution (for example, \"use grep then bash\"), the model could occasionally return text-only answers with no tool calls, creating false-positive seeming success unless the loop enforced the requirement.**
  **Fix:** Add explicit-tool-requirement detection in `agentic_loop.py` (`user_explicitly_requires_tool_execution`) and nudge for real tool calls before accepting text-only completion, but only until at least one tool round executes. Add regression coverage in `tests/test_agentic_loop.py` (`test_run_agentic_loop_retries_when_user_explicitly_requires_tools`). Validate with: `PYTHONPATH=. python3 tests/test_agentic_loop.py` and `PYTHONPATH=. python3 scripts/live_sendblue_smoke.py --prompt \"Live Sendblue smoke test scenario 3...\" --require-tool grep --require-tool bash`.
- **Pitfall: In live iMessage traffic, the model could narrate a tapback in plain text (for example, `sending like to ...`) instead of calling `send_tapback`, even when valid `message_handle` context was present.**
  **Fix:** Extend action-intent detection in `agentic_loop.py` to treat tapback narration as a retry-worthy action leak and update the retry nudge to explicitly instruct `send_tapback` with the provided `message_handle`. Validate with: `PYTHONPATH=. python3 tests/test_agentic_loop.py`, then watch a foreground `LOG_LEVEL=DEBUG python3 main.py` run for the next live Sendblue tapback-only message.
- **Pitfall: Even after a correct live `send_tapback` call, the model could still emit leftover acknowledgement/meta text (`done`, `sent a like tapback...`, `no more loops...`) which Sendblue would deliver as an unwanted second message.**
  **Fix:** Suppress tapback-only acknowledgement text in `agentic_loop.py`, strip leaked internal prompt residue in `handler.py`, and make `send_imessage()` in `integrations.py` skip empty payloads. Add regression coverage in `tests/test_agentic_loop.py`, `tests/test_memory_maintenance.py`, and `tests/test_sendblue_debounce.py`. Validate with: `PYTHONPATH=. python3 tests/test_agentic_loop.py`, `PYTHONPATH=. python3 tests/test_memory_maintenance.py`, `PYTHONPATH=. python3 tests/test_sendblue_debounce.py`, and `PYTHONPATH=. python3 - <<'PY' ... run_smoke(... required_tools=['send_tapback'], require_outbound_message=False) ... PY`.
- **Pitfall: Fully suppressing tapback-only acknowledgement text fixed prompt leakage but created a worse UX regression: users saw only the tapback and no visible reply, which reads like agent failure in iMessage/Telegram.**
  **Fix:** Canonicalize tapback-only completions to a stable minimal reply (`reacted.`) in `agentic_loop.py`, and mirror that fallback in `handler.py` so prompt residue is still stripped without going silent. Keep `send_imessage()` empty-payload skipping as a safeguard. Validate with: `PYTHONPATH=. python3 tests/test_agentic_loop.py`, `PYTHONPATH=. python3 tests/test_memory_maintenance.py`, `PYTHONPATH=. python3 tests/test_sendblue_debounce.py`, and `PYTHONPATH=. python3 - <<'PY' ... run_smoke(... expected_substrings=['reacted.'], required_tools=['send_tapback'], require_outbound_message=True) ... PY`.
- **Pitfall: A synthetic fallback like `reacted.` still feels wrong in chat because it replies about the tapback instead of replying to the user's actual message.**
  **Fix:** Replace the tapback fallback with a forced text-only follow-up turn in `agentic_loop.py`: once `send_tapback` has executed, if the model tries to end with `done`/reaction narration, strip tools from the retry payload and instruct it to answer the user's message naturally without mentioning the tapback. Remove handler-side tapback canonicalization so the loop owns this behavior. Validate with: `PYTHONPATH=. python3 tests/test_agentic_loop.py`, `PYTHONPATH=. python3 tests/test_memory_maintenance.py`, `PYTHONPATH=. python3 tests/test_sendblue_debounce.py`, and `PYTHONPATH=. python3 - <<'PY' ... run_smoke(... required_tools=['send_tapback'], require_outbound_message=True) ... PY` expecting a normal conversational reply like `glad we're on the same page`.
- **Pitfall: The executor had no built-in way to escalate hard mid-run strategy decisions to a stronger model, so it either stalled in deliberation text or improvised architecture choices without a shared-context advisor pass.**
  **Fix:** Add a first-class `consult_advisor` tool backed by `AgentHandler.consult_advisor()`, resolve executor/advisor models separately (`PREFLIGHT_MODEL`/`EXECUTOR_MODEL` for executor, `MAIN_MODEL`/`ADVISOR_MODEL` for advisor), pass live turn context through the agentic loop, and nudge hard-decision text toward advisor consultation. Validate with: `PYTHONPATH=. python3 tests/test_agentic_loop.py`, `PYTHONPATH=. python3 tests/test_process_response.py`, and `PYTHONPATH=. python3 tests/test_simple.py`.

### Session Pitfalls + Fixes (2026-06-04)

- **Pitfall: Every inbound message replied with "Error: Request timed out — the task was too complex to complete..." This looked like a complexity/loop bug, but the real cause was the configured `MODEL_ID`.** The `.env` model `minimaxai/minimax-m2.7` is *listed* in NVIDIA's `/v1/models` catalog but the inference endpoint **black-holes** requests for it (no response headers, no error — the connection just hangs). Because the NVIDIA POST in `api_call_with_retry()` never returns, `handle_imessage()` blocks for the full `REQUEST_TIMEOUT_SECONDS` (default 600s) in `integrations.py`, then `asyncio.wait_for` fires the `asyncio.TimeoutError` fallback at `integrations.py:~4163` ("too complex" message). No NVIDIA request ever appears in logs; the last line is `handler: Context window: N tokens used`.
  **Diagnosis shortcut:** if every turn stalls right after the `Context window:` log with no `Raw API response` debug line, curl the configured model directly: `curl -m 30 -X POST https://integrate.api.nvidia.com/v1/chat/completions -H "Authorization: Bearer $NVIDIA_API_KEY" -d '{"model":"<MODEL_ID>","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'`. A `curl (28) timed out` with 0 bytes = the model is unavailable upstream even though it's catalog-listed. Verify the API key is fine by testing a known-good model (e.g. `meta/llama-3.1-8b-instruct` responds instantly).
  **Fix:** Point `MODEL_ID` at a model that actually responds. Switched to `stepfun-ai/step-3.7-flash` (fast, good, native multimodal; ~2s first-token, HTTP 200). Restart `main.py` so import-time `MODEL_ID`/`ADVISOR_MODEL_ID`/`PRIMARY_MODEL_ID` rebind. Validate live: POST a test payload to `http://localhost:3847/webhook/receive` and confirm logs show `finish_reason: stop` and an outbound `"is_outbound": true, "status": "DELIVERED"` within seconds (not the timeout fallback).
  **Hardening note (not yet applied):** `REQUEST_TIMEOUT_SECONDS=600` makes an unresponsive model look like a hang for 10 minutes. Consider a much shorter per-API-call timeout on the `aiohttp` POST in `api_call_with_retry()` so a dead/black-holing model fails fast and visibly instead of stalling the whole turn.

- **Change (2026-06-04): Removed the code-side tool-call retry guards from `agentic_loop.py` and moved them into the system prompt.** Deleted `user_explicitly_requires_tool_execution` (forced-tool-when-user-asked), plus the pseudo-tool-syntax, skill-URL, action-intent-narration, bare-reaction/reaction-ack, and tapback-followup retry nudges and all their helpers/regexes (`contains_action_intent_narration`, `contains_pseudo_tool_syntax`, `is_bare_reaction_word`, `text_contains_reaction_emoji`, `conversation_exposes_reaction_targets`, `looks_like_short_reaction_ack`, `needs_tapback_followup_reply`, `latest_user_skill_url`, `_ACTION_INTENT_PATTERNS`, `_PSEUDO_TOOL_*`, `_BARE_REACTION_PATTERN`, `_SHORT_REACTION_ACK_PREFIX_PATTERN`, `_SKILL_URL_PATTERN`/`_SKILL_URL_HESITATION_PATTERNS`, `TAPBACK_ACK_PATTERN`, `REACTION_TOOL_NAMES`/`REACTION_WORD_NAMES`, and the `max_action_intent_retries`/`max_tapback_reply_retries` plumbing). These were extra model round-trips that added latency; the model is now trusted to call tools, with the rules enforced via `prompts/system_prompt.md` (`ACT DIRECTLY` + new `REACTIONS` section). **Kept on purpose:** `infer_tool_calls_from_content()` leaked-call recovery (pure parsing, no extra round-trip) and the tool-leak output-sanitization guard (stops raw tool JSON reaching the user). Tests/script updated (`tests/test_agentic_loop.py` dropped the three deleted-helper unit tests; `scripts/live_sendblue_smoke.py` dropped the dead diagnostic fields). Validated: full suite green via `AGENTZERO_LIVE_TESTS=0 PYTHONPATH=. python3 run_all_tests.py` (23/23) and `AGENTZERO_LIVE_TESTS=1 python3 tests/test_agentic_loop.py` (all live L1–L8 pass, incl. a live in-process probe showing `TOOLS CALLED: ['bash']`).

- **Test regressions fixed in the same pass (pre-existing, from the uncommitted refactor that removed features the tests still imported):**
  - `tests/test_process_response.py` imported `_apply_cache_busting_headers` from `api.py`, which no longer exists (the per-request no-cache/X-Request-Id logic was removed). Dropped the import, the `test_cache_busting_headers_are_applied` test, and its `main()` call.
  - `tests/test_memory_maintenance.py` imported `REQUEST_FRESHNESS_INSTRUCTION` from `handler.py`, which no longer exists (the one-time freshness token was removed). It was import-only/unused in the body; dropped from the import list.
  - `tests/test_agentic_loop.py::test_live_agent_calls_grep_tool` used a brittle `len(text) > 5` assertion that failed when the model answered a count tersely (e.g. `"0"`). Replaced with: reply is non-empty AND contains a digit (matches the test's intent — it asks for a number of matches).
  - Note: NVIDIA free tier is 40 RPM and the in-process rate limiter is per-process, so running live tests while `main.py` is also running will produce spurious `Rate limit exceeded after retries` failures. Stop the daemon before running live suites.

- **Change (2026-06-04): Hardened the API layer so multi-step agentic tasks (e.g. build-a-site-then-publish) don't abort on a single transient round.** Two fixes in `api.py`, both generic (no tie to any specific skill/provider feature):
  1. **Streaming SSE reader was line-splitting incorrectly.** `_read_streaming_chat_response` iterated `resp.content` directly, but an aiohttp `StreamReader` yields arbitrary network chunks, not complete lines — so a single SSE `data:` line carrying a large tool-call `arguments` payload (e.g. an HTML file for `write`) could be split across reads and handed to `json.loads` half-formed → `Unterminated string`. Now it buffers raw bytes, splits on `\n`, only parses complete lines, and defensively skips an unparseable line instead of aborting the stream. Regression: `tests/test_process_response.py::test_streaming_handles_data_line_split_across_reads`.
  2. **Transient 200-body provider errors weren't retried.** Providers intermittently return a 200 whose body is `{"error": {"message": "Unterminated string ..."}}` (the upstream model output was truncated/malformed while processing a large tool-call payload). The old retry filter only matched `rate limit`/`rate_limit`/`server_error`, so these aborted the whole run with a fatal `Error:`. Added `_is_transient_api_error()` (matches markers like `unterminated string`, `expecting value`, `internal server error`, `timeout`, `overloaded`, `try again`, …) and retry on `resp.status >= 500`; genuine client errors (e.g. `invalid 'messages'`) still fail fast. Regressions: `test_api_call_retries_transient_200_body_error`, `test_api_call_does_not_retry_permanent_client_error`.
  **Validated end-to-end:** the autonomous build-and-publish golden flow went from intermittent failure (`Unterminated string` at the post-`write` round) to **3/3 live successes**, each producing a real shareable URL and a final reply of the form "done — your … page is live at <link>". Reproduce with: `AGENTZERO_LIVE_TESTS=1 PYTHONPATH=. python3 scripts/live_sendblue_smoke.py --prompt "make me a simple fun website landing page and publish it online so i can share it, then send me the link" --require-tool bash --no-require-read-receipt --no-require-typing-indicator`. Note: the iMessage path is non-streaming (`response_chunk_callback` is None; iMessage uses `interim_response_callback`), so fix #2 is the one that carries the golden flow; fix #1 protects the streaming path used when a chunk callback is wired.

- **Change (2026-06-04): Tool-call history compaction — stop re-sending large tool-call argument payloads on later rounds.** Per `docs/DOSSIER_tool_call_history_compaction.md`. After a tool round executes, `agentic_loop.py` now retains a *compacted copy* of the assistant message: large string fields inside each tool call's JSON `arguments` (most commonly a whole file passed to `write`) are replaced with a readable `<elided N chars — 'field' written in round R>` placeholder via `_compact_executed_tool_call_args()` / `_compact_arguments_string()`. The live `message` passed to `execute_tool_calls` is never mutated (it already ran with full fidelity) — only the history copy shrinks, so subsequent rounds don't re-upload the payload. This cuts per-turn input tokens and removes the request bloat that was stressing the provider into the transient truncation errors fixed earlier the same day. Keyed on argument *size*, not tool identity (generic). Env-gated: `AGENTZERO_COMPACT_TOOL_ARGS` (default `1`/on), `AGENTZERO_COMPACT_TOOL_ARG_THRESHOLD` (default 2000 chars), `AGENTZERO_COMPACT_FIELD_THRESHOLD` (default 1000 chars) — small calls (bash one-liners, reads, reactions) are never touched. Idempotent; falls back to truncating non-JSON args. Tests: `tests/test_agentic_loop.py` C1–C6. Live-validated on the golden build-and-publish flow: compaction logs `reclaimed ~5–12 KB` after the `write` round and the site still publishes with the link reported. (One live run failed on `Rate limit exceeded after retries` — the 40 RPM free-tier ceiling from back-to-back runs, not a compaction regression.) Validate: `AGENTZERO_LIVE_TESTS=0 PYTHONPATH=. python3 tests/test_agentic_loop.py` and a `LOG_LEVEL=INFO` smoke publish run (grep `Agentic loop: compacted`).

- **Change (2026-06-04): Made relative reminders ("remind me in a minute") reliable.** Root cause of the flakiness: `reminder_create` only accepted `run_at` (an absolute Unix epoch), so the model had to convert "in a minute" into an epoch — but the system prompt's `current_time` was **local, timezone-less, and epoch-less** (`datetime.now().strftime(...)`), while the scheduler compares against UTC. The model sometimes got the epoch right and sometimes landed it in the past (→ "run_at must be in the future") or hours off. Fixes:
  1. **New `delay_seconds` param** threaded through `tools.reminder_create_tool` → `handler.create_reminder_task` → `reminder_tasks.ReminderScheduler.create_task` (+ helper `_coerce_delay_seconds`). The server computes `run_at = now + delay_seconds` — no client-side clock/timezone math. An explicit `run_at` still takes precedence; `delay_seconds <= 0` is rejected with a clear error.
  2. **Tool schema + system prompt steer relative requests to `delay_seconds`** ("in a minute" → 60, "in 5 minutes" → 300). Added a `REMINDERS` section to `prompts/system_prompt.md` (must actually call `reminder_create`, use `delay_seconds` for relative timing, `cron` only for recurring).
  3. **Unambiguous time context:** `handler._build_system_content` now injects local time **with timezone** + UTC + the current Unix epoch, so any remaining absolute scheduling has a correct reference.
  Delivery was already robust (`integrations.deliver_scheduled_session_output` reconstructs the iMessage phone / Telegram chat_id from the `session_id` even without a registered in-memory target, so reminders survive a daemon restart). Tests: `tests/test_reminder_tasks.py` (`test_delay_seconds_creates_one_off_at_relative_time`, `test_delay_seconds_rejects_non_positive`, `test_run_at_takes_precedence_over_delay_seconds`, `test_reminder_create_tool_passes_delay_seconds`). Live-validated golden flow: "hey can you remind me in a minute to take out the trash" → model calls `reminder_create(delay_seconds=60, message="take out the trash")`, replies "got it, I'll remind you in a minute", and the reminder is delivered back to the session **after ~65s** (≈1 min + the scheduler poll interval). Validate: `AGENTZERO_LIVE_TESTS=0 PYTHONPATH=. python3 tests/test_reminder_tasks.py`.

### Session Pitfalls + Fixes (2026-06-05)

- **Pitfall: A live publish flow surfaced `Error: unterminated string starting at: line 1 column 69 (char 67) [iteration=3, tools_executed=['activate_skill', 'bash', 'write']]` directly to the user.** Root cause: on the round *after* `write` succeeded, the provider returned a 200 body `{"error":{"message":"Unterminated string ..."}}` — its **own** failure parsing the model's next tool-call `arguments` JSON (a serialization hiccup, not our parse). The api.py transient-retry from 2026-06-04 *did* fire (`_is_transient_api_error` matches `unterminated string`), but all 5 retries re-sent the **identical payload** → the model re-emitted the **identical malformed output** → every retry failed deterministically, exhausting the budget. `agentic_loop.py:448` then formatted the raw parser error and the user got JSON-parser guts as their reply even though the website file had been written. Two-part fix:
  1. **Break determinism on parse-class retries (`api.py`).** Added `_is_parse_class_api_error()` (the JSON-decode subset of the transient markers) and `_perturb_sampling_for_retry()`. When a retry is for a parse-class error, bump `temperature` by `+0.15` per attempt (capped 0.95) and drop any fixed `seed` *before* re-sending, so the model emits different tokens the provider can serialize. Also caught `(json.JSONDecodeError, ValueError)` around `resp.json()` in `api_call_with_retry` — our own parse of a truncated provider body is a `ValueError`, NOT an `aiohttp.ClientError`, so it previously **escaped the retry loop entirely** as an uncaught exception; now it retries with the same perturbation. Regressions: `tests/test_process_response.py::test_parse_class_retry_perturbs_sampling`, `::test_api_call_recovers_from_body_parse_failure`.
  2. **Never leak parser guts mid-task (`agentic_loop.py`).** When a round returns a transient/parse `Error:` AND ≥1 tool round already executed real work (`executed_tool_names` non-empty), call `_attempt_graceful_finish()` — a tools-stripped follow-up that asks the model to summarise what it completed in plain text (stripping tools removes the large/complex serialization the provider was choking on). If that also fails it falls back to the raw error path. Helpers: `_looks_recoverable()`, `_attempt_graceful_finish()`. Regression: `tests/test_agentic_loop.py::test_graceful_finish_on_transient_error_after_tool_round`. Validate: `AGENTZERO_LIVE_TESTS=0 PYTHONPATH=. python3 tests/test_process_response.py`, `AGENTZERO_LIVE_TESTS=0 PYTHONPATH=. python3 tests/test_agentic_loop.py`, full suite `AGENTZERO_LIVE_TESTS=0 PYTHONPATH=. python3 run_all_tests.py` (23/23 green).

### Session Pitfalls + Fixes (2026-06-11)

- **Pitfall: A live multi-step run surfaced `Error: Rate limit exceeded after retries [iteration=9, tools_executed=['bash', 'bash', 'bash', 'bash'...]]` directly to the user after the agent had already done real work.** Root cause was the same anti-pattern as the 2026-06-05 parser-guts leak, but rate-limit class was not covered: a later round 429'd, `api_call_with_retry` exhausted its 5 retries and returned `{"error":{"message":"Rate limit exceeded after retries"}}`, and the `agentic_loop.py` graceful-finish guard did **not** fire because `_looks_recoverable()` checked `_RECOVERABLE_ERROR_MARKERS`, which had no rate-limit marker — so the raw error dumped to the user instead of summarising the completed tool work. Two-part fix:
  1. **Treat exhausted rate limits as recoverable (`agentic_loop.py`).** Added `"rate limit"` / `"rate_limit"` to `_RECOVERABLE_ERROR_MARKERS` so a 429-exhausted error after real tool work triggers `_attempt_graceful_finish()` (tools-stripped summary). The graceful-finish call re-runs through `api_call_with_retry`, whose own retry/backoff gives the rolling free-tier window time to clear; worst case it also fails and we fall back to the raw-error path — never worse than before. Regression: `tests/test_agentic_loop.py::test_graceful_finish_on_rate_limit_after_tool_round`.
  2. **Harden the 429 backoff (`api.py`).** Added `_retry_after_seconds()` (parses both numeric and HTTP-date `Retry-After`; the server is authoritative about when its window clears) and `_MAX_RATELIMIT_BACKOFF = 20.0` (was a hardcoded `15.0` cap). The 429 branch now prefers `Retry-After` when present, else exponential backoff, both capped at the constant — a 5-retry sequence (3, 9, 20, 20 ≈ 52s) now covers most of a saturated rolling-60s window without blowing past turn/benchmark timeouts. Regressions: `tests/test_process_response.py::test_429_honors_retry_after_header`, `::test_429_caps_oversized_retry_after`.
  **Root-cause note:** hitting the 429 *at all* with a per-process 30 RPM limiter (`_MIN_INTERVAL=2.0`, under the 40 RPM ceiling) almost always means **two processes sharing the free tier** (e.g. the `main.py` daemon running while a smoke/test run fires) — the limiter is per-process, so collectively they exceed 40 RPM. Stop the daemon before live runs. Validate: `AGENTZERO_LIVE_TESTS=0 PYTHONPATH=. python3 tests/test_process_response.py`, `AGENTZERO_LIVE_TESTS=0 PYTHONPATH=. python3 tests/test_agentic_loop.py`, full suite `AGENTZERO_LIVE_TESTS=0 PYTHONPATH=. python3 run_all_tests.py` (23/23 green).

- **Change (2026-06-11): Added an orchestrated 3-stage pipeline for non-trivial turns (orchestrator-plan → blind worker → orchestrator-finalize).** Per the user's request to split planning from execution. In `handler.py`, `handle()` now branches (after the consortium block, before the standard single-model path) into a new `AgentHandler._run_orchestrated_task()` when `ORCHESTRATED_PIPELINE_ENABLED` is on and the turn is non-trivial:
  1. **Orchestrator plan** — advisor model (`ADVISOR_MODEL_ID`), `tools=[]`, `max_tokens=ORCHESTRATOR_PLAN_MAX_TOKENS` (800). Mirrors `consult_advisor()`. System prompt `prompts/orchestrator_plan.md` turns the user request into a self-contained worker brief. Empty/error → `raise` so the caller falls back to the single-model path.
  2. **Blind worker** (`_run_blind_worker`) — executor model (`EXECUTOR_MODEL_ID`, previously a dead constant), full work tools via `_build_request_payload_template`, runs the real agentic loop (`api_call_with_retry` → `process_response`). System prompt `prompts/worker_system.md` is **blind to the orchestration** (no orchestrator/finalizer/delivery awareness) and is told to *do the work and report results*, not chat. The orchestrator brief is the worker's final user message; rolling history gives continuity. Self-heal wraps this stage exactly as the normal path does.
  3. **Orchestrator finalize** — advisor model again, `max_tokens=ORCHESTRATOR_FINALIZE_MAX_TOKENS` (2000). System prompt `prompts/orchestrator_finalize.md` writes the final user-facing reply from the worker's results. Empty/error → return worker results directly (graceful degradation). Stays **synchronous** — `handle()` still returns one string, so every channel (iMessage/Telegram/OpenAI-compat) is unaffected.
  - **Double-delivery guard (important):** the worker payload strips user-facing/reentrant tools via `_WORKER_EXCLUDED_TOOLS` (`send_message`, `declare_message_count`, `send_tapback`, `send_telegram_reaction`, `consortium_start/stop/status`) so only the orchestrator delivers. Belt-and-suspenders: the worker runs under a sentinel runtime session `"orchestrated_worker__"+session_id` which doesn't start with `imessage_`/`tg_`, so even a leaked `send_message` buffers instead of texting (`tools.py:1281/1307`). A `{"delivered_via_tool": true}` envelope is also unwrapped defensively.
  - **Triviality gate:** `_is_trivial_query()` — greetings/acks regex, or `len < 60` with a `None`/generic task — bypass the pipeline (greetings shouldn't pay for plan+worker+finalize). `Task.type` is the string value (planning.py:158), verified.
  - **Cost caveat (surfaced to the user):** this adds ~2 advisor calls/task on the same NVIDIA 40 RPM key, so it does **not** reduce rate-limit pressure — the win lands only when `EXECUTOR_MODEL` is pointed at a *different* (cheaper/faster) model than `ADVISOR_MODEL`; today both default to `moonshotai/kimi-k2.6`. Env-gated `AGENTZERO_ORCHESTRATED_PIPELINE` (default `1`; set `0` to revert).
  - **Known dormant bug (not fixed, out of scope):** `tests/_live_harness.py:172` `live_agent_handle()` calls `handle(messages=…, session=…)` — wrong signature; only fires under `AGENTZERO_LIVE_TESTS=1`. Fix to `handle({"messages": messages}, session_id=…)` if/when live `handle()` harness tests are run.
  Tests: new `tests/test_orchestrated_pipeline.py` (registered in `run_all_tests.py`) covers the triviality gate against the real analyzer, the worker tool filter, the flag, the sentinel-session buffering, and the full 3-stage flow + finalize/plan failure paths with mocked `api_call_with_retry`/`process_response`. Validate: `AGENTZERO_LIVE_TESTS=0 PYTHONPATH=. python3 tests/test_orchestrated_pipeline.py` and full suite `AGENTZERO_LIVE_TESTS=0 PYTHONPATH=. python3 run_all_tests.py` (24/24 green).

- **Change (2026-06-11): Reworked the logging system — timestamped formatting, clear user/assistant turn lines, in-loop coverage, and errors that bubble from anywhere.** New module `logging_setup.py` centralizes config: `configure_logging()` installs a single timestamped formatter (`%(asctime)s [%(levelname)-7s] %(name)s: %(message)s`), sets the root level from `LOG_LEVEL`, is idempotent (clears existing handlers so re-runs don't stack), and quiets noisy third-party loggers (`aiohttp`, `telegram`, `urllib3`, `httpx`/`httpcore`, `apscheduler`, `asyncio`) to WARNING. `main.py` now calls it instead of bare `logging.basicConfig`. Helpers `preview()` (one-line trimmed text), `log_user_turn()` / `log_assistant_turn()` log on a dedicated `agentzero.turn` logger as `USER ▶ [session] …`, `ASSISTANT ◀ [session] …`, and `ASSISTANT ✖ [session] …` for errors.
  - **Turn logging + catch-all bubbling (`handler.py`):** the public `handle()` is now a thin wrapper around the renamed `_handle_impl()` — it logs the inbound user turn, runs the pipeline inside try/except, logs the outbound reply, and converts ANY exception raised anywhere in the pipeline into a logged (`logger.exception`) clean `Error: …` string instead of crashing the caller. No internal callers used `self.handle`, so the rename is safe.
  - **In-loop coverage (`agentic_loop.py`):** added an iteration-start DEBUG line, surfaced the per-round tool-execution line to INFO with the actual tool names (`Loop iter N: executed K tool call(s): bash, write`), and wrapped the loop body in `_run_agentic_loop_inner` with a catch-all `except Exception` that logs and returns `Error: <exc> [agentic_loop, tools_executed=…]` so a crash anywhere in tool execution/parsing bubbles cleanly (the shared mutable `executed_tool_names` list is still accurate in the handler).
  - **Pipeline coverage (`handler.py`):** the orchestrated pipeline logs each stage boundary at INFO (`planning (advisor=…)`, `worker executing (executor=…, N tools)`, `finalizing reply (advisor=…)`) alongside the existing brief-ready/worker-done/finalize-complete lines.
  Validate: `AGENTZERO_LIVE_TESTS=0 PYTHONPATH=. python3 run_all_tests.py` (24/24 green) and eyeball `LOG_LEVEL=INFO python3 main.py` startup / a live turn for the new `USER ▶` / `ASSISTANT ◀` / `Loop iter` lines.

## Key Functions Reference

| Function                   | Purpose                                    | Location               |
| -------------------------- | ------------------------------------------ | ---------------------- |
| `initialize_agent()`       | Build memory/planning/handler stack        | `main.py:243`          |
| `AgentHandler.handle()`    | Main request processing pipeline           | `handler.py:981`       |
| `api_call_with_retry()`    | API call with retry + stream assembly      | `api.py:237`           |
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
| `/memorystats` | Show current memory cadence and dream-profile status for the active session/chat context.                                                                                                |
| `/memorycadence` | Alias of `/memorystats`.                                                                                                                                                               |

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

- **Pitfall: The model could output a bare tapback reaction word (like/love/dislike/laugh/emphasize/question) as its entire text response instead of calling `send_tapback`, causing the reaction word to be sent as a regular iMessage.**
  **Fix:** Add `is_bare_reaction_word()` detection in `agentic_loop.py` and treat bare reaction words like action-intent narration — nudge the model to either call `send_tapback` or give a normal conversational reply. Also add a safety net in `handler.py`'s `_finalize_visible_response()` that suppresses bare reaction words on iMessage sessions so they never reach the user as text. Add regression coverage in `tests/test_agentic_loop.py` (`test_is_bare_reaction_word`, `test_run_agentic_loop_retries_bare_reaction_word`). Validate with: `PYTHONPATH=. python3 tests/test_agentic_loop.py`.
- **Pitfall: Telegram reaction support was only partially wired, so the model could not see concrete Telegram `chat_id`/`message_id` targets, the public payload schema did not expose `send_telegram_reaction`, and live Telegram turns lacked Sendblue-style typing/reaction parity.**
  **Fix:** Add `send_telegram_reaction` to `BASE_PAYLOAD["tools"]`, thread `telegram_chat_id` / `telegram_message_id` metadata from `integrations.py` into `AgentHandler.handle()`, inject `[Available Telegram reaction targets ...]` context in `handler.py`, and add Telegram-side auto-reaction plus sustained typing indicators during long-running replies. Add regression coverage in `tests/test_telegram_message_flow.py`, `tests/test_memory_maintenance.py`, `tests/test_agentic_loop.py`, and `tests/test_simple.py`. Validate with: `PYTHONPATH=. python3 tests/test_telegram_message_flow.py`, `PYTHONPATH=. python3 tests/test_memory_maintenance.py`, `PYTHONPATH=. python3 tests/test_agentic_loop.py`, and `PYTHONPATH=. python3 tests/test_simple.py`.
- **Pitfall: Telegram voice notes were silently dropped from the user-content path because `_extract_telegram_attachment_urls()` only preserved photos/image documents, so Telegram never reached the existing voice-memo transcription pipeline that Sendblue used.**
  **Fix:** Extend Telegram attachment extraction in `integrations.py` to include `voice`, `audio`, and audio documents, generalize the voice-memo transcription helpers with channel-specific config prefixes, and route Telegram turns through `_build_telegram_user_content()` so voice notes are transcribed before multimodal image handling. Add regression coverage in `tests/test_telegram_voice_memo.py` and register it in `run_all_tests.py`. Validate with: `PYTHONPATH=. python3 tests/test_telegram_voice_memo.py`, `PYTHONPATH=. python3 tests/test_sendblue_voice_memo.py`, and `PYTHONPATH=. python3 tests/test_telegram_message_flow.py`.
- **Pitfall: XML-style pseudo-tool markup like `<read><filepath>...` in planning-heavy live runs bypassed the narrow pseudo-tool detector, so the loop treated it as a generic text-only failure instead of the targeted pseudo-tool retry path.**
  **Fix:** Broaden `contains_pseudo_tool_syntax()` in `agentic_loop.py` to detect XML-style fake tool tags in addition to `<tool(args)>` markup, and add regression coverage in `tests/test_agentic_loop.py` for `<read><filepath>...` style output. Validate with: `PYTHONPATH=. python3 tests/test_agentic_loop.py`.
- **Pitfall: Live recovery-rate batch prompts could create their own context/payload failures when they allowed broad repo searches or whole-file reads, which polluted the measurement with harness artifacts instead of executor behavior.**
  **Fix:** Constrain `scripts/live_sendblue_batch.py` to grep-first, file-scoped prompts plus line-limited `read()` guidance for planning-heavy cases. Validate the bounded harness with: `LOG_LEVEL=WARNING PYTHONPATH=. python3 scripts/live_sendblue_batch.py --case planning_heavy_repo_triage`.
- **Pitfall: Executor-first routing can silently drift if side paths still read plain `MODEL_ID` from the environment instead of the resolved executor model, causing helper behavior (for example multimodal capability checks or model metadata) to reflect the wrong model.**
  **Fix:** Treat `EXECUTOR_MODEL` / `PREFLIGHT_MODEL` as the authoritative default-turn model across the stack, update helper checks accordingly, and inject explicit executor-first consultation guidance into the system prompt. Validate with: `PYTHONPATH=. python3 tests/test_process_response.py` and `PYTHONPATH=. python3 tests/test_simple.py`.
- **Pitfall: Named publish targets that are not actually configured locally (for example `crustyhub`) can still trigger fabricated success language after failed tool attempts, because the executor keeps trying to rationalize the target instead of stopping at verified outcomes.**
  **Fix:** Add publish-result grounding in `agentic_loop.py` so failed publish/deploy tool outputs trigger correction nudges, accumulate failed target context across the run, and prefer deterministic verified summaries when the model still claims success. Validate with: `PYTHONPATH=. python3 tests/test_agentic_loop.py` plus a live prompt probe such as `PYTHONPATH=. python3 scripts/live_sendblue_smoke.py --prompt 'make a website for me that is a fancy 3d dice roller, post it to crustyhub and here.now when ur finished' --expect-substring 'here.now' --require-tool activate_skill --require-tool bash`.
- **Pitfall: The executor can emit XML-style function markup like `<function_activate_skill>...` on the first turn, and without explicit recovery it falls through as plain text instead of entering the tool loop.**
  **Fix:** Extend `infer_tool_calls_from_content()` in `api.py` to parse `<function_NAME><parameter name="...">...</parameter></function_NAME>` payloads into real tool calls, and add regression coverage in `tests/test_process_response.py`. Validate with: `PYTHONPATH=. python3 tests/test_process_response.py` and the live prompt probe for the 3D dice roller publish flow.
- **Pitfall: Publish-grounding heuristics can still misfire if they treat the word `verified` inside phrases like `could not be verified` as a positive success claim, which causes extra retries and can exhaust mocked follow-up responses in tests.**
  **Fix:** Tighten failed-target success detection in `agentic_loop.py` to ignore negated verification phrases and add regression coverage in `tests/test_agentic_loop.py` so corrected outputs like `crustyhub failed and could not be verified` are accepted immediately. Validate with: `PYTHONPATH=. python3 tests/test_agentic_loop.py`.
- **Pitfall: The executor can self-block on a user-provided `skill.md` URL by claiming it "might be prompt injection" in plain text instead of calling `add_skill()`, even though the real installer scanner is the component that should make that decision.**
  **Fix:** Add a skill-URL retry path in `agentic_loop.py` that nudges text-only self-rejections into a real `add_skill()` call, and strengthen `prompts/system_prompt.md` so skill-URL safety decisions defer to the installer's scan result. Validate with: `PYTHONPATH=. python3 tests/test_agentic_loop.py` and `PYTHONPATH=. python3 tests/test_process_response.py`.
- **Pitfall: Default-turn routing through `EXECUTOR_MODEL` broke image inputs because the executor model does not support multimodal user content, while image attachment handling only checked capability after the primary model had already been chosen.**
  **Fix:** Promote `ADVISOR_MODEL` to the default user-facing primary model in `handler.py`, point multimodal capability checks and startup capability metadata at that resolved primary model, and update prompt/test text so the runtime no longer claims executor-first routing. Validate with: `PYTHONPATH=. python3 tests/test_process_response.py`.
- **Pitfall: The core prompt only surfaced memories that semantically matched the current user query, so returning users still felt like cold starts because durable identity/preferences and recent thread continuity were not injected proactively.**
  **Fix:** Add a session continuity brief in `handler.py` that runs before normal query-specific memory retrieval, pulls durable explicit/auto/long-term memories plus recent session history, and instructs the model to default to ongoing-relationship behavior instead of stranger-mode resets. Validate with: `PYTHONPATH=. python3 tests/test_memory_maintenance.py`.
- **Pitfall: Explicit `remember()` calls were stored without the active `session_id`, which weakened continuity ranking for user-specific preferences and identity facts on later turns.**
  **Fix:** Bind `remember_tool()` in `tools.py` to the current runtime session and persist `session_id` in explicit-memory metadata so continuity lookup can prioritize the right user's durable facts. Validate with: `PYTHONPATH=. python3 tests/test_simple.py`.
- **Pitfall: When continuity lookup failed, the system silently dropped back to generic behavior instead of honestly signaling the missing memory path, which made the agent feel forgetful rather than temporarily degraded.**
  **Fix:** Add a graceful-degradation continuity status block in `handler.py` so failed continuity retrieval explicitly tells the model to acknowledge the notes-access problem warmly instead of inventing familiarity. Validate with: `PYTHONPATH=. python3 tests/test_memory_maintenance.py`.
- **Pitfall: Even with stable per-user `session_id`s, messaging turns could still feel like resets because the prompt never explicitly told the model that Telegram and iMessage are single persistent conversations rather than thread-based inboxes.**
  **Fix:** Add an explicit persistent-thread rule in `handler.py` for `tg_` / `imessage_` sessions, reinforce it in the session continuity brief/fallback, and cover it with messaging-specific prompt assertions in `tests/test_memory_maintenance.py`. Validate with: `PYTHONPATH=. python3 tests/test_memory_maintenance.py`.
- **Pitfall: Conversation history queries sorted only by `created_at`, so user/assistant rows written within the same second could come back in nondeterministic order and then be reversed into malformed chronological context.**
  **Fix:** Add `id DESC` as a deterministic tiebreaker in `memory.py` conversation-history queries and cover rolling-context ordering in `tests/test_memory_maintenance.py`. Validate with: `PYTHONPATH=. python3 tests/test_memory_maintenance.py` and `PYTHONPATH=. python3 scripts/live_telegram_smoke.py`.
- **Pitfall: Messaging continuity/query-memory injection could bleed unrelated old memories from other sessions/users into Telegram or iMessage turns because memory selection was not scoped to the active persistent messaging session.**
  **Fix:** Scope continuity and query-memory injection in `handler.py` to memories whose metadata `session_id` matches the active `tg_` / `imessage_` session, then cover the filter in `tests/test_memory_maintenance.py` and recheck with the Telegram shim. Validate with: `PYTHONPATH=. python3 tests/test_memory_maintenance.py` and `PYTHONPATH=. python3 scripts/live_telegram_smoke.py`.
- **Pitfall: Live Telegram replies could succeed while post-response auto-memory silently failed with `Session is closed`, because the normal `handler.handle()` path exited the `aiohttp.ClientSession` context before calling `_run_memory_maintenance()`.**
  **Fix:** Move normal-path conversation persistence and `_run_memory_maintenance()` inside the active `aiohttp.ClientSession` scope in `handler.py`, and add regression coverage in `tests/test_memory_maintenance.py` (`test_handle_runs_memory_maintenance_before_closing_http_session`). Validate with: `PYTHONPATH=. python3 tests/test_memory_maintenance.py` and `LOG_LEVEL=DEBUG timeout 60s python3 main.py`.
- **Pitfall: Live Telegram image attachments could 404 before conversion because `_telegram_file_url()` always prepended the Telegram file API base, even when the bot library already returned an absolute `file_path`, producing malformed URLs like `https://.../https://...`.**
  **Fix:** Normalize Telegram `file_path` values in `integrations.py`: return absolute URLs as-is and only prepend the bot file base for relative paths. Add regression coverage in `tests/test_telegram_voice_memo.py` (`test_telegram_file_url_preserves_absolute_file_paths`). Validate with: `PYTHONPATH=. python3 tests/test_telegram_voice_memo.py` and a live `LOG_LEVEL=DEBUG timeout 120s python3 main.py` Telegram photo probe.
- **Pitfall: After the Telegram image fix, the model could still leak markdown pseudo-tool directives like `*send_telegram_reaction: ...*` into live replies instead of making a real reaction tool call, because pseudo-tool detection only covered angle-bracket/XML styles.**
  **Fix:** Extend pseudo-tool detection in `agentic_loop.py` to catch wrapped markdown-style tool directives, and add a final-response safeguard in `handler.py` that strips leaked `send_tapback` / `send_telegram_reaction` directive lines if they somehow reach visible output. Add regression coverage in `tests/test_agentic_loop.py` (`test_run_agentic_loop_retries_markdown_reaction_pseudo_tool`) and `tests/test_memory_maintenance.py` (`test_handle_strips_pseudo_reaction_directives_from_response`). Validate with: `PYTHONPATH=. python3 tests/test_agentic_loop.py` and `PYTHONPATH=. python3 tests/test_memory_maintenance.py`.
- **Pitfall: When users said things like `your name is Alice, remember that`, the model could call `remember()` with the wrong subject (`User's name is Alice`), which poisoned continuity and made later Telegram/iMessage turns confuse the user's identity with the assistant's.**
  **Fix:** Add subject-aware memory normalization in `tools.py` and `handler.py`: use the live user turn to rewrite obvious assistant-identity memories into canonical content (`The assistant's name is Alice.`), persist `assistant_identity` / `assistant_name` metadata, and reinforce the rule in the memory tool descriptions and auto-memory extractor prompt. Validate with: `PYTHONPATH=. python3 tests/test_simple.py` and `PYTHONPATH=. python3 tests/test_memory_maintenance.py`.
- **Pitfall: Even after users corrected assistant identity, stale contradictory memories like `User's name is Alice` could still be injected into prompts and keep the confusion alive.**
  **Fix:** Add an `[Assistant identity note]` in `handler.py` sourced from recent session memories, and suppress conflicting user-name memories when a same-session assistant-name memory exists. Cover with `tests/test_memory_maintenance.py` (`test_handle_injects_assistant_identity_and_filters_conflicting_user_name_memory`). Validate with: `PYTHONPATH=. python3 tests/test_memory_maintenance.py`.
- **Pitfall: Fixing assistant-name persistence at write time alone was not enough for sessions that already contained a bad `User's name is Alice` memory, because the correction might not be stored yet.**
  **Fix:** Let `handler.py` infer assistant identity directly from recent user turns (for example `your name is alice`) and use that inferred identity to inject an assistant-name note and suppress conflicting user-name memories until durable memory catches up. Cover with `tests/test_memory_maintenance.py` (`test_handle_infers_assistant_identity_from_recent_history`). Validate with: `PYTHONPATH=. python3 tests/test_memory_maintenance.py`.
- **Pitfall: Claude Agent SDK `output_format` parameter caused query failures; structured output had to be parsed from the `ResultMessage.result` text instead of `structured_output` field.**
  **Fix:** Remove `output_format` from `ClaudeAgentOptions`, use plain text prompting with explicit JSON structure instructions, and parse the JSON from `ResultMessage.result` using regex to extract the JSON object when Claude wraps it in explanatory text. The `_run_query` method in `self_heal.py` now extracts JSON objects from result text using `\{[\s\S]*\}` pattern matching. Validate with: `PYTHONPATH=. python3 tests/test_self_heal.py`.
- **Pitfall: The self-heal subsystem relied on the Claude Agent SDK (`claude_agent_sdk`) which required a separate subscription and could fail silently with `ImportError`. Replacing it with the agent's own NVIDIA API eliminates the external dependency.**
  **Fix:** Replace `ClaudeAgentSDKHealer` with `AgentSelfHealer` in `self_heal.py` that calls the agent's own NVIDIA API via `api_call_with_retry` with a minimal payload (`tools=[]`, `temperature=0.3`, `stream=False`). The healer parses JSON patch responses using the same `\{[\s\S]*\}` extraction pattern. Update `SELF_HEAL_MODEL` env var (was `SELF_HEAL_CLAUDE_MODEL`). Validate with: `PYTHONPATH=. python3 tests/test_self_heal.py`.
- **Pitfall: NVCF asset upload for large images returned 403 on the S3 PUT because the same `aiohttp.ClientSession` (which may carry auth headers) was reused for the pre-signed upload URL, and the content type was not corrected to `image/jpeg` after ImageMagick conversion.**
  **Fix:** Use a fresh `aiohttp.ClientSession()` for the S3 PUT (no auth headers on pre-signed URLs), always pass `image/jpeg` as the upload content type (since ImageMagick always converts to JPEG), and keep `NVCF_MODEL_IDS` empty until a model genuinely requires NVCF asset refs. All models in `MULTIMODAL_MODEL_IDS` use base64 `image_url` blocks instead. Validate with: `PYTHONPATH=. python3 tests/test_multimodal_integrations.py`.
- **Pitfall: Large inbound images from iMessage/Telegram could exceed API payload limits because ImageMagick conversion only re-encoded to JPEG at quality 95 without resizing.**
  **Fix:** Add `MAX_IMAGE_DIMENSION = 2048` and pass `-resize 2048x2048>` to ImageMagick so images larger than 2048px on any side are downsampled, and reduce quality from 95 to 92. This keeps base64 payloads within reasonable limits for all multimodal models. Validate with: `PYTHONPATH=. python3 tests/test_multimodal_integrations.py`.
- **Pitfall: Multimodal integration tests relied on setting `os.environ["MODEL_ID"]` to control model behavior, but `PRIMARY_MODEL_ID` is cached at import time and never re-read from env, so the tests silently tested the wrong model path.**
  **Fix:** Rewrite multimodal tests to use `unittest.mock.patch` on `_model_needs_nvcf_assets()` and `_model_supports_multimodal()` instead of mutating env vars, and add direct unit tests for `_build_user_message_content_from_normalized` with NVCF and multimodal paths. Validate with: `PYTHONPATH=. python3 tests/test_multimodal_integrations.py`.
- **Pitfall: The self-heal prompt told the model "Read the relevant source files" but the API payload had `tools=[]`, so the model could not actually read any files.**
  **Fix:** Include relevant source file contents inline in the prompt (`AgentSelfHealer._build_prompt` reads `tools.py` from the worktree and includes it). Validate with: `PYTHONPATH=. python3 tests/test_regression_self_heal_e2e.py`.
- **Pitfall: `_extract_content_value` only checked for a `content` key in tool results, so tools like `bash_tool` (uses `stdout`) and `grep_tool` (uses `matches`) would always be flagged as empty-output regressions.**
  **Fix:** Also check for `stdout` and `matches` keys, and treat explicitly `null` values as empty strings. Validate with: `PYTHONPATH=. python3 tests/test_regressions.py`.
- **Pitfall: The `run_agentic_loop()` and `process_response()` entry points lacked a `session_id` parameter, so regression reports had no session identity for deduplication.**
  **Fix:** Add `session_id` parameter to both functions, thread it through `agentic_loop.py` and `api.py`, and include it in `RegressionReport`. Validate with: `PYTHONPATH=. python3 tests/test_regression_self_heal_e2e.py`.
- **Pitfall: Mocked API self-heal tests showed the full pipeline works, but live NVIDIA API calls with complex prompts (tools.py inlined) can take >180s, causing timeout.**
  **Fix:** Increase `SELF_HEAL_TIMEOUT_SECONDS` env var to 300, or use a faster model via `SELF_HEAL_MODEL`. The pipeline itself (detection → format → heal_manager → worktree → fix) is verified with mocked responses. Validate with: `PYTHONPATH=. python3 tests/test_regression_self_heal_e2e.py`.

### Session Pitfalls + Fixes (2026-06-21)

- **Pitfall: Every tool-free inference path was failing because the provider now rejects an empty `tools` array.** The live error log showed 24× `Error: orchestrator plan failed: {'message': '1 validation error: ... Value error, \`tools\` must not be an empty array. Either provide at least one tool or omit the field entirely.'}`. Many call sites express "no tools" as `payload["tools"] = []` (orchestrator plan at `handler.py:3960`, advisor/consult, consortium judges, `_run_direct_ai_inference` reminders, `self_heal.py:324`). The provider used to tolerate `[]` but now 422s the whole request, so the orchestrator plan stage silently fell back to the single-model path and other tool-free paths (advisor, reminders, self-heal) failed outright.
  **Fix:** Sanitize centrally at the single send point in `api.py::api_call_with_retry` — after copying `json_data`, if `request_payload.get("tools")` is falsy, `pop("tools")` and `pop("tool_choice")` so the wire request omits the field while callers can keep passing `tools=[]`. This covers every current and future caller (all inference routes through `api_call_with_retry`, including `agentic_loop._attempt_graceful_finish`, which already dropped the key). Regression: `tests/test_process_response.py::test_api_call_omits_empty_tools_array` and `::test_api_call_keeps_nonempty_tools_array`. Validate with: `AGENTZERO_LIVE_TESTS=0 PYTHONPATH=. python3 tests/test_process_response.py` and full suite `AGENTZERO_LIVE_TESTS=0 PYTHONPATH=. python3 run_all_tests.py` (24/24 green).
  **Operational note:** `api.py` binds at import time, so a running `main.py` daemon (`agentzero.pid`) must be restarted (`python3 main.py --stop` then start) to pick up this fix.

- **Pitfall: A user asked the bot to "export every memory" and it replied "no memories found ... recall returning 0 results across all queries," which looked like the memory store had been wiped.** It had NOT: forensics on `agent_memory.db` showed 140 memories, 2063 conversations, 215 topics, 14 reminders fully intact. Root cause: `memory.search_memories()` is keyword-LIKE matching (no embeddings at query time), and vague/enumeration queries ("export every memory", "memory", "name" — note `name` is in the stopword list) extract no usable keyword that matches any stored `content`, so they returned `[]`. The model issued generic queries, got 0 every time, and wrongly concluded the store was empty. There was also no enumeration tool exposed — `get_recent_memories` existed in `tools.py`/`TOOLS` but was never added to `BASE_PAYLOAD["tools"]`, so the model literally could not list/dump all memories. (Diagnostic gotcha: a naive `for name in cursor.execute("SELECT name FROM sqlite_master..."): cursor.execute("COUNT...")` reuses one cursor and the inner query clobbers the outer loop, making it look like only the first table exists — use a separate cursor or fetch the table list first.)
  **Fix:** (1) In `tools.py::recall_tool`, when keyword search returns no memories, fall back to `get_recent_memories(limit=max(top_k,10))` and return them with an explicit `note` ("No direct keyword matches ... The memory store is NOT empty. Use get_recent_memories to enumerate") and `similarity: 0.0`, so an explicit recall never falsely reads as an empty store and the model won't claim the fallback rows matched the query. The continuity/query-injection path (`handler.py` `search_memories(query=user_query, ...)`) is deliberately left unchanged to avoid surfacing unrelated memories on trivial turns. (2) Expose `get_recent_memories` in `handler.py` `BASE_PAYLOAD["tools"]` (optional `limit`, no required args) as the first-class enumerate/export path. Validate with: `AGENTZERO_LIVE_TESTS=0 PYTHONPATH=. python3 run_all_tests.py` (24/24) and a direct probe: `recall_tool(query="name")` now returns recent memories with a `note`, `get_recent_memories_tool(limit=200)` returns all 140.

- **Audit (2026-06-21): swept every payload-exposed tool for "works technically, breaks on a natural call" defects (the class the memory recall bug belonged to).** Method: introspect `handler.BASE_PAYLOAD["tools"]` schemas vs the actual `tools.TOOLS` function signatures via `inspect.signature` — flag (a) tools exposed to the model but absent from `TOOLS` (would silently produce no tool result), (b) schema params the function can't accept and that have no `**kwargs` (natural call → `TypeError`), and (c) function-required params the schema doesn't mark required (natural call omits them). Findings + fixes:
  - **`consult_advisor` / `consult_reviewer`: exposed in `BASE_PAYLOAD` but never in `TOOLS` and never dispatched** (no handler in `agentic_loop`/`execute_tool_calls`), so any call silently appended no tool result. They were also unused and token-wasting. **Nuked entirely** per user: removed both `BASE_PAYLOAD` schema blocks, the `AgentHandler.consult_advisor()` / `consult_reviewer()` methods, the complex/low-complexity prompt nudges that referenced them, and the now-orphaned constants `REVIEWER_MODEL_ID`, `ADVISOR_RESPONSE_MAX_TOKENS`, `REVIEWER_RESPONSE_MAX_TOKENS` (kept `ADVISOR_MODEL_ID` — still used by the orchestrated pipeline). Payload tool count 26 → 24.
  - **`consortium_start`: schema advertised `context`, `members`, `max_rounds` but `consortium_start_tool(task, task_id)` accepts none of them and had no `**kwargs`** — if the model passed what the schema told it to, the call `TypeError`d. Fix: trimmed the schema to the only real param (`task`) and added tolerant `**_ignored` to `consortium_start_tool` as a safety net (same pattern as the earlier `grep_tool`/`read_file_tool` natural-arg fixes).
  - Re-ran the audit after fixes: zero exposed-but-missing tools, zero schema/signature mismatches across all 24 exposed tools. Validate with: `AGENTZERO_LIVE_TESTS=0 PYTHONPATH=. python3 run_all_tests.py` (24/24 green) and the inline audit script (introspects BASE_PAYLOAD vs TOOLS).

- **Change (2026-06-21): Added interactive browser automation via CloakBrowser (stealth Chromium) and removed the never-wired Obscura browser.** Context: the user asked to "use this browser over the one we have currently" (https://github.com/CloakHQ/CloakBrowser). Investigation found there was **no browser tool wired into the agent at all** — Obscura existed only as docs (`docs/OBSCURA_*.md`) + an 80MB binary (`/usr/local/bin/obscura`), never implemented in `tools.py`/`handler.py`. CloakBrowser (`pip install cloakbrowser`, v0.4.0) is a Playwright/Puppeteer drop-in; the async API was verified live before coding: `cloakbrowser.ensure_binary()` → path, `await cloakbrowser.launch_async(headless=True)` → Playwright async `Browser`, then `browser.new_page()` / `page.goto(url, wait_until=...)` (`.status`) / `page.title()` / `page.inner_text(sel)` / `page.fill(sel,text)` / `page.click(sel)` / `page.screenshot(path=...)` / `browser.close()`. No license key needed for basic use. The chromium binary auto-downloads (~200MB) to `~/.cloakbrowser/` on first `ensure_binary()` (cache shared across venvs).
  - **Implementation (`tools.py`):** a process-wide persistent session `_browser_session = {"browser", "page"}` guarded by an `asyncio.Lock`, plus 7 async tools — `browser_open`, `browser_navigate` (alias `browser_goto`, auto-opens the session), `browser_click`, `browser_type` (alias `browser_fill`, uses `page.fill`), `browser_read` (returns visible text, capped by `max_chars`), `browser_screenshot` (saves PNG to `workspace/browser_screenshots/`, returns path — tool results are text-only in this arch, so screenshots are paths not inline images), `browser_close`. Action tools (click/type/read/screenshot) return `{"success": False, "error": "No browser page open. Call browser_navigate first."}` when no session exists rather than auto-launching. `cloakbrowser` is imported lazily inside `_ensure_browser_page()` (it pulls in playwright) so module import stays cheap and `ImportError` is reported cleanly. All tools take `**_ignored` for natural-arg tolerance (per the 2026-06-21 audit pattern).
  - **Wiring:** registered in `TOOLS` + `validate_tool_args()` (`tools.py`), and exposed in `BASE_PAYLOAD["tools"]` (`handler.py`, inserted after `web_search`). Payload tool count 24 → 31 (7 new schemas; aliases `browser_goto`/`browser_fill` are in `TOOLS` but intentionally NOT in the visible schema — the api.py payload-scoped filter would reject them, so the model is taught only the canonical names).
  - **Obscura removed** per the user: deleted `docs/OBSCURA_SUMMARY.md`, `docs/OBSCURA_INSTALL.md`, and `sudo rm /usr/local/bin/obscura`. (A stray copy under the untracked `selfheal_unknown_makynw71/` self-heal worktree was left as-is — not part of the project.) `cloakbrowser` added to `requirements.txt`.
  - **Tests (`tests/test_simple.py`):** `test_browser_tools_registered` (registry + required-arg validation + BASE_PAYLOAD exposure), `test_browser_tools_require_open_session` (graceful no-session errors, no launch), and a live-gated `test_live_browser_navigate` (`AGENTZERO_LIVE_TESTS=1`) that launches → navigates example.com → reads → screenshots → closes. Validate: `AGENTZERO_LIVE_TESTS=0 PYTHONPATH=. python3 run_all_tests.py` (24/24 green) and `AGENTZERO_LIVE_TESTS=1 PYTHONPATH=. python3 -c "import asyncio; from tests.test_simple import test_live_browser_navigate as t; asyncio.run(t())"` (passes: navigate status 200, read 129 chars, screenshot written).
  - **Operational note:** `tools.py`/`handler.py` bind at import time, so the running `main.py --daemon` (pid in `agentzero.pid`) must be restarted (`python3 main.py --stop` then start) to expose the browser tools to live traffic. Also: the session is shared process-wide (single-user agent) — iMessage and Telegram share one browser; acceptable here but note for any future multi-user split.

- **Change (2026-06-21): Native multimodal — stop forwarding images to the Mistral describer, and feed browser screenshots into the model's context.** The `.env` model (`MODEL_ID=stepfun-ai/step-3.7-flash`) is natively multimodal, but the code had two gaps: (1) `_build_user_message_content_async` in `integrations.py` *always* forwarded inbound images to the `IMAGE_DESCRIBER_MODEL` (`mistralai/mistral-small-4-119b-2603`) and fed back text descriptions — even for multimodal primaries — because `step-3.7-flash` wasn't in `MULTIMODAL_MODEL_IDS` (so `_model_supports_multimodal()` returned False); and (2) `browser_screenshot` only returned a file *path*, which the model can't see.
  - **Fix 1 — native inbound images (`integrations.py`):** added `stepfun-ai/step-3.7-flash` to `MULTIMODAL_MODEL_IDS`, and rewrote `_build_user_message_content_async` so that when `_model_supports_multimodal()` is True it builds native `image_url` content blocks via `_attachment_url_to_base64_data_url` (resized JPEG base64 data URLs) and returns a `list[dict]` — the describer is NOT called. Non-multimodal models keep the existing Mistral-describe-to-text fallback. If every conversion fails it degrades to a text note rather than dropping the turn. Return type widened to `str | list[dict]` (channel callers already accept block lists; the iMessage/Telegram builders at `integrations.py` `_build_imessage_user_content`/`_build_telegram_user_content` pass it straight through). The sync `_build_user_message_content` / `_build_user_message_content_from_normalized` were left unchanged — they have no non-test callers.
  - **Fix 2 — browser screenshots into context (`agentic_loop.py`):** after `messages.extend(tool_results)`, `_inject_browser_screenshots()` scans the round's tool results for successful `browser_screenshot` calls (matched by `tool_call_id` → name), and *only if the running model (`base_payload["model"]`) is in `MULTIMODAL_MODEL_IDS`* appends a `role:"user"` message carrying the screenshot as an `image_url` block (read from disk, resized JPEG base64 via the lazily-imported `integrations._resize_image_to_jpeg_sync`). New helpers: `_model_supports_multimodal_blocks`, `_browser_screenshot_paths`, `_screenshot_file_to_data_url`, `_inject_browser_screenshots` (all lazy-import `integrations` to avoid the `agentic_loop → integrations → handler → agentic_loop` import cycle). Caveat: injected screenshots persist in `messages` and re-send on later rounds (bounded by the JPEG resize); no eviction added.
  - **Gotcha:** `PRIMARY_MODEL_ID`/`MODEL_ID` resolve from `os.environ["MODEL_ID"]` which is only set once `.env` is loaded (done in `main.py` via dotenv). A bare `import handler` without loading `.env` shows the hardcoded fallback (`moonshotai/kimi-k2.6`), not the real model — load `.env` first when probing.
  - **Tests:** `tests/test_multimodal_integrations.py` updated to the new contract — the two async describer tests now run with `_model_supports_multimodal=False` (describer is the non-multimodal fallback), `test_build_user_message_content_async_multimodal_returns_image_blocks` asserts a block list with `image_url` blocks and that the describer is never called, and `test_async_builder_falls_back_to_text_when_conversion_fails` covers the all-fail degradation. `tests/test_agentic_loop.py::test_inject_browser_screenshots` covers inject-for-multimodal vs skip-for-text-model. Validate: `AGENTZERO_LIVE_TESTS=0 PYTHONPATH=. python3 run_all_tests.py` (24/24 green). **Live-validated:** POSTed an `image_url` block (a real browser screenshot, resized to 23 KB JPEG) to `step-3.7-flash` → HTTP 200 and the model's `reasoning_content` accurately described the screenshot ("a screenshot of a plain webpage … heading 'Example Domain' …"), confirming both the native-inbound path and screenshot perception. (`step-3.7-flash` is a reasoning model that emits `reasoning_content` before `content`; a too-small `max_tokens` can leave `content` null with `finish_reason: length` — not a multimodal bug.)
  - **Operational note:** restart the daemon (`python3 main.py --stop` then start) for `integrations.py`/`agentic_loop.py` import-time changes to take effect.

- **Change (2026-06-21): Taught the model it has a browser so "use your browser" routes to `browser_*`, not `web_search` (Exa).** Tool schemas alone weren't enough steering — added a `BROWSER` section to `prompts/system_prompt.md` that names the `browser_*` tools, states the model controls a real stealth browser (and can see screenshots), and explicitly says to use them for "use your browser"/open/navigate/click/fill/log-in/see-a-page requests while reserving `web_search` for quick text lookups (it cannot open or act on a page). **Live-validated** with `scripts/live_sendblue_smoke.py --prompt "use your browser to go to example.com and tell me the exact heading on the page" --require-tool browser_navigate`: the model called `browser_navigate` → `browser_read` → replied "The exact main heading on example.com is Example Domain." with **no `web_search` call** (`missing_tools: []`; the harness `success:false` is only the default expected-substring check, which doesn't apply to a custom `--prompt`).

- **Pitfall: Voice-memo transcription was failing on every inbound audio with `Unavailable model requested given these parameters: language_code=en; sample_rate=0; type=offline;`.** Two root causes in `integrations.py`: (1) `NVIDIA_WHISPER_FUNCTION_ID` pointed at a dead/deprecated NVCF function (`d8dd4e9b-fbf5-4fb0-9dba-8cf436c8d965`) — that exact id reproduces the "Unavailable model requested" error live; and (2) `RecognitionConfig` never set `sample_rate_hertz` or `encoding`, so the hosted model couldn't be matched (`sample_rate=0`). The flow also tried the **raw** m4a/opus bytes first and only converted to WAV for native iMessage m4a — so Telegram opus/ogg never converted at all.
  **Fix:** switched to NVIDIA **Parakeet 1.1B RNNT multilingual ASR** (`NVIDIA_WHISPER_FUNCTION_ID = "71203149-d3b7-4460-8231-1be2543a1fca"`, same `grpc.nvcf.nvidia.com:443` gateway). Parakeet is PCM-only, so `_transcribe_audio_bytes_with_whisper` now **always** converts inbound audio to 16 kHz mono LINEAR_PCM WAV via `_convert_m4a_audio_with_ffmpeg_sync` (ffmpeg auto-detects m4a/caf/opus/ogg/mp3/…) before transcribing — dropping the doomed raw-first attempt and fixing Telegram opus/ogg. `_transcribe_audio_bytes_with_whisper_sync` gained a `sample_rate_hertz` param (new constant `VOICE_MEMO_SAMPLE_RATE_HZ = 16000`) and now sets `encoding=riva.client.AudioEncoding.LINEAR_PCM` + `sample_rate_hertz=16000` in the config. (Riva proto field is `encoding`, NOT `audio_encoding`; `AudioEncoding.LINEAR_PCM` is accessible by attribute even though it's absent from `dir()`.)
  **Empirically determined the config** before coding: probed the live gateway with a 16 kHz tone WAV — Parakeet + `sample_rate_hertz=16000` + `LINEAR_PCM` → request accepted; Parakeet + `sample_rate=0` → "input format doesn't match"; old id + correct config → the original "Unavailable model requested" error. **Live round-trip validated** end-to-end: a stereo 44.1 kHz Ogg Vorbis speech clip ran through the real pipeline (download → ffmpeg → Parakeet) and returned an accurate transcript ("This is an example sound file in … vorbis format from wikipedia, the free encyclopedia"). Tests updated in `tests/test_sendblue_voice_memo.py` (`test_transcribe_audio_bytes_converts_to_wav_then_transcribes`, `test_transcribe_audio_bytes_returns_none_when_conversion_fails`, and `test_transcribe_audio_bytes_uses_parakeet_defaults` now asserts the Parakeet id + `16000`) and `tests/test_telegram_voice_memo.py` (mocks the converter, asserts trailing `16000`). `_is_native_imessage_m4a` is no longer used in the transcribe flow but is kept (still unit-tested). Validate: `AGENTZERO_LIVE_TESTS=0 PYTHONPATH=. python3 tests/test_sendblue_voice_memo.py`, `… tests/test_telegram_voice_memo.py`, full suite 24/24. Overridable via `SENDBLUE_/TELEGRAM_VOICE_MEMO_FUNCTION_ID` / `_LANGUAGE` env. **Restart the daemon** for `integrations.py` import-time changes to take effect.
