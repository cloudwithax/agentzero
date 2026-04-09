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
