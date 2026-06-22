# Dossier: Tool-Call History Compaction

**Status:** ✅ Implemented 2026-06-04 (`agentic_loop.py` `_compact_executed_tool_call_args` / `_compact_arguments_string`; tests `tests/test_agentic_loop.py` C1–C6). Live-validated: compaction fires each golden publish run (reclaiming ~5–12 KB after the `write` round) with no behavioral regression — the site still publishes and the link is reported.
**Author:** Agent pairing session, 2026-06-04
**Related:** `agentic_loop.py`, `api.py` (`execute_tool_calls`), `handler.py` (`_build_rolling_context`)
**Origin:** Follow-up to the build-and-publish golden-test hardening (see AGENTS.md, 2026-06-04). The fixes there (transient-error retry + streaming buffering) made the flow *reliable*; this dossier addresses the *root stressor* that triggered the failure in the first place.

---

## 1. Problem statement

During a multi-step task, the agentic loop accumulates the full conversation in `messages` and re-sends it on every round. The accumulation happens here:

```python
# agentic_loop.py  (end of each tool round)
messages.append(message)        # assistant turn, INCLUDING its tool_calls
messages.extend(tool_results)   # role:"tool" results
```

When the model calls a tool with a **large argument payload** — most notably `write(filepath, content)` where `content` is an entire HTML/CSS/JS file — that payload is carried in `message["tool_calls"][i]["function"]["arguments"]` (a JSON string). It then lives in `messages` for the rest of the run and is re-uploaded on **every subsequent API call**.

Concretely, the golden build-and-publish flow does:

1. `activate_skill` → 2. `activate_skill` → 3. `write` (big HTML, e.g. 5–40 KB) → 4. `bash` (run publish) → 5. final reply.

From round 3 onward, every request re-sends the full HTML twice over (once in the assistant `write` tool-call arguments, and partially again in the tool result echo). This is what stressed the provider into intermittently returning a truncated/malformed `{"error":{"message":"Unterminated string ..."}}` body — the failure the retry fix now papers over.

### Why it matters
- **Reliability:** larger requests raise the odds of the transient upstream errors we just had to add retries for. Shrinking the payload removes the trigger, not just the symptom.
- **Cost / latency:** every extra round re-bills the same tens of KB of input tokens. A 4-round publish re-sends the HTML ~3 times. Longer sessions compound this.
- **Context budget:** `handler._build_rolling_context` (context_window=128000, buffer=2000) drops *old history messages* to fit. A single huge `write` can crowd out genuinely useful recent turns from the rolling window.

---

## 2. Root cause

The loop treats tool-call arguments and tool results as immutable history. That is correct for *semantic* content (the model needs to remember what it did), but the **verbatim bytes** of a written file are not needed once the tool has executed:

- The model does not need the full HTML re-fed to know "I wrote index.html." A short descriptor (`<wrote 8,412 bytes to index.html>`) preserves the *fact* without the *payload*.
- Tool *results* for `write` are already small (`{"success": true, ...}`); the bloat is overwhelmingly the **assistant's own tool-call `arguments`**.

So the highest-value, lowest-risk target is: **after a tool round executes, replace large tool-call `arguments` in the retained assistant message with a compact placeholder**, while keeping the real arguments only for the round in which they execute.

---

## 3. Goals / non-goals

**Goals**
- G1. Stop re-sending large tool-call argument payloads on rounds after they executed.
- G2. Preserve the model's awareness of what it did (tool name, target, size) so behavior is unchanged.
- G3. Keep it generic — no special-casing of any one skill or hosting provider; key on argument *size*, not tool identity. (`write` is the common case but `bash` heredocs, `edit`, etc. can also be large.)
- G4. Be measurable: log how many bytes/tokens were reclaimed.

**Non-goals**
- N1. Not touching the *current* round's payload — the tool must execute with full fidelity.
- N2. Not compacting tool *results* in v1 (they're usually small; revisit only if data shows otherwise).
- N3. Not a general context-summarization/dream-cycle feature — this is mechanical, lossless-of-meaning trimming, not semantic summarization.
- N4. Not changing `_build_rolling_context` token math (it benefits automatically once messages are smaller).

---

## 4. Proposed design

### 4.1 Where
A single new helper invoked in the loop **after** `execute_tool_calls` returns and **before/at** the point the assistant message is appended to history:

```python
# agentic_loop.py, replacing the bare append at the end of the round
messages.append(_compact_executed_tool_call_args(message))
messages.extend(tool_results)
```

Crucially we compact the **copy that goes into history**, not the `message` object passed to `execute_tool_calls` (which already ran with full args). Use a shallow copy + replaced `tool_calls` list so we never mutate the live object or the provider's response.

### 4.2 What gets compacted
For each tool call in the assistant message, if the serialized `arguments` string exceeds a threshold (`COMPACT_TOOL_ARG_THRESHOLD`, default ~2,000 chars), replace large **string-valued** fields inside the parsed arguments with a placeholder descriptor; leave small scalar fields (filepath, flags, slugs) intact.

Example transform (conceptual):

```
BEFORE  write {"filepath":"index.html","content":"<!doctype html><html>... 8KB ..."}
AFTER   write {"filepath":"index.html","content":"<elided 8412 chars — written in round 3>"}
```

Rules:
- Only fields whose string value length ≥ `COMPACT_FIELD_THRESHOLD` (default 1,000 chars) are elided.
- Placeholder is human/model-readable and states the size + round, so the model still "knows" it wrote a large file.
- Non-string args (numbers, bools, small strings, nested small objects) are untouched.
- If `arguments` isn't valid JSON (rare/recovered calls), fall back to truncating the raw string with a clear `…[elided N chars]` suffix.
- Idempotent: re-compacting an already-compacted message is a no-op (placeholders are below threshold).

### 4.3 Config (env-gated, default ON)
| Var | Default | Meaning |
|---|---|---|
| `AGENTZERO_COMPACT_TOOL_ARGS` | `1` | Master switch (set `0` to disable). |
| `AGENTZERO_COMPACT_TOOL_ARG_THRESHOLD` | `2000` | Min serialized-arguments length (chars) before a call is considered. |
| `AGENTZERO_COMPACT_FIELD_THRESHOLD` | `1000` | Min individual string-field length (chars) to elide that field. |

Defaults chosen so normal small calls (bash one-liners, reads, reactions) are never touched; only genuine file-sized payloads are.

---

## 5. Alternatives considered

| Option | Verdict |
|---|---|
| **A. Compact assistant tool-call args after execution (proposed).** | ✅ Best. Targets the actual bloat, preserves meaning, generic by size, low risk. |
| B. Compact tool *results* instead. | ❌ Results for `write` are already tiny; misses the real payload (the args). Revisit only if profiling shows large results (e.g. big `read`/`grep` outputs — those are a separate, weaker case). |
| C. Drop the assistant tool-call message entirely after execution, keep only the result. | ❌ Breaks the OpenAI tool-call contract (a `role:"tool"` message must follow an assistant message that contains the matching `tool_call_id`); some providers 400 on orphaned tool results. |
| D. Don't store written content in args at all — make `write` take a reference. | ❌ Large protocol change; the model emits args directly, can't pre-reference. |
| E. Rely solely on `_build_rolling_context` trimming. | ❌ That trims *oldest* whole messages; a huge `write` in the *recent* window survives and still gets re-sent every round until it ages out. Doesn't address per-round re-upload. |
| F. Semantic summarization of history. | ❌ Overkill, lossy, slow, and risks changing behavior. Out of scope (N3). |

---

## 6. Edge cases & risks

- **R1. Model needs to re-read what it wrote.** Mitigation: it can `read(filepath)` — the file is on disk. The placeholder names the filepath. Low risk because the model rarely needs the verbatim content of its own prior write mid-task; it works forward from "I wrote X."
- **R2. Provider validation of tool-call arguments on resend.** Some providers re-validate that resent assistant `tool_calls[].arguments` is well-formed JSON. Our placeholder keeps it valid JSON (we re-`json.dumps` the modified dict). The raw-string fallback path (invalid-JSON case) is the only one that yields non-JSON args, and those came from recovery paths that providers already tolerate as free text. Verify against the live provider.
- **R3. `tool_call_id` continuity.** We do not touch IDs, names, or the result messages — only the *values inside* `arguments`. The assistant↔tool pairing is preserved.
- **R4. Multi-tool rounds.** A round can contain several tool calls; compact each independently.
- **R5. Streaming/buffer interplay.** Orthogonal — compaction runs on the assembled message after the round, well after streaming assembly.
- **R6. Idempotency under retries.** The loop may re-append on retry paths; ensure compaction is idempotent (placeholders are sub-threshold) and only applied to the history copy.

---

## 7. Implementation plan

1. **`agentic_loop.py`:**
   - Add `_compact_executed_tool_call_args(message: dict) -> dict` (pure function, returns a new dict; never mutates input).
   - Add the three `AGENTZERO_COMPACT_*` env reads as module constants.
   - At the end of the tool round, change `messages.append(message)` → `messages.append(_compact_executed_tool_call_args(message))`.
   - `log()`/`logger.info` a one-line reclaim summary: `compacted N tool arg(s), reclaimed ~K chars`.
2. **Keep the live `message` passed to `execute_tool_calls` untouched** (it already executed with full args; the only change is what we *retain*).
3. **No handler changes required** — `_build_rolling_context` automatically sees smaller messages.

Estimated size: ~50–70 LoC + tests. Single file.

---

## 8. Testing plan

**Unit (`tests/test_agentic_loop.py`):**
- `test_compact_tool_args_elides_large_write_content` — a `write` with 8 KB `content` → retained arguments contain a `<elided …>` placeholder, `filepath` preserved, result JSON still parses.
- `test_compact_tool_args_preserves_small_calls` — `bash {"command":"echo hi"}` is byte-for-byte unchanged.
- `test_compact_tool_args_does_not_mutate_input` — the original `message` object is unchanged (full args intact) after compaction.
- `test_compact_tool_args_idempotent` — compacting twice == compacting once.
- `test_compact_tool_args_invalid_json_fallback` — non-JSON args get truncated with an explicit suffix, no exception.
- `test_compact_tool_args_multi_call_round` — each call in a multi-tool message handled independently.

**Live (smoke):**
- Re-run the golden publish flow ≥3×; assert still 3/3 success **and** capture per-run total request bytes (before/after) to quantify the reclaim. Expect the post-`write` rounds to shrink by ~the file size.
- A long multi-write session (write 3 files, then publish) to confirm cumulative savings and no "model forgot what it wrote" regressions.

**Observability:** log reclaimed bytes per round; optionally surface a session total in the smoke harness's `agentic_metrics`.

---

## 9. Success criteria
- Post-`write` request payloads shrink by approximately the written file size (measured).
- Golden publish flow stays 3/3 (or better) live.
- No behavioral regression: the model still proceeds to publish and reports the link correctly.
- Full non-live suite stays green (`run_all_tests.py`).

## 10. Open questions
- Q1. Should we *also* compact very large tool *results* (e.g. a `read` of a big file, or verbose `grep`)? Defer to v2 pending profiling — same helper generalizes to result `content`.
- Q2. Threshold tuning: 2,000 chars is a guess. Instrument a few real sessions and pick a value that never touches conversational/tool-control calls but always catches files.
- Q3. Do we want the placeholder to include a content hash (so the model can detect "same file as before")? Probably unnecessary for v1.
