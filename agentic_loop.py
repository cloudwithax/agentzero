"""Agentic loop for multi-step task execution.

Implements a nanocode-style loop (https://github.com/1rgs/nanocode) that keeps
calling the model and executing tool calls until the model decides to stop or a
safety cap is reached.  This replaces the ad-hoc inner ``while`` loop that was
inline inside ``process_response`` and adds:

* An explicit ``max_iterations`` cap so runaway tasks can't loop forever.
* A forced-finish message when the cap is hit so the user gets a coherent
  answer rather than silence.
* Cleaner separation of the "execute tools → feed results back → continue"
  cycle so it is easy to reason about and test.
* Consistent tool-leak detection on the final text response.
"""

import asyncio
import base64
import logging
import json
import os
import re
from typing import Any, Awaitable, Callable, Optional

import aiohttp

from api import (
    _extract_allowed_tool_names,
    _message_content_to_text,
    api_call_with_retry,
    detect_tool_leak,
    execute_tool_calls,
    infer_tool_calls_from_content,
    safe_strip_markdown,
)
from tools import (
    get_send_message_buffer,
    init_declared_message_count,
    init_send_message_buffer,
    reset_declared_message_count,
    reset_send_message_buffer,
    reset_tool_runtime_messages,
    set_tool_runtime_messages,
)

# Matches a literal <DONE> completion-protocol marker on its own line or
# surrounded by whitespace. Stripped from visible text after send_message
# delivers the real reply.
_DONE_MARKER_RE = re.compile(r"(?im)^\s*<\s*DONE\s*>\s*$")


def _model_supports_multimodal_blocks(model_id: Optional[str]) -> bool:
    """True when ``model_id`` accepts inline image_url content blocks."""
    if not model_id:
        return False
    try:
        from integrations import MULTIMODAL_MODEL_IDS
    except Exception:
        return False
    return model_id.lower() in MULTIMODAL_MODEL_IDS


def _browser_screenshot_paths(
    message: dict[str, Any], tool_results: list[dict[str, Any]]
) -> list[str]:
    """Return file paths produced by successful browser_screenshot tool calls."""
    id_to_name = {
        tc.get("id"): tc.get("function", {}).get("name")
        for tc in message.get("tool_calls", []) or []
    }
    paths: list[str] = []
    for res in tool_results:
        if id_to_name.get(res.get("tool_call_id")) != "browser_screenshot":
            continue
        content = res.get("content")
        if not isinstance(content, str):
            continue
        try:
            data = json.loads(content)
        except Exception:
            continue
        if isinstance(data, dict) and data.get("success") and data.get("path"):
            paths.append(data["path"])
    return paths


def _screenshot_file_to_data_url(path: str) -> Optional[str]:
    """Read a screenshot file and return a base64 image data URL (resized JPEG)."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
        if not raw:
            return None
        try:
            from integrations import _resize_image_to_jpeg_sync

            resized = _resize_image_to_jpeg_sync(raw)
        except Exception:
            resized = None
        if resized:
            return "data:image/jpeg;base64," + base64.b64encode(resized).decode("ascii")
        return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
    except Exception as e:
        logger.warning("Failed to load browser screenshot %s: %s", path, e)
        return None


def _inject_browser_screenshots(
    messages: list[dict[str, Any]],
    message: dict[str, Any],
    tool_results: list[dict[str, Any]],
    model_id: Optional[str],
) -> int:
    """Append captured browser screenshots as a multimodal user message.

    Screenshots produced by ``browser_screenshot`` are saved to disk and the
    tool only returns a path (which the model cannot see). For natively
    multimodal models, inject the actual image so the next round can view it.
    Returns the number of screenshots injected.
    """
    if not _model_supports_multimodal_blocks(model_id):
        return 0
    paths = _browser_screenshot_paths(message, tool_results)
    if not paths:
        return 0
    blocks: list[dict[str, Any]] = [
        {"type": "text", "text": "[Browser screenshot — here is what the page looks like:]"}
    ]
    for path in paths:
        data_url = _screenshot_file_to_data_url(path)
        if data_url:
            blocks.append({"type": "image_url", "image_url": {"url": data_url}})
    if len(blocks) <= 1:
        return 0
    messages.append({"role": "user", "content": blocks})
    return len(blocks) - 1


def _strip_done_marker(text: str) -> str:
    if not text:
        return text
    return _DONE_MARKER_RE.sub("", text).strip()


def _assemble_send_message_buffer_response(
    buffer: list[dict[str, Any]],
    fallback_text: str,
    accumulated_attachments: list[str],
) -> Optional[str]:
    """Build the loop's final response from a non-empty send_message buffer.

    Channel-bound entries (imessage/telegram) were already delivered by the
    send_message tool, so we return a ``delivered_via_tool`` envelope and
    integration code skips re-sending.  For non-channel sessions (CLI,
    OpenAI-compat, tests) we concatenate the buffered bubbles into the
    visible response so callers see the actual reply text.
    """
    if not buffer:
        return None

    channels = {str(record.get("channel") or "").strip() for record in buffer}
    bound_channels = {"imessage", "telegram"}
    delivered_via_tool = bool(channels & bound_channels)

    buffered_attachments: list[str] = list(accumulated_attachments)
    bubble_texts: list[str] = []
    for record in buffer:
        text = str(record.get("text") or "").strip()
        if text:
            bubble_texts.append(text)
        for url in record.get("attachments") or []:
            normalized = str(url).strip()
            if normalized and normalized not in buffered_attachments:
                buffered_attachments.append(normalized)

    if delivered_via_tool:
        envelope: dict[str, Any] = {
            "delivered_via_tool": True,
            "text": "",
            "attachments": [],
        }
        return json.dumps(envelope)

    visible_text = "\n\n".join(bubble_texts) if bubble_texts else fallback_text
    if buffered_attachments:
        return json.dumps(
            {"text": visible_text, "attachments": buffered_attachments}
        )
    return visible_text

logger = logging.getLogger(__name__)

# Safety cap: maximum number of tool-call rounds before we force a final answer.
DEFAULT_MAX_ITERATIONS = 10

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    """Read a non-negative int env var, falling back to ``default`` on error."""
    try:
        return max(minimum, int(os.environ.get(name, str(default)).strip()))
    except (ValueError, AttributeError):
        return default


# ── Tool-call history compaction ──────────────────────────────────────────────
# After a tool round executes, the assistant message (including the verbatim
# tool-call ``arguments``) is retained in ``messages`` and re-sent on every
# subsequent API call.  Large arguments — most commonly a whole file passed to
# ``write`` — bloat every following request, wasting tokens and stressing the
# provider into transient truncation errors.  Once a tool has executed, the
# model only needs to *remember that it ran*, not the full payload, so we
# replace large string fields with a compact, readable placeholder in the
# retained copy.  Keyed on argument size (generic), not on tool identity.
_COMPACT_TOOL_ARGS = os.environ.get("AGENTZERO_COMPACT_TOOL_ARGS", "1").strip() != "0"
# Min serialized-arguments length before a tool call is considered for compaction.
_COMPACT_TOOL_ARG_THRESHOLD = _env_int(
    "AGENTZERO_COMPACT_TOOL_ARG_THRESHOLD", 2000, minimum=200
)
# Min individual string-field length before that field is elided.
_COMPACT_FIELD_THRESHOLD = _env_int(
    "AGENTZERO_COMPACT_FIELD_THRESHOLD", 1000, minimum=100
)


def _compact_executed_tool_call_args(
    message: dict[str, Any], round_number: int
) -> dict[str, Any]:
    """Return a history-safe copy of an assistant message with large tool-call
    ``arguments`` elided.

    The input ``message`` is never mutated — the tool calls it carries have
    already executed with full fidelity; only the copy retained in history is
    shrunk.  Large string fields inside each call's JSON ``arguments`` are
    replaced with a ``<elided N chars …>`` placeholder so the model keeps the
    fact of the action (tool name, target, size) without re-uploading the
    payload on every later round.  Idempotent: placeholders are below the
    threshold, so re-compacting is a no-op.
    """
    if not _COMPACT_TOOL_ARGS:
        return message
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return message

    new_tool_calls: list[dict[str, Any]] = []
    changed = False
    reclaimed = 0
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            new_tool_calls.append(tool_call)
            continue
        function = tool_call.get("function")
        raw_args = function.get("arguments") if isinstance(function, dict) else None
        if not isinstance(raw_args, str) or len(raw_args) < _COMPACT_TOOL_ARG_THRESHOLD:
            new_tool_calls.append(tool_call)
            continue

        compacted_args, field_reclaimed = _compact_arguments_string(
            raw_args, round_number
        )
        if compacted_args == raw_args:
            new_tool_calls.append(tool_call)
            continue

        changed = True
        reclaimed += field_reclaimed
        new_function = dict(function)
        new_function["arguments"] = compacted_args
        new_call = dict(tool_call)
        new_call["function"] = new_function
        new_tool_calls.append(new_call)

    if not changed:
        return message

    logger.info(
        "Agentic loop: compacted tool-call arguments in retained history "
        "(round %d, reclaimed ~%d chars).",
        round_number,
        reclaimed,
    )
    new_message = dict(message)
    new_message["tool_calls"] = new_tool_calls
    return new_message


def _compact_arguments_string(raw_args: str, round_number: int) -> tuple[str, int]:
    """Elide large string fields in a JSON ``arguments`` string.

    Returns ``(compacted_string, chars_reclaimed)``.  Falls back to truncating
    the raw string when it is not valid JSON (e.g. recovered/leaked calls).
    """
    try:
        parsed = json.loads(raw_args)
    except (json.JSONDecodeError, TypeError):
        # Not valid JSON — truncate the raw blob but keep a readable marker.
        head = raw_args[:_COMPACT_FIELD_THRESHOLD]
        reclaimed = len(raw_args) - len(head)
        return (
            f"{head}…[elided {reclaimed} chars — executed in round {round_number}]",
            reclaimed,
        )

    if not isinstance(parsed, dict):
        return raw_args, 0

    reclaimed = 0
    new_parsed: dict[str, Any] = {}
    for key, value in parsed.items():
        if isinstance(value, str) and len(value) >= _COMPACT_FIELD_THRESHOLD:
            reclaimed += len(value)
            new_parsed[key] = (
                f"<elided {len(value)} chars — '{key}' written in round {round_number}>"
            )
        else:
            new_parsed[key] = value

    if reclaimed == 0:
        return raw_args, 0
    return json.dumps(new_parsed), reclaimed


def extract_outbound_attachments(
    tool_results: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
) -> list[str]:
    """Extract attachment URLs produced by image-generating tool calls.

    Looks for successful tool results that contain a ``url`` field pointing
    to an image resource.  These URLs are suitable for embedding in outbound
    Telegram / Sendblue messages alongside the assistant's text reply.
    """
    # Build a set of tool-call IDs that came from image-producing tools.
    image_tool_names = {"generate_image"}
    image_call_ids = {
        tc.get("id")
        for tc in tool_calls
        if tc.get("function", {}).get("name") in image_tool_names
    }

    urls: list[str] = []
    for result in tool_results:
        if not isinstance(result, dict):
            continue
        # Only inspect results that correspond to image-producing calls.
        if result.get("tool_call_id") not in image_call_ids:
            continue
        payload = result.get("content", "")
        try:
            parsed = json.loads(payload) if isinstance(payload, str) else payload
        except Exception:
            continue
        if not isinstance(parsed, dict):
            continue
        if parsed.get("success") is False:
            continue
        url = str(parsed.get("url") or "").strip()
        if url and url not in urls:
            urls.append(url)
    return urls


# Markers that mean the failed round is a transient provider hiccup (most often
# the provider failing to serialize the model's own tool-call output) rather
# than a permanent client error.  Used to decide whether a tools-stripped
# graceful finish is worth attempting before surfacing a raw error.
_RECOVERABLE_ERROR_MARKERS = (
    "unterminated string",
    "expecting value",
    "expecting ',' delimiter",
    "expecting property name",
    "invalid \\escape",
    "extra data",
    "jsondecode",
    "internal server error",
    "internal error",
    "service unavailable",
    "temporarily unavailable",
    "bad gateway",
    "gateway timeout",
    "timeout",
    "timed out",
    "overloaded",
    "try again",
    "please retry",
    "upstream",
    # A rate limit exhausted in api_call_with_retry is also recoverable: a
    # tools-stripped graceful finish re-runs through api_call_with_retry, whose
    # own retry/backoff gives the rolling window time to clear.  Worst case it
    # also fails and we fall back to the raw-error path — never worse than today.
    "rate limit",
    "rate_limit",
)


def _looks_recoverable(error_msg: str) -> bool:
    """Return True when a failed round looks transient enough to retry as text."""
    if not error_msg:
        return False
    lowered = error_msg.lower()
    return any(marker in lowered for marker in _RECOVERABLE_ERROR_MARKERS)


async def _attempt_graceful_finish(
    *,
    messages: list[dict[str, Any]],
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
    base_payload: dict[str, Any],
    executed_tool_names: list[str],
) -> Optional[str]:
    """Make a tools-stripped call so the model summarises completed work.

    Used when a tool round already did real work but the *next* round failed on
    a transient provider parse/serialize error.  Stripping tools removes the
    large/complex tool-call serialization that the provider was choking on, so
    the model can usually produce a clean plain-text summary.  Returns the
    summary text, or ``None`` if the recovery attempt also fails (caller then
    falls back to the raw error path).
    """
    finish_messages = list(messages)
    finish_messages.append(
        {
            "role": "user",
            "content": (
                "[System: The previous step hit a temporary provider error. Do "
                "NOT call any tools. In plain text, tell the user what you have "
                "completed so far and give them a clear final answer now.]"
            ),
        }
    )
    finish_payload = {k: v for k, v in base_payload.items() if k != "tools"}
    finish_payload["messages"] = finish_messages
    try:
        finish_data = await api_call_with_retry(
            session,
            base_url,
            finish_payload,
            {"Authorization": f"Bearer {api_key}"},
        )
    except Exception:  # noqa: BLE001 — recovery is best-effort
        logger.exception("Graceful-finish recovery attempt raised.")
        return None

    if "error" in finish_data or not finish_data.get("choices"):
        return None

    text = _message_content_to_text(
        finish_data["choices"][0]["message"].get("content", "")
    ).strip()
    if not text or detect_tool_leak(text):
        return None

    logger.info(
        "Agentic loop: recovered from transient round error via tools-stripped "
        "finish (tools_executed=%s).",
        executed_tool_names,
    )
    return text


async def run_agentic_loop(
    messages: list[dict[str, Any]],
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
    base_payload: dict[str, Any],
    stream_chunk_callback: Optional[Callable[[str], Awaitable[None]]] = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_tool_leak_retries: int = 1,
    initial_response_data: Optional[dict[str, Any]] = None,
) -> str:
    """Run the agentic loop until the model produces a final text answer.

    while tool_calls:
        execute tool calls
        call model again
    return final text

    If ``initial_response_data`` is provided it is used as the first response
    (no extra API call needed for that round).  Callers that have already
    fetched the first response should pass it here.

    Additional safeguards:
    * If the model emits raw bash/JSON code instead of a structured tool call
      the leaked content is recovered and executed as a ``bash`` tool call.
    * If the final text response still contains leaked tool content, a one-shot
      formatting-guard retry is attempted.
    * Once ``max_iterations`` rounds are used a forced-finish user message is
      injected and tools are stripped from the final payload so the model
      *must* produce a text summary.
    """
    allowed_tool_names = _extract_allowed_tool_names(base_payload)
    allowed_tool_names_set = set(allowed_tool_names)

    result = await _run_agentic_loop_inner(
        messages=messages,
        session=session,
        base_url=base_url,
        api_key=api_key,
        base_payload=base_payload,
        stream_chunk_callback=stream_chunk_callback,
        max_iterations=max_iterations,
        max_tool_leak_retries=max_tool_leak_retries,
        initial_response_data=initial_response_data,
        allowed_tool_names=allowed_tool_names,
        allowed_tool_names_set=allowed_tool_names_set,
    )
    return result


async def _run_agentic_loop_inner(
    messages: list[dict[str, Any]],
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
    base_payload: dict[str, Any],
    stream_chunk_callback: Optional[Callable[[str], Awaitable[None]]],
    max_iterations: int,
    max_tool_leak_retries: int,
    initial_response_data: Optional[dict[str, Any]],
    allowed_tool_names: list[str],
    allowed_tool_names_set: set[str],
) -> str:
    tool_leak_retries_used = 0
    executed_tool_rounds = 0
    executed_tool_names: list[str] = []
    pending_response_data: Optional[dict[str, Any]] = initial_response_data
    accumulated_attachments: list[str] = []

    # Track send_message tool deliveries so the final return can reflect
    # what the model actually sent to the user.  Channel-bound deliveries
    # (iMessage/Telegram) happen out-of-band inside the tool; non-channel
    # sessions rely on the loop to assemble buffered bubbles into the
    # visible response.
    send_buffer_token = init_send_message_buffer()
    declared_count_token = init_declared_message_count()
    try:
        return await _run_agentic_loop_body(
            messages=messages,
            session=session,
            base_url=base_url,
            api_key=api_key,
            base_payload=base_payload,
            stream_chunk_callback=stream_chunk_callback,
            max_iterations=max_iterations,
            max_tool_leak_retries=max_tool_leak_retries,
            initial_response_data=initial_response_data,
            allowed_tool_names=allowed_tool_names,
            allowed_tool_names_set=allowed_tool_names_set,
            tool_leak_retries_used=tool_leak_retries_used,
            executed_tool_rounds=executed_tool_rounds,
            executed_tool_names=executed_tool_names,
            pending_response_data=pending_response_data,
            accumulated_attachments=accumulated_attachments,
        )
    except Exception as exc:
        # Catch-all so a failure ANYWHERE in the loop (tool execution, payload
        # mutation, parsing, etc.) surfaces as a clean Error string that bubbles
        # to the caller instead of crashing the turn.
        logger.exception(
            "Agentic loop crashed (tools_executed=%s)", executed_tool_names
        )
        tools_context = (
            f", tools_executed={executed_tool_names}" if executed_tool_names else ""
        )
        return f"Error: {exc} [agentic_loop{tools_context}]"
    finally:
        reset_send_message_buffer(send_buffer_token)
        reset_declared_message_count(declared_count_token)


async def _run_agentic_loop_body(
    *,
    messages: list[dict[str, Any]],
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
    base_payload: dict[str, Any],
    stream_chunk_callback: Optional[Callable[[str], Awaitable[None]]],
    max_iterations: int,
    max_tool_leak_retries: int,
    initial_response_data: Optional[dict[str, Any]],
    allowed_tool_names: list[str],
    allowed_tool_names_set: set[str],
    tool_leak_retries_used: int,
    executed_tool_rounds: int,
    executed_tool_names: list[str],
    pending_response_data: Optional[dict[str, Any]],
    accumulated_attachments: list[str],
) -> str:
    for iteration in range(max_iterations + 1):  # +1 so the forced-finish call is free
        forced_finish = iteration == max_iterations
        logger.debug(
            "Loop iter %d/%d starting (forced_finish=%s)",
            iteration + 1,
            max_iterations,
            forced_finish,
        )

        if pending_response_data is not None:
            # Use the pre-fetched response for this iteration (no API call needed).
            response_data = pending_response_data
            pending_response_data = None
        else:
            # Build the payload for this round.
            current_payload = base_payload.copy()
            current_payload["messages"] = messages

            if forced_finish:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "[System: You have used the maximum number of tool-call rounds. "
                            "Do NOT call any more tools. Summarise what you have accomplished "
                            "and give the user a direct final answer now.]"
                        ),
                    }
                )
                current_payload = {
                    k: v for k, v in base_payload.items() if k != "tools"
                }
                current_payload["messages"] = messages
                logger.warning(
                    "Agentic loop reached max_iterations=%d — forcing final answer.",
                    max_iterations,
                )

            # ── API call ──────────────────────────────────────────────────────
            response_data = await api_call_with_retry(
                session,
                base_url,
                current_payload,
                {"Authorization": f"Bearer {api_key}"},
                stream=stream_chunk_callback is not None and not forced_finish,
                stream_chunk_callback=(
                    stream_chunk_callback if not forced_finish else None
                ),
            )

        # ── Error handling ────────────────────────────────────────────────────
        if "error" in response_data:
            error_msg = response_data["error"].get("message", "Unknown API error")
            logger.error(
                "Agentic loop API error (iteration %d): %s", iteration, error_msg
            )

            # If tool rounds already executed real work (e.g. a website was
            # written) and the failure is a transient provider parse/serialize
            # error on the *next* round, don't dump raw parser guts at the user.
            # Make one tools-stripped attempt to have the model summarise what it
            # accomplished in plain text instead.
            if executed_tool_names and not forced_finish and _looks_recoverable(error_msg):
                recovered = await _attempt_graceful_finish(
                    messages=messages,
                    session=session,
                    base_url=base_url,
                    api_key=api_key,
                    base_payload=base_payload,
                    executed_tool_names=executed_tool_names,
                )
                if recovered:
                    return recovered

            tools_context = (
                f", tools_executed={executed_tool_names}" if executed_tool_names else ""
            )
            return f"Error: {error_msg} [iteration={iteration}{tools_context}]"

        if not response_data.get("choices"):
            logger.error(
                "Agentic loop: no choices in response (iteration %d)", iteration
            )
            tools_context = (
                f", tools_executed={executed_tool_names}" if executed_tool_names else ""
            )
            return f"Error: No response from API [iteration={iteration}{tools_context}]"

        message = response_data["choices"][0]["message"]

        # ── Recover leaked tool calls ─────────────────────────────────────────
        if not message.get("tool_calls"):
            inferred = infer_tool_calls_from_content(
                _message_content_to_text(message.get("content", ""))
            )
            if inferred:
                message = dict(message)  # avoid mutating the response object
                message["tool_calls"] = inferred
                message["content"] = None
                logger.warning(
                    "Agentic loop iteration %d: recovered %d tool call(s) from leaked content.",
                    iteration,
                    len(inferred),
                )

        tool_calls = message.get("tool_calls") or []

        # ── No tool calls → the model is done ────────────────────────────────
        if not tool_calls:
            content_text = _message_content_to_text(message.get("content", ""))

            # Tool-call behavior (call tools instead of narrating, no pseudo-tool
            # syntax, react via send_tapback/send_telegram_reaction, install skill
            # URLs, don't claim success on tool errors) is enforced via the system
            # prompt rather than code-side retry nudges. The model is trusted to
            # call tools when appropriate; whatever it returns here is final.

            # Check for leaked tool content in the final response.
            if detect_tool_leak(content_text):
                if tool_leak_retries_used < max_tool_leak_retries:
                    tool_leak_retries_used += 1
                    logger.warning(
                        "Agentic loop: tool-leak detected in final response, "
                        "formatting-guard retry %d/%d.",
                        tool_leak_retries_used,
                        max_tool_leak_retries,
                    )
                    messages.append(message)
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "[System: Your reply exposed internal tool call "
                                "content. Reply in plain text only — no tool "
                                "calls, code fences, or JSON.]"
                            ),
                        }
                    )
                    retry_payload = base_payload.copy()
                    retry_payload["messages"] = messages
                    retry_data = await api_call_with_retry(
                        session,
                        base_url,
                        retry_payload,
                        {"Authorization": f"Bearer {api_key}"},
                    )
                    if retry_data.get("choices"):
                        retry_msg = retry_data["choices"][0]["message"]
                        retry_text = _message_content_to_text(
                            retry_msg.get("content", "")
                        )
                        # If the retry still leaks, return the safe fallback.
                        if detect_tool_leak(retry_text):
                            logger.warning(
                                "Agentic loop: tool-leak persists after retry, "
                                "returning safe fallback."
                            )
                            return (
                                "Sorry, there was an internal formatting issue. "
                                "Please send that again."
                            )
                        content_text = retry_text
                else:
                    logger.warning(
                        "Agentic loop: max tool-leak retries exhausted, returning safe fallback."
                    )
                    return (
                        "Sorry, there was an internal formatting issue. "
                        "Please send that again."
                    )

            final_text = safe_strip_markdown(content_text) if content_text else ""
            # Strip the <DONE> completion-protocol marker from visible text;
            # it is a signal to the loop, not user-facing content.
            final_text = _strip_done_marker(final_text)

            # If the model delivered via send_message this turn, the buffer
            # is authoritative for the visible reply.  Channel-bound sends
            # were already dispatched out-of-band; assemble accordingly.
            buffered_response = _assemble_send_message_buffer_response(
                get_send_message_buffer(),
                fallback_text=final_text,
                accumulated_attachments=accumulated_attachments,
            )
            if buffered_response is not None:
                return buffered_response

            if accumulated_attachments:
                return json.dumps(
                    {
                        "text": final_text,
                        "attachments": accumulated_attachments,
                    }
                )
            return final_text

        # ── Execute tool calls and feed results back ──────────────────────────
        runtime_messages_token = set_tool_runtime_messages([*messages, message])
        try:
            tool_results = await execute_tool_calls(
                message, allowed_tool_names=allowed_tool_names
            )
        finally:
            reset_tool_runtime_messages(runtime_messages_token)
        recent_attachments = extract_outbound_attachments(tool_results, tool_calls)
        for url in recent_attachments:
            if url not in accumulated_attachments:
                accumulated_attachments.append(url)
        executed_tool_rounds += 1
        executed_tool_names.extend(
            tool_call.get("function", {}).get("name", "")
            for tool_call in tool_calls
            if tool_call.get("function", {}).get("name")
        )

        logger.info(
            "Loop iter %d: executed %d tool call(s): %s",
            executed_tool_rounds,
            len(tool_calls),
            ", ".join(
                tc.get("function", {}).get("name", "?") for tc in tool_calls
            ),
        )

        # Retain a compacted copy so large tool-call argument payloads (e.g. a
        # whole file passed to write) are not re-sent on every later round.
        # ``message`` itself already executed with full args above.
        messages.append(
            _compact_executed_tool_call_args(message, round_number=executed_tool_rounds)
        )
        messages.extend(tool_results)

        # Surface browser screenshots to natively multimodal models so the
        # model can actually see the page it just captured.
        injected = _inject_browser_screenshots(
            messages, message, tool_results, base_payload.get("model")
        )
        if injected:
            logger.info(
                "Agentic loop: injected %d browser screenshot(s) into multimodal context",
                injected,
            )

    # Unreachable — the forced_finish branch always returns — but satisfies mypy.
    return "Task completed."
