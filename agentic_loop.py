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

import logging
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

# Patterns that indicate the model is narrating intended actions instead of
# executing them. When detected in a text-only response (no tool_calls), the
# loop nudges the model to actually make the calls.
_ACTION_INTENT_PATTERNS = re.compile(
    r"(?i)"
    r"(?:activating\s+(?:the\s+)?(?:`[^`]+`|[\w-]+)\s+skill)"
    r"|(?:stand\s+by\s+while\s+i)"
    r"|(?:let\s+me\s+(?:actually\s+)?(?:(?:\w+\s+){0,4})?(?:run|execute|call|create|write|inspect|check|verify|investigate|debug|review|push|publish|post|upload|deploy))"
    r"|(?:i(?:'m|\s+am)\s+(?:going\s+to|about\s+to|now\s+going\s+to)\s+(?:run|execute|call|create|write|activate|inspect|check|verify|investigate|debug|review|push|publish|post|upload|deploy))"
    r"|(?:sending\s+`?(?:like|love|dislike|laugh|emphasize|question)`?\s+to\b)"
    r"|(?:react(?:ing|ion)\s+with\s+`?(?:like|love|dislike|laugh|emphasize|question)`?\b)"
    r"|(?:\b(?:running|writing|creating|building|scaffolding|fixing|inspecting|checking|verifying|investigating|debugging|reviewing|pushing|publishing|posting|uploading|deploying)\b.{0,80}\b(?:now|rn)\b)"
    r"|(?:running\s+the\s+.{3,60}\s+(?:script|command|tool))"
    r"|(?:i\s+need\s+to:\s*\n)"
)
_TOOL_REQUIREMENT_PATTERNS = re.compile(
    r"(?is)"
    r"(?:tool\s+execution\s+is\s+mandatory)"
    r"|(?:must\s+(?:use|call|execute|run).{0,40}(?:tool|tool_call))"
    r"|(?:do\s+not\s+answer.{0,80}(?:without|unless).{0,40}(?:tool|tool_call))"
    r"|(?:if\s+you\s+do\s+not\s+execute.{0,60}fail)"
)
TAPBACK_ACK_PATTERN = re.compile(
    r"(?is)^\s*"
    r"(?:done|ok(?:ay)?|sent)[.!]?\s*$"
    r"|^\s*(?:done[.!]?\s*)?"
    r"(?:"
    r"(?:(?:i\s+)?(?:sent|sending|reacted)\b.{0,220}\b(?:tapback|reaction)\b.{0,220})"
    r"|"
    r"(?:(?:i\s+)?(?:sent|sending|reacted)\b.{0,220}\b(?:like|love|dislike|laugh|emphasize|question)\b.{0,220}\b(?:to\s+your|message)\b.{0,220})"
    r")"
    r"\s*$"
)
_BARE_REACTION_PATTERN = re.compile(
    r"^(?:like|love|dislike|laugh|emphasize|question)[.!]?\s*$",
    re.IGNORECASE,
)

logger = logging.getLogger(__name__)

# Safety cap: maximum number of tool-call rounds before we force a final answer.
DEFAULT_MAX_ITERATIONS = 20
DEFAULT_MAX_ACTION_INTENT_RETRIES = 3
DEFAULT_MAX_TAPBACK_REPLY_RETRIES = 1


def contains_action_intent_narration(text: str) -> bool:
    """Return True when text looks like narrated work instead of executed work."""
    if not text:
        return False
    return bool(_ACTION_INTENT_PATTERNS.search(text))


def user_explicitly_requires_tool_execution(
    messages: list[dict[str, Any]], allowed_tool_names: set[str]
) -> bool:
    """Return True when the latest real user turn explicitly requires tools."""
    latest_user_text = ""
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        candidate = _message_content_to_text(message.get("content", ""))
        stripped = candidate.strip()
        if stripped.startswith("[System:"):
            continue
        latest_user_text = candidate
        break

    if not latest_user_text:
        return False

    if _TOOL_REQUIREMENT_PATTERNS.search(latest_user_text):
        return True

    lowered = latest_user_text.lower()
    if "tool" in lowered and re.search(r"(?i)\b(?:use|call|execute|run)\b", lowered):
        return True

    for tool_name in allowed_tool_names:
        if re.search(
            rf"(?i)\b(?:use|call|execute|run)\s+(?:the\s+)?`?{re.escape(tool_name)}`?(?:\s+tool)?\b",
            latest_user_text,
        ):
            return True

    return False


def is_bare_reaction_word(text: str) -> bool:
    """Return True when text is just a tapback reaction word with no other content."""
    if not text:
        return False
    return bool(_BARE_REACTION_PATTERN.match(text.strip()))


def needs_tapback_followup_reply(text: str, executed_tool_names: list[str]) -> bool:
    """Return True when a tapback-only turn still needs a normal text reply."""
    if not text or not executed_tool_names:
        return False

    normalized_tool_names = {name for name in executed_tool_names if name}
    if normalized_tool_names != {"send_tapback"}:
        return False

    return bool(TAPBACK_ACK_PATTERN.match(text.strip()))


async def run_agentic_loop(
    messages: list[dict[str, Any]],
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
    base_payload: dict[str, Any],
    stream_chunk_callback: Optional[Callable[[str], Awaitable[None]]] = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_tool_leak_retries: int = 1,
    max_action_intent_retries: int = DEFAULT_MAX_ACTION_INTENT_RETRIES,
    max_tapback_reply_retries: int = DEFAULT_MAX_TAPBACK_REPLY_RETRIES,
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
    tool_leak_retries_used = 0
    action_intent_retries_used = 0
    tapback_reply_retries_used = 0
    executed_tool_rounds = 0
    executed_tool_names: list[str] = []
    pending_response_data: Optional[dict[str, Any]] = initial_response_data

    for iteration in range(max_iterations + 1):  # +1 so the forced-finish call is free
        forced_finish = iteration == max_iterations

        if pending_response_data is not None:
            # Use the pre-fetched response for this iteration (no API call needed).
            response_data = pending_response_data
            pending_response_data = None
        else:
            # Build the payload for this round.
            current_payload = base_payload.copy()
            current_payload["messages"] = messages

            if forced_finish:
                # Inject a system nudge and strip tools so the model must answer in text.
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
                stream_chunk_callback=stream_chunk_callback
                if not forced_finish
                else None,
            )

        # ── Error handling ────────────────────────────────────────────────────
        if "error" in response_data:
            error_msg = response_data["error"].get("message", "Unknown API error")
            logger.error(
                "Agentic loop API error (iteration %d): %s", iteration, error_msg
            )
            return f"Error: {error_msg}"

        if not response_data.get("choices"):
            logger.error(
                "Agentic loop: no choices in response (iteration %d)", iteration
            )
            return "Error: No response from API"

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
            explicit_tool_required = (
                executed_tool_rounds == 0
                and user_explicitly_requires_tool_execution(
                    messages, allowed_tool_names_set
                )
            )

            # Detect "narrating actions without executing them" — the model
            # describes what it will do but didn't make any tool calls.
            # Give it one chance to actually execute.
            if (
                not forced_finish
                and content_text
                and (
                    contains_action_intent_narration(content_text)
                    or is_bare_reaction_word(content_text)
                    or explicit_tool_required
                )
                and action_intent_retries_used < max_action_intent_retries
            ):
                action_intent_retries_used += 1
                if is_bare_reaction_word(content_text):
                    logger.warning(
                        "Agentic loop: model returned bare reaction word "
                        "'%s' instead of calling send_tapback; nudging model "
                        "to execute properly (attempt %d/%d).",
                        content_text.strip(),
                        action_intent_retries_used,
                        max_action_intent_retries,
                    )
                    messages.append(message)
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "[System: You just replied with a single "
                                "reaction word. That is not a valid reply — "
                                "tapback reactions must be sent via the "
                                "send_tapback tool. Either call "
                                "send_tapback with the message_handle and "
                                "reaction type, or reply to the user's "
                                "message in normal conversational text. Do "
                                "not output bare reaction words as text.]"
                            ),
                        },
                    )
                elif explicit_tool_required:
                    logger.warning(
                        "Agentic loop: user explicitly required tool execution "
                        "but model returned text-only output; nudging model to execute "
                        "(attempt %d/%d).",
                        action_intent_retries_used,
                        max_action_intent_retries,
                    )
                else:
                    logger.warning(
                        "Agentic loop: detected action-intent narration without "
                        "tool calls, nudging model to execute (attempt %d/%d).",
                        action_intent_retries_used,
                        max_action_intent_retries,
                    )
                messages.append(message)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "[System: You just described actions you intend to take "
                            "but did not make any tool calls. Do NOT narrate what "
                            "you will do — actually do it. Make the tool_calls now. "
                            "If you said you're activating a skill, call "
                            "activate_skill. If you said you're running a command, "
                            "call bash. If you said you're sending a tapback or "
                            "reaction, call send_tapback with the provided "
                            "message_handle. If you said you're publishing, pushing, "
                            "posting, uploading, or deploying, make the relevant "
                            "skill and shell tool calls now. Act, don't narrate.]"
                        ),
                    },
                )
                continue

            if (
                needs_tapback_followup_reply(content_text, executed_tool_names)
                and tapback_reply_retries_used < max_tapback_reply_retries
            ):
                tapback_reply_retries_used += 1
                logger.warning(
                    "Agentic loop: tapback tool executed but model replied with "
                    "acknowledgement-only text; requesting a normal conversational "
                    "follow-up (attempt %d/%d).",
                    tapback_reply_retries_used,
                    max_tapback_reply_retries,
                )
                messages.append(message)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "[System: You already handled the tapback internally. "
                            "Now reply to the user's message normally in plain text. "
                            "Do not mention the tapback, reaction, send_tapback, tool "
                            "calls, or any internal action. Continue the conversation "
                            "naturally and answer the user's actual message.]"
                        ),
                    }
                )
                retry_payload = {k: v for k, v in base_payload.items() if k != "tools"}
                retry_payload["messages"] = messages
                retry_data = await api_call_with_retry(
                    session,
                    base_url,
                    retry_payload,
                    {"Authorization": f"Bearer {api_key}"},
                )
                if retry_data.get("choices"):
                    retry_msg = retry_data["choices"][0]["message"]
                    content_text = _message_content_to_text(
                        retry_msg.get("content", "")
                    )

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
                                "Your last reply exposed internal tool call content. "
                                "Do not show tool calls, shell commands, code fences, or "
                                "JSON internals. Reply naturally in plain text for the end user."
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

            return safe_strip_markdown(content_text) if content_text else ""

        # ── Execute tool calls and feed results back ──────────────────────────
        tool_results = await execute_tool_calls(
            message, allowed_tool_names=allowed_tool_names
        )
        executed_tool_rounds += 1
        executed_tool_names.extend(
            tool_call.get("function", {}).get("name", "")
            for tool_call in tool_calls
            if tool_call.get("function", {}).get("name")
        )

        logger.debug(
            "Agentic loop iteration %d: executed %d tool call(s).",
            iteration + 1,
            len(tool_calls),
        )

        messages.append(message)
        messages.extend(tool_results)

    # Unreachable — the forced_finish branch always returns — but satisfies mypy.
    return "Task completed."
