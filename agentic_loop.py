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

logger = logging.getLogger(__name__)

# Safety cap: maximum number of tool-call rounds before we force a final answer.
DEFAULT_MAX_ITERATIONS = 20


async def run_agentic_loop(
    messages: list[dict[str, Any]],
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
    base_payload: dict[str, Any],
    stream_chunk_callback: Optional[Callable[[str], Awaitable[None]]] = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_tool_leak_retries: int = 1,
) -> str:
    """Run the agentic loop until the model produces a final text answer.

    The loop mirrors the nanocode pattern:

    1. Call the model.
    2. If the response contains tool calls, execute them and append the results
       as a ``tool`` role message, then go to step 1.
    3. If there are no tool calls, return the assistant's text content.

    Additional safeguards:
    * If the model emits raw bash/JSON code instead of a structured tool call
      the leaked content is recovered and executed as a ``bash`` tool call.
    * If the final text response still contains leaked tool content, a one-shot
      formatting-guard retry is attempted.
    * Once ``max_iterations`` rounds are used a forced-finish user message is
      injected and tools are stripped from the final payload so the model
      *must* produce a text summary.

    Args:
        messages: Full conversation so far, including the current user turn.
            **Mutated in-place** — callers should pass a copy if they need to
            preserve the original list.
        session: Active ``aiohttp.ClientSession``.
        base_url: Chat-completions endpoint URL.
        api_key: Bearer token / API key.
        base_payload: Base request dict (model, temperature, tools schema …).
            Copied on each iteration; never mutated.
        stream_chunk_callback: Optional coroutine called with each text delta
            during streaming turns.
        max_iterations: Hard cap on tool-call rounds.  When reached the model
            is prompted to summarise and tools are disabled for that final call.
        max_tool_leak_retries: How many times to retry when leaked tool content
            is detected in the final text response.

    Returns:
        The final plain-text assistant response (markdown stripped).
    """
    allowed_tool_names = _extract_allowed_tool_names(base_payload)
    tool_leak_retries_used = 0

    for iteration in range(max_iterations + 1):  # +1 so the forced-finish call is free
        forced_finish = iteration == max_iterations

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

        # ── API call ──────────────────────────────────────────────────────────
        response_data = await api_call_with_retry(
            session,
            base_url,
            current_payload,
            {"Authorization": f"Bearer {api_key}"},
            stream=stream_chunk_callback is not None and not forced_finish,
            stream_chunk_callback=stream_chunk_callback if not forced_finish else None,
        )

        # ── Error handling ────────────────────────────────────────────────────
        if "error" in response_data:
            error_msg = response_data["error"].get("message", "Unknown API error")
            logger.error("Agentic loop API error (iteration %d): %s", iteration, error_msg)
            return f"Error: {error_msg}"

        if not response_data.get("choices"):
            logger.error("Agentic loop: no choices in response (iteration %d)", iteration)
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

        logger.debug(
            "Agentic loop iteration %d: executed %d tool call(s).",
            iteration + 1,
            len(tool_calls),
        )

        messages.append(message)
        messages.extend(tool_results)

    # Unreachable — the forced_finish branch always returns — but satisfies mypy.
    return "Task completed."
