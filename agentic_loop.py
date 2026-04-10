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
import json
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
from tools import reset_tool_runtime_messages, set_tool_runtime_messages

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
_HARD_DECISION_PATTERNS = re.compile(
    r"(?i)"
    r"(?:need\s+to\s+decide)"
    r"|(?:deciding\s+between)"
    r"|(?:not\s+sure\s+which)"
    r"|(?:unsure\s+which)"
    r"|(?:which\s+approach)"
    r"|(?:best\s+approach)"
    r"|(?:trade[-\s]?off)"
    r"|(?:hard\s+decision)"
    r"|(?:stuck\s+on\s+(?:the\s+)?approach)"
    r"|(?:blocked\s+on\s+(?:the\s+)?approach)"
)
_PSEUDO_TOOL_TAG_PATTERN = re.compile(r"<[A-Za-z_][\w]*\([^<>]*\)>")
_PSEUDO_TOOL_XML_PATTERN = re.compile(
    r"(?is)<(?:"
    r"read|write|edit|glob|grep|bash|web_search|webfetch|codesearch|"
    r"activate_skill|add_skill|consult_advisor|consult_reviewer|"
    r"send_tapback|send_telegram_reaction|send_reaction|"
    r"consortium_[a-z_]+|reminder_[a-z_]+"
    r")\b[^>]*>"
)
_PSEUDO_FUNCTION_XML_PATTERN = re.compile(r"(?is)<function_[A-Za-z_][\w-]*\b")
_PSEUDO_TOOL_MARKDOWN_PATTERN = re.compile(
    r"(?im)(?:^|[\r\n])\s*[*_`~][*_`~ ]*"
    r"(?:read|write|edit|glob|grep|bash|web_search|webfetch|codesearch|"
    r"activate_skill|add_skill|consult_advisor|consult_reviewer|"
    r"send_tapback|send_telegram_reaction|send_reaction|"
    r"consortium_[a-z_]+|reminder_[a-z_]+)"
    r"\s*(?:[:(])"
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
REACTION_TOOL_NAMES = {"send_tapback", "send_telegram_reaction", "send_reaction"}
_PUBLISH_REQUEST_PATTERNS = re.compile(
    r"(?i)\b(?:publish|deploy|push|upload|post\s+it|put\s+it\s+online|here\.now|site)\b"
)
_SKILL_URL_PATTERN = re.compile(
    r"https?://\S*?skill\S*?\.md(?:\b|[?#]|$)",
    re.IGNORECASE,
)
_URL_PATTERN = re.compile(r"https?://[^\s'\"<>()]+")
_SUCCESS_CLAIM_PATTERNS = re.compile(
    r"(?i)\b(?:done|completed|live at|published|pushed|deployed|uploaded|all set|finished|succeeded|successful|up-to-date|verified|complete)\b"
)
_FAILURE_ACK_PATTERNS = re.compile(
    r"(?i)\b(?:failed|failure|couldn't|could not|unable|did not|not available|not found|unverified|couldn't verify)\b"
)
_SKILL_URL_HESITATION_PATTERNS = re.compile(
    r"(?i)"
    r"(?:prompt[-\s]?injection)"
    r"|(?:suspicious)"
    r"|(?:unsafe|untrusted)"
    r"|(?:can't|cannot|won't|will not|should not).{0,80}(?:install|add|fetch)"
    r"|(?:manual(?:ly)?\s+review)"
)


def contains_action_intent_narration(text: str) -> bool:
    """Return True when text looks like narrated work instead of executed work."""
    if not text:
        return False
    return bool(_ACTION_INTENT_PATTERNS.search(text))


def latest_user_requests_publish(messages: list[dict[str, Any]]) -> bool:
    """Return True when the latest real user turn asks for publish/deploy work."""
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

    return bool(_PUBLISH_REQUEST_PATTERNS.search(latest_user_text))


def latest_user_skill_url(messages: list[dict[str, Any]]) -> str:
    """Return the latest user-provided skill URL, if present."""
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        candidate = _message_content_to_text(message.get("content", ""))
        stripped = candidate.strip()
        if stripped.startswith("[System:"):
            continue
        match = _SKILL_URL_PATTERN.search(candidate)
        if match:
            return match.group(0)
        break

    return ""


def summarize_recent_tool_failures(
    tool_results: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
) -> list[str]:
    """Return human-readable summaries for failed tool results in one round."""
    tool_call_by_id = {tool_call.get("id"): tool_call for tool_call in tool_calls}
    failures: list[str] = []

    for result in tool_results:
        if not isinstance(result, dict):
            continue
        tool_call_id = result.get("tool_call_id")
        tool_call = tool_call_by_id.get(tool_call_id, {})
        tool_name = tool_call.get("function", {}).get("name", "tool")
        call_context = ""
        raw_arguments = tool_call.get("function", {}).get("arguments", "")
        try:
            parsed_arguments = json.loads(raw_arguments) if raw_arguments else {}
        except Exception:
            parsed_arguments = {}
        if isinstance(parsed_arguments, dict):
            if tool_name == "bash":
                call_context = str(parsed_arguments.get("command") or "").strip()
            elif tool_name in {"activate_skill", "add_skill"}:
                call_context = str(
                    parsed_arguments.get("name") or parsed_arguments.get("url") or ""
                ).strip()

        payload = result.get("content", "")
        try:
            parsed = json.loads(payload) if isinstance(payload, str) else payload
        except Exception:
            parsed = None

        if isinstance(parsed, dict):
            if parsed.get("success") is False:
                error_text = str(parsed.get("error") or parsed.get("stderr") or "").strip()
                summary = tool_name
                if call_context:
                    summary += f" ({call_context[:180]})"
                summary += f": {error_text or 'reported failure'}"
                failures.append(summary)
                continue

            returncode = parsed.get("returncode")
            if isinstance(returncode, int) and returncode != 0:
                stderr_text = str(parsed.get("stderr") or parsed.get("stdout") or "").strip()
                summary = tool_name
                if call_context:
                    summary += f" ({call_context[:180]})"
                summary += f": returncode {returncode}"
                if stderr_text:
                    summary += f" ({stderr_text[:180]})"
                failures.append(summary)

    return failures


def extract_verified_urls(tool_results: list[dict[str, Any]]) -> list[str]:
    """Extract externally verifiable URLs from successful tool output."""
    urls: list[str] = []
    for result in tool_results:
        if not isinstance(result, dict):
            continue
        payload = result.get("content", "")
        try:
            parsed = json.loads(payload) if isinstance(payload, str) else payload
        except Exception:
            parsed = None
        if not isinstance(parsed, dict):
            continue

        success = parsed.get("success")
        returncode = parsed.get("returncode")
        if success is False:
            continue
        if isinstance(returncode, int) and returncode != 0:
            continue

        searchable_fragments = [
            str(parsed.get("stdout") or ""),
            str(parsed.get("message") or ""),
            str(parsed.get("content") or ""),
        ]
        for fragment in searchable_fragments:
            for match in _URL_PATTERN.findall(fragment):
                if match not in urls:
                    urls.append(match)
    return urls


def likely_unverified_success_claim(text: str) -> bool:
    """Return True when text reads like a blanket success claim without failure acknowledgment."""
    if not text:
        return False
    return bool(_SUCCESS_CLAIM_PATTERNS.search(text)) and not bool(
        _FAILURE_ACK_PATTERNS.search(text)
    )


def response_claims_failed_targets_succeeded(
    text: str, failure_summaries: list[str]
) -> bool:
    """Return True when text positively claims success for a target that had failed tools."""
    if not text or not failure_summaries:
        return False

    normalized = text.lower()
    if re.search(r"(?i)\b(?:both\s+targets|both\s+destinations|functionally\s+complete)\b", text):
        return True

    target_tokens: set[str] = set()
    for summary in failure_summaries:
        lowered = summary.lower()
        for host in re.findall(r"[a-z0-9-]+(?:\.[a-z0-9-]+)+", lowered):
            target_tokens.add(host)
            target_tokens.add(host.split(".")[0])
        for token in re.findall(r"\b[a-z][a-z0-9-]{4,}\b", lowered):
            if token not in {"returncode", "reported", "failure", "resolve", "origin", "error"}:
                target_tokens.add(token)

    for token in target_tokens:
        if token not in normalized:
            continue
        for token_match in re.finditer(rf"(?i){re.escape(token)}", text):
            window_start = max(0, token_match.start() - 80)
            window_end = min(len(text), token_match.end() + 80)
            window = text[window_start:window_end]
            lowered_window = window.lower()

            if re.search(
                r"(?i)\b(?:pushed|published|deployed|uploaded|posted|succeeded|successful|live|complete|done|up-to-date)\b",
                window,
            ):
                return True

            if "verified" in lowered_window and not re.search(
                r"(?i)\b(?:not|could not|couldn't|unable to|failed to|unverified)\b.{0,24}\bverified\b",
                window,
            ):
                return True

    return False


def build_safe_publish_summary(
    verified_urls: list[str],
    failure_summaries: list[str],
) -> str:
    """Build a deterministic publish summary from verified successes and failures."""
    verified_here_now = [url for url in verified_urls if "here.now" in url]
    other_verified = [url for url in verified_urls if url not in verified_here_now]

    parts: list[str] = []
    if verified_here_now:
        parts.append(f"verified live deployment: {verified_here_now[0]}")
    elif other_verified:
        parts.append(f"verified destination: {other_verified[0]}")
    else:
        parts.append("no publish target was fully verified")

    if failure_summaries:
        parts.append(
            "could not verify at least one requested publish target because tool steps failed: "
            + "; ".join(failure_summaries[:2])
        )

    return ". ".join(parts) + "."


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


def contains_hard_decision_language(text: str) -> bool:
    """Return True when text surfaces a strategic branch that should consult the advisor."""
    if not text:
        return False
    return bool(_HARD_DECISION_PATTERNS.search(text))


def contains_pseudo_tool_syntax(text: str) -> bool:
    """Return True when text contains fake angle-bracket tool markup."""
    if not text:
        return False
    return bool(
        _PSEUDO_TOOL_TAG_PATTERN.search(text)
        or _PSEUDO_TOOL_XML_PATTERN.search(text)
        or _PSEUDO_FUNCTION_XML_PATTERN.search(text)
        or _PSEUDO_TOOL_MARKDOWN_PATTERN.search(text)
    )


def needs_tapback_followup_reply(text: str, executed_tool_names: list[str]) -> bool:
    """Return True when a reaction-only turn still needs a normal text reply."""
    if not text or not executed_tool_names:
        return False

    normalized_tool_names = {name for name in executed_tool_names if name}
    if len(normalized_tool_names) != 1 or not normalized_tool_names.issubset(
        REACTION_TOOL_NAMES
    ):
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
    accumulated_failed_tool_summaries: list[str] = []
    accumulated_verified_urls: list[str] = []

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
            requested_skill_url = (
                latest_user_skill_url(messages)
                if "add_skill" in allowed_tool_names_set
                else ""
            )
            pseudo_tool_syntax_detected = contains_pseudo_tool_syntax(content_text)
            hard_decision_detected = (
                "consult_advisor" in allowed_tool_names_set
                and contains_hard_decision_language(content_text)
            )

            if (
                not forced_finish
                and content_text
                and pseudo_tool_syntax_detected
                and action_intent_retries_used < max_action_intent_retries
            ):
                action_intent_retries_used += 1
                logger.warning(
                    "Agentic loop: detected invalid pseudo-tool syntax instead of real tool calls; "
                    "nudging model to emit structured tool_calls (attempt %d/%d).",
                    action_intent_retries_used,
                    max_action_intent_retries,
                )
                messages.append(message)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "[System: Your last reply used invalid pseudo-tool markup like "
                            '<read(filepath="foo")>. That text is never executed. Do not '
                            "emit angle-bracket tags, XML-style calls, or any plain-text "
                            "representation of a tool call. Use actual structured tool_calls only. "
                            "If this is a planning-heavy repo task, your next response must either "
                            "make real inspection tool calls (read/glob/grep/bash) or call "
                            "consult_advisor for the strategic question.]"
                        ),
                    },
                )
                continue

            if (
                not forced_finish
                and content_text
                and hard_decision_detected
                and action_intent_retries_used < max_action_intent_retries
            ):
                action_intent_retries_used += 1
                logger.warning(
                    "Agentic loop: detected hard-decision text without an advisor consultation; "
                    "nudging model to call consult_advisor (attempt %d/%d).",
                    action_intent_retries_used,
                    max_action_intent_retries,
                )
                messages.append(message)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "[System: You surfaced a hard strategic decision. Do not stop "
                            "here. Call consult_advisor with the exact decision you need "
                            "resolved plus any options, constraints, or failed attempts. "
                            "After the tool returns, continue executing.]"
                        ),
                    },
                )
                continue

            if (
                not forced_finish
                and requested_skill_url
                and executed_tool_rounds == 0
                and action_intent_retries_used < max_action_intent_retries
            ):
                action_intent_retries_used += 1
                hesitation = bool(_SKILL_URL_HESITATION_PATTERNS.search(content_text))
                logger.warning(
                    "Agentic loop: user provided a skill URL but model returned text instead "
                    "of calling add_skill; nudging install flow%s (attempt %d/%d).",
                    " after self-blocking" if hesitation else "",
                    action_intent_retries_used,
                    max_action_intent_retries,
                )
                messages.append(message)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "[System: The user provided a skill URL. Do not pre-screen, reject, "
                            "or manually fetch that URL yourself. Call add_skill with the exact "
                            "URL now so the actual skill installer can fetch it, run the real "
                            "prompt-injection scan, validate the SKILL.md format, install it, "
                            "and auto-activate it if appropriate. Only after the add_skill tool "
                            "returns should you describe success or failure. URL: "
                            f"{requested_skill_url}]"
                        ),
                    }
                )
                continue

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
                        "'%s' instead of calling a reaction tool; nudging model "
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
                                "reactions must be sent via the correct "
                                "tool for the current channel. For iMessage, "
                                "call send_tapback with message_handle. For "
                                "Telegram, call send_telegram_reaction with "
                                "chat_id and message_id. Otherwise reply to "
                                "the user's message in normal conversational "
                                "text. Do not output bare reaction words as text.]"
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
                            "For planning-heavy repo work, planning text alone is not progress: "
                            "inspect with read/glob/grep/bash, or call a consultation tool if the "
                            "blocker is strategic or review-related. "
                            "If you said you're activating a skill, call "
                            "activate_skill. If you said you're running a command, "
                            "call bash. If you said you're sending a tapback or "
                            "If the blocker is a hard strategic decision, call "
                            "consult_advisor first, then continue executing. If you need a "
                            "correctness/risk pass on the current approach, call "
                            "consult_reviewer first, then continue executing. "
                            "reaction, call the correct reaction tool for the "
                            "current channel: send_tapback with message_handle "
                            "for iMessage, or send_telegram_reaction with "
                            "chat_id/message_id for Telegram. If you said "
                            "you're publishing, pushing, "
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
                    "Agentic loop: reaction tool executed but model replied with "
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
                            "[System: You already handled the reaction internally. "
                            "Now reply to the user's message normally in plain text. "
                            "Do not mention the tapback, Telegram reaction, "
                            "send_tapback, send_telegram_reaction, tool calls, "
                            "or any internal action. Continue the conversation "
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

            if (
                not forced_finish
                and latest_user_requests_publish(messages)
                and accumulated_failed_tool_summaries
                and (
                    likely_unverified_success_claim(content_text)
                    or response_claims_failed_targets_succeeded(
                        content_text, accumulated_failed_tool_summaries
                    )
                )
                and action_intent_retries_used < max_action_intent_retries
            ):
                action_intent_retries_used += 1
                logger.warning(
                    "Agentic loop: final response claimed publish/deploy success despite "
                    "failed tool results; requesting a grounded correction (attempt %d/%d).",
                    action_intent_retries_used,
                    max_action_intent_retries,
                )
                messages.append(message)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "[System: One or more publish/deploy steps failed in tool output. "
                            "Failed tool results are ground truth. Do not claim those targets "
                            "succeeded, and do not invent URLs, remotes, or platform endpoints. "
                            "Reply with only the destinations that were actually verified by "
                            "successful tool output, and explicitly name any target that failed "
                            "or could not be verified. If further recovery is possible, make the "
                            "needed tool_calls first. Recent failed tools: "
                            + "; ".join(accumulated_failed_tool_summaries[:3])
                            + "]"
                        ),
                    }
                )
                continue

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

            if (
                latest_user_requests_publish(messages)
                and accumulated_failed_tool_summaries
                and (
                    likely_unverified_success_claim(content_text)
                    or response_claims_failed_targets_succeeded(
                        content_text, accumulated_failed_tool_summaries
                    )
                    or forced_finish
                )
            ):
                return build_safe_publish_summary(
                    verified_urls=accumulated_verified_urls,
                    failure_summaries=accumulated_failed_tool_summaries,
                )

            return safe_strip_markdown(content_text) if content_text else ""

        # ── Execute tool calls and feed results back ──────────────────────────
        runtime_messages_token = set_tool_runtime_messages([*messages, message])
        try:
            tool_results = await execute_tool_calls(
                message, allowed_tool_names=allowed_tool_names
            )
        finally:
            reset_tool_runtime_messages(runtime_messages_token)
        recent_failures = summarize_recent_tool_failures(
            tool_results=tool_results,
            tool_calls=tool_calls,
        )
        recent_verified_urls = extract_verified_urls(tool_results)
        for failure in recent_failures:
            if failure not in accumulated_failed_tool_summaries:
                accumulated_failed_tool_summaries.append(failure)
        for url in recent_verified_urls:
            if url not in accumulated_verified_urls:
                accumulated_verified_urls.append(url)
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
