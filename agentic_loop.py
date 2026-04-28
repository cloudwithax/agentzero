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
from tools import (
    get_send_message_buffer,
    init_send_message_buffer,
    reset_send_message_buffer,
    reset_tool_runtime_messages,
    set_tool_runtime_messages,
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
_SHORT_REACTION_ACK_PREFIX_PATTERN = re.compile(
    r"^(?:sent|sending|reacted|done)\b",
    re.IGNORECASE,
)

logger = logging.getLogger(__name__)

# Safety cap: maximum number of tool-call rounds before we force a final answer.
DEFAULT_MAX_ITERATIONS = 20
DEFAULT_MAX_ACTION_INTENT_RETRIES = 3
DEFAULT_MAX_TAPBACK_REPLY_RETRIES = 1
DEFAULT_MAX_SEND_MESSAGE_RETRIES = 3
REACTION_TOOL_NAMES = {"send_tapback", "send_telegram_reaction", "send_reaction"}
REACTION_WORD_NAMES = {
    "love",
    "like",
    "dislike",
    "laugh",
    "emphasize",
    "question",
    "party",
    "clap",
    "cry",
    "sob",
    "scream",
    "mindblown",
    "pray",
    "cool",
    "100",
    "hearts",
    "starry",
    "angry",
    "devil",
    "ghost",
    "clown",
    "shrug",
    "eyes",
    "kiss",
    "hug",
    "salute",
    "nerd",
    "trophy",
    "heartbreak",
    "heartonfire",
    "vomit",
    "poo",
    "ok",
    "whale",
    "dove",
    "unicorn",
    "moai",
    "banana",
    "strawberry",
    "champagne",
    "hotdog",
    "yawn",
    "woozy",
    "sleep",
    "scared",
    "handshake",
    "halo",
    "grin",
    "alien",
    "lightning",
    "moon",
    "cursing",
    "zany",
    "lipstick",
    "nailpolish",
    "middlefinger",
    "coder",
    "pill",
    "pumpkin",
    "cupid",
    "hearteyes",
    "writing",
    "santa",
    "christmas",
    "snowman",
    "seenoevil",
}
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

# ── Tool context optimization thresholds ──────────────────────────────────────
_OPTIMIZE_AFTER_ROUNDS = 3  # Skip optimization for first 2 rounds
_OPTIMIZE_EVERY_N_ROUNDS = 2  # Re-optimize every 2 rounds (3, 5, 7, 9...)
_OPTIMIZE_MIN_TOOL_TOKENS = 4000  # Skip if tool context is small


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
                error_text = str(
                    parsed.get("error") or parsed.get("stderr") or ""
                ).strip()
                summary = tool_name
                if call_context:
                    summary += f" ({call_context[:180]})"
                summary += f": {error_text or 'reported failure'}"
                failures.append(summary)
                continue

            returncode = parsed.get("returncode")
            if isinstance(returncode, int) and returncode != 0:
                stderr_text = str(
                    parsed.get("stderr") or parsed.get("stdout") or ""
                ).strip()
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


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}


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
    if re.search(
        r"(?i)\b(?:both\s+targets|both\s+destinations|functionally\s+complete)\b", text
    ):
        return True

    target_tokens: set[str] = set()
    for summary in failure_summaries:
        lowered = summary.lower()
        for host in re.findall(r"[a-z0-9-]+(?:\.[a-z0-9-]+)+", lowered):
            target_tokens.add(host)
            target_tokens.add(host.split(".")[0])
        for token in re.findall(r"\b[a-z][a-z0-9-]{4,}\b", lowered):
            if token not in {
                "returncode",
                "reported",
                "failure",
                "resolve",
                "origin",
                "error",
            }:
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


def text_contains_reaction_emoji(text: str) -> bool:
    """Return True when text contains any Telegram reaction emoji."""
    if not text:
        return False
    try:
        from integrations import TELEGRAM_REACTION_EMOJI_MAP

        return any(emoji in text for emoji in TELEGRAM_REACTION_EMOJI_MAP.values())
    except ImportError:
        return False


def conversation_exposes_reaction_targets(messages: list[dict[str, Any]]) -> bool:
    """Return True when the prompt context includes concrete reaction targets."""
    for message in messages:
        content = _message_content_to_text(message.get("content", ""))
        if (
            "[Available Telegram reaction targets" in content
            or "[Available iMessage tapback handles" in content
        ):
            return True
    return False


def looks_like_short_reaction_ack(text: str) -> bool:
    """Return True for terse `sent! ❤️`-style acknowledgements."""
    normalized = (text or "").strip()
    if not normalized or len(normalized) > 32:
        return False
    if not _SHORT_REACTION_ACK_PREFIX_PATTERN.match(normalized):
        return False

    remainder = _SHORT_REACTION_ACK_PREFIX_PATTERN.sub("", normalized, count=1).strip()
    if not remainder:
        return False

    compact = re.sub(r"[\s.!?]+", "", remainder)
    if not compact:
        return False

    if re.fullmatch(r"[^\w]+", compact):
        return True

    lowered = compact.lower()
    return lowered in REACTION_WORD_NAMES


def needs_tapback_followup_reply(text: str, executed_tool_names: list[str]) -> bool:
    """Return True when a reaction-only turn still needs a normal text reply."""
    if not text or not executed_tool_names:
        return False

    normalized_tool_names = {name for name in executed_tool_names if name}
    if len(normalized_tool_names) != 1 or not normalized_tool_names.issubset(
        REACTION_TOOL_NAMES
    ):
        return False

    return bool(
        TAPBACK_ACK_PATTERN.match(text.strip())
    ) or looks_like_short_reaction_ack(text)


def _estimate_tool_context_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate token count for assistant+tool messages only (len/4 convention)."""
    total = 0
    for msg in messages:
        role = msg.get("role", "")
        if role not in ("assistant", "tool"):
            continue
        content = msg.get("content") or ""
        if isinstance(content, str):
            total += len(content) // 4
        # Count tool_calls arguments
        for tc in msg.get("tool_calls") or []:
            args = tc.get("function", {}).get("arguments", "")
            if args:
                total += len(args) // 4
    return total


def _extract_tool_rounds(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Walk messages and identify tool call rounds.

    Each round = one assistant message with tool_calls + its consecutive tool
    role result messages.  Non-tool messages (system, user, text-only assistant)
    are tagged as "passthrough" and always kept.

    Returns list of dicts:
      - passthrough: {"type": "passthrough", "messages": [...], "start_pos": int, "end_pos": int}
      - round:       {"type": "round", "round_index": int, "assistant_msg": dict,
                       "tool_result_msgs": [...], "start_pos": int, "end_pos": int}
    """
    segments: list[dict[str, Any]] = []
    passthrough_buf: list[dict[str, Any]] = []
    passthrough_start: int | None = None
    round_index = 0
    i = 0

    while i < len(messages):
        msg = messages[i]

        # Detect assistant message with tool_calls
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            # Flush passthrough buffer
            if passthrough_buf:
                segments.append(
                    {
                        "type": "passthrough",
                        "messages": list(passthrough_buf),
                        "start_pos": passthrough_start,
                        "end_pos": i - 1,
                    }
                )
                passthrough_buf.clear()
                passthrough_start = None

            assistant_pos = i
            tool_results: list[dict[str, Any]] = []
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                tool_results.append(messages[j])
                j += 1

            segments.append(
                {
                    "type": "round",
                    "round_index": round_index,
                    "assistant_msg": msg,
                    "tool_result_msgs": tool_results,
                    "start_pos": assistant_pos,
                    "end_pos": j - 1,
                }
            )
            round_index += 1
            i = j
        else:
            if passthrough_start is None:
                passthrough_start = i
            passthrough_buf.append(msg)
            i += 1

    # Flush remaining passthrough
    if passthrough_buf:
        segments.append(
            {
                "type": "passthrough",
                "messages": list(passthrough_buf),
                "start_pos": passthrough_start,
                "end_pos": len(messages) - 1,
            }
        )

    return segments


async def _optimize_tool_context(
    *,
    messages: list[dict[str, Any]],
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
) -> list[dict[str, Any]]:
    """Use the advisor model to prune/summarize old tool call rounds."""
    from prompt_templates import get_template

    # Lazy imports to avoid circular dependency (same pattern as api.py)
    from handler import ADVISOR_MODEL_ID, BASE_PAYLOAD

    segments = _extract_tool_rounds(messages)
    rounds = [s for s in segments if s["type"] == "round"]

    if len(rounds) < 2:
        return messages  # Nothing to optimize

    # Build compact representation for the advisor
    compact_rounds: list[dict[str, Any]] = []
    for r in rounds:
        tool_calls_compact = []
        for tc in r["assistant_msg"].get("tool_calls") or []:
            fn = tc.get("function", {})
            args_preview = (fn.get("arguments") or "")[:200]
            tool_calls_compact.append(
                {
                    "name": fn.get("name", "unknown"),
                    "arguments_preview": args_preview,
                }
            )

        results_compact = []
        for tr in r["tool_result_msgs"]:
            content_raw = tr.get("content") or ""
            # Detect success/failure
            success = True
            try:
                parsed = (
                    json.loads(content_raw)
                    if isinstance(content_raw, str)
                    else content_raw
                )
                if isinstance(parsed, dict):
                    if parsed.get("success") is False:
                        success = False
                    rc = parsed.get("returncode")
                    if isinstance(rc, int) and rc != 0:
                        success = False
            except Exception:
                pass
            results_compact.append(
                {
                    "tool_call_id": tr.get("tool_call_id", ""),
                    "content_preview": content_raw[:500]
                    if isinstance(content_raw, str)
                    else str(content_raw)[:500],
                    "success": success,
                }
            )

        compact_rounds.append(
            {
                "round_index": r["round_index"],
                "tool_calls": tool_calls_compact,
                "results": results_compact,
            }
        )

    # Call the advisor model
    system_prompt = get_template("tool_context_optimizer")
    user_content = json.dumps(compact_rounds, indent=2)

    payload = BASE_PAYLOAD.copy()
    payload["model"] = ADVISOR_MODEL_ID
    payload["tools"] = []
    payload["max_tokens"] = 4096
    payload["temperature"] = 0.1
    payload["messages"] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    response_data = await api_call_with_retry(
        session,
        base_url,
        payload,
        {"Authorization": f"Bearer {api_key}"},
    )

    if "error" in response_data or not response_data.get("choices"):
        logger.warning(
            "Tool context optimizer: API error, returning original messages."
        )
        return messages

    raw_output = _message_content_to_text(
        response_data["choices"][0]["message"].get("content", "")
    )

    # Parse JSON — strip code fences if present
    cleaned = raw_output.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning(
            "Tool context optimizer: failed to parse JSON response, keeping original."
        )
        return messages

    decisions_list = parsed.get("decisions")
    if not isinstance(decisions_list, list):
        logger.warning(
            "Tool context optimizer: invalid decisions structure, keeping original."
        )
        return messages

    # Index decisions by round_index
    decisions_by_idx: dict[int, dict[str, Any]] = {}
    for d in decisions_list:
        idx = d.get("round_index")
        if isinstance(idx, int):
            decisions_by_idx[idx] = d

    # Hard override: force KEEP on the most recent round
    if rounds:
        last_round_idx = rounds[-1]["round_index"]
        decisions_by_idx[last_round_idx] = {
            "round_index": last_round_idx,
            "action": "KEEP",
        }

    # Rebuild messages applying decisions
    rebuilt: list[dict[str, Any]] = []
    for segment in segments:
        if segment["type"] == "passthrough":
            rebuilt.extend(segment["messages"])
            continue

        ridx = segment["round_index"]
        decision = decisions_by_idx.get(ridx, {"action": "KEEP"})
        action = decision.get("action", "KEEP").upper()

        if action == "DROP":
            logger.debug(
                "Tool context optimizer: dropping round %d — %s",
                ridx,
                decision.get("reason", "no reason"),
            )
            continue

        if action == "SUMMARIZE":
            summary = decision.get("summary", "Previous tool interaction.")
            # Keep assistant msg verbatim (preserves tool_calls for format validity)
            rebuilt.append(segment["assistant_msg"])
            # Replace each tool result content with summary
            for tr in segment["tool_result_msgs"]:
                summarized = dict(tr)
                summarized["content"] = f"[Context summary] {summary}"
                rebuilt.append(summarized)
            logger.debug(
                "Tool context optimizer: summarized round %d.",
                ridx,
            )
            continue

        # KEEP (default)
        rebuilt.append(segment["assistant_msg"])
        rebuilt.extend(segment["tool_result_msgs"])

    logger.info(
        "Tool context optimizer: %d messages -> %d messages (%d rounds evaluated).",
        len(messages),
        len(rebuilt),
        len(rounds),
    )
    return rebuilt


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
    max_send_message_retries: int = DEFAULT_MAX_SEND_MESSAGE_RETRIES,
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
    send_message_available = "send_message" in allowed_tool_names_set
    send_buffer_token = init_send_message_buffer() if send_message_available else None

    try:
        result = await _run_agentic_loop_inner(
            messages=messages,
            session=session,
            base_url=base_url,
            api_key=api_key,
            base_payload=base_payload,
            stream_chunk_callback=stream_chunk_callback,
            max_iterations=max_iterations,
            max_tool_leak_retries=max_tool_leak_retries,
            max_action_intent_retries=max_action_intent_retries,
            max_tapback_reply_retries=max_tapback_reply_retries,
            max_send_message_retries=max_send_message_retries,
            initial_response_data=initial_response_data,
            allowed_tool_names=allowed_tool_names,
            allowed_tool_names_set=allowed_tool_names_set,
            send_message_available=send_message_available,
        )
        return result
    finally:
        if send_buffer_token is not None:
            reset_send_message_buffer(send_buffer_token)


async def _run_agentic_loop_inner(
    messages: list[dict[str, Any]],
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
    base_payload: dict[str, Any],
    stream_chunk_callback: Optional[Callable[[str], Awaitable[None]]],
    max_iterations: int,
    max_tool_leak_retries: int,
    max_action_intent_retries: int,
    max_tapback_reply_retries: int,
    max_send_message_retries: int,
    initial_response_data: Optional[dict[str, Any]],
    allowed_tool_names: list[str],
    allowed_tool_names_set: set[str],
    send_message_available: bool,
) -> str:
    tool_leak_retries_used = 0
    action_intent_retries_used = 0
    tapback_reply_retries_used = 0
    send_message_retries_used = 0
    force_send_message_next = False
    executed_tool_rounds = 0
    executed_tool_names: list[str] = []
    pending_response_data: Optional[dict[str, Any]] = initial_response_data
    accumulated_failed_tool_summaries: list[str] = []
    accumulated_verified_urls: list[str] = []
    accumulated_attachments: list[str] = []

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
                # Inject a system nudge so the model gives a final answer.
                # If send_message is enforced, keep ONLY send_message (so the
                # model is forced to deliver via the tool); otherwise strip
                # tools entirely so the model must answer in text.
                if send_message_available:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "[System: You have used the maximum number of "
                                "tool-call rounds. Stop investigating. Deliver "
                                "your final answer to the user NOW by calling "
                                "send_message — that is the only allowed tool. "
                                "Plain-text replies are not delivered.]"
                            ),
                        }
                    )
                    base_tools = base_payload.get("tools") or []
                    forced_tools = [
                        tool
                        for tool in base_tools
                        if isinstance(tool, dict)
                        and tool.get("function", {}).get("name") == "send_message"
                    ]
                    current_payload = {
                        k: v for k, v in base_payload.items() if k != "tools"
                    }
                    if forced_tools:
                        current_payload["tools"] = forced_tools
                        current_payload["tool_choice"] = {
                            "type": "function",
                            "function": {"name": "send_message"},
                        }
                else:
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
            elif force_send_message_next:
                # Previous round returned plain text instead of send_message;
                # the API-level tool_choice forces compliance this round.
                base_tools = base_payload.get("tools") or []
                forced_tools = [
                    tool
                    for tool in base_tools
                    if isinstance(tool, dict)
                    and tool.get("function", {}).get("name") == "send_message"
                ]
                if forced_tools:
                    current_payload = {
                        k: v for k, v in base_payload.items() if k != "tools"
                    }
                    current_payload["tools"] = forced_tools
                    current_payload["tool_choice"] = {
                        "type": "function",
                        "function": {"name": "send_message"},
                    }
                    current_payload["messages"] = messages
                force_send_message_next = False

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
            reaction_ack_without_tool = (
                not forced_finish
                and executed_tool_rounds == 0
                and conversation_exposes_reaction_targets(messages)
                and text_contains_reaction_emoji(content_text)
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
                            "[System: You wrote pseudo-tool syntax but produced zero "
                            "tool calls. Text like <read(...)> is never executed. "
                            "Make real structured tool_calls. Do not reply with "
                            "text — make the tool call.]"
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
                            "[System: You described a decision but produced zero "
                            "tool calls. Call consult_advisor with the decision "
                            "and options. Do not reply with text — make the "
                            "tool call.]"
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
                            "[System: You were given a skill URL but produced zero "
                            "tool calls. Call add_skill with the URL: "
                            f"{requested_skill_url} — do not reply with text, "
                            "make the tool call.]"
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
                    or reaction_ack_without_tool
                    or explicit_tool_required
                )
                and action_intent_retries_used < max_action_intent_retries
            ):
                action_intent_retries_used += 1
                if is_bare_reaction_word(content_text) or reaction_ack_without_tool:
                    reason = "send a reaction"
                    tool_hint = (
                        "Call send_telegram_reaction with chat_id, message_id, "
                        "and reaction from the available targets above"
                    )
                else:
                    reason = "take action"
                    tool_hint = "Make the tool_calls you described"
                logger.warning(
                    "Agentic loop: model tried to %s but produced zero "
                    "tool calls (attempt %d/%d): %s",
                    reason,
                    action_intent_retries_used,
                    max_action_intent_retries,
                    content_text.strip()[:80],
                )
                messages.append(message)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"[System: You tried to {reason} but produced zero "
                            f"tool calls. {tool_hint}. Do not reply with "
                            "text — make the tool call.]"
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
                            "[System: Reaction sent. Now reply to the user's "
                            "actual message in plain text. Do not mention "
                            "reactions, tool calls, or internal actions.]"
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
                            "[System: You claimed success but these tools failed: "
                            + "; ".join(accumulated_failed_tool_summaries[:3])
                            + ". Do not claim failed targets succeeded. Report "
                            "only verified results, or make tool_calls to retry.]"
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

            # ── Enforce send_message: nudge if model returned plain text ──
            if send_message_available:
                send_buffer = get_send_message_buffer()
                if not send_buffer and final_text.strip():
                    if (
                        not forced_finish
                        and send_message_retries_used < max_send_message_retries
                    ):
                        send_message_retries_used += 1
                        logger.warning(
                            "Agentic loop: model returned plain text instead of "
                            "calling send_message; forcing send_message tool "
                            "choice (attempt %d/%d).",
                            send_message_retries_used,
                            max_send_message_retries,
                        )
                        messages.append(message)
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "[System: Plain-text replies are not "
                                    "delivered to the user. You MUST call the "
                                    "send_message tool to deliver your "
                                    "response. Make one send_message call per "
                                    "message bubble — split only when there "
                                    "are distinct beats (setup/punchline, "
                                    "answer/aside, question/context). Do not "
                                    "reply with text — make the tool call(s) "
                                    "now.]"
                                ),
                            }
                        )
                        force_send_message_next = True
                        continue

                    logger.error(
                        "Agentic loop: model refused to call send_message after "
                        "%d nudges (forced_finish=%s); returning error to user.",
                        send_message_retries_used,
                        forced_finish,
                    )
                    return (
                        "Sorry, I hit an internal delivery error. Please send "
                        "that again."
                    )

                if send_buffer:
                    first_channel = send_buffer[0].get("channel", "buffered")
                    if first_channel == "buffered":
                        # Non-messaging channel (CLI, OpenAI-compat): no live
                        # send happened — assemble the buffered messages into
                        # the response so the caller actually sees them.
                        joined = "\n\n".join(
                            str(record.get("text", ""))
                            for record in send_buffer
                            if str(record.get("text", "")).strip()
                        )
                        if accumulated_attachments:
                            return json.dumps(
                                {
                                    "text": joined,
                                    "attachments": accumulated_attachments,
                                }
                            )
                        return joined

                    # Messaging channel: every bubble already shipped via
                    # the live tool dispatch, so suppress the legacy
                    # post-loop send path.
                    return json.dumps(
                        {
                            "text": "",
                            "attachments": [],
                            "delivered_via_tool": True,
                        }
                    )

            if accumulated_attachments:
                return json.dumps({
                    "text": final_text,
                    "attachments": accumulated_attachments,
                })
            return final_text

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
        recent_attachments = extract_outbound_attachments(tool_results, tool_calls)
        for failure in recent_failures:
            if failure not in accumulated_failed_tool_summaries:
                accumulated_failed_tool_summaries.append(failure)
        for url in recent_verified_urls:
            if url not in accumulated_verified_urls:
                accumulated_verified_urls.append(url)
        for url in recent_attachments:
            if url not in accumulated_attachments:
                accumulated_attachments.append(url)
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

        # ── Context optimization ──────────────────────────────────────────
        if (
            executed_tool_rounds >= _OPTIMIZE_AFTER_ROUNDS
            and executed_tool_rounds % _OPTIMIZE_EVERY_N_ROUNDS == 0
            and _estimate_tool_context_tokens(messages) >= _OPTIMIZE_MIN_TOOL_TOKENS
        ):
            try:
                messages = await _optimize_tool_context(
                    messages=messages,
                    session=session,
                    base_url=base_url,
                    api_key=api_key,
                )
            except Exception:
                logger.warning(
                    "Tool context optimization failed; continuing with full context.",
                    exc_info=True,
                )

    # Unreachable — the forced_finish branch always returns — but satisfies mypy.
    return "Task completed."
