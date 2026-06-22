"""API call functions and response processing."""

import asyncio
import json
import logging
import re
import time
import traceback
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Awaitable, Callable, Optional

import aiohttp
import re as _re

import strip_markdown

from tools import TOOLS, validate_tool_args

logger = logging.getLogger(__name__)


# ── Global rate limiter (token bucket) ────────────────────────────────────────
# NVIDIA free-tier: 40 RPM → 1 request every 1.5s
# Bumped to 2.0s (30 RPM effective) for safety cushion against upstream jitter.
_RPM_LIMIT = 40
_MIN_INTERVAL = 2.0  # seconds between requests; floor at 1.5s for 40 RPM
_rate_lock = asyncio.Lock()
_last_request_time: float = 0.0

# Upper bound on a single 429 backoff sleep.  A saturated free-tier window is a
# rolling 60s, so the old 15s cap could exhaust all retries before the window
# cleared.  20s gives a 5-retry sequence (3, 9, 20, 20) ≈ 52s of coverage while
# staying short enough not to blow past benchmark/turn timeouts.
_MAX_RATELIMIT_BACKOFF = 20.0


# Substrings that mark a provider error as transient/server-side rather than a
# permanent client mistake.  When a 200 body carries one of these, retrying the
# request almost always succeeds, so we treat them like rate limits.
_TRANSIENT_API_ERROR_MARKERS = (
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
    "connection reset",
    "econnreset",
)


def _is_transient_api_error(error_msg: str) -> bool:
    """Return True when a provider error string looks transient/retryable."""
    if not error_msg:
        return False
    lowered = error_msg.lower()
    return any(marker in lowered for marker in _TRANSIENT_API_ERROR_MARKERS)


# A subset of transient markers that indicate the provider failed to *parse /
# serialize the model's own output* (a malformed tool-call ``arguments`` JSON),
# rather than a capacity/network blip.  For these, re-sending the identical
# payload reproduces the identical bad generation, so the retry must perturb
# sampling to make the model emit different tokens.
_PARSE_CLASS_ERROR_MARKERS = (
    "unterminated string",
    "expecting value",
    "expecting ',' delimiter",
    "expecting property name",
    "invalid \\escape",
    "extra data",
    "jsondecode",
)


def _is_parse_class_api_error(error_msg: str) -> bool:
    """Return True when the error looks like a malformed-model-output parse failure."""
    if not error_msg:
        return False
    lowered = error_msg.lower()
    return any(marker in lowered for marker in _PARSE_CLASS_ERROR_MARKERS)


def _perturb_sampling_for_retry(payload: dict[str, Any], attempt: int) -> None:
    """Nudge sampling params in-place so a deterministic parse failure can recover.

    The provider keeps choking on the same serialized tool call because the
    identical payload yields the identical greedy generation.  Bumping the
    temperature a little (and clearing any fixed seed) changes the emitted
    tokens enough for the provider to serialize them, without meaningfully
    changing task behavior.
    """
    base_temp = payload.get("temperature")
    try:
        base_temp = float(base_temp)
    except (TypeError, ValueError):
        base_temp = 0.6
    # Escalate gently: +0.15 per attempt, capped so we never go incoherent.
    payload["temperature"] = round(min(base_temp + 0.15 * attempt, 0.95), 3)
    # A fixed seed would defeat the perturbation — drop it for the retry.
    payload.pop("seed", None)


def _retry_after_seconds(resp: Any) -> Optional[float]:
    """Parse a ``Retry-After`` header into seconds, or None when absent/unusable.

    The server is authoritative about when its rate-limit window will clear, so
    when it sends ``Retry-After`` we prefer that over blind exponential backoff.
    Handles both the numeric (delay-seconds) and HTTP-date header forms.
    """
    headers = getattr(resp, "headers", None)
    if not headers:
        return None
    try:
        raw = headers.get("Retry-After") or headers.get("retry-after")
    except AttributeError:
        return None
    if not raw:
        return None
    raw = str(raw).strip()

    # Numeric form: a delay in seconds.
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass

    # HTTP-date form: an absolute timestamp to wait until.
    try:
        retry_dt = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if retry_dt is None:
        return None
    if retry_dt.tzinfo is None:
        retry_dt = retry_dt.replace(tzinfo=timezone.utc)
    return max(0.0, (retry_dt - datetime.now(timezone.utc)).total_seconds())


async def _rate_limit_wait() -> None:
    """Block until we're allowed to make the next API request."""
    global _last_request_time
    async with _rate_lock:
        now = time.monotonic()
        elapsed = now - _last_request_time
        if elapsed < _MIN_INTERVAL:
            wait = _MIN_INTERVAL - elapsed
            logger.debug("Rate limiter: waiting %.2fs", wait)
            await asyncio.sleep(wait)
        _last_request_time = time.monotonic()


def _message_content_to_text(content: Any) -> str:
    """Extract plain text from content that may include multimodal blocks."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item.get("content"), str):
                parts.append(item["content"])
        return "\n".join(part for part in parts if part).strip()

    return "" if content is None else str(content)


TOOL_LEAK_PATTERNS = [
    "```bash",
    "```sh",
    "```shell",
    "tool_calls",
    "<|tool_call|>",
    "<|tool_calls|>",
    "<function_",
]

# Common code fence openings that indicate leaked code output.
_CODE_LEAK_FENCE_PATTERNS = [
    "```html",
    "```css",
    "```javascript",
    "```js",
    "```jsx",
    "```ts",
    "```tsx",
    "```python",
    "```py",
    "```json",
    "```xml",
    "```svg",
    "```go",
    "```rs",
    "```rust",
    "```java",
    "```cpp",
    "```c++",
    "```c",
    "```php",
    "```rb",
    "```ruby",
    "```sql",
    "```yaml",
    "```yml",
    "```toml",
    "```ini",
    "```dockerfile",
    "```makefile",
    "```md",
    "```markdown",
    "```txt",
    "```text",
    "```diff",
]

# Matches fenced code blocks (``` ... ```) and inline code (`...`).
_CODE_FENCE_RE = _re.compile(r"(```[\s\S]*?```|`[^`\n]+`)")


def safe_strip_markdown(text: str) -> str:
    """Strip markdown formatting while preserving content inside code fences/spans.

    ``strip_markdown`` treats ``__word__`` as bold markup and removes the
    underscores, which mangles Python dunder names like ``__name__``.  This
    wrapper extracts all code spans/blocks first, strips markdown on the
    remaining prose, then splices the original code back in.
    """
    if not text:
        return text

    placeholders: list[str] = []

    def _stash(match: _re.Match) -> str:
        placeholders.append(match.group(0))
        return f"\x00CODE{len(placeholders) - 1}\x00"

    protected = _CODE_FENCE_RE.sub(_stash, text)
    stripped = strip_markdown.strip_markdown(protected)

    for idx, original in enumerate(placeholders):
        stripped = stripped.replace(f"\x00CODE{idx}\x00", original)

    return stripped


def detect_tool_leak(content: str) -> bool:
    """Detect internal tool-call/command leakage in assistant text output."""
    if not content:
        return False

    content_lower = content.lower()
    if any(pattern in content_lower for pattern in TOOL_LEAK_PATTERNS):
        return True

    # Detect leaked code fences (HTML, CSS, JS, Python, etc.)
    if any(pattern in content_lower for pattern in _CODE_LEAK_FENCE_PATTERNS):
        return True

    # Extra guard for raw shell command dumps without explicit tool_calls.
    stripped = content_lower.strip()
    if stripped.startswith("curl ") or stripped.startswith("curl -x"):
        return True
    if _re.search(r"<[a-z_][\w]*\([^<>]*\)>", stripped):
        return True

    return False


def _extract_inferred_call(
    candidate: dict[str, Any], call_id: str
) -> dict[str, Any] | None:
    """Normalize a candidate object into a tool_call payload."""
    function_obj: dict[str, Any] | None = None

    if isinstance(candidate.get("function"), dict):
        function_obj = candidate["function"]
    elif isinstance(candidate.get("function_call"), dict):
        function_obj = candidate["function_call"]
    elif "name" in candidate:
        function_obj = {
            "name": candidate.get("name"),
            "arguments": candidate.get("arguments", {}),
        }

    if not function_obj:
        return None

    func_name = function_obj.get("name")
    func_args = function_obj.get("arguments", {})
    if not isinstance(func_name, str) or func_name not in TOOLS:
        return None

    if isinstance(func_args, str):
        try:
            func_args = json.loads(func_args)
        except Exception:
            return None

    if not isinstance(func_args, dict):
        return None

    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": func_name,
            "arguments": json.dumps(func_args),
        },
    }


def _parse_json_tool_calls_blob(blob: str) -> list[dict[str, Any]]:
    """Parse JSON-ish tool-call payloads from model text output."""
    candidates: list[str] = []
    stripped = blob.strip()
    candidates.append(stripped)

    fenced_match = re.fullmatch(
        r"```(?:json|javascript|js)?\s*\n([\s\S]*?)\n```", stripped
    )
    if fenced_match:
        candidates.append(fenced_match.group(1).strip())

    tag_match = re.search(r"<\|tool_call\|>\s*([\s\S]+)", stripped)
    if tag_match:
        candidates.append(tag_match.group(1).strip())

    parsed_calls: list[dict[str, Any]] = []
    for idx, candidate_text in enumerate(candidates, start=1):
        try:
            data = json.loads(candidate_text)
        except Exception:
            continue

        raw_items: list[Any]
        if isinstance(data, dict) and isinstance(data.get("tool_calls"), list):
            raw_items = data["tool_calls"]
        elif isinstance(data, list):
            raw_items = data
        elif isinstance(data, dict):
            raw_items = [data]
        else:
            continue

        for call_num, raw_item in enumerate(raw_items, start=1):
            if not isinstance(raw_item, dict):
                continue
            inferred = _extract_inferred_call(
                raw_item, call_id=f"inferred_json_{idx}_{call_num}"
            )
            if inferred:
                parsed_calls.append(inferred)

        if parsed_calls:
            return parsed_calls

    return []


def _coerce_xml_parameter_value(raw_value: str) -> Any:
    """Coerce simple XML parameter text into a JSON-like Python value."""
    value = raw_value.strip()
    if not value:
        return ""

    if re.fullmatch(r"-?\d+", value):
        try:
            return int(value)
        except Exception:
            return value

    if re.fullmatch(r"-?\d+\.\d+", value):
        try:
            return float(value)
        except Exception:
            return value

    lowered = value.lower()
    if lowered in {"true", "false", "null"}:
        try:
            return json.loads(lowered)
        except Exception:
            return value

    if value.startswith("{") or value.startswith("[") or value.startswith('"'):
        try:
            return json.loads(value)
        except Exception:
            return value

    return value


def _parse_xml_tool_call_blob(blob: str) -> list[dict[str, Any]]:
    """Parse XML-style function markup emitted by some models."""
    stripped = blob.strip()
    match = re.fullmatch(
        r"(?is)<function_(?P<name>[a-zA-Z_][\w-]*)>\s*(?P<body>.*?)\s*</function_(?P=name)>",
        stripped,
    )
    if not match:
        return []

    tool_name = match.group("name").replace("-", "_")
    if tool_name not in TOOLS:
        return []

    body = match.group("body").strip()
    parameters: dict[str, Any] = {}
    for param_match in re.finditer(
        r'(?is)<parameter\s+name=["\'](?P<name>[^"\']+)["\']\s*>(?P<value>.*?)</parameter>',
        body,
    ):
        param_name = param_match.group("name").strip()
        if not param_name:
            continue
        parameters[param_name] = _coerce_xml_parameter_value(param_match.group("value"))

    if not parameters and body:
        return []

    return [
        {
            "id": "inferred_xml_1",
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(parameters),
            },
        }
    ]


def infer_tool_calls_from_content(content: str) -> list[dict[str, Any]]:
    """Infer strict fallback tool calls when model emits command blocks as plain text.

    This only accepts a single pure ```bash fenced block to avoid accidental execution
    from normal explanatory text.
    """
    if not content:
        return []

    stripped = content.strip()

    # Check for bare tool_name{json} pseudo-tool syntax
    bare_json_calls = _parse_bare_json_tool_calls(stripped)
    if bare_json_calls:
        return bare_json_calls

    match = re.fullmatch(r"```(?:bash|sh|shell)\s*\n([\s\S]*?)\n```", stripped)
    if match:
        command = match.group(1).strip()
        if command:
            return [
                {
                    "id": "inferred_bash_1",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"command": command}),
                    },
                }
            ]

    xml_calls = _parse_xml_tool_call_blob(content)
    if xml_calls:
        return xml_calls

    return _parse_json_tool_calls_blob(content)


def _parse_bare_json_tool_calls(content: str) -> list[dict[str, Any]]:
    """Parse bare tool_name{json} pseudo-tool syntax from model text.

    Handles formats like:
        declare_message_count{"count": 1}
        send_message{"text": "Hello"}
        recall{"query": "Alex name and location"}
    """
    _BARE_TOOL_RE = re.compile(
        r"(?:^|[\r\n])\s*"
        r"([a-z][a-z0-9_]*)"
        r"\s*(\{(?:[^{}]|(?:\{[^{}]*\}))*\})",
        re.MULTILINE | re.DOTALL,
    )
    matches = _BARE_TOOL_RE.findall(content)
    if not matches:
        return []

    calls: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for idx, (name, args_str) in enumerate(matches):
        name = name.strip()
        if name in seen_names:
            continue
        # Skip if it looks like a variable assignment or other JS/Python code
        if re.search(r"^[a-z]+\s*=\s*", content):
            continue
        try:
            parsed_args = json.loads(args_str.strip())
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(parsed_args, dict):
            continue
        calls.append(
            {
                "id": f"inferred_bare_json_{idx + 1}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(parsed_args),
                },
            }
        )
        seen_names.add(name)

    return calls if len(calls) <= 3 else []  # safety: don't recover too many


def _extract_allowed_tool_names(payload: dict[str, Any]) -> set[str]:
    """Extract allowed tool names from a payload's declared tool schema."""
    allowed: set[str] = set()
    for tool in payload.get("tools", []):
        if not isinstance(tool, dict):
            continue

        function_obj = tool.get("function", {})
        if not isinstance(function_obj, dict):
            continue

        name = function_obj.get("name")
        if isinstance(name, str) and name:
            allowed.add(name)

    return allowed


def _coerce_stream_delta_text(content: Any) -> str:
    """Extract text content from a streamed delta payload."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text_value = item.get("text")
            if isinstance(text_value, str):
                parts.append(text_value)
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)

    return ""


def _merge_stream_delta(
    assembled_message: dict[str, Any], delta: dict[str, Any]
) -> str:
    """Merge a chat-completions SSE delta into an assembled assistant message."""
    role = delta.get("role")
    if isinstance(role, str) and role:
        assembled_message["role"] = role

    content_delta = _coerce_stream_delta_text(delta.get("content"))
    if content_delta:
        assembled_message["content"] = (
            assembled_message.get("content", "") + content_delta
        )

    tool_call_deltas = delta.get("tool_calls")
    if isinstance(tool_call_deltas, list):
        tool_calls = assembled_message.setdefault("tool_calls", [])
        for tool_call_delta in tool_call_deltas:
            if not isinstance(tool_call_delta, dict):
                continue

            index = tool_call_delta.get("index", len(tool_calls))
            if not isinstance(index, int) or index < 0:
                index = len(tool_calls)

            while len(tool_calls) <= index:
                tool_calls.append(
                    {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                )

            current_tool_call = tool_calls[index]
            tool_call_id = tool_call_delta.get("id")
            if isinstance(tool_call_id, str) and tool_call_id:
                current_tool_call["id"] = tool_call_id

            tool_call_type = tool_call_delta.get("type")
            if isinstance(tool_call_type, str) and tool_call_type:
                current_tool_call["type"] = tool_call_type

            function_delta = tool_call_delta.get("function")
            if not isinstance(function_delta, dict):
                continue

            current_function = current_tool_call.setdefault(
                "function", {"name": "", "arguments": ""}
            )
            function_name = function_delta.get("name")
            if isinstance(function_name, str) and function_name:
                current_function["name"] += function_name

            function_arguments = function_delta.get("arguments")
            if isinstance(function_arguments, str) and function_arguments:
                current_function["arguments"] += function_arguments

    return content_delta


def _finalize_stream_message(assembled_message: dict[str, Any]) -> dict[str, Any]:
    """Normalize the assembled streamed assistant message to standard shape."""
    finalized_message = {
        "role": assembled_message.get("role", "assistant"),
        "content": assembled_message.get("content", "") or None,
    }

    tool_calls = assembled_message.get("tool_calls") or []
    normalized_tool_calls: list[dict[str, Any]] = []
    for index, tool_call in enumerate(tool_calls, start=1):
        if not isinstance(tool_call, dict):
            continue
        function_obj = tool_call.get("function") or {}
        if not isinstance(function_obj, dict):
            function_obj = {}

        function_name = function_obj.get("name") or ""
        arguments = function_obj.get("arguments") or "{}"
        if not function_name:
            continue

        normalized_tool_calls.append(
            {
                "id": tool_call.get("id") or f"stream_call_{index}",
                "type": tool_call.get("type") or "function",
                "function": {
                    "name": function_name,
                    "arguments": arguments,
                },
            }
        )

    if normalized_tool_calls:
        finalized_message["tool_calls"] = normalized_tool_calls

    return finalized_message


async def _read_streaming_chat_response(
    resp: aiohttp.ClientResponse,
    stream_chunk_callback: Optional[Callable[[str], Awaitable[None]]] = None,
) -> dict[str, Any]:
    """Read an SSE chat-completions response and assemble a standard response payload."""
    assembled_message: dict[str, Any] = {
        "role": "assistant",
        "content": "",
        "tool_calls": [],
    }

    async def _handle_sse_line(line: str) -> bool:
        """Process one complete SSE line. Returns True when the stream is done."""
        if not line or not line.startswith("data:"):
            return False

        payload = line[5:].strip()
        if not payload:
            return False
        if payload == "[DONE]":
            return True

        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError as exc:
            # A genuinely malformed/partial SSE line should never abort the
            # whole stream (and thus the agentic run).  Skip it and keep
            # assembling — large tool-call argument payloads in particular
            # must not be lost to a single bad chunk.
            logger.warning("Skipping unparseable SSE chunk: %s", exc)
            return False

        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            return False

        choice = choices[0]
        if not isinstance(choice, dict):
            return False

        delta = choice.get("delta")
        if not isinstance(delta, dict):
            return False

        content_delta = _merge_stream_delta(assembled_message, delta)
        if content_delta and stream_chunk_callback:
            await stream_chunk_callback(content_delta)
        return False

    # ``resp.content`` is a byte StreamReader that yields arbitrary network
    # chunks, NOT complete lines — a single SSE ``data:`` line carrying a large
    # tool-call ``arguments`` payload (e.g. an HTML file for ``write``) can be
    # split across reads.  Buffer raw bytes and only parse complete newline-
    # terminated lines so partial JSON is never handed to ``json.loads``.
    buffer = b""
    done = False
    async for raw_chunk in resp.content.iter_any():
        if not raw_chunk:
            continue
        buffer += raw_chunk
        while b"\n" in buffer:
            line_bytes, buffer = buffer.split(b"\n", 1)
            line = line_bytes.decode("utf-8", errors="replace").strip()
            if await _handle_sse_line(line):
                done = True
                break
        if done:
            break

    # Flush any trailing complete line left in the buffer (no final newline).
    if not done and buffer.strip():
        await _handle_sse_line(buffer.decode("utf-8", errors="replace").strip())

    return {"choices": [{"message": _finalize_stream_message(assembled_message)}]}


async def api_call_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    json_data: dict[str, Any],
    headers: dict[str, str],
    max_retries: int = 5,
    backoff: float = 3.0,
    stream: bool = False,
    stream_chunk_callback: Optional[Callable[[str], Awaitable[None]]] = None,
) -> dict[str, Any]:
    """Make an API call with retry logic for transient errors."""
    request_headers = headers.copy()
    request_headers["Accept"] = (
        "text/event-stream"
        if stream
        else request_headers.get("Accept", "application/json")
    )
    request_payload = json_data.copy()
    request_payload["stream"] = stream

    # The provider rejects an empty ``tools`` array outright ("`tools` must not
    # be an empty array. Either provide at least one tool or omit the field
    # entirely."), which kills every tool-free inference path (orchestrator
    # plan, advisor, judges, reminders, self-heal — all pass ``tools=[]``).
    # Normalize here so callers can keep expressing "no tools" as ``[]`` while
    # the wire request omits the field. ``tool_choice`` is meaningless without
    # tools, so drop it too.
    if not request_payload.get("tools"):
        request_payload.pop("tools", None)
        request_payload.pop("tool_choice", None)

    for attempt in range(max_retries):
        await _rate_limit_wait()
        try:
            async with session.post(
                url, json=request_payload, headers=request_headers
            ) as resp:
                if stream:
                    if resp.status >= 400:
                        try:
                            response_data = await resp.json(content_type=None)
                        except Exception:
                            error_text = await resp.text()
                            response_data = {
                                "error": {"message": error_text or str(resp.status)}
                            }
                    elif resp.content_type != "text/event-stream":
                        response_data = await resp.json(content_type=None)
                    else:
                        response_data = await _read_streaming_chat_response(
                            resp,
                            stream_chunk_callback=stream_chunk_callback,
                        )
                else:
                    response_data = await resp.json(content_type=None)

                logger.debug(
                    "Raw API response:\n%s",
                    json.dumps(response_data, indent=2, default=str),
                )

                # Handle NVIDIA/non-standard rate-limit format: {"status": 429, "title": "..."}
                if resp.status == 429 or response_data.get("status") == 429:
                    if attempt < max_retries - 1:
                        # Prefer the server's Retry-After hint; otherwise fall
                        # back to exponential backoff.  Either way cap the wait so
                        # we don't blow past benchmark/turn timeouts.
                        retry_after = _retry_after_seconds(resp)
                        if retry_after is not None:
                            wait_time = min(retry_after, _MAX_RATELIMIT_BACKOFF)
                        else:
                            wait_time = min(
                                backoff ** (attempt + 1), _MAX_RATELIMIT_BACKOFF
                            )
                        logger.warning(
                            "Rate limit (429) hit, retrying in %.1fs (attempt %d/%d)%s...",
                            wait_time,
                            attempt + 1,
                            max_retries,
                            " [Retry-After]" if retry_after is not None else "",
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    return {"error": {"message": "Rate limit exceeded after retries"}}

                # Check for rate limiting or server errors (standard OpenAI format)
                if "error" in response_data:
                    error = response_data["error"]
                    error_msg = error.get("message", "Unknown error")
                    error_type = error.get("type", "")

                    # Retry on rate limits and transient server-side errors.
                    # Providers intermittently return a 200 body carrying an
                    # error (e.g. a malformed/truncated upstream response such as
                    # "Unterminated string ...") when processing large tool-call
                    # payloads.  These are transient and almost always succeed on
                    # a retry, so a single bad round must not abort a multi-step
                    # task (e.g. build-a-site-then-publish).
                    if (
                        "rate limit" in error_msg.lower()
                        or error_type in ["rate_limit", "server_error"]
                        or _is_transient_api_error(error_msg)
                        or resp.status >= 500
                    ):
                        if attempt < max_retries - 1:
                            wait_time = min(backoff ** (attempt + 1), 15.0)
                            # A parse-class failure means the provider couldn't
                            # serialize the model's own (malformed) tool call.
                            # The same payload reproduces the same bad output, so
                            # perturb sampling before the next attempt.
                            if _is_parse_class_api_error(error_msg):
                                _perturb_sampling_for_retry(request_payload, attempt + 1)
                                logger.warning(
                                    "Parse-class API error (%r); retrying in %ss with "
                                    "temperature=%s (attempt %d/%d)...",
                                    error_msg[:160],
                                    wait_time,
                                    request_payload.get("temperature"),
                                    attempt + 1,
                                    max_retries,
                                )
                            else:
                                logger.warning(
                                    "Transient API error (%r), retrying in %ss "
                                    "(attempt %d/%d)...",
                                    error_msg[:160],
                                    wait_time,
                                    attempt + 1,
                                    max_retries,
                                )
                            await asyncio.sleep(wait_time)
                            continue

                    # Return error without retry for other errors
                    return response_data

                return response_data

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt < max_retries - 1:
                wait_time = backoff**attempt
                logger.warning(f"API call failed: {e}, retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"API call failed after {max_retries} attempts: {e}")
                return {"error": {"message": str(e)}}

        except (json.JSONDecodeError, ValueError) as e:
            # Our own parse of the provider's response body failed (truncated /
            # malformed body).  This is a ValueError, NOT an aiohttp.ClientError,
            # so without this clause it would escape the retry loop entirely and
            # surface a raw parser error to the user.  Treat it as transient and
            # perturb sampling like any other parse-class failure.
            if attempt < max_retries - 1:
                wait_time = backoff**attempt
                _perturb_sampling_for_retry(request_payload, attempt + 1)
                logger.warning(
                    "Failed to parse API response body (%r), retrying in %ss "
                    "with temperature=%s (attempt %d/%d)...",
                    str(e)[:160],
                    wait_time,
                    request_payload.get("temperature"),
                    attempt + 1,
                    max_retries,
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(
                    "Failed to parse API response after %d attempts: %s",
                    max_retries,
                    e,
                )
                return {"error": {"message": str(e)}}

    return {"error": {"message": "Max retries exceeded"}}


async def execute_tool_calls(
    message: dict[str, Any],
    allowed_tool_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Execute tool calls from a message and return results."""
    tool_results = []

    # Handle standard OpenAI-style tool_calls
    for tool_call in message.get("tool_calls", []):
        func_name = tool_call["function"]["name"]
        try:
            func_args = json.loads(tool_call["function"]["arguments"])
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(
                "Failed to parse tool-call arguments for %s: %s",
                func_name,
                e,
            )
            tool_results.append(
                {
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "content": json.dumps(
                        {
                            "success": False,
                            "error": f"Invalid JSON arguments: {e}",
                        }
                    ),
                }
            )
            continue

        if allowed_tool_names is not None and func_name not in allowed_tool_names:
            tool_results.append(
                {
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "content": json.dumps(
                        {
                            "success": False,
                            "error": f"Tool not available in current payload: {func_name}",
                        }
                    ),
                }
            )
            continue

        # Validate the tool call arguments
        is_valid, error = validate_tool_args(func_name, func_args)
        if not is_valid:
            tool_results.append(
                {
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "content": json.dumps({"success": False, "error": error}),
                }
            )
            continue

        if func_name in TOOLS:
            try:
                result = await TOOLS[func_name](**func_args)

                # When activate_skill succeeds with content, surface the
                # skill instructions as prominent text so the model treats
                # them as actionable directives rather than a passive JSON
                # acknowledgment.  This aligns with the Agent Skills spec's
                # progressive-disclosure model: once a skill is triggered,
                # its instructions must enter the context window in a form
                # the model will follow.
                if (
                    func_name == "activate_skill"
                    and isinstance(result, dict)
                    and result.get("success")
                    and result.get("content")
                    and not result.get("already_active")
                ):
                    skill_name = result.get("name", "unknown")
                    skill_dir = result.get("skill_dir", "")
                    skill_content = result["content"]
                    tool_result_text = (
                        f"[Skill '{skill_name}' activated successfully]\n"
                        f"Skill directory: {skill_dir}\n\n"
                        f"You MUST follow the instructions below for the current task.\n"
                        f"--- BEGIN SKILL INSTRUCTIONS ---\n"
                        f"{skill_content}\n"
                        f"--- END SKILL INSTRUCTIONS ---\n\n"
                        f"Now proceed with the user's request, following the skill "
                        f"instructions above."
                    )
                    tool_results.append(
                        {
                            "tool_call_id": tool_call["id"],
                            "role": "tool",
                            "content": tool_result_text,
                        }
                    )
                else:
                    tool_results.append(
                        {
                            "tool_call_id": tool_call["id"],
                            "role": "tool",
                            "content": json.dumps(result),
                        }
                    )
            except Exception as e:
                logger.error(f"Tool execution error for {func_name}: {e}")
                tb_text = traceback.format_exc()
                enriched_error = {
                    "success": False,
                    "error": str(e),
                    "exception_type": type(e).__name__,
                    "traceback_snippet": tb_text[-2000:] if tb_text else None,
                }
                tool_results.append(
                    {
                        "tool_call_id": tool_call["id"],
                        "role": "tool",
                        "content": json.dumps(enriched_error),
                    }
                )

    return tool_results


async def process_response(
    response_data: dict[str, Any],
    messages: list[dict[str, Any]],
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
    base_payload: dict[str, Any],
    max_tool_leak_retries: int = 1,
    stream_chunk_callback: Optional[Callable[[str], Awaitable[None]]] = None,
) -> str:
    """Process API response via the agentic loop.

    The first response has already been fetched by the caller; this function
    seeds the agentic loop with it and lets ``run_agentic_loop`` handle all
    subsequent rounds of tool calling until the model produces a final text
    answer or the iteration cap is reached.
    """
    # Lazy import to avoid a circular dependency at module load time.
    from agentic_loop import run_agentic_loop  # noqa: PLC0415

    # Handle top-level API errors before entering the loop.
    if "error" in response_data:
        error_msg = response_data["error"].get("message", "Unknown API error")
        logger.error("API error: %s", error_msg)
        return f"Error: {error_msg}"

    if not response_data.get("choices"):
        logger.error("No choices in initial response: %s", response_data)
        return "Error: No response from API"

    # Hand the initial response directly to the agentic loop.
    # The loop owns the full while-tool-calls cycle and will handle this
    # response as its first iteration without making a redundant API call.
    return await run_agentic_loop(
        messages=messages,
        session=session,
        base_url=base_url,
        api_key=api_key,
        base_payload=base_payload,
        stream_chunk_callback=stream_chunk_callback,
        max_tool_leak_retries=max_tool_leak_retries,
        initial_response_data=response_data,
    )
