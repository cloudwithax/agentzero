"""API call functions and response processing."""

import asyncio
import json
import logging
import re
import uuid
from typing import Any, Awaitable, Callable, Optional

import aiohttp
import strip_markdown

from tools import TOOLS, validate_tool_args

logger = logging.getLogger(__name__)

ASSET_REFERENCE_PATTERN = re.compile(r"asset_id,([A-Za-z0-9-]+)")


def _apply_cache_busting_headers(headers: dict[str, str]) -> dict[str, str]:
    """Add per-request no-cache headers so upstream layers avoid reusing responses."""
    request_headers = headers.copy()
    request_headers["Cache-Control"] = "no-cache, no-store, max-age=0"
    request_headers["Pragma"] = "no-cache"
    request_headers["Expires"] = "0"
    request_headers["X-Request-Id"] = request_headers.get(
        "X-Request-Id", str(uuid.uuid4())
    )
    return request_headers


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


def _append_asset_ids_from_text(
    text: str,
    seen: set[str],
    ordered_asset_ids: list[str],
) -> None:
    for match in ASSET_REFERENCE_PATTERN.finditer(text or ""):
        asset_id = match.group(1).strip()
        if asset_id and asset_id not in seen:
            seen.add(asset_id)
            ordered_asset_ids.append(asset_id)


def _extract_nvcf_asset_ids(messages: Any) -> list[str]:
    """Extract asset IDs from conversation messages for NVCF header wiring."""
    if not isinstance(messages, list):
        return []

    seen: set[str] = set()
    ordered_asset_ids: list[str] = []

    for message in messages:
        if not isinstance(message, dict):
            continue

        content = message.get("content")
        if isinstance(content, str):
            _append_asset_ids_from_text(content, seen, ordered_asset_ids)
            continue

        if not isinstance(content, list):
            continue

        for item in content:
            if not isinstance(item, dict):
                continue

            text_value = item.get("text")
            if isinstance(text_value, str):
                _append_asset_ids_from_text(text_value, seen, ordered_asset_ids)

            content_value = item.get("content")
            if isinstance(content_value, str):
                _append_asset_ids_from_text(content_value, seen, ordered_asset_ids)

            image_obj = item.get("image_url")
            if isinstance(image_obj, dict):
                image_url = image_obj.get("url")
                if isinstance(image_url, str):
                    _append_asset_ids_from_text(image_url, seen, ordered_asset_ids)

    return ordered_asset_ids


TOOL_LEAK_PATTERNS = [
    "```bash",
    "```sh",
    "```shell",
    "tool_calls",
    "<|tool_call|>",
    "<|tool_calls|>",
]


def detect_tool_leak(content: str) -> bool:
    """Detect internal tool-call/command leakage in assistant text output."""
    if not content:
        return False

    content_lower = content.lower()
    if any(pattern in content_lower for pattern in TOOL_LEAK_PATTERNS):
        return True

    # Extra guard for raw shell command dumps without explicit tool_calls.
    stripped = content_lower.strip()
    if stripped.startswith("curl ") or stripped.startswith("curl -x"):
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


def infer_tool_calls_from_content(content: str) -> list[dict[str, Any]]:
    """Infer strict fallback tool calls when model emits command blocks as plain text.

    This only accepts a single pure ```bash fenced block to avoid accidental execution
    from normal explanatory text.
    """
    if not content:
        return []

    stripped = content.strip()
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

    return _parse_json_tool_calls_blob(content)


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

    async for raw_line in resp.content:
        line = raw_line.decode("utf-8").strip()
        if not line or not line.startswith("data:"):
            continue

        payload = line[5:].strip()
        if not payload:
            continue
        if payload == "[DONE]":
            break

        chunk = json.loads(payload)
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            continue

        choice = choices[0]
        if not isinstance(choice, dict):
            continue

        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue

        content_delta = _merge_stream_delta(assembled_message, delta)
        if content_delta and stream_chunk_callback:
            await stream_chunk_callback(content_delta)

    return {"choices": [{"message": _finalize_stream_message(assembled_message)}]}


async def api_call_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    json_data: dict[str, Any],
    headers: dict[str, str],
    max_retries: int = 3,
    backoff: float = 2.0,
    stream: bool = False,
    stream_chunk_callback: Optional[Callable[[str], Awaitable[None]]] = None,
) -> dict[str, Any]:
    """Make an API call with retry logic for transient errors."""
    request_headers = _apply_cache_busting_headers(headers)
    request_headers["Accept"] = (
        "text/event-stream"
        if stream
        else request_headers.get("Accept", "application/json")
    )
    request_payload = json_data.copy()
    request_payload["stream"] = stream

    has_asset_header = any(
        key.lower() == "nvcf-input-asset-references" for key in request_headers
    )
    if not has_asset_header:
        asset_ids = _extract_nvcf_asset_ids(request_payload.get("messages"))
        if asset_ids:
            request_headers["NVCF-INPUT-ASSET-REFERENCES"] = ",".join(asset_ids)

    for attempt in range(max_retries):
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
                    response_data = await resp.json()

                # Check for rate limiting or server errors
                if "error" in response_data:
                    error = response_data["error"]
                    error_msg = error.get("message", "Unknown error")
                    error_type = error.get("type", "")

                    # Retry on rate limits and certain server errors
                    if "rate limit" in error_msg.lower() or error_type in [
                        "rate_limit",
                        "server_error",
                    ]:
                        if attempt < max_retries - 1:
                            wait_time = backoff**attempt
                            logger.warning(
                                f"Rate limit hit, retrying in {wait_time}s..."
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
        func_args = json.loads(tool_call["function"]["arguments"])

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
                tool_results.append(
                    {
                        "tool_call_id": tool_call["id"],
                        "role": "tool",
                        "content": json.dumps(result),
                    }
                )
            except Exception as e:
                logger.error(f"Tool execution error for {func_name}: {e}")
                tool_results.append(
                    {
                        "tool_call_id": tool_call["id"],
                        "role": "tool",
                        "content": json.dumps({"success": False, "error": str(e)}),
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
    """Process API response. Handles multiple rounds of tool calls."""
    tool_leak_retry_count = 0

    while True:
        # Handle API errors
        if "error" in response_data:
            error_msg = response_data["error"].get("message", "Unknown API error")
            logger.error(f"API error: {error_msg}")
            return f"Error: {error_msg}"

        if "choices" not in response_data or not response_data["choices"]:
            logger.error(f"No choices in response: {response_data}")
            return "Error: No response from API"

        message = response_data["choices"][0]["message"]

        # Fallback: recover tool calls when model emits raw bash code fences.
        if not message.get("tool_calls"):
            inferred_tool_calls = infer_tool_calls_from_content(
                _message_content_to_text(message.get("content", ""))
            )
            if inferred_tool_calls:
                message["tool_calls"] = inferred_tool_calls
                message["content"] = None
                logger.warning("Recovered tool call from leaked content")

        # Keep processing tool calls until there are none
        while "tool_calls" in message and message["tool_calls"]:
            # Execute tool calls
            allowed_tool_names = _extract_allowed_tool_names(base_payload)
            tool_results = await execute_tool_calls(
                message,
                allowed_tool_names=allowed_tool_names,
            )

            # Build conversation history
            messages.append(message)
            messages.extend(tool_results)

            # Make follow-up call with tool results (with retry)
            current_payload = base_payload.copy()
            current_payload["messages"] = messages

            response_data = await api_call_with_retry(
                session,
                base_url,
                current_payload,
                {"Authorization": f"Bearer {api_key}"},
                stream=stream_chunk_callback is not None,
                stream_chunk_callback=stream_chunk_callback,
            )

            # Handle errors in follow-up call
            if "error" in response_data:
                error_msg = response_data["error"].get("message", "Unknown API error")
                logger.error(f"API error in follow-up: {error_msg}")
                return f"Error: {error_msg}"

            if "choices" not in response_data or not response_data["choices"]:
                logger.error("No choices in follow-up response")
                return "Error: No response from API"

            message = response_data["choices"][0]["message"]

        content_text = _message_content_to_text(message.get("content", ""))
        if detect_tool_leak(content_text):
            if tool_leak_retry_count < max_tool_leak_retries:
                tool_leak_retry_count += 1
                logger.warning(
                    "Detected leaked tool-call content, retrying with formatting guard "
                    f"(attempt {tool_leak_retry_count}/{max_tool_leak_retries})..."
                )

                leak_guard_message = {
                    "role": "user",
                    "content": (
                        "Your last reply exposed internal tool call content. "
                        "Do not show tool calls, shell commands, code fences, or JSON internals. "
                        "Reply naturally in plain text for the end user."
                    ),
                }
                messages.append(message)
                messages.append(leak_guard_message)

                current_payload = base_payload.copy()
                current_payload["messages"] = messages

                response_data = await api_call_with_retry(
                    session,
                    base_url,
                    current_payload,
                    {"Authorization": f"Bearer {api_key}"},
                    stream=stream_chunk_callback is not None,
                    stream_chunk_callback=stream_chunk_callback,
                )
                continue

            logger.warning("Max tool-leak retries exceeded, returning safe fallback")
            return (
                "Sorry, there was an internal formatting issue. Please send that again."
            )

        return strip_markdown.strip_markdown(content_text) if content_text else ""
