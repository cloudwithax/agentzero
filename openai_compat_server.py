"""OpenAI-compatible HTTP server for AgentZero."""

import asyncio
import json
import logging
import re
import time
import uuid
from typing import Any, Optional, Protocol

from aiohttp import web

from handler import MODEL_ID as HANDLER_MODEL_ID

logger = logging.getLogger(__name__)

_ALLOWED_ROLES = {"system", "user", "assistant", "tool"}
_MESSAGE_BLOCK_PATTERN = re.compile(
    r"<message>\s*(?P<message>.*?)\s*</message>",
    re.IGNORECASE | re.DOTALL,
)
_TYPING_DIRECTIVE_PATTERN = re.compile(
    r"<typing(?:\s+seconds\s*=\s*['\"]?\d+(?:\.\d+)?['\"]?)?\s*/>",
    re.IGNORECASE,
)
_MESSAGE_TAG_PATTERN = re.compile(r"</?message>", re.IGNORECASE)


class HandlerLike(Protocol):
    """Minimal protocol needed from AgentHandler for compatibility routing."""

    async def handle(
        self,
        request,
        session_id: Optional[str] = None,
        interim_response_callback=None,
        response_chunk_callback=None,
        request_metadata: Optional[dict[str, Any]] = None,
    ) -> Any: ...


def _openai_error(
    message: str,
    *,
    status: int = 400,
    error_type: str = "invalid_request_error",
    code: Optional[str] = None,
) -> web.Response:
    """Return an OpenAI-compatible JSON error response."""
    payload = {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": code,
        }
    }
    return web.json_response(payload, status=status)


def _extract_bearer_token(request: web.Request) -> str:
    """Extract bearer token from Authorization header."""
    authorization = request.headers.get("Authorization", "")
    if not authorization:
        return ""

    parts = authorization.split(" ", 1)
    if len(parts) != 2:
        return ""

    scheme, token = parts
    if scheme.lower() != "bearer":
        return ""

    return token.strip()


def _is_authorized(request: web.Request, expected_api_key: str) -> bool:
    """Validate inbound API key against configured key."""
    provided = _extract_bearer_token(request)
    return bool(provided and provided == expected_api_key)


def _content_to_text(content: Any) -> str:
    """Extract text from string or multimodal content blocks."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue

            block_type = str(block.get("type", "")).strip().lower()
            if block_type in {"text", "input_text"} and isinstance(
                block.get("text"), str
            ):
                parts.append(block["text"])
            elif isinstance(block.get("content"), str):
                parts.append(block["content"])
        return "\n".join(part for part in parts if part).strip()

    if content is None:
        return ""

    return str(content)


def _estimate_tokens_from_text(text: str) -> int:
    """Estimate token count with a simple character heuristic."""
    normalized = (text or "").strip()
    if not normalized:
        return 0
    return max(1, len(normalized) // 4)


def _estimate_prompt_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate prompt tokens from all message contents."""
    total = 0
    for message in messages:
        total += _estimate_tokens_from_text(_content_to_text(message.get("content")))
        total += 3  # rough overhead per message object
    return total


def _normalize_content_block(block: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Normalize one OpenAI content block into AgentZero-compatible format."""
    block_type = str(block.get("type", "")).strip().lower()

    if block_type in {"text", "input_text"}:
        text = block.get("text")
        if not isinstance(text, str):
            text = block.get("content")
        if not isinstance(text, str):
            return None
        return {"type": "text", "text": text}

    if block_type in {"image_url", "input_image"}:
        raw_image_url = block.get("image_url")
        image_url = ""

        if isinstance(raw_image_url, str):
            image_url = raw_image_url.strip()
        elif isinstance(raw_image_url, dict) and isinstance(
            raw_image_url.get("url"), str
        ):
            image_url = raw_image_url["url"].strip()
        elif isinstance(block.get("url"), str):
            image_url = block["url"].strip()

        if not image_url:
            return None

        return {
            "type": "image_url",
            "image_url": {"url": image_url},
        }

    if isinstance(block.get("text"), str):
        return {"type": "text", "text": block["text"]}

    if isinstance(block.get("content"), str):
        return {"type": "text", "text": block["content"]}

    return None


def _normalize_messages(
    raw_messages: Any,
) -> tuple[Optional[list[dict[str, Any]]], Optional[str]]:
    """Validate and normalize inbound chat messages."""
    if not isinstance(raw_messages, list) or not raw_messages:
        return None, "'messages' must be a non-empty array"

    normalized: list[dict[str, Any]] = []

    for index, message in enumerate(raw_messages):
        if not isinstance(message, dict):
            return None, f"messages[{index}] must be an object"

        role = str(message.get("role", "")).strip().lower()
        if role not in _ALLOWED_ROLES:
            return (
                None,
                f"messages[{index}].role must be one of {sorted(_ALLOWED_ROLES)}",
            )

        content = message.get("content", "")
        normalized_content: Any
        if isinstance(content, list):
            blocks = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                normalized_block = _normalize_content_block(block)
                if normalized_block:
                    blocks.append(normalized_block)
            normalized_content = blocks if blocks else ""
        elif content is None:
            normalized_content = ""
        elif isinstance(content, str):
            normalized_content = content
        else:
            normalized_content = str(content)

        normalized_message: dict[str, Any] = {
            "role": role,
            "content": normalized_content,
        }

        if role == "tool" and isinstance(message.get("tool_call_id"), str):
            normalized_message["tool_call_id"] = message["tool_call_id"]

        normalized.append(normalized_message)

    if not any(msg.get("role") == "user" for msg in normalized):
        return None, "messages must include at least one user message"

    return normalized, None


def _normalize_session_id(candidate: str) -> str:
    """Normalize session IDs for safe storage keys."""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", candidate.strip())
    cleaned = cleaned.strip("_")
    return cleaned[:96]


def _resolve_session_id(request: web.Request, body: dict[str, Any]) -> str:
    """Resolve session ID from request headers/body, with fallback."""
    header_session = request.headers.get("X-Session-Id", "").strip()
    normalized_header_session = _normalize_session_id(header_session)
    if normalized_header_session:
        return f"openai_{normalized_header_session}"

    body_session = body.get("session_id")
    if isinstance(body_session, str):
        normalized_body_session = _normalize_session_id(body_session)
        if normalized_body_session:
            return f"openai_{normalized_body_session}"

    metadata = body.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("session_id"), str):
        normalized_metadata_session = _normalize_session_id(metadata["session_id"])
        if normalized_metadata_session:
            return f"openai_{normalized_metadata_session}"

    user_identifier = body.get("user")
    if isinstance(user_identifier, str):
        normalized_user_identifier = _normalize_session_id(user_identifier)
        if normalized_user_identifier:
            return f"openai_user_{normalized_user_identifier}"

    return f"openai_{uuid.uuid4().hex[:16]}"


def _build_usage(
    prompt_messages: list[dict[str, Any]],
    completion_text: str,
) -> dict[str, int]:
    """Build a rough OpenAI usage object."""
    prompt_tokens = _estimate_prompt_tokens(prompt_messages)
    completion_tokens = _estimate_tokens_from_text(completion_text)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _sanitize_outbound_text_for_openai(raw_text: Any) -> str:
    """Strip integration-only delivery tags from assistant text for OpenAI clients."""
    normalized = "" if raw_text is None else str(raw_text)
    if not normalized.strip():
        return ""

    message_chunks = [
        match.group("message").strip()
        for match in _MESSAGE_BLOCK_PATTERN.finditer(normalized)
        if isinstance(match.group("message"), str) and match.group("message").strip()
    ]
    if message_chunks:
        return "\n\n".join(message_chunks).strip()

    # If no valid <message> blocks exist, still drop typing directives and dangling tags.
    sanitized = _TYPING_DIRECTIVE_PATTERN.sub("", normalized)
    sanitized = _MESSAGE_TAG_PATTERN.sub("", sanitized)
    return sanitized.strip()


async def _healthcheck_endpoint(_request: web.Request) -> web.Response:
    """Simple unauthenticated health endpoint."""
    return web.json_response({"status": "ok"})


async def _models_endpoint(request: web.Request) -> web.Response:
    """Return available models in OpenAI list format."""
    if not _is_authorized(request, request.app["api_key"]):
        return _openai_error(
            "Invalid API key provided",
            status=401,
            error_type="authentication_error",
            code="invalid_api_key",
        )

    created = int(time.time())
    model_alias = request.app["model_alias"]
    backing_model = request.app["backing_model"]

    data = [
        {
            "id": model_alias,
            "object": "model",
            "created": created,
            "owned_by": "agentzero",
        }
    ]

    if backing_model != model_alias:
        data.append(
            {
                "id": backing_model,
                "object": "model",
                "created": created,
                "owned_by": "agentzero",
            }
        )

    return web.json_response({"object": "list", "data": data})


async def _chat_completions_endpoint(request: web.Request) -> web.StreamResponse:
    """Handle OpenAI-style /v1/chat/completions requests."""
    if not _is_authorized(request, request.app["api_key"]):
        return _openai_error(
            "Invalid API key provided",
            status=401,
            error_type="authentication_error",
            code="invalid_api_key",
        )

    try:
        body = await request.json()
    except Exception:
        return _openai_error(
            "Request body must be valid JSON",
            status=400,
            code="invalid_json",
        )

    if not isinstance(body, dict):
        return _openai_error(
            "Request body must be a JSON object",
            status=400,
            code="invalid_json",
        )

    requested_model = str(body.get("model", "")).strip()
    if not requested_model:
        return _openai_error(
            "'model' is required",
            status=400,
            code="model_required",
        )

    allowed_models = {
        request.app["model_alias"],
        request.app["backing_model"],
    }
    if requested_model not in allowed_models:
        return _openai_error(
            f"Model '{requested_model}' not found. Available models: {sorted(allowed_models)}",
            status=400,
            code="model_not_found",
        )

    normalized_messages, message_error = _normalize_messages(body.get("messages"))
    if message_error:
        return _openai_error(message_error, status=400, code="invalid_messages")

    assert normalized_messages is not None

    should_stream = bool(body.get("stream", False))
    created = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    handler: HandlerLike = request.app["handler"]
    session_id = _resolve_session_id(request, body)
    request_metadata: dict[str, Any] = {
        "channel": "openai_compat",
        "model": requested_model,
    }

    user_identifier = body.get("user")
    if isinstance(user_identifier, str) and user_identifier.strip():
        request_metadata["client_user"] = user_identifier.strip()

    user_agent = request.headers.get("User-Agent", "").strip()
    if user_agent:
        request_metadata["user_agent"] = user_agent

    request_payload = {"messages": normalized_messages}

    if not should_stream:
        try:
            completion_text = await handler.handle(
                request_payload,
                session_id=session_id,
                request_metadata=request_metadata,
            )
        except Exception as exc:
            logger.exception("OpenAI compatibility request failed")
            return _openai_error(
                f"Internal server error: {exc}",
                status=500,
                error_type="server_error",
                code="internal_error",
            )

        completion_text = _sanitize_outbound_text_for_openai(completion_text)
        usage = _build_usage(normalized_messages, completion_text)

        return web.json_response(
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": requested_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": completion_text,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": usage,
            }
        )

    stream_response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await stream_response.prepare(request)

    async def _write_event(payload: dict[str, Any]) -> None:
        chunk = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        await stream_response.write(chunk.encode("utf-8"))

    await _write_event(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": requested_model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }
            ],
        }
    )

    streamed_chunks: list[str] = []

    async def _on_chunk(chunk: str) -> None:
        if not chunk:
            return

        # Buffer streamed deltas so delivery tags can be removed safely even if split.
        streamed_chunks.append(chunk)

    try:
        completion_text = await handler.handle(
            request_payload,
            session_id=session_id,
            response_chunk_callback=_on_chunk,
            request_metadata=request_metadata,
        )
    except Exception as exc:
        logger.exception("OpenAI compatibility streaming request failed")
        error_payload = {
            "error": {
                "message": f"Internal server error: {exc}",
                "type": "server_error",
                "param": None,
                "code": "internal_error",
            }
        }
        await stream_response.write(
            f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n".encode("utf-8")
        )
        await stream_response.write(b"data: [DONE]\n\n")
        await stream_response.write_eof()
        return stream_response

    raw_stream_text = "".join(streamed_chunks)
    raw_completion_text = str(completion_text or "")
    merged_text = raw_completion_text or raw_stream_text
    completion_text = _sanitize_outbound_text_for_openai(merged_text)

    if completion_text:
        await _write_event(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": requested_model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": completion_text},
                        "finish_reason": None,
                    }
                ],
            }
        )

    await _write_event(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": requested_model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
    )
    await stream_response.write(b"data: [DONE]\n\n")
    await stream_response.write_eof()
    return stream_response


def create_openai_compatible_app(
    handler: HandlerLike,
    *,
    api_key: str,
    model_alias: str,
    backing_model: Optional[str] = None,
) -> web.Application:
    """Build and configure the OpenAI-compatible aiohttp application."""
    normalized_api_key = (api_key or "").strip()
    if not normalized_api_key:
        raise ValueError("api_key must be set")

    normalized_model_alias = (
        model_alias or "agentzero-main"
    ).strip() or "agentzero-main"
    normalized_backing_model = (
        backing_model or HANDLER_MODEL_ID
    ).strip() or HANDLER_MODEL_ID

    app = web.Application()
    app["handler"] = handler
    app["api_key"] = normalized_api_key
    app["model_alias"] = normalized_model_alias
    app["backing_model"] = normalized_backing_model

    app.router.add_get("/healthz", _healthcheck_endpoint)
    app.router.add_get("/v1/models", _models_endpoint)
    app.router.add_post("/v1/chat/completions", _chat_completions_endpoint)

    return app


async def start_openai_compatible_server(
    handler: HandlerLike,
    *,
    host: str,
    port: int,
    api_key: str,
    model_alias: str,
) -> None:
    """Run the OpenAI-compatible server indefinitely."""
    app = create_openai_compatible_app(
        handler,
        api_key=api_key,
        model_alias=model_alias,
    )

    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(runner, host, port)
    await site.start()

    logger.info(
        "OpenAI-compatible server started on %s:%s (model alias: %s, backing model: %s)",
        host,
        port,
        app["model_alias"],
        app["backing_model"],
    )

    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        await runner.cleanup()
