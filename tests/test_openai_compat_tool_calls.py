#!/usr/bin/env python3
"""Live-API test for OpenAI compat server tool-call round-trips."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aiohttp import ClientSession, web
from aiohttp.test_utils import TestClient, TestServer

from openai_compat_server import create_openai_compatible_app
from tests._live_harness import (
    LIVE,
    skip_if_not_live,
    _make_handler,
    _make_store,
)


class LiveHandler:
    """Wraps the real handler for the compat server's handle() contract."""

    def __init__(self, real_handler, session_id: str = "compat_test"):
        self._handler = real_handler
        self._session_id = session_id

    async def handle(self, request, session_id=None, interim_response_callback=None,
                     response_chunk_callback=None, request_metadata=None):
        sid = session_id or self._session_id
        messages = request.get("messages", [])
        result = await self._handler.handle(
            messages=messages, session_id=sid,
            interim_response_callback=interim_response_callback,
            response_chunk_callback=response_chunk_callback,
            request_metadata=request_metadata,
        )
        return result


def _extract_response_text(response: str) -> str:
    """Extract plain text from a handler response that might be JSON or raw text."""
    try:
        parsed = json.loads(response)
        if isinstance(parsed, dict):
            return parsed.get("text", str(parsed))
        return str(parsed)
    except (json.JSONDecodeError, TypeError):
        return response


async def test_live_compat_server_models_endpoint() -> None:
    """GET /v1/models should list available models."""
    if not LIVE:
        print("\nTest L1: Compat server models [SKIP]")
        return
    print("\nTest L1: OpenAI compat server /v1/models")
    skip_if_not_live()

    store = _make_store()
    handler_obj = _make_handler(store)
    wrapped = LiveHandler(handler_obj)

    app = create_openai_compatible_app(wrapped, api_key="test-api-key", model_alias="agentzero-test")
    async with TestServer(app) as server:
        async with TestClient(server) as client:
            resp = await client.get(
                "/v1/models",
                headers={"Authorization": "Bearer test-api-key"}
            )
            assert resp.status == 200, f"Expected 200, got {resp.status}"
            data = await resp.json()
            assert "data" in data
            assert len(data["data"]) >= 1
            print(f"  PASS — {len(data['data'])} model(s) listed")


async def test_live_compat_server_chat_completion() -> None:
    """POST /v1/chat/completions should call real agent and return response."""
    if not LIVE:
        print("\nTest L2: Compat server chat [SKIP]")
        return
    print("\nTest L2: OpenAI compat server POST /v1/chat/completions")
    skip_if_not_live()

    store = _make_store()
    handler_obj = _make_handler(store)
    wrapped = LiveHandler(handler_obj)

    app = create_openai_compatible_app(wrapped, api_key="test-api-key", model_alias="agentzero-test")
    async with TestServer(app) as server:
        async with TestClient(server) as client:
            payload = {
                "model": "agentzero-main",
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Just say 'compat test ok' and nothing else. Stop after that."
                        ),
                    }
                ],
                "stream": False,
            }
            resp = await client.post(
                "/v1/chat/completions",
                json=payload,
                headers={
                    "Authorization": "Bearer test-api-key",
                    "Content-Type": "application/json",
                },
            )
            data = await resp.json()
            assert resp.status == 200, f"Expected 200, got {resp.status}: {data}"
            choice = data.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content", "")
            assert len(content) > 3, f"Empty content: {data}"
            print(f"  PASS — response: {content[:120]}")


async def test_live_compat_server_tool_call_roundtrip() -> None:
    """POST with tool-call trigger should execute tools and return final response."""
    if not LIVE:
        print("\nTest L3: Compat server tool calls [SKIP]")
        return
    print("\nTest L3: OpenAI compat server tool-call roundtrip")
    skip_if_not_live()

    store = _make_store()
    handler_obj = _make_handler(store)
    wrapped = LiveHandler(handler_obj)

    app = create_openai_compatible_app(wrapped, api_key="test-api-key", model_alias="agentzero-test")
    async with TestServer(app) as server:
        async with TestClient(server) as client:
            payload = {
                "model": "agentzero-main",
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Run `printf compat-tool-ok` using bash. "
                            "Reply with exactly the bash output and nothing else. "
                            "Stop after that."
                        ),
                    }
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "description": "Run a bash command",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "command": {"type": "string"}
                                },
                                "required": ["command"],
                            },
                        },
                    }
                ],
                "stream": False,
            }
            resp = await client.post(
                "/v1/chat/completions",
                json=payload,
                headers={
                    "Authorization": "Bearer test-api-key",
                    "Content-Type": "application/json",
                },
            )
            data = await resp.json()
            assert resp.status == 200, f"Expected 200, got {resp.status}: {data}"
            choice = data.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content", "")
            assert "compat-tool-ok" in content, f"Missing tool output: {content[:300]}"
            print(f"  PASS — response: {content[:120]}")


async def test_live_compat_server_unauthorized() -> None:
    """Requests without valid API key should be rejected."""
    print("\nTest L4: Compat server auth rejection")
    store = _make_store()
    handler_obj = _make_handler(store)
    wrapped = LiveHandler(handler_obj)
    app = create_openai_compatible_app(wrapped, api_key="test-api-key", model_alias="agentzero-test")

    async with TestServer(app) as server:
        async with TestClient(server) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "agentzero", "messages": [{"role": "user", "content": "hi"}]},
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 401, f"Expected 401, got {resp.status}"
    print("  PASS")


async def main() -> None:
    print("=" * 60)
    print("OpenAI compat server tests")
    print("=" * 60)
    await test_live_compat_server_unauthorized()
    await test_live_compat_server_models_endpoint()
    await test_live_compat_server_chat_completion()
    await test_live_compat_server_tool_call_roundtrip()
    print("\n" + "=" * 60)
    print("OpenAI compat server tests complete")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
