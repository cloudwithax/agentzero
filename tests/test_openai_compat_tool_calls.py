#!/usr/bin/env python3
"""Test tool calls through the OpenAI compat server path end-to-end."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aiohttp.test_utils import TestClient, TestServer
from unittest.mock import AsyncMock, patch

from openai_compat_server import create_openai_compatible_app


class ToolCallingHandler:
    """Handler that simulates a tool-call round-trip like the real handler does."""

    def __init__(self):
        self.calls = []

    async def handle(
        self,
        request,
        session_id=None,
        interim_response_callback=None,
        response_chunk_callback=None,
        request_metadata=None,
    ):
        self.calls.append({
            "request": request,
            "session_id": session_id,
        })

        messages = request.get("messages", [])

        # Simulate what handler.handle() does:
        # 1. Build a payload with tools
        # 2. Call the real API (mocked)
        # 3. Process response with tool-call loop
        from api import process_response
        from handler import BASE_PAYLOAD, BASE_URL, API_KEY

        payload = BASE_PAYLOAD.copy()
        payload["messages"] = messages

        # First response: model wants to call bash
        first_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_test_123",
                                "type": "function",
                                "function": {
                                    "name": "bash",
                                    "arguments": json.dumps({"command": "echo hello"}),
                                },
                            }
                        ],
                    }
                }
            ]
        }

        # Second response: model gives final answer
        second_response = {
            "choices": [
                {"message": {"role": "assistant", "content": "Command output: hello"}}
            ]
        }

        call_count = [0]

        class MockResponse:
            def __init__(self, payload):
                self.payload = payload
                self.status = 200

            async def json(self, content_type=None):
                return self.payload

        class MockPostCM:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                call_count[0] += 1
                if call_count[0] == 1:
                    return MockResponse(first_response)
                return MockResponse(second_response)

            async def __aexit__(self, *args):
                return False

        mock_session = AsyncMock()
        mock_session.post = MockPostCM

        # This is what handler.handle() does
        content = await process_response(
            first_response,
            payload["messages"],
            mock_session,
            BASE_URL,
            API_KEY,
            payload,
        )

        return content


async def test_tool_calls_through_openai_compat():
    """Tool calls should work end-to-end through the OpenAI compat server."""
    print("Test: Tool calls through OpenAI compat server")

    handler = ToolCallingHandler()
    app = create_openai_compatible_app(
        handler,
        api_key="test-key",
        model_alias="agentzero-main",
        backing_model="test-model",
    )
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()

    try:
        response = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-key"},
            json={
                "model": "agentzero-main",
                "messages": [{"role": "user", "content": "Run a command"}],
            },
        )
        payload = await response.json()

        assert response.status == 200, f"Expected 200, got {response.status}"
        content = payload["choices"][0]["message"]["content"]
        assert content == "Command output: hello", f"Got: {content}"
        print("  ✓ Passed")
    finally:
        await client.close()


async def main():
    await test_tool_calls_through_openai_compat()
    print("All OpenAI compat tool call tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
