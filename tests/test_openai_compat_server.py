#!/usr/bin/env python3
"""Regression tests for OpenAI-compatible AgentZero server."""

import asyncio

from aiohttp.test_utils import TestClient, TestServer

from openai_compat_server import create_openai_compatible_app


class DummyHandler:
    """Small handler stub that mimics AgentHandler.handle contract."""

    def __init__(
        self,
        *,
        response_text: str = "hello world",
        streamed_chunks: list[str] | None = None,
    ) -> None:
        self.calls: list[dict] = []
        self.response_text = response_text
        self.streamed_chunks = streamed_chunks or ["hello ", "world"]

    async def handle(
        self,
        request,
        session_id=None,
        interim_response_callback=None,
        response_chunk_callback=None,
        request_metadata=None,
    ):
        self.calls.append(
            {
                "request": request,
                "session_id": session_id,
                "request_metadata": request_metadata,
            }
        )

        if response_chunk_callback:
            for chunk in self.streamed_chunks:
                await response_chunk_callback(chunk)

        return self.response_text


async def _start_test_client(handler: DummyHandler) -> TestClient:
    app = create_openai_compatible_app(
        handler,
        api_key="test-openai-key",
        model_alias="agentzero-main",
        backing_model="moonshotai/kimi-k2-instruct-0905",
    )
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    return client


async def test_requires_bearer_auth() -> None:
    """/v1/models should reject requests without valid bearer key."""
    print("Test 1: Auth required")
    client = await _start_test_client(DummyHandler())

    try:
        response = await client.get("/v1/models")
        payload = await response.json()

        assert response.status == 401, f"Expected 401, got {response.status}"
        assert payload["error"]["code"] == "invalid_api_key"
        print("  ✓ Passed")
    finally:
        await client.close()


async def test_chat_completion_calls_main_handler() -> None:
    """Non-streaming chat completions should route into AgentHandler.handle."""
    print("Test 2: Non-stream completion")
    handler = DummyHandler()
    client = await _start_test_client(handler)

    try:
        response = await client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer test-openai-key",
                "X-Session-Id": "session-42",
            },
            json={
                "model": "agentzero-main",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "Say hello.",
                            }
                        ],
                    }
                ],
            },
        )
        payload = await response.json()

        assert response.status == 200, f"Expected 200, got {response.status}"
        assert payload["object"] == "chat.completion"
        assert payload["choices"][0]["message"]["content"] == "hello world"

        assert len(handler.calls) == 1
        recorded_call = handler.calls[0]
        assert recorded_call["session_id"] == "openai_session-42"
        assert recorded_call["request"]["messages"][0]["role"] == "user"

        normalized_content = recorded_call["request"]["messages"][0]["content"]
        assert isinstance(normalized_content, list)
        assert normalized_content[0]["type"] == "text"
        assert normalized_content[0]["text"] == "Say hello."
        print("  ✓ Passed")
    finally:
        await client.close()


async def test_chat_completion_streaming_sse() -> None:
    """Streaming chat completions should emit OpenAI-style SSE with sanitized text."""
    print("Test 3: Streaming completion")
    client = await _start_test_client(DummyHandler())

    try:
        response = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-openai-key"},
            json={
                "model": "agentzero-main",
                "stream": True,
                "messages": [{"role": "user", "content": "Stream please."}],
            },
        )

        body = await response.text()

        assert response.status == 200, f"Expected 200, got {response.status}"
        assert "chat.completion.chunk" in body
        assert '"content": "hello world"' in body
        assert "data: [DONE]" in body
        print("  ✓ Passed")
    finally:
        await client.close()


async def test_chat_completion_strips_message_tags_non_stream() -> None:
    """Non-streaming responses should remove delivery wrappers from content."""
    print("Test 4: Strip message tags (non-stream)")
    handler = DummyHandler(
        response_text=(
            '<typing seconds="1.2"/>\n'
            "<message>first chunk</message>\n"
            "<message>second chunk</message>"
        )
    )
    client = await _start_test_client(handler)

    try:
        response = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-openai-key"},
            json={
                "model": "agentzero-main",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        payload = await response.json()

        assert response.status == 200, f"Expected 200, got {response.status}"
        content = payload["choices"][0]["message"]["content"]
        assert "<message>" not in content
        assert "<typing" not in content
        assert content == "first chunk\n\nsecond chunk"
        print("  ✓ Passed")
    finally:
        await client.close()


async def test_chat_completion_strips_message_tags_streaming() -> None:
    """Streaming responses should not leak <message>/<typing> directives."""
    print("Test 5: Strip message tags (streaming)")
    handler = DummyHandler(
        response_text=('<typing seconds="0.8"/>' "<message>streamed hello</message>"),
        streamed_chunks=["<message>streamed ", "hello</message>"],
    )
    client = await _start_test_client(handler)

    try:
        response = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-openai-key"},
            json={
                "model": "agentzero-main",
                "stream": True,
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        body = await response.text()

        assert response.status == 200, f"Expected 200, got {response.status}"
        assert "<message>" not in body
        assert "<typing" not in body
        assert '"content": "streamed hello"' in body
        assert "data: [DONE]" in body
        print("  ✓ Passed")
    finally:
        await client.close()


async def main() -> None:
    await test_requires_bearer_auth()
    await test_chat_completion_calls_main_handler()
    await test_chat_completion_streaming_sse()
    await test_chat_completion_strips_message_tags_non_stream()
    await test_chat_completion_strips_message_tags_streaming()
    print("All OpenAI compatibility tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
