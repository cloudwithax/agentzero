"""Live-API tests for API response handling and multi-round flows.

Deterministic tests (headers, templates, streaming parsing) remain as-is.
Mock-based flow tests are replaced with live-API integration tests.
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api import (
    _apply_cache_busting_headers,
    api_call_with_retry,
    execute_tool_calls,
    infer_tool_calls_from_content,
)
from handler import API_KEY, BASE_PAYLOAD, BASE_URL, FINAL_RESPONSE_MAX_TOKENS
from prompt_templates import get_template
from tests._live_harness import (
    LIVE,
    live_run_agentic_loop,
    live_agent_handle,
    parse_loop_result,
    skip_if_not_live,
    _make_handler,
    _make_store,
)

# ─── Deterministic tests ──────────────────────────────────────────────────────


def test_cache_busting_headers_are_applied() -> None:
    """Every API request should carry explicit no-cache headers and a unique ID."""
    print("\nTest 0a: Cache-busting headers")

    first = _apply_cache_busting_headers({"Authorization": "Bearer test"})
    second = _apply_cache_busting_headers({"Authorization": "Bearer test"})

    assert first["Authorization"] == "Bearer test"
    assert first["Cache-Control"] == "no-cache, no-store, max-age=0"
    assert first["Pragma"] == "no-cache"
    assert first["Expires"] == "0"
    assert first["X-Request-Id"]
    assert second["X-Request-Id"]
    assert first["X-Request-Id"] != second["X-Request-Id"]
    print("  PASS")


def test_system_prompt_distinguishes_repo_paths_from_workspace_paths() -> None:
    """The prompt should tell the executor not to look for repo code under workspace/."""
    print("\nTest 0aa: System prompt repo path guidance")

    rendered = get_template(
        "system_prompt",
        {
            "current_time": "2026-04-09 12:00:00",
            "workspace_path": "/home/clxud/agentzero/workspace",
            "identity": "You are a helpful AI assistant.",
        },
    )

    assert "REPO CODE" in rendered
    assert "Do not assume repo code lives under the workspace" in rendered
    assert "handler.py" in rendered
    assert "self-reject or safety-review" in rendered
    print("  PASS")


def test_infer_tool_calls_from_xml_function_markup() -> None:
    """XML-style function markup should recover into a real tool call."""
    print("\nTest 0ab: Recover XML function markup")

    inferred = infer_tool_calls_from_content(
        """
<function_activate_skill>
<parameter name="name">frontend-design</parameter>
</function_activate_skill>
""".strip()
    )

    assert len(inferred) == 1
    assert inferred[0]["function"]["name"] == "activate_skill"
    assert json.loads(inferred[0]["function"]["arguments"]) == {
        "name": "frontend-design"
    }
    print("  PASS")


class MockStreamingContent:
    def __init__(self, lines):
        self._lines = [line.encode("utf-8") for line in lines]

    def __aiter__(self):
        self._iter = iter(self._lines)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


async def test_api_call_with_retry_streams_text_deltas() -> None:
    """Streaming API calls should emit text deltas and assemble final content."""
    print("\nTest 0b: Streaming text deltas")

    class MockResponse:
        status = 200
        content_type = "text/event-stream"

        def __init__(self):
            self.content = MockStreamingContent(
                [
                    'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',
                    'data: {"choices":[{"delta":{"content":"Hel"}}]}\n',
                    'data: {"choices":[{"delta":{"content":"lo"}}]}\n',
                    "data: [DONE]\n",
                ]
            )

    class MockPostCM:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return MockResponse()

        async def __aexit__(self, *args):
            return False

    collected_chunks: list[str] = []

    async def collect_chunk(chunk: str) -> None:
        collected_chunks.append(chunk)

    mock_session = AsyncMock()
    mock_session.post = MockPostCM

    response_data = await api_call_with_retry(
        mock_session,
        BASE_URL,
        {"model": "test-model", "messages": []},
        {"Authorization": "Bearer test"},
        stream=True,
        stream_chunk_callback=collect_chunk,
    )

    message = response_data["choices"][0]["message"]
    assert message["content"] == "Hello", f"Unexpected streamed content: {message}"
    assert collected_chunks == ["Hel", "lo"], f"Unexpected chunks: {collected_chunks}"
    print("  PASS")


async def test_api_call_with_retry_streams_tool_calls() -> None:
    """Streaming API calls should assemble tool call names and arguments correctly."""
    print("\nTest 0c: Streaming tool-call assembly")

    first_chunk = json.dumps({"choices": [{"delta": {"role": "assistant"}}]})
    second_chunk = json.dumps(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_stream",
                                "type": "function",
                                "function": {"name": "ba"},
                            }
                        ]
                    }
                }
            ]
        }
    )
    third_chunk = json.dumps(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {
                                    "name": "sh",
                                    "arguments": '{"command": "echo hi"}',
                                },
                            }
                        ]
                    }
                }
            ]
        }
    )

    class MockResponse:
        status = 200
        content_type = "text/event-stream"

        def __init__(self):
            self.content = MockStreamingContent(
                [
                    f"data: {first_chunk}\n",
                    f"data: {second_chunk}\n",
                    f"data: {third_chunk}\n",
                    "data: [DONE]\n",
                ]
            )

    class MockPostCM:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return MockResponse()

        async def __aexit__(self, *args):
            return False

    mock_session = AsyncMock()
    mock_session.post = MockPostCM

    response_data = await api_call_with_retry(
        mock_session,
        BASE_URL,
        {"model": "test-model", "messages": []},
        {"Authorization": "Bearer test"},
        stream=True,
    )

    tool_calls = response_data["choices"][0]["message"].get("tool_calls") or []
    assert len(tool_calls) == 1, f"Unexpected tool calls: {tool_calls}"
    assert tool_calls[0]["function"]["name"] == "bash"
    assert tool_calls[0]["function"]["arguments"] == '{"command": "echo hi"}'
    print("  PASS")


async def test_api_call_with_retry_does_not_add_nvcf_header() -> None:
    """NVCF image-asset request headers should never be injected."""
    print("\nTest 0d: No NVCF header injection")

    captured_headers: dict[str, str] = {}

    class MockResponse:
        status = 200

        async def json(self, content_type=None):
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    class MockPostCM:
        def __init__(self, *args, **kwargs):
            captured_headers.update(kwargs.get("headers", {}))

        async def __aenter__(self):
            return MockResponse()

        async def __aexit__(self, *args):
            return False

    mock_session = AsyncMock()
    mock_session.post = MockPostCM

    await api_call_with_retry(
        mock_session,
        BASE_URL,
        {"model": "test-model", "messages": []},
        {"Authorization": "Bearer test"},
        stream=False,
    )

    assert "x-nvcf-payload" not in [k.lower() for k in captured_headers]
    assert "NVCF-INPUT-ASSET-REFERENCES" not in captured_headers
    assert "NVCF-FUNCTION-ASSET-IDS" not in captured_headers
    print("  PASS")


# ─── Live-API integration tests ────────────────────────────────────────────────


async def test_live_single_tool_call_flow() -> None:
    """Simple tool call → result → follow-up → done."""
    skip_if_not_live()
    print("\nTest L1: Single tool call flow via live API")

    result = await live_run_agentic_loop(
        messages=[
            {
                "role": "user",
                "content": (
                    "Run `printf tool-flow-ok` using bash. Reply with the output "
                    "prefixed by 'result: '. Stop after that."
                ),
            }
        ],
        max_iterations=5,
    )

    parsed = parse_loop_result(result)
    text = parsed.get("text", result)
    assert "tool-flow-ok" in text, f"Missing expected output in: {text[:200]}"
    print(f"  PASS — reply: {text[:120]}")


async def test_live_multi_tool_roundtrip() -> None:
    """Multiple tools called across iterations should all work."""
    skip_if_not_live()
    print("\nTest L2: Multi-tool roundtrip via live API")

    result = await live_run_agentic_loop(
        messages=[
            {
                "role": "user",
                "content": (
                    "1. Use read to read the first 2 lines of AGENTS.md.\n"
                    "2. Use bash to run `printf roundtrip-ok`.\n"
                    "3. Reply with both results as:\n"
                    "read: <first 2 lines>\n"
                    "bash: roundtrip-ok\n"
                    "Stop after step 3."
                ),
            }
        ],
        max_iterations=8,
    )

    parsed = parse_loop_result(result)
    text = parsed.get("text", result)
    assert "roundtrip-ok" in text, f"Missing bash output in: {text[:300]}"
    assert len(text) > 30, f"Reply too short: {text[:300]}"
    print(f"  PASS — reply length={len(text)}")


async def test_live_completion_protocol_with_handler() -> None:
    """Full handler.handle() should produce a response via the completion protocol."""
    skip_if_not_live()
    print("\nTest L3: Full handler.handle() completion via live API")

    store = _make_store()
    handler = _make_handler(store)

    response = await live_agent_handle(
        handler,
        user_text="Just say 'hello from handler' and nothing else. Stop after that.",
        session_id="test_process",
    )

    assert len(response) > 3, f"Response too short: {response[:200]}"
    # Should not contain leaked pseudo-tool markup
    assert "<read" not in response.lower(), f"Pseudo-tool leak: {response[:200]}"
    print(f"  PASS — reply: {response[:120]}")


def test_live_max_tokens_configuration() -> None:
    """BASE_PAYLOAD should have sufficient max_tokens for real responses."""
    print("\nTest L4: BASE_PAYLOAD max_tokens configuration")

    assert BASE_PAYLOAD.get("max_tokens") >= FINAL_RESPONSE_MAX_TOKENS, (
        f"max_tokens {BASE_PAYLOAD.get('max_tokens')} < {FINAL_RESPONSE_MAX_TOKENS}"
    )
    # Verify tools list includes the new declare_message_count
    tool_names = [
        t.get("function", {}).get("name")
        for t in BASE_PAYLOAD.get("tools", [])
        if isinstance(t, dict)
    ]
    assert "declare_message_count" in tool_names, "Missing declare_message_count tool"
    assert "send_message" in tool_names, "Missing send_message tool"
    print("  PASS")


# ─── Main ──────────────────────────────────────────────────────────────────────


async def main() -> None:
    print("=" * 60)
    print("Testing process_response - Edge Cases")
    print("=" * 60)

    test_cache_busting_headers_are_applied()
    test_system_prompt_distinguishes_repo_paths_from_workspace_paths()
    test_infer_tool_calls_from_xml_function_markup()
    await test_api_call_with_retry_streams_text_deltas()
    await test_api_call_with_retry_streams_tool_calls()
    await test_api_call_with_retry_does_not_add_nvcf_header()

    if LIVE:
        print("\n" + "=" * 60)
        print("Live-API process_response tests")
        print("=" * 60)
        await test_live_single_tool_call_flow()
        await test_live_multi_tool_roundtrip()
        await test_live_completion_protocol_with_handler()
    test_live_max_tokens_configuration()

    print("\n" + "=" * 60)
    print("All edge case tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
