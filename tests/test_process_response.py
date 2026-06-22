"""Live-API tests for API response handling and multi-round flows.

Deterministic tests (headers, templates, streaming parsing) remain as-is.
Mock-based flow tests are replaced with live-API integration tests.
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api import (
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
    """Mocks an aiohttp StreamReader.

    ``chunks`` are the raw byte reads ``iter_any()`` yields — these need NOT
    align to line boundaries, mirroring real network behavior where a single
    SSE ``data:`` line can be split across reads.
    """

    def __init__(self, chunks):
        self._chunks = [chunk.encode("utf-8") for chunk in chunks]

    async def iter_any(self):
        for chunk in self._chunks:
            yield chunk

    def __aiter__(self):
        self._iter = iter(self._chunks)
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


async def test_streaming_handles_data_line_split_across_reads() -> None:
    """A large tool-call argument SSE line split across network reads must not
    crash the stream (regression: this produced 'Unterminated string' errors
    that aborted multi-step runs such as building + publishing a site)."""
    print("\nTest 0c2: Streaming survives split data lines")

    # A write() call carrying a big HTML payload — the kind of argument that
    # makes the SSE line large enough to be fragmented by the network.
    big_html = "<html><body>" + ("<p>hello world</p>" * 400) + "</body></html>"
    tool_chunk = json.dumps(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_write",
                                "type": "function",
                                "function": {
                                    "name": "write",
                                    "arguments": json.dumps(
                                        {"filepath": "index.html", "content": big_html}
                                    ),
                                },
                            }
                        ]
                    }
                }
            ]
        }
    )
    full_line = f"data: {tool_chunk}\n"
    # Split the single SSE line into three arbitrary byte reads that do NOT
    # align to line boundaries — exactly what aiohttp's StreamReader does.
    cut1, cut2 = 40, len(full_line) // 2
    reads = [full_line[:cut1], full_line[cut1:cut2], full_line[cut2:], "data: [DONE]\n"]

    class MockResponse:
        status = 200
        content_type = "text/event-stream"

        def __init__(self):
            self.content = MockStreamingContent(reads)

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

    assert "error" not in response_data, f"Stream errored on split line: {response_data}"
    tool_calls = response_data["choices"][0]["message"].get("tool_calls") or []
    assert len(tool_calls) == 1, f"Tool call lost across split reads: {tool_calls}"
    parsed = json.loads(tool_calls[0]["function"]["arguments"])
    assert parsed["filepath"] == "index.html"
    assert parsed["content"] == big_html, "Reassembled argument payload was corrupted"
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


async def test_api_call_omits_empty_tools_array() -> None:
    """An empty ``tools`` array must be stripped from the wire request.

    The provider rejects ``"tools": []`` outright ("`tools` must not be an empty
    array. Either provide at least one tool or omit the field entirely."), which
    used to kill every tool-free inference path (orchestrator plan, advisor,
    judges, reminders, self-heal). They all pass ``tools=[]``; the API layer must
    omit the field so those requests succeed."""
    print("\nTest 0d2: Empty tools array omitted from request")

    captured_payload: dict = {}

    class MockResponse:
        status = 200

        async def json(self, content_type=None):
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    class MockPostCM:
        def __init__(self, *args, **kwargs):
            captured_payload.update(kwargs.get("json", {}))

        async def __aenter__(self):
            return MockResponse()

        async def __aexit__(self, *args):
            return False

    mock_session = AsyncMock()
    mock_session.post = MockPostCM

    await api_call_with_retry(
        mock_session,
        BASE_URL,
        {
            "model": "test-model",
            "messages": [],
            "tools": [],
            "tool_choice": "auto",
        },
        {"Authorization": "Bearer test"},
        stream=False,
    )

    assert "tools" not in captured_payload, (
        f"Empty tools array leaked to request: {captured_payload.get('tools')!r}"
    )
    assert "tool_choice" not in captured_payload, (
        "tool_choice must be dropped alongside an empty tools array"
    )
    print("  PASS")


async def test_api_call_keeps_nonempty_tools_array() -> None:
    """A populated ``tools`` array must pass through untouched."""
    print("\nTest 0d3: Non-empty tools array preserved")

    captured_payload: dict = {}
    tools = [{"type": "function", "function": {"name": "bash"}}]

    class MockResponse:
        status = 200

        async def json(self, content_type=None):
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    class MockPostCM:
        def __init__(self, *args, **kwargs):
            captured_payload.update(kwargs.get("json", {}))

        async def __aenter__(self):
            return MockResponse()

        async def __aexit__(self, *args):
            return False

    mock_session = AsyncMock()
    mock_session.post = MockPostCM

    await api_call_with_retry(
        mock_session,
        BASE_URL,
        {"model": "test-model", "messages": [], "tools": tools},
        {"Authorization": "Bearer test"},
        stream=False,
    )

    assert captured_payload.get("tools") == tools, "Populated tools must be preserved"
    print("  PASS")


async def test_api_call_retries_transient_200_body_error() -> None:
    """A transient server error returned in a 200 body (e.g. a truncated upstream
    response on a large tool-call payload) must be retried, not surfaced as a
    fatal error that aborts a multi-step run."""
    print("\nTest 0e: Retry transient 200-body error")

    attempts = {"n": 0}

    class MockResponse:
        status = 200

        async def json(self, content_type=None):
            attempts["n"] += 1
            if attempts["n"] == 1:
                # First call: provider returns an error body instead of choices.
                return {
                    "error": {
                        "message": "Unterminated string starting at: line 1 column 59 (char 58)"
                    }
                }
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

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
        stream=False,
        backoff=1.0,
    )

    assert attempts["n"] == 2, f"Expected a retry, got {attempts['n']} attempt(s)"
    assert "error" not in response_data, f"Transient error not recovered: {response_data}"
    assert response_data["choices"][0]["message"]["content"] == "ok"
    print("  PASS")


async def test_api_call_does_not_retry_permanent_client_error() -> None:
    """A genuine client-side error must NOT be retried (no wasted round-trips)."""
    print("\nTest 0f: No retry on permanent client error")

    attempts = {"n": 0}

    class MockResponse:
        status = 400

        async def json(self, content_type=None):
            attempts["n"] += 1
            return {"error": {"message": "invalid 'messages': field required"}}

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
        stream=False,
        backoff=1.0,
    )

    assert attempts["n"] == 1, f"Permanent error should not retry, got {attempts['n']}"
    assert "error" in response_data
    print("  PASS")


async def test_parse_class_retry_perturbs_sampling() -> None:
    """A parse-class error means the provider couldn't serialize the model's own
    tool call.  Re-sending the identical payload reproduces the identical bad
    output, so the retry must perturb sampling (bump temperature, drop seed)."""
    print("\nTest 0g: Parse-class retry perturbs sampling")

    seen_temps: list = []
    attempts = {"n": 0}

    class MockResponse:
        status = 200

        def __init__(self, sent_payload):
            self._sent = sent_payload

        async def json(self, content_type=None):
            attempts["n"] += 1
            seen_temps.append(self._sent.get("temperature"))
            if attempts["n"] > 1:
                assert "seed" not in self._sent, "seed must be dropped on parse-class retry"
            if attempts["n"] == 1:
                return {
                    "error": {
                        "message": "Unterminated string starting at: line 1 column 69 (char 67)"
                    }
                }
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    class MockPostCM:
        def __init__(self, url, json=None, headers=None):
            self._payload = json

        async def __aenter__(self):
            return MockResponse(self._payload)

        async def __aexit__(self, *args):
            return False

    mock_session = AsyncMock()
    mock_session.post = MockPostCM

    response_data = await api_call_with_retry(
        mock_session,
        BASE_URL,
        {"model": "test-model", "messages": [], "temperature": 0.6, "seed": 42},
        {"Authorization": "Bearer test"},
        stream=False,
        backoff=1.0,
    )

    assert attempts["n"] == 2, f"Expected one retry, got {attempts['n']}"
    assert response_data["choices"][0]["message"]["content"] == "ok"
    # First attempt uses the original temperature, the retry must be higher.
    assert seen_temps[0] == 0.6, f"First temp should be original, got {seen_temps[0]}"
    assert seen_temps[1] > seen_temps[0], (
        f"Retry temperature should be perturbed upward: {seen_temps}"
    )
    print("  PASS")


async def test_api_call_recovers_from_body_parse_failure() -> None:
    """If our own resp.json() raises on a truncated/malformed body, it must be
    caught and retried (it is a ValueError, not an aiohttp.ClientError), never
    escaping the retry loop as an uncaught exception."""
    print("\nTest 0h: Recover from response-body parse failure")

    attempts = {"n": 0}

    class MockResponse:
        status = 200

        async def json(self, content_type=None):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise json.JSONDecodeError("Unterminated string", "{bad", 4)
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

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
        {"model": "test-model", "messages": [], "temperature": 0.6},
        {"Authorization": "Bearer test"},
        stream=False,
        backoff=1.0,
    )

    assert attempts["n"] == 2, f"Expected a retry after body parse failure, got {attempts['n']}"
    assert "error" not in response_data, f"Should recover, got {response_data}"
    assert response_data["choices"][0]["message"]["content"] == "ok"
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


async def _run_429_with_retry_after(retry_after_value: str) -> tuple[dict, list, int]:
    """Drive api_call_with_retry through one 429 (carrying the given Retry-After)
    then a success, capturing every sleep wait.  Returns (response, waits, attempts)."""
    import api

    attempts = {"n": 0}

    class MockResponse:
        def __init__(self, attempt):
            self._attempt = attempt
            if attempt == 1:
                self.status = 429
                self.headers = {"Retry-After": retry_after_value}
            else:
                self.status = 200
                self.headers = {}

        async def json(self, content_type=None):
            if self._attempt == 1:
                return {"error": {"message": "Too Many Requests"}}
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    class MockPostCM:
        def __init__(self, *args, **kwargs):
            attempts["n"] += 1
            self._resp = MockResponse(attempts["n"])

        async def __aenter__(self):
            return self._resp

        async def __aexit__(self, *a):
            return False

    mock_session = AsyncMock()
    mock_session.post = MockPostCM

    waits: list[float] = []

    async def fake_sleep(seconds):
        waits.append(seconds)

    with patch("api.asyncio.sleep", fake_sleep):
        response_data = await api_call_with_retry(
            mock_session,
            BASE_URL,
            {"model": "test-model", "messages": []},
            {"Authorization": "Bearer test"},
            stream=False,
            backoff=1.0,
        )
    return response_data, waits, attempts["n"]


async def test_429_honors_retry_after_header() -> None:
    """A 429 carrying a Retry-After hint must wait exactly that long (server is
    authoritative) and then recover, not blindly use exponential backoff."""
    print("\nTest 0i: 429 honors Retry-After header")

    response_data, waits, n_attempts = await _run_429_with_retry_after("7")

    assert n_attempts == 2, f"Expected one retry then success, got {n_attempts}"
    assert response_data["choices"][0]["message"]["content"] == "ok"
    assert 7.0 in waits, f"Retry-After=7 should be honored, waits={waits}"
    print("  PASS")


async def test_429_caps_oversized_retry_after() -> None:
    """An absurd Retry-After must be capped at _MAX_RATELIMIT_BACKOFF so a single
    429 can't stall the whole turn past its timeout."""
    print("\nTest 0j: 429 caps oversized Retry-After")

    import api

    response_data, waits, n_attempts = await _run_429_with_retry_after("999")

    assert n_attempts == 2, f"Expected one retry then success, got {n_attempts}"
    assert response_data["choices"][0]["message"]["content"] == "ok"
    assert api._MAX_RATELIMIT_BACKOFF in waits, (
        f"Oversized Retry-After should be capped to {api._MAX_RATELIMIT_BACKOFF}, "
        f"waits={waits}"
    )
    assert all(w <= api._MAX_RATELIMIT_BACKOFF for w in waits), (
        f"No 429 wait may exceed the cap, waits={waits}"
    )
    print("  PASS")


async def main() -> None:
    print("=" * 60)
    print("Testing process_response - Edge Cases")
    print("=" * 60)

    test_system_prompt_distinguishes_repo_paths_from_workspace_paths()
    test_infer_tool_calls_from_xml_function_markup()
    await test_api_call_with_retry_streams_text_deltas()
    await test_api_call_with_retry_streams_tool_calls()
    await test_streaming_handles_data_line_split_across_reads()
    await test_api_call_with_retry_does_not_add_nvcf_header()
    await test_api_call_omits_empty_tools_array()
    await test_api_call_keeps_nonempty_tools_array()
    await test_api_call_retries_transient_200_body_error()
    await test_api_call_does_not_retry_permanent_client_error()
    await test_parse_class_retry_perturbs_sampling()
    await test_api_call_recovers_from_body_parse_failure()
    await test_429_honors_retry_after_header()
    await test_429_caps_oversized_retry_after()

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
