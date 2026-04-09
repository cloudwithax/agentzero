"""Test process_response function directly - no complex mocking needed."""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api import (
    _apply_cache_busting_headers,
    api_call_with_retry,
    execute_tool_calls,
    infer_tool_calls_from_content,
    process_response,
)
from handler import BASE_URL, API_KEY, BASE_PAYLOAD, FINAL_RESPONSE_MAX_TOKENS
from prompt_templates import get_template


def test_cache_busting_headers_are_applied() -> None:
    """Every API request should carry explicit no-cache headers and a unique ID."""
    print("Test 0a: Cache-busting headers")

    first = _apply_cache_busting_headers({"Authorization": "Bearer test"})
    second = _apply_cache_busting_headers({"Authorization": "Bearer test"})

    assert first["Authorization"] == "Bearer test"
    assert first["Cache-Control"] == "no-cache, no-store, max-age=0"
    assert first["Pragma"] == "no-cache"
    assert first["Expires"] == "0"
    assert first["X-Request-Id"]
    assert second["X-Request-Id"]
    assert first["X-Request-Id"] != second["X-Request-Id"]
    print("  ✓ Passed")


def test_system_prompt_distinguishes_repo_paths_from_workspace_paths() -> None:
    """The prompt should tell the executor not to look for repo code under workspace/."""
    print("Test 0aa: System prompt repo path guidance")

    rendered = get_template(
        "system_prompt",
        {
            "current_time": "2026-04-09 12:00:00",
            "workspace_path": "/home/clxud/agentzero/workspace",
            "identity": "You are a helpful AI assistant.",
        },
    )

    assert "Repository code path rule:" in rendered
    assert "Do NOT assume repository files live under" in rendered
    assert "handler.py" in rendered
    assert "agentic_loop.py" in rendered
    assert "consult_reviewer()" in rendered
    assert "All normal user-facing turns run on the advisor model by default." in rendered
    assert "TOOL RESULTS ARE GROUND TRUTH:" in rendered
    assert "Do not self-reject or manually \"safety review\" a user-provided skill URL" in rendered
    print("  ✓ Passed")


def test_infer_tool_calls_from_xml_function_markup() -> None:
    """XML-style function markup should recover into a real tool call."""
    print("Test 0ab: Recover XML function markup")

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
    print("  ✓ Passed")


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


async def test_api_call_with_retry_streams_text_deltas():
    """Streaming API calls should emit text deltas and assemble final content."""
    print("Test 0b: Streaming text deltas")

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
    print("  ✓ Passed")


async def test_api_call_with_retry_streams_tool_calls():
    """Streaming API calls should assemble tool call names and arguments correctly."""
    print("Test 0c: Streaming tool-call assembly")

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
    assert (
        tool_calls[0]["function"]["name"] == "bash"
    ), f"Unexpected tool call: {tool_calls[0]}"
    assert tool_calls[0]["function"]["arguments"] == '{"command": "echo hi"}'
    print("  ✓ Passed")


async def test_api_call_with_retry_does_not_add_nvcf_header():
    """NVCF image-asset request headers should never be injected."""
    print("Test 0d: No NVCF header injection")

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
        {
            "model": "test-model",
            "messages": [
                {
                    "role": "user",
                    "content": "data:image/png;asset_id,aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                }
            ],
        },
        {"Authorization": "Bearer test"},
        stream=False,
    )

    assert not any(
        key.lower() == "nvcf-input-asset-references" for key in captured_headers
    ), f"Unexpected NVCF header injection: {captured_headers}"
    print("  ✓ Passed")


async def test_regular_response():
    """Test response with just content (no tool calls)."""
    print("Test 1: Regular text response")

    response_data = {
        "choices": [
            {"message": {"role": "assistant", "content": "Hello! How can I help?"}}
        ]
    }

    mock_session = AsyncMock()
    messages = [{"role": "user", "content": "Hi"}]

    content = await process_response(
        response_data, messages, mock_session, BASE_URL, API_KEY, BASE_PAYLOAD.copy()
    )

    assert content == "Hello! How can I help?", f"Got: {content}"
    print(f"  Content: {content}")
    print("  ✓ Passed")


async def test_empty_content():
    """Test response with empty/null content."""
    print("Test 2: Empty/null content")

    response_data = {"choices": [{"message": {"role": "assistant", "content": None}}]}

    mock_session = AsyncMock()
    messages = [{"role": "user", "content": "Test"}]

    content = await process_response(
        response_data, messages, mock_session, BASE_URL, API_KEY, BASE_PAYLOAD.copy()
    )

    assert content == "", f"Expected empty string, got: {content}"
    print(f"  Content: '{content}'")
    print("  ✓ Passed (handled gracefully)")


async def test_missing_content_key():
    """Test response missing content key entirely."""
    print("Test 3: Missing content key")

    response_data = {
        "choices": [
            {
                "message": {
                    "role": "assistant"
                    # No content key
                }
            }
        ]
    }

    mock_session = AsyncMock()
    messages = [{"role": "user", "content": "Test"}]

    content = await process_response(
        response_data, messages, mock_session, BASE_URL, API_KEY, BASE_PAYLOAD.copy()
    )

    assert content == "", f"Expected empty string, got: {content}"
    print(f"  Content: '{content}'")
    print("  ✓ Passed (handled gracefully)")


async def test_tool_calls():
    """Test response with single tool call."""
    print("Test 4: Single tool call")

    first_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": json.dumps({"command": "echo 'success'"}),
                            },
                        }
                    ],
                }
            }
        ]
    }

    second_response = {
        "choices": [
            {"message": {"role": "assistant", "content": "Command output: success"}}
        ]
    }

    class MockResponse:
        status = 200

        async def json(self, content_type=None):
            return second_response

    class MockPostCM:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return MockResponse()

        async def __aexit__(self, *args):
            return False

    mock_session = AsyncMock()
    mock_session.post = (
        MockPostCM  # Return the class, it will be instantiated with args
    )

    messages = [{"role": "user", "content": "Run command"}]

    content = await process_response(
        first_response, messages, mock_session, BASE_URL, API_KEY, BASE_PAYLOAD.copy()
    )

    # Verify messages list was properly extended
    # Original: 1 user message, now should have tool call + result added
    assert len(messages) == 3, f"Expected 3 messages, got {len(messages)}"
    assert messages[1].get("tool_calls") is not None, "Missing tool_calls in history"
    assert messages[2]["role"] == "tool", "Missing tool result"

    print(f"  Final content: {content}")
    print("  ✓ Passed (tool executed, follow-up called)")


async def test_multiple_tool_calls():
    """Test response with multiple tool calls."""
    print("Test 5: Multiple tool calls")

    response_data = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": json.dumps({"command": "echo 'first'"}),
                            },
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": json.dumps({"command": "echo 'second'"}),
                            },
                        },
                    ],
                }
            }
        ]
    }

    second_response = {
        "choices": [{"message": {"role": "assistant", "content": "Both done"}}]
    }

    class MockResponse:
        status = 200

        async def json(self, content_type=None):
            return second_response

    class MockPostCM:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return MockResponse()

        async def __aexit__(self, *args):
            return False

    mock_session = AsyncMock()
    mock_session.post = MockPostCM

    messages = [{"role": "user", "content": "Run two commands"}]

    await process_response(
        response_data, messages, mock_session, BASE_URL, API_KEY, BASE_PAYLOAD.copy()
    )

    # Verify messages list was properly extended
    # Should have: user message + assistant tool_calls + 2 tool results = 4
    assert len(messages) == 4, f"Expected 4 messages, got {len(messages)}"

    tool_results = [m for m in messages if m["role"] == "tool"]
    assert len(tool_results) == 2, f"Expected 2 tool results, got {len(tool_results)}"

    print("  Both tools executed and results added to conversation")
    print("  ✓ Passed")


async def test_unknown_tool():
    """Test handling of unknown tool names."""
    print("Test 6: Unknown tool name")

    response_data = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_999",
                            "type": "function",
                            "function": {
                                "name": "nonexistent_tool",
                                "arguments": json.dumps({"param": "value"}),
                            },
                        }
                    ],
                }
            }
        ]
    }

    second_response = {
        "choices": [
            {"message": {"role": "assistant", "content": "I don't know that tool"}}
        ]
    }

    class MockResponse:
        status = 200

        async def json(self, content_type=None):
            return second_response

    class MockPostCM:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return MockResponse()

        async def __aexit__(self, *args):
            return False

    mock_session = AsyncMock()
    mock_session.post = MockPostCM

    messages = [{"role": "user", "content": "Test"}]

    await process_response(
        response_data, messages, mock_session, BASE_URL, API_KEY, BASE_PAYLOAD.copy()
    )

    # Unknown tool should not crash, just won't produce result
    # Check messages list directly
    tool_results = [m for m in messages if m["role"] == "tool"]

    print(f"  Tool results for unknown tool: {len(tool_results)}")
    # Unknown tool produces no result entry
    print("  ✓ Passed (no crash)")


async def test_tool_result_format():
    """Verify tool results are properly formatted."""
    print("Test 7: Tool result JSON format")

    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_test",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": json.dumps({"command": "echo 'test'"}),
                },
            }
        ],
    }

    results = await execute_tool_calls(message)

    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    result = results[0]

    assert result["role"] == "tool", f"Wrong role: {result['role']}"
    assert result["tool_call_id"] == "call_test", f"Wrong ID: {result['tool_call_id']}"
    assert "content" in result, "Missing content"

    # Content should be valid JSON
    parsed = json.loads(result["content"])
    assert "success" in parsed, "Tool result missing success field"

    print(f"  Result format: {result}")
    print("  ✓ Passed (proper JSON format)")


async def test_message_history_preservation():
    """Verify original messages are preserved and extended with tool calls."""
    print("Test 8: Message history preservation")

    original_messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Run command"},
    ]
    original_len = len(original_messages)

    response_data = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_hist",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": '{"command": "echo test"}',
                            },
                        }
                    ],
                }
            }
        ]
    }

    second_response = {
        "choices": [{"message": {"role": "assistant", "content": "Done"}}]
    }

    class MockResponse:
        status = 200

        async def json(self, content_type=None):
            return second_response

    class MockPostCM:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return MockResponse()

        async def __aexit__(self, *args):
            return False

    mock_session = AsyncMock()
    mock_session.post = MockPostCM

    messages = original_messages.copy()

    await process_response(
        response_data, messages, mock_session, BASE_URL, API_KEY, BASE_PAYLOAD.copy()
    )

    # Messages should be extended with assistant tool_calls + tool result
    # Original: 2 messages, after: 2 + 1 (assistant) + 1 (tool) = 4
    assert (
        len(messages) == original_len + 2
    ), f"Messages not extended properly: {len(messages)}"
    assert messages[2].get("tool_calls") is not None, "Missing tool_calls"
    assert messages[3]["role"] == "tool", "Missing tool result"

    print(f"  History length: {len(messages)} (was {original_len})")
    print("  ✓ Passed")


async def test_tool_leak_retry():
    """Recover and execute bash tool call from leaked bash code block."""
    print("Test 9: Tool-leak recovery to tool call")

    first_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "```bash\ncurl -X POST https://example.com\n```",
                }
            }
        ]
    }

    second_response = {
        "choices": [
            {"message": {"role": "assistant", "content": "All set, I fixed it."}}
        ]
    }

    class MockResponse:
        def __init__(self, payload):
            self.payload = payload
            self.status = 200

        async def json(self, content_type=None):
            return self.payload

    class MockPostCM:
        def __init__(self, payload):
            self.payload = payload

        async def __aenter__(self):
            return MockResponse(self.payload)

        async def __aexit__(self, *args):
            return False

    class MockSession:
        def __init__(self):
            self.calls = 0

        def post(self, *args, **kwargs):
            self.calls += 1
            return MockPostCM(second_response)

    mock_session = MockSession()
    messages = [{"role": "user", "content": "please fix"}]

    content = await process_response(
        first_response, messages, mock_session, BASE_URL, API_KEY, BASE_PAYLOAD.copy()
    )

    assert content == "All set, I fixed it.", f"Unexpected content: {content}"
    assert (
        mock_session.calls == 1
    ), f"Expected one follow-up call, got {mock_session.calls}"
    tool_results = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_results) == 1, "Expected inferred bash tool result in history"
    print("  ✓ Passed")


async def test_tool_leak_fallback():
    """Return fallback text for non-recoverable leaked tool-call content."""
    print("Test 10: Tool-leak fallback")

    leaked_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "<|tool_call|>not-valid-json",
                }
            }
        ]
    }

    class MockResponse:
        status = 200

        async def json(self, content_type=None):
            return leaked_response

    class MockPostCM:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return MockResponse()

        async def __aexit__(self, *args):
            return False

    mock_session = AsyncMock()
    mock_session.post = MockPostCM

    messages = [{"role": "user", "content": "please fix"}]

    content = await process_response(
        leaked_response,
        messages,
        mock_session,
        BASE_URL,
        API_KEY,
        BASE_PAYLOAD.copy(),
        max_tool_leak_retries=1,
    )

    assert (
        "internal formatting issue" in content.lower()
    ), f"Unexpected content: {content}"
    print("  ✓ Passed")


async def test_json_tool_call_recovery():
    """Recover and execute tool call from tool_call tag JSON payload."""
    print("Test 11: JSON tool-call recovery")

    first_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": '<|tool_call|>{"name":"bash","arguments":{"command":"echo recovered"}}',
                }
            }
        ]
    }

    second_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Recovered and executed successfully.",
                }
            }
        ]
    }

    class MockResponse:
        def __init__(self, payload):
            self.payload = payload
            self.status = 200

        async def json(self, content_type=None):
            return self.payload

    class MockPostCM:
        def __init__(self, payload):
            self.payload = payload

        async def __aenter__(self):
            return MockResponse(self.payload)

        async def __aexit__(self, *args):
            return False

    class MockSession:
        def __init__(self):
            self.calls = 0

        def post(self, *args, **kwargs):
            self.calls += 1
            return MockPostCM(second_response)

    mock_session = MockSession()
    messages = [{"role": "user", "content": "run this please"}]

    content = await process_response(
        first_response, messages, mock_session, BASE_URL, API_KEY, BASE_PAYLOAD.copy()
    )

    assert (
        content == "Recovered and executed successfully."
    ), f"Unexpected content: {content}"
    assert (
        mock_session.calls == 1
    ), f"Expected one follow-up call, got {mock_session.calls}"
    tool_results = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_results) == 1, "Expected recovered tool result in history"
    print("  ✓ Passed")


async def test_complex_multi_round_agentic_flow() -> None:
    """Complex tasks should survive narration and multiple tool rounds."""
    print("Test 12: Complex multi-round agentic flow")

    initial_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": (
                        "let me inspect the reminder state and run the publish step now."
                    ),
                }
            }
        ]
    }

    followup_responses = [
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_check_state",
                                "type": "function",
                                "function": {
                                    "name": "bash",
                                    "arguments": json.dumps(
                                        {"command": "echo reminder state clean"}
                                    ),
                                },
                            }
                        ],
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_publish",
                                "type": "function",
                                "function": {
                                    "name": "bash",
                                    "arguments": json.dumps(
                                        {
                                            "command": "echo https://cedar-canopy-ytnp.here.now/"
                                        }
                                    ),
                                },
                            }
                        ],
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": (
                            "Reminder state is clean and the site is live at "
                            "https://cedar-canopy-ytnp.here.now/."
                        ),
                    }
                }
            ]
        },
    ]

    tool_results = [
        [
            {
                "tool_call_id": "call_check_state",
                "role": "tool",
                "content": json.dumps(
                    {"success": True, "stdout": "reminder state clean\n"}
                ),
            }
        ],
        [
            {
                "tool_call_id": "call_publish",
                "role": "tool",
                "content": json.dumps(
                    {
                        "success": True,
                        "stdout": "https://cedar-canopy-ytnp.here.now/\n",
                    }
                ),
            }
        ],
    ]

    async def fake_api_call_with_retry(*args, **kwargs):
        assert followup_responses, "Unexpected extra follow-up API call"
        return followup_responses.pop(0)

    execute_tool_calls = AsyncMock(side_effect=tool_results)
    messages = [{"role": "user", "content": "Check reminders, publish, and summarize"}]

    with (
        patch("agentic_loop.api_call_with_retry", side_effect=fake_api_call_with_retry),
        patch("agentic_loop.execute_tool_calls", execute_tool_calls),
    ):
        content = await process_response(
            initial_response,
            messages,
            AsyncMock(),
            BASE_URL,
            API_KEY,
            BASE_PAYLOAD.copy(),
        )

    assert "Reminder state is clean" in content
    assert "https://cedar-canopy-ytnp.here.now/" in content
    assert execute_tool_calls.await_count == 2
    tool_messages = [message for message in messages if message.get("role") == "tool"]
    assert len(tool_messages) == 2, f"Expected two tool results, got {tool_messages}"
    assert any(
        "did not make any tool calls" in message.get("content", "")
        for message in messages
        if message.get("role") == "user"
    ), "Expected narration nudge in message history"
    print("  ✓ Passed")


async def test_initial_payload_has_sufficient_max_tokens_for_complex_requests() -> None:
    """Regression: complex multi-step requests (e.g. 'make a website') must not be
    truncated by a low max_tokens limit on the initial API call.

    Before the fix, handler.py applied FINAL_RESPONSE_MAX_TOKENS (150) to the
    initial payload, which truncated the model's tool-call response for complex
    tasks and caused 'Error: No response from API'.
    """
    print("Test 13: Initial payload has sufficient max_tokens for complex requests")

    from handler import AgentHandler

    tmp = Path("/tmp/test_max_tokens_regression.db")
    tmp.touch()

    try:
        from memory import EnhancedMemoryStore
        from capabilities import CapabilityProfile, AdaptiveFormatter
        from examples import AdaptiveFewShotManager, ExampleBank
        from planning import TaskPlanner, TaskAnalyzer

        store = EnhancedMemoryStore(db_path=str(tmp), api_key="test")
        profile = CapabilityProfile(set(), model_name="test")
        planner = TaskPlanner(profile)
        analyzer = TaskAnalyzer()
        formatter = AdaptiveFormatter(profile)
        examples_manager = AdaptiveFewShotManager(ExampleBank())

        handler_instance = AgentHandler(
            memory_store=store,
            capability_profile=profile,
            example_bank=examples_manager,
            task_planner=planner,
            task_analyzer=analyzer,
            adaptive_formatter=formatter,
        )

        captured_payloads: list[dict] = []

        async def _capture_payload(_session, _url, payload, _headers, **_kw):
            captured_payloads.append(dict(payload))
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "I'll build that for you.",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "write",
                                        "arguments": json.dumps(
                                            {
                                                "filepath": "workspace/index.html",
                                                "content": "<html>...</html>",
                                            }
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }

        async def _fake_execute_tools(message, **_kw):
            return [
                {
                    "tool_call_id": "call_1",
                    "role": "tool",
                    "content": json.dumps({"success": True}),
                }
            ]

        with (
            patch(
                "handler.api_call_with_retry",
                new=AsyncMock(side_effect=_capture_payload),
            ),
            patch(
                "agentic_loop.execute_tool_calls",
                new=AsyncMock(side_effect=_fake_execute_tools),
            ),
            patch.object(store, "search_memories", new=AsyncMock(return_value=[])),
        ):
            result = await handler_instance.handle(
                {
                    "messages": [
                        {"role": "user", "content": "make a 3d dice roller website"}
                    ]
                },
                session_id="test_session",
            )

            assert len(captured_payloads) >= 1, "Expected at least one API call"
            initial = captured_payloads[0]
            initial_max_tokens = initial.get("max_tokens", 0)

            assert initial_max_tokens >= 8000, (
                f"Initial payload max_tokens={initial_max_tokens} is too low for "
                f"complex multi-step requests; the model needs enough budget to emit "
                f"complete tool calls (function name + JSON arguments). "
                f"FINAL_RESPONSE_MAX_TOKENS ({FINAL_RESPONSE_MAX_TOKENS}) should NOT "
                f"be applied to the initial payload."
            )

            assert "Error: No response from API" not in (result or "")

        print("  ✓ Passed")

    finally:
        tmp.unlink(missing_ok=True)


async def test_consult_advisor_uses_advisor_model_and_shared_context() -> None:
    """Advisor consultations should use the advisor model, no tools, and shared context."""
    print("Test 14: consult_advisor uses advisor model and shared context")

    from handler import ADVISOR_MODEL_ID, AgentHandler
    from memory import EnhancedMemoryStore
    from capabilities import CapabilityProfile, AdaptiveFormatter
    from examples import AdaptiveFewShotManager, ExampleBank
    from planning import TaskPlanner, TaskAnalyzer

    tmp = Path("/tmp/test_consult_advisor_regression.db")
    tmp.touch()

    try:
        store = EnhancedMemoryStore(db_path=str(tmp), api_key="test")
        profile = CapabilityProfile(set(), model_name="test")
        planner = TaskPlanner(profile)
        analyzer = TaskAnalyzer()
        formatter = AdaptiveFormatter(profile)
        examples_manager = AdaptiveFewShotManager(ExampleBank())

        handler_instance = AgentHandler(
            memory_store=store,
            capability_profile=profile,
            example_bank=examples_manager,
            task_planner=planner,
            task_analyzer=analyzer,
            adaptive_formatter=formatter,
        )

        captured_payloads: list[dict[str, object]] = []

        async def _capture_payload(_session, _url, payload, _headers, **_kw):
            captured_payloads.append(dict(payload))
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                "Decision:\nAdd the advisor tool before editing the loop.\n"
                                "Why:\nIt gives the executor a stable contract to call.\n"
                                "Next steps:\n1. Register the tool.\n2. Wire shared context.\n"
                                "Risks:\nAvoid losing live turn context."
                            ),
                        }
                    }
                ]
            }

        shared_messages = [
            {"role": "system", "content": "Base executor system prompt."},
            {
                "role": "user",
                "content": "Make advisor consultation the default pattern.",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_read",
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": json.dumps({"filepath": "agentic_loop.py"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_read",
                "content": json.dumps({"success": True, "content": "loop contents"}),
            },
        ]

        with patch(
            "handler.api_call_with_retry",
            new=AsyncMock(side_effect=_capture_payload),
        ):
            result = await handler_instance.consult_advisor(
                question="What is the least risky implementation order?",
                context="I need the smallest coherent change set.",
                session_id="advisor_test_session",
                shared_messages=shared_messages,
            )

        assert result["success"] is True, f"Unexpected advisor result: {result}"
        assert result["advisor_model"] == ADVISOR_MODEL_ID
        assert captured_payloads, "Expected one advisor payload"

        payload = captured_payloads[0]
        assert payload["model"] == ADVISOR_MODEL_ID
        assert payload["tools"] == []
        payload_messages = payload["messages"]
        assert payload_messages[0]["role"] == "system"
        assert "executor/advisor strategy" in payload_messages[0]["content"]
        assert any(message.get("role") == "tool" for message in payload_messages)
        assert any(message.get("tool_calls") for message in payload_messages)
        assert payload_messages[-1]["role"] == "user"
        assert (
            "Decision needed: What is the least risky implementation order?"
            in payload_messages[-1]["content"]
        )
        assert "smallest coherent change set" in payload_messages[-1]["content"]
        assert result["advice"].startswith("Decision:"), result["advice"]

        print("  ✓ Passed")
    finally:
        tmp.unlink(missing_ok=True)


async def test_consult_reviewer_uses_reviewer_model_and_shared_context() -> None:
    """Reviewer consultations should use the reviewer model, no tools, and shared context."""
    print("Test 15: consult_reviewer uses reviewer model and shared context")

    from handler import REVIEWER_MODEL_ID, AgentHandler
    from memory import EnhancedMemoryStore
    from capabilities import CapabilityProfile, AdaptiveFormatter
    from examples import AdaptiveFewShotManager, ExampleBank
    from planning import TaskPlanner, TaskAnalyzer

    tmp = Path("/tmp/test_consult_reviewer_regression.db")
    tmp.touch()

    try:
        store = EnhancedMemoryStore(db_path=str(tmp), api_key="test")
        profile = CapabilityProfile(set(), model_name="test")
        planner = TaskPlanner(profile)
        analyzer = TaskAnalyzer()
        formatter = AdaptiveFormatter(profile)
        examples_manager = AdaptiveFewShotManager(ExampleBank())

        handler_instance = AgentHandler(
            memory_store=store,
            capability_profile=profile,
            example_bank=examples_manager,
            task_planner=planner,
            task_analyzer=analyzer,
            adaptive_formatter=formatter,
        )

        captured_payloads: list[dict[str, object]] = []

        async def _capture_payload(_session, _url, payload, _headers, **_kw):
            captured_payloads.append(dict(payload))
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                "Verdict:\nThe implementation slice is coherent.\n"
                                "Findings:\nShared context is the main risk boundary.\n"
                                "Fixes:\nMirror advisor plumbing first.\n"
                                "Residual risks:\nDo not auto-invoke until telemetry exists."
                            ),
                        }
                    }
                ]
            }

        shared_messages = [
            {"role": "system", "content": "Base executor system prompt."},
            {
                "role": "user",
                "content": "Implement consult_reviewer with the smallest viable slice.",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_read",
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": json.dumps({"filepath": "handler.py"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_read",
                "content": json.dumps({"success": True, "content": "handler contents"}),
            },
        ]

        with patch(
            "handler.api_call_with_retry",
            new=AsyncMock(side_effect=_capture_payload),
        ):
            result = await handler_instance.consult_reviewer(
                question="Review the first consult_reviewer implementation slice.",
                context="Goal: mirror advisor plumbing without auto-invocation.",
                session_id="reviewer_test_session",
                shared_messages=shared_messages,
            )

        assert result["success"] is True, f"Unexpected reviewer result: {result}"
        assert result["reviewer_model"] == REVIEWER_MODEL_ID
        assert captured_payloads, "Expected one reviewer payload"

        payload = captured_payloads[0]
        assert payload["model"] == REVIEWER_MODEL_ID
        assert payload["tools"] == []
        payload_messages = payload["messages"]
        assert payload_messages[0]["role"] == "system"
        assert "reviewer model" in payload_messages[0]["content"]
        assert any(message.get("role") == "tool" for message in payload_messages)
        assert any(message.get("tool_calls") for message in payload_messages)
        assert payload_messages[-1]["role"] == "user"
        assert (
            "Review target: Review the first consult_reviewer implementation slice."
            in payload_messages[-1]["content"]
        )
        assert "mirror advisor plumbing" in payload_messages[-1]["content"]
        assert result["review"].startswith("Verdict:"), result["review"]

        print("  ✓ Passed")
    finally:
        tmp.unlink(missing_ok=True)


async def main():
    print("=" * 60)
    print("Testing process_response - Edge Cases")
    print("=" * 60)
    print()

    test_cache_busting_headers_are_applied()
    test_system_prompt_distinguishes_repo_paths_from_workspace_paths()

    await test_api_call_with_retry_streams_text_deltas()
    await test_api_call_with_retry_streams_tool_calls()
    await test_api_call_with_retry_does_not_add_nvcf_header()
    await test_regular_response()
    await test_empty_content()
    await test_missing_content_key()
    await test_tool_calls()
    await test_multiple_tool_calls()
    await test_unknown_tool()
    await test_tool_result_format()
    await test_message_history_preservation()
    await test_tool_leak_retry()
    await test_tool_leak_fallback()
    await test_json_tool_call_recovery()
    await test_complex_multi_round_agentic_flow()
    await test_initial_payload_has_sufficient_max_tokens_for_complex_requests()
    await test_consult_advisor_uses_advisor_model_and_shared_context()
    await test_consult_reviewer_uses_reviewer_model_and_shared_context()

    print()
    print("=" * 60)
    print("All edge case tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
