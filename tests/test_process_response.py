"""Test process_response function directly - no complex mocking needed."""

import asyncio
import json
import sys
from unittest.mock import AsyncMock

sys.path.insert(0, "/home/clxud/Documents/github/agentzero")
from api import process_response, execute_tool_calls
from handler import BASE_URL, API_KEY, BASE_PAYLOAD


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

    # Create proper async context manager mock
    class MockResponse:
        async def json(self):
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
        async def json(self):
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
        async def json(self):
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
        async def json(self):
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
    assert len(messages) == original_len + 2, (
        f"Messages not extended properly: {len(messages)}"
    )
    assert messages[2].get("tool_calls") is not None, "Missing tool_calls"
    assert messages[3]["role"] == "tool", "Missing tool result"

    print(f"  History length: {len(messages)} (was {original_len})")
    print("  ✓ Passed")


async def main():
    print("=" * 60)
    print("Testing process_response - Edge Cases")
    print("=" * 60)
    print()

    await test_regular_response()
    await test_empty_content()
    await test_missing_content_key()
    await test_tool_calls()
    await test_multiple_tool_calls()
    await test_unknown_tool()
    await test_tool_result_format()
    await test_message_history_preservation()

    print()
    print("=" * 60)
    print("All edge case tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
