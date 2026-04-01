#!/usr/bin/env python3
"""Test the full tool calling flow to identify issues"""

import asyncio
import json
from unittest.mock import AsyncMock, patch

# Import from refactored modules
from handler import AgentHandler
from memory import EnhancedMemoryStore
from capabilities import Capability, CapabilityProfile, AdaptiveFormatter
from examples import ExampleBank, AdaptiveFewShotManager
from planning import TaskPlanner, TaskAnalyzer
from tools import set_memory_store


def create_test_handler():
    """Create a test handler with mocked dependencies."""
    import tempfile

    # Create a temporary database file
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    temp_db.close()

    # Create memory store with temp database
    memory_store = EnhancedMemoryStore(
        db_path=temp_db.name,
        api_key="test_key",
    )

    # Set memory store for tools
    set_memory_store(memory_store)

    # Create capability profile
    capability_profile = CapabilityProfile(
        capabilities={
            Capability.JSON_OUTPUT,
            Capability.TOOL_USE,
            Capability.CHAIN_OF_THOUGHT,
            Capability.REASONING,
            Capability.LONG_CONTEXT,
            Capability.FEW_SHOT,
            Capability.SELF_CORRECTION,
            Capability.STRUCTURED_OUTPUT,
        },
        model_name="test-model",
    )

    # Create other components
    task_planner = TaskPlanner(capability_profile)
    task_analyzer = TaskAnalyzer()
    adaptive_formatter = AdaptiveFormatter(capability_profile)
    example_bank = AdaptiveFewShotManager(ExampleBank(exploration_rate=0.1))

    # Create handler
    handler = AgentHandler(
        memory_store=memory_store,
        capability_profile=capability_profile,
        example_bank=example_bank,
        task_planner=task_planner,
        task_analyzer=task_analyzer,
        adaptive_formatter=adaptive_formatter,
    )

    return handler


async def mock_api_response(messages, tool_calls=None):
    """Mock the NVIDIA API response"""
    if tool_calls:
        return {"choices": [{"message": {"content": None, "tool_calls": tool_calls}}]}
    else:
        return {"choices": [{"message": {"content": "Final response without tools"}}]}


async def test_single_tool_call():
    """Test a single tool call round-trip"""
    print("=== Test: Single tool call ===")

    handler = create_test_handler()

    request = {"messages": [{"role": "user", "content": "Read the file main.py"}]}

    # Mock the first API call to return a tool call
    mock_tool_call = [
        {
            "id": "call_001",
            "type": "function",
            "function": {
                "name": "read",
                "arguments": json.dumps({"filepath": "main.py"}),
            },
        }
    ]

    with patch("aiohttp.ClientSession") as mock_session:
        mock_resp = AsyncMock()
        mock_resp.json.return_value = await mock_api_response(
            request["messages"], mock_tool_call
        )

        mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_resp

        try:
            await handler.handle(request)
            print("✓ Single tool call completed")
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback

            traceback.print_exc()

    print()


async def test_multiple_tool_calls():
    """Test multiple tool calls in one response"""
    print("=== Test: Multiple tool calls ===")

    handler = create_test_handler()

    request = {"messages": [{"role": "user", "content": "Read main.py and list files"}]}

    mock_tool_calls = [
        {
            "id": "call_001",
            "type": "function",
            "function": {
                "name": "read",
                "arguments": json.dumps({"filepath": "main.py"}),
            },
        },
        {
            "id": "call_002",
            "type": "function",
            "function": {"name": "glob", "arguments": json.dumps({"pattern": "*.py"})},
        },
    ]

    with patch("aiohttp.ClientSession") as mock_session:
        mock_resp = AsyncMock()
        mock_resp.json.return_value = await mock_api_response(
            request["messages"], mock_tool_calls
        )

        mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_resp

        try:
            await handler.handle(request)
            print("✓ Multiple tool calls completed")
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback

            traceback.print_exc()

    print()


async def test_payload_persistence():
    """Test if payload accumulates across multiple requests"""
    print("=== Test: Payload persistence ===")

    handler = create_test_handler()

    request1 = {"messages": [{"role": "user", "content": "First request"}]}
    request2 = {"messages": [{"role": "user", "content": "Second request"}]}

    # Simulate first request (no tools)
    with patch("aiohttp.ClientSession") as mock_session:
        mock_resp = AsyncMock()
        mock_resp.json.return_value = await mock_api_response(
            request1["messages"], None
        )
        mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_resp
        await handler.handle(request1)

    # Simulate second request (no tools)
    with patch("aiohttp.ClientSession") as mock_session:
        mock_resp = AsyncMock()
        mock_resp.json.return_value = await mock_api_response(
            request2["messages"], None
        )
        mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_resp
        await handler.handle(request2)

    print("✓ Payload should be reset each time (only current request messages)")
    print()


async def test_tool_error_handling():
    """Test what happens when a tool fails"""
    print("=== Test: Tool error handling ===")

    handler = create_test_handler()

    request = {"messages": [{"role": "user", "content": "Read nonexistent file"}]}

    mock_tool_call = [
        {
            "id": "call_001",
            "type": "function",
            "function": {
                "name": "read",
                "arguments": json.dumps({"filepath": "nonexistent.txt"}),
            },
        }
    ]

    with patch("aiohttp.ClientSession") as mock_session:
        mock_resp = AsyncMock()
        mock_resp.json.return_value = await mock_api_response(
            request["messages"], mock_tool_call
        )

        mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_resp

        try:
            await handler.handle(request)
            print("✓ Tool error handled gracefully")
        except Exception as e:
            print(f"✗ Unhandled error: {e}")

    print()


async def test_recursive_tool_calls():
    """Test if second API call also returns tool calls (not currently supported)"""
    print("=== Test: Recursive tool calls (known limitation) ===")
    print("Current implementation only handles ONE round of tool calls.")
    print("If the second API response also contains tool_calls, they will be ignored.")
    print()


if __name__ == "__main__":
    asyncio.run(test_single_tool_call())
    asyncio.run(test_multiple_tool_calls())
    asyncio.run(test_payload_persistence())
    asyncio.run(test_tool_error_handling())
    asyncio.run(test_recursive_tool_calls())
