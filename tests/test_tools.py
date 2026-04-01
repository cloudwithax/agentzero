#!/usr/bin/env python3
"""Test script to debug tool calling implementation"""

import asyncio
import json
from tools import (
    TOOLS,
    read_file_tool,
    glob_tool,
    grep_tool,
    bash_tool,
)


async def test_tools():
    print("Testing individual tools...")

    # Test read
    result = await read_file_tool("main.py")
    print(
        f"Read result: success={result.get('success')}, content length={len(result.get('content', '')) if result.get('success') else 'N/A'}"
    )

    # Test glob
    result = await glob_tool("*.py")
    print(
        f"Glob result: success={result.get('success')}, matches={result.get('matches', [])}"
    )

    # Test grep
    result = await grep_tool("import")
    print(
        f"Grep result: success={result.get('success')}, matches count={len(result.get('matches', []))}"
    )

    # Test bash
    result = await bash_tool("echo 'Hello World'")
    print(
        f"Bash result: success={result.get('success')}, stdout={result.get('stdout', '').strip()}"
    )

    print("\nTesting tool execution flow...")


async def simulate_tool_call_flow():
    """Simulate how the agent would handle a tool call from the API response"""
    # Simulate a tool call from the model
    mock_tool_call = {
        "id": "call_123",
        "function": {"name": "read", "arguments": json.dumps({"filepath": "main.py"})},
    }

    func_name = mock_tool_call["function"]["name"]
    func_args = json.loads(mock_tool_call["function"]["arguments"])

    print(f"Simulating tool call: {func_name} with args {func_args}")

    if func_name in TOOLS:
        result = await TOOLS[func_name](**func_args)
        print(f"Tool result: {result}")

        # Format as tool response message
        tool_response = {
            "tool_call_id": mock_tool_call["id"],
            "role": "tool",
            "content": json.dumps(result),
        }
        print(f"Tool response message: {tool_response}")
        return tool_response

    return None


async def test_payload_mutation():
    """Test if payload mutation causes issues"""
    from handler import BASE_PAYLOAD

    print("\nTesting payload mutation...")
    payload = BASE_PAYLOAD.copy()
    original_messages = payload.get("messages", [])

    # Simulate adding messages
    payload["messages"] = [{"role": "user", "content": "test"}]
    print(f"After first set: messages count = {len(payload['messages'])}")

    payload["messages"].append({"role": "assistant", "content": "ok"})
    print(f"After append: messages count = {len(payload['messages'])}")

    # Check if this accumulates across calls
    payload["messages"] = [{"role": "user", "content": "test2"}]
    print(f"After reset: messages count = {len(payload['messages'])}")

    # Restore
    payload["messages"] = original_messages


if __name__ == "__main__":
    asyncio.run(test_tools())
    asyncio.run(simulate_tool_call_flow())
    asyncio.run(test_payload_mutation())
