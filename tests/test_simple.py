#!/usr/bin/env python3
"""Simple test to verify tool calling logic"""

import asyncio
import json
from tools import TOOLS
from handler import BASE_PAYLOAD


async def test_tools_work():
    """Test that all tools are callable"""
    print("Testing tools...")

    # Test read
    result = await TOOLS["read"](filepath="main.py")
    assert result["success"], f"Read failed: {result}"
    print(f"✓ read works, got {len(result['content'])} bytes")

    # Test glob
    result = await TOOLS["glob"](pattern="*.py")
    assert result["success"]
    assert "main.py" in result["matches"]
    print(f"✓ glob works, found {len(result['matches'])} files")

    # Test grep
    result = await TOOLS["grep"](pattern="import")
    assert result["success"]
    assert len(result["matches"]) > 0
    print(f"✓ grep works, found {len(result['matches'])} matches")

    # Test bash
    result = await TOOLS["bash"](command="echo 'test'")
    assert result["success"]
    assert "test" in result["stdout"]
    print(f"✓ bash works, stdout: {result['stdout'].strip()}")

    # Consortium task-management tools should be publicly available.
    assert "consortium_start" in TOOLS
    assert "consortium_stop" in TOOLS
    assert "consortium_status" in TOOLS
    print("✓ consortium management tools are registered")

    print("All tools functional!\n")


async def test_payload_isolation():
    """Test that BASE_PAYLOAD is not mutated"""
    print("Testing payload isolation...")

    original_keys = set(BASE_PAYLOAD.keys())
    original_messages = BASE_PAYLOAD.get("messages", [])
    tool_names = {
        tool.get("function", {}).get("name") for tool in BASE_PAYLOAD.get("tools", [])
    }
    assert "consortium_start" in tool_names, "consortium_start missing from payload"
    assert "consortium_stop" in tool_names, "consortium_stop missing from payload"
    assert "consortium_status" in tool_names, "consortium_status missing from payload"
    assert "consortium_agree" not in tool_names, "consortium_agree should not be public"

    # Simulate what handle() does
    test_payload = BASE_PAYLOAD.copy()
    test_payload["messages"] = [{"role": "user", "content": "test"}]

    # Check BASE_PAYLOAD is unchanged
    assert (
        "messages" not in BASE_PAYLOAD or BASE_PAYLOAD["messages"] == original_messages
    )
    assert set(BASE_PAYLOAD.keys()) == original_keys
    print("✓ BASE_PAYLOAD is not mutated by copy()")

    # Check test_payload has messages
    assert "messages" in test_payload
    assert len(test_payload["messages"]) == 1
    print("✓ New payload can be modified independently\n")


async def test_tool_call_parsing():
    """Test parsing tool call arguments"""
    print("Testing tool call parsing...")

    # Simulate a tool call from the API
    tool_call = {
        "id": "call_123",
        "function": {"name": "read", "arguments": json.dumps({"filepath": "main.py"})},
    }

    func_name = tool_call["function"]["name"]
    func_args = json.loads(tool_call["function"]["arguments"])

    assert func_name == "read"
    assert func_args["filepath"] == "main.py"

    result = await TOOLS[func_name](**func_args)
    assert result["success"]
    print("✓ Tool call parsed and executed successfully\n")


async def test_tool_result_format():
    """Test that tool results are formatted correctly for the API"""
    print("Testing tool result format...")

    result = await TOOLS["read"](filepath="main.py")
    tool_response = {
        "tool_call_id": "call_123",
        "role": "tool",
        "content": json.dumps(result),
    }

    # Verify it's valid JSON
    parsed = json.loads(tool_response["content"])
    assert "success" in parsed
    print(
        f"✓ Tool response format correct: {tool_response['role']} with JSON content\n"
    )


if __name__ == "__main__":
    asyncio.run(test_tools_work())
    asyncio.run(test_payload_isolation())
    asyncio.run(test_tool_call_parsing())
    asyncio.run(test_tool_result_format())
    print("All tests passed!")
