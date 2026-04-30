#!/usr/bin/env python3
"""Simple test to verify tool calling logic"""

import asyncio
import json
import os
import tempfile
from tools import TOOLS
from tools import (
    reset_tool_runtime_session,
    reset_tool_runtime_messages,
    set_memory_store,
    set_tool_runtime_messages,
    set_tool_runtime_session,
)
from handler import BASE_PAYLOAD
from memory import EnhancedMemoryStore


async def test_tools_work():
    """Test that all tools are callable"""
    print("Testing tools...")

    # Test read
    result = await TOOLS["read"](filepath="main.py")
    assert result["success"], f"Read failed: {result}"
    print(f"✓ read works, got {len(result['content'])} bytes")

    sliced = await TOOLS["read"](filepath="AGENTS.md", limit=1)
    assert sliced["success"], f"Read slice failed: {sliced}"
    assert sliced["content"].startswith("# AgentZero"), f"Unexpected slice: {sliced}"
    print("✓ read supports model-style limit slicing")

    sliced_with_offset = await TOOLS["read"](filepath="AGENTS.md", offset=1, limit=1)
    assert sliced_with_offset["success"], (
        f"Read offset slice failed: {sliced_with_offset}"
    )
    assert sliced_with_offset["content"].startswith("# AgentZero"), (
        f"Unexpected offset slice: {sliced_with_offset}"
    )
    print("✓ read supports human-friendly offset slicing")

    # Test glob
    result = await TOOLS["glob"](pattern="*.py")
    assert result["success"]
    assert "main.py" in result["matches"]
    print(f"✓ glob works, found {len(result['matches'])} files")

    # Test grep
    result = await TOOLS["grep"](pattern="import", path="tests")
    assert result["success"]
    assert len(result["matches"]) > 0
    print(f"✓ grep works, found {len(result['matches'])} matches")

    include_result = await TOOLS["grep"](
        pattern="Pitfall:",
        path=".",
        include="AGENTS.md",
        max_matches=1,
    )
    assert include_result["success"], f"Grep include failed: {include_result}"
    assert len(include_result["matches"]) == 1, (
        f"Unexpected grep include matches: {include_result}"
    )
    print("✓ grep supports include/max_matches arguments")

    # Test bash
    result = await TOOLS["bash"](command="echo 'test'")
    assert result["success"]
    assert "test" in result["stdout"]
    print(f"✓ bash works, stdout: {result['stdout'].strip()}")

    # Consortium task-management tools should be publicly available.
    assert "consortium_start" in TOOLS
    assert "consortium_stop" in TOOLS
    assert "consortium_status" in TOOLS
    assert "reminder_create" in TOOLS
    assert "reminder_list" in TOOLS
    assert "reminder_status" in TOOLS
    assert "reminder_cancel" in TOOLS
    assert "reminder_run_now" in TOOLS
    assert "send_tapback" in TOOLS
    assert "send_telegram_reaction" in TOOLS
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
    assert "reminder_create" in tool_names, "reminder_create missing from payload"
    assert "reminder_list" in tool_names, "reminder_list missing from payload"
    assert "reminder_status" in tool_names, "reminder_status missing from payload"
    assert "reminder_cancel" in tool_names, "reminder_cancel missing from payload"
    assert "reminder_run_now" in tool_names, "reminder_run_now missing from payload"
    assert "send_tapback" in tool_names, "send_tapback missing from payload"
    assert "send_telegram_reaction" in tool_names, (
        "send_telegram_reaction missing from payload"
    )
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


async def test_remember_tool_persists_runtime_session_id():
    """Explicit remember() calls should stay tied to the active runtime session."""
    print("Testing remember runtime session binding...")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.close()

    try:
        store = EnhancedMemoryStore(db_path=tmp.name, api_key="")
        set_memory_store(store)

        session_token = set_tool_runtime_session("tg_memory_test")
        try:
            result = await TOOLS["remember"](
                content="User goes by Alice.",
                importance="high",
            )
        finally:
            reset_tool_runtime_session(session_token)

        assert result["success"], f"Remember failed: {result}"
        recent = store.get_recent_memories(limit=1)
        assert recent, "Expected explicit memory to be stored"
        assert recent[0].metadata.get("session_id") == "tg_memory_test", recent[0]
        print("✓ remember stores active session_id for continuity\n")
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)


async def test_remember_tool_rewrites_assistant_name_from_user_turn():
    """Explicit remember() should correct assistant-name perspective mistakes."""
    print("Testing remember assistant-name perspective correction...")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.close()

    try:
        store = EnhancedMemoryStore(db_path=tmp.name, api_key="")
        set_memory_store(store)

        session_token = set_tool_runtime_session("tg_identity_test")
        messages_token = set_tool_runtime_messages(
            [
                {
                    "role": "user",
                    "content": "your name is alice, remember that",
                }
            ]
        )
        try:
            result = await TOOLS["remember"](
                content="User's name is Alice.",
                importance="high",
            )
        finally:
            reset_tool_runtime_messages(messages_token)
            reset_tool_runtime_session(session_token)

        assert result["success"], f"Remember failed: {result}"
        recent = store.get_recent_memories(limit=1)
        assert recent, "Expected explicit memory to be stored"
        assert recent[0].content == "The assistant's name is Alice.", recent[0]
        assert recent[0].metadata.get("subject") == "assistant_identity", recent[0]
        assert recent[0].metadata.get("slot") == "assistant_name", recent[0]
        assert recent[0].metadata.get("assistant_name") == "Alice", recent[0]
        print("✓ remember corrects assistant-name identity memories\n")
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)


# ─── Live-API integration tests ───────────────────────────────────────────────


async def test_live_agent_uses_read_tool() -> None:
    """Agent should use the read tool and return file contents."""
    from tests._live_harness import LIVE, live_run_agentic_loop, parse_loop_result, skip_if_not_live
    if not LIVE:
        print("[SKIP] Live read tool test (AGENTZERO_LIVE_TESTS=1 required)")
        return
    skip_if_not_live()
    print("Testing live agent read tool...")
    result = await live_run_agentic_loop(
        messages=[{
            "role": "user",
            "content": "Read the first line of AGENTS.md using the read tool. Reply with exactly that line. Stop after.",
        }],
        max_iterations=5,
    )
    parsed = parse_loop_result(result)
    text = parsed.get("text", result)
    assert len(text) > 5, f"Reply too short: {text[:200]}"
    print(f"  PASS — reply: {text[:100]}")


async def test_live_agent_uses_bash_tool() -> None:
    """Agent should use bash and return the output."""
    from tests._live_harness import LIVE, live_run_agentic_loop, parse_loop_result, skip_if_not_live
    if not LIVE:
        print("[SKIP] Live bash tool test")
        return
    skip_if_not_live()
    print("Testing live agent bash tool...")
    result = await live_run_agentic_loop(
        messages=[{
            "role": "user",
            "content": "Run `printf simple-bash-ok` using bash. Reply with exactly the output. Stop after.",
        }],
        max_iterations=5,
    )
    parsed = parse_loop_result(result)
    text = parsed.get("text", result)
    assert "simple-bash-ok" in text, f"Missing bash output: {text[:200]}"
    print(f"  PASS — reply: {text[:100]}")


async def test_live_agent_uses_write_tool() -> None:
    """Agent should write a file and confirm success."""
    from tests._live_harness import LIVE, live_run_agentic_loop, parse_loop_result, skip_if_not_live
    if not LIVE:
        print("[SKIP] Live write tool test")
        return
    skip_if_not_live()
    print("Testing live agent write tool...")
    result = await live_run_agentic_loop(
        messages=[{
            "role": "user",
            "content": (
                "Write a file at workspace/simple_test.txt containing exactly 'simple-write-ok'. "
                "Then reply with just 'written' and the file contents. Stop after."
            ),
        }],
        max_iterations=5,
    )
    parsed = parse_loop_result(result)
    text = parsed.get("text", result)
    assert "simple-write-ok" in text or "written" in text.lower(), f"Unexpected: {text[:200]}"
    print(f"  PASS — reply: {text[:100]}")


async def test_live_agent_uses_glob_tool() -> None:
    """Agent should use glob to find files."""
    from tests._live_harness import LIVE, live_run_agentic_loop, parse_loop_result, skip_if_not_live
    if not LIVE:
        print("[SKIP] Live glob tool test")
        return
    skip_if_not_live()
    print("Testing live agent glob tool...")
    result = await live_run_agentic_loop(
        messages=[{
            "role": "user",
            "content": "Use glob to find all *.py files in the current directory. Reply with the count. Stop after.",
        }],
        max_iterations=5,
    )
    parsed = parse_loop_result(result)
    text = parsed.get("text", result)
    assert len(text) > 3, f"Reply too short: {text[:200]}"
    print(f"  PASS — reply: {text[:100]}")


if __name__ == "__main__":
    asyncio.run(test_tools_work())
    asyncio.run(test_payload_isolation())
    asyncio.run(test_tool_call_parsing())
    asyncio.run(test_tool_result_format())
    asyncio.run(test_remember_tool_persists_runtime_session_id())
    asyncio.run(test_remember_tool_rewrites_assistant_name_from_user_turn())
    print("Deterministic tests passed!")

    print("\n--- Live-API integration tests ---")
    asyncio.run(test_live_agent_uses_read_tool())
    asyncio.run(test_live_agent_uses_bash_tool())
    asyncio.run(test_live_agent_uses_write_tool())
    asyncio.run(test_live_agent_uses_glob_tool())
    print("\nAll tests passed!")
