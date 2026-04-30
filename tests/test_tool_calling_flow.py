#!/usr/bin/env python3
"""Live-API test for the full tool calling flow."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests._live_harness import (
    LIVE,
    live_run_agentic_loop,
    live_agent_handle,
    parse_loop_result,
    skip_if_not_live,
    _make_handler,
    _make_store,
)
from handler import BASE_PAYLOAD
from tools import set_memory_store


async def test_live_multi_tool_call_flow() -> None:
    """Real API: agent calls bash, grep, read in one turn."""
    if not LIVE:
        print("\nTest L1: Multi-tool flow [SKIP]")
        return
    print("\nTest L1: Multi-tool call flow via live API")
    skip_if_not_live()

    result = await live_run_agentic_loop(
        messages=[
            {
                "role": "user",
                "content": (
                    "Step 1: use grep to search for 'AgentHandler' in handler.py. "
                    "Step 2: use bash to run `printf flow-test-passed`. "
                    "Step 3: use read to read the first line of AGENTS.md. "
                    "Reply with all three results as:\n"
                    "grep: <match count>\n"
                    "bash: flow-test-passed\n"
                    "read: <first line>\n"
                    "Stop after step 3."
                ),
            }
        ],
        max_iterations=5,
    )

    parsed = parse_loop_result(result)
    text = parsed.get("text", result)
    assert "flow-test-passed" in text, f"Missing bash output: {text[:300]}"
    assert len(text) > 30, f"Reply too short: {text[:300]}"
    print(f"  PASS — reply length={len(text)}")


async def test_live_handler_handle_tool_flow() -> None:
    """Real handler.handle() should call tools and produce a reply."""
    if not LIVE:
        print("\nTest L2: Handler tool flow [SKIP]")
        return
    print("\nTest L2: Handler.handle() tool flow via live API")
    skip_if_not_live()

    store = _make_store()
    handler = _make_handler(store)

    response = await live_agent_handle(
        handler,
        user_text=(
            "Use the bash tool to run `printf handler-tool-ok`. "
            "Reply with only the bash output. Stop after that."
        ),
        session_id="test_tool_flow",
    )

    assert "handler-tool-ok" in response, f"Missing expected output: {response[:200]}"
    print(f"  PASS — response: {response[:100]}")


async def test_live_payload_tools_config() -> None:
    """Verify all required tools are in BASE_PAYLOAD."""
    print("\nTest L3: BASE_PAYLOAD tool configuration")
    tool_names = [
        t.get("function", {}).get("name")
        for t in BASE_PAYLOAD.get("tools", [])
        if isinstance(t, dict)
    ]
    required = ["read", "write", "bash", "grep", "glob", "send_message", "declare_message_count"]
    for name in required:
        assert name in tool_names, f"Missing tool: {name}"
    print("  PASS")


async def main() -> None:
    print("=" * 60)
    print("Tool calling flow tests")
    print("=" * 60)
    await test_live_multi_tool_call_flow()
    await test_live_handler_handle_tool_flow()
    test_live_payload_tools_config()
    print("\n" + "=" * 60)
    print("Tool calling flow tests complete")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
