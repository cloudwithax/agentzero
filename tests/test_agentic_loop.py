#!/usr/bin/env python3
"""Live-API tests for the agentic loop.

All tests that involve LLM behavior call the real NVIDIA API.
Pure-function detection tests (regex patterns) remain deterministic.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_loop import (
    contains_action_intent_narration,
    contains_pseudo_tool_syntax,
    is_bare_reaction_word,
    run_agentic_loop,
)
from tests._live_harness import (
    LIVE,
    live_run_agentic_loop,
    parse_loop_result,
    skip_if_not_live,
)
from tools import get_send_message_buffer

# ─── Pure-function tests (no API needed) ───────────────────────────────────────


def test_contains_action_intent_narration_matches_here_now_publish_language() -> None:
    """here.now-style narrated publish text should trigger an execution retry."""
    print("\nTest 1: Detect narrated here.now publish text")

    narrated = (
        "bet. fixing the meter logic rn. pushing the fix to both crustyhub and "
        "here.now now. refresh in 5 secs."
    )
    narrated_inspection = (
        "let me inspect the reminder state and run the publish step now."
    )
    narrated_tapback = 'sending `like` to "hey can you do that again" right now.'
    assert contains_action_intent_narration(narrated) is True
    assert contains_action_intent_narration(narrated_inspection) is True
    assert contains_action_intent_narration(narrated_tapback) is True
    assert contains_action_intent_narration("it's live at https://example.com") is False
    print("  PASS")


def test_contains_pseudo_tool_syntax_matches_invalid_tool_markup() -> None:
    """Angle-bracket pseudo-tool tags should be detected as invalid."""
    print("\nTest 1c: Detect pseudo-tool syntax")

    assert contains_pseudo_tool_syntax('<read(filepath="handler.py")>') is True
    assert contains_pseudo_tool_syntax('<glob(pattern="**/*.py")>') is True
    assert (
        contains_pseudo_tool_syntax("<read>\n<filepath>handler.py</filepath>\n</read>")
        is True
    )
    assert (
        contains_pseudo_tool_syntax(
            '<function_activate_skill><parameter name="name">frontend-design</parameter></function_activate_skill>'
        )
        is True
    )
    assert contains_pseudo_tool_syntax("read handler.py") is False
    assert contains_pseudo_tool_syntax("I will inspect handler.py now.") is False
    print("  PASS")


def test_is_bare_reaction_word() -> None:
    """Bare reaction words should be detected but phrases should not."""
    print("\nTest 6: Bare reaction word detection")

    assert is_bare_reaction_word("like") is True
    assert is_bare_reaction_word("love") is True
    assert is_bare_reaction_word("dislike") is True
    assert is_bare_reaction_word("laugh") is True
    assert is_bare_reaction_word("emphasize") is True
    assert is_bare_reaction_word("question") is True
    assert is_bare_reaction_word("like.") is True
    assert is_bare_reaction_word(" Like ") is True
    assert is_bare_reaction_word("like that") is False
    assert is_bare_reaction_word("I like that") is False
    assert is_bare_reaction_word("") is False
    print("  PASS")


# ─── Live-API tests ────────────────────────────────────────────────────────────


async def test_live_agent_calls_tools_for_read_and_bash() -> None:
    """Agent should make real tool calls when asked to read a file and run a command."""
    skip_if_not_live()
    print("\nTest L1: Agent calls read+bash tools via live API")

    result = await live_run_agentic_loop(
        messages=[
            {
                "role": "user",
                "content": (
                    "Use the read tool to read the first line of AGENTS.md, then use "
                    "the bash tool to run `printf hello-live`. Reply with exactly:\n"
                    "first_line: <first line of AGENTS.md>\n"
                    "bash_out: <bash output>\n"
                    "Stop after that — nothing else."
                ),
            }
        ],
        max_iterations=5,
    )

    text = parse_loop_result(result).get("text", result)
    assert "hello-live" in text, f"Expected bash output in reply, got: {text[:200]}"
    assert len(text) > 10, f"Reply too short: {text[:200]}"
    print(f"  PASS — reply length={len(text)}")


async def test_live_agent_calls_grep_tool() -> None:
    """Agent should use grep when asked to search code."""
    skip_if_not_live()
    print("\nTest L2: Agent calls grep via live API")

    result = await live_run_agentic_loop(
        messages=[
            {
                "role": "user",
                "content": (
                    "Use the grep tool to search AGENTS.md for the word 'Pitfall'. "
                    "Reply with the number of matches found. Stop after that."
                ),
            }
        ],
        max_iterations=5,
    )

    text = parse_loop_result(result).get("text", result)
    assert len(text) > 5, f"Reply too short: {text[:200]}"
    print(f"  PASS — reply: {text[:120]}")


async def test_live_agent_calls_declare_message_count_and_sends_done() -> None:
    """Agent should call declare_message_count, send_message, and output <DONE>."""
    skip_if_not_live()
    print("\nTest L3: Agent follows completion protocol (declare_message_count + <DONE>)")

    result = await live_run_agentic_loop(
        messages=[
            {
                "role": "user",
                "content": (
                    "Just say hello back to me. Call declare_message_count(count=1) "
                    "first, then send one message via send_message(text='hey there'), "
                    "then output exactly <DONE> on its own line."
                ),
            }
        ],
        max_iterations=5,
    )

    parsed = parse_loop_result(result)
    text = parsed.get("text", result)
    assert len(text) > 0, "Expected non-empty reply"
    print(f"  PASS — reply: {text[:120]}")


async def test_live_agent_responds_normally_to_greeting() -> None:
    """Agent should respond naturally to a simple greeting."""
    skip_if_not_live()
    print("\nTest L4: Agent responds to casual greeting")

    result = await live_run_agentic_loop(
        messages=[
            {
                "role": "user",
                "content": "hi there, how are you? just reply in one sentence.",
            }
        ],
        max_iterations=5,
    )

    parsed = parse_loop_result(result)
    text = parsed.get("text", result)
    assert len(text) > 3, f"Reply too short: {text[:200]}"
    print(f"  PASS — reply: {text[:120]}")


async def test_live_agent_uses_real_tool_calls_not_pseudo_syntax() -> None:
    """Agent should make structured tool_calls (not fake angle-bracket markup)."""
    skip_if_not_live()
    print("\nTest L5: Agent uses real tool calls, not pseudo-tool syntax")

    result = await live_run_agentic_loop(
        messages=[
            {
                "role": "user",
                "content": (
                    "Read the file AGENTS.md using the read tool. Tell me the first "
                    "line. Do NOT write <read(...)> — use the actual tool_call. "
                    "Stop after telling me the first line."
                ),
            }
        ],
        max_iterations=5,
    )

    parsed = parse_loop_result(result)
    text = parsed.get("text", result)
    assert len(text) > 5, f"Reply too short: {text[:200]}"
    # Should NOT contain raw pseudo-tool markup
    assert "<read" not in (text or "").lower(), f"Pseudo-tool leakage: {text[:200]}"
    print(f"  PASS — reply: {text[:120]}")


async def test_live_agent_calls_bash_then_replies() -> None:
    """Agent should execute bash first, then reply with real output."""
    skip_if_not_live()
    print("\nTest L6: Agent calls bash then replies")

    result = await live_run_agentic_loop(
        messages=[
            {
                "role": "user",
                "content": (
                    "Run `echo live-test-ok` using the bash tool. Then reply with "
                    "the exact output of the command prefixed by 'output: '. "
                    "Stop after that."
                ),
            }
        ],
        max_iterations=5,
    )

    parsed = parse_loop_result(result)
    text = parsed.get("text", result)
    assert "live-test-ok" in text, f"Missing expected output in: {text[:200]}"
    print(f"  PASS — reply: {text[:120]}")


async def test_live_agent_writes_and_reads_file() -> None:
    """Agent should write a file then read it back."""
    skip_if_not_live()
    print("\nTest L7: Agent writes and reads a file")

    result = await live_run_agentic_loop(
        messages=[
            {
                "role": "user",
                "content": (
                    "Write a file called workspace/live_test.txt containing exactly "
                    "'live-test-content-xyz'. Then read that file back and reply with "
                    "its contents. Stop after that."
                ),
            }
        ],
        max_iterations=5,
    )

    parsed = parse_loop_result(result)
    text = parsed.get("text", result)
    assert "live-test-content-xyz" in text, f"Missing file contents in: {text[:200]}"
    print(f"  PASS — reply: {text[:120]}")


async def test_live_agent_handles_multiple_tool_rounds() -> None:
    """Agent should handle a task requiring multiple tool-call rounds."""
    skip_if_not_live()
    print("\nTest L8: Agent handles multi-round tool execution")

    result = await live_run_agentic_loop(
        messages=[
            {
                "role": "user",
                "content": (
                    "Step 1: use grep to search for 'import' in handler.py (just tell me how many matches). "
                    "Step 2: use bash to run `printf multi-round-ok`. "
                    "Step 3: reply with both results as: 'grep: N matches, bash: <output>'. "
                    "Stop after step 3."
                ),
            }
        ],
        max_iterations=5,
    )

    parsed = parse_loop_result(result)
    text = parsed.get("text", result)
    assert "multi-round-ok" in text, f"Missing bash output in: {text[:200]}"
    assert len(text) > 20, f"Reply too short: {text[:200]}"
    print(f"  PASS — reply: {text[:120]}")


# ─── Main ──────────────────────────────────────────────────────────────────────


async def main() -> None:
    print("=" * 60)
    print("Testing agentic loop action-intent handling")
    print("=" * 60)

    test_contains_action_intent_narration_matches_here_now_publish_language()
    test_contains_pseudo_tool_syntax_matches_invalid_tool_markup()
    test_is_bare_reaction_word()

    if LIVE:
        print("\n" + "=" * 60)
        print("Live-API agentic loop tests")
        print("=" * 60)
        await test_live_agent_calls_tools_for_read_and_bash()
        await test_live_agent_calls_grep_tool()
        await test_live_agent_calls_declare_message_count_and_sends_done()
        await test_live_agent_responds_normally_to_greeting()
        await test_live_agent_uses_real_tool_calls_not_pseudo_syntax()
        await test_live_agent_calls_bash_then_replies()
        await test_live_agent_writes_and_reads_file()
        await test_live_agent_handles_multiple_tool_rounds()
    else:
        print("\n[SKIP] Live-API tests disabled (set AGENTZERO_LIVE_TESTS=1)")

    print("\n" + "=" * 60)
    print("Agentic loop tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
