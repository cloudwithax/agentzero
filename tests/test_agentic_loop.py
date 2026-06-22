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
    _assemble_send_message_buffer_response,
    _compact_executed_tool_call_args,
    _inject_browser_screenshots,
    _strip_done_marker,
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


def _write_call_message(content: str, *, filepath: str = "index.html") -> dict:
    """Build an assistant message carrying a single write() tool call."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_write_1",
                "type": "function",
                "function": {
                    "name": "write",
                    "arguments": json.dumps({"filepath": filepath, "content": content}),
                },
            }
        ],
    }


def test_compact_tool_args_elides_large_write_content() -> None:
    """A large write() content field is replaced with a placeholder in history."""
    print("\nTest C1: Compact large tool-call arguments")

    big = "<!doctype html>" + ("<p>hello world</p>" * 500)
    message = _write_call_message(big)
    compacted = _compact_executed_tool_call_args(message, round_number=3)

    args = json.loads(compacted["tool_calls"][0]["function"]["arguments"])
    assert args["filepath"] == "index.html", "filepath must be preserved"
    assert "elided" in args["content"], f"content not elided: {args['content'][:80]}"
    assert len(args["content"]) < 200, "placeholder should be small"
    assert "round 3" in args["content"]
    print("  PASS")


def test_compact_tool_args_preserves_small_calls() -> None:
    """Small tool calls (e.g. bash one-liners) must be byte-for-byte unchanged."""
    print("\nTest C2: Compaction leaves small calls untouched")

    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_bash_1",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": json.dumps({"command": "echo hi"}),
                },
            }
        ],
    }
    result = _compact_executed_tool_call_args(message, round_number=1)
    assert result is message, "small calls should pass through unchanged (same object)"
    print("  PASS")


def test_compact_tool_args_does_not_mutate_input() -> None:
    """Compaction must never mutate the original message (it already executed)."""
    print("\nTest C3: Compaction does not mutate input")

    big = "x" * 5000
    message = _write_call_message(big)
    original_args = message["tool_calls"][0]["function"]["arguments"]
    _compact_executed_tool_call_args(message, round_number=2)
    assert (
        message["tool_calls"][0]["function"]["arguments"] == original_args
    ), "input message was mutated"
    assert big in original_args, "original full content lost"
    print("  PASS")


def test_compact_tool_args_idempotent() -> None:
    """Compacting an already-compacted message is a no-op."""
    print("\nTest C4: Compaction is idempotent")

    big = "y" * 5000
    once = _compact_executed_tool_call_args(_write_call_message(big), round_number=1)
    twice = _compact_executed_tool_call_args(once, round_number=1)
    assert (
        twice["tool_calls"][0]["function"]["arguments"]
        == once["tool_calls"][0]["function"]["arguments"]
    )
    print("  PASS")


def test_compact_tool_args_invalid_json_fallback() -> None:
    """Non-JSON arguments are truncated with a marker, never raising."""
    print("\nTest C5: Invalid-JSON arguments fall back to truncation")

    raw = "this is not json " + ("z" * 5000)
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_x",
                "type": "function",
                "function": {"name": "write", "arguments": raw},
            }
        ],
    }
    result = _compact_executed_tool_call_args(message, round_number=4)
    new_args = result["tool_calls"][0]["function"]["arguments"]
    assert "elided" in new_args and len(new_args) < len(raw)
    print("  PASS")


def test_compact_tool_args_multi_call_round() -> None:
    """Each tool call in a multi-call round is handled independently."""
    print("\nTest C6: Multi-call round compaction")

    big = "h" * 6000
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {
                    "name": "write",
                    "arguments": json.dumps({"filepath": "a.html", "content": big}),
                },
            },
            {
                "id": "c2",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": json.dumps({"command": "ls"}),
                },
            },
        ],
    }
    result = _compact_executed_tool_call_args(message, round_number=1)
    a1 = json.loads(result["tool_calls"][0]["function"]["arguments"])
    a2 = json.loads(result["tool_calls"][1]["function"]["arguments"])
    assert "elided" in a1["content"], "large write not compacted"
    assert a2["command"] == "ls", "small bash call should be untouched"
    print("  PASS")


def test_strip_done_marker_removes_completion_signal() -> None:
    """The <DONE> completion-protocol marker should be stripped from visible text."""
    print("\nTest 7: <DONE> marker stripping")

    assert _strip_done_marker("<DONE>") == ""
    assert _strip_done_marker("hello\n<DONE>\n") == "hello"
    assert _strip_done_marker("reply\n\n<DONE>") == "reply"
    assert _strip_done_marker("") == ""
    assert _strip_done_marker("no marker here") == "no marker here"
    print("  PASS")


def test_assemble_send_message_buffer_response_for_buffered_channel() -> None:
    """Non-channel sessions should see send_message bubbles assembled into text."""
    print("\nTest 8: send_message buffer assembly for CLI/test sessions")

    buffer = [{"text": "hey there", "attachments": [], "channel": "buffered"}]
    result = _assemble_send_message_buffer_response(buffer, "", [])
    assert result == "hey there", f"got: {result!r}"

    buffer = [
        {"text": "first", "attachments": [], "channel": "buffered"},
        {"text": "second", "attachments": [], "channel": "buffered"},
    ]
    result = _assemble_send_message_buffer_response(buffer, "", [])
    assert result == "first\n\nsecond", f"got: {result!r}"

    buffer = [
        {"text": "see", "attachments": ["http://img/a.jpg"], "channel": "buffered"}
    ]
    parsed = json.loads(_assemble_send_message_buffer_response(buffer, "", []))
    assert parsed["text"] == "see"
    assert parsed["attachments"] == ["http://img/a.jpg"]
    print("  PASS")


def test_assemble_send_message_buffer_response_for_channel_bound_delivery() -> None:
    """Channel-bound sessions should return a delivered_via_tool envelope."""
    print("\nTest 9: send_message channel-bound envelope")

    for channel in ("imessage", "telegram"):
        buffer = [{"text": "hi", "attachments": [], "channel": channel}]
        parsed = json.loads(_assemble_send_message_buffer_response(buffer, "", []))
        assert parsed.get("delivered_via_tool") is True, f"failed for {channel}"
        assert parsed.get("text") == "", f"text leaked for {channel}: {parsed!r}"

    # Empty buffer falls through (returns None) so the existing return paths win.
    assert _assemble_send_message_buffer_response([], "fallback", []) is None
    print("  PASS")


def test_inject_browser_screenshots() -> None:
    """Screenshots are injected as image blocks only for multimodal models."""
    print("\nTest 10: browser screenshot injection")
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
        fh.write(b"\x89PNG\r\n\x1a\nfake-screenshot-bytes")
        path = fh.name
    try:
        message = {
            "tool_calls": [{"id": "c1", "function": {"name": "browser_screenshot"}}]
        }
        tool_results = [
            {
                "tool_call_id": "c1",
                "role": "tool",
                "content": json.dumps({"success": True, "path": path}),
            }
        ]

        # Multimodal model → inject one image as a user message.
        messages: list = []
        injected = _inject_browser_screenshots(
            messages, message, tool_results, "stepfun-ai/step-3.7-flash"
        )
        assert injected == 1, injected
        assert messages[-1]["role"] == "user"
        blocks = messages[-1]["content"]
        assert any(b.get("type") == "image_url" for b in blocks), blocks

        # Non-multimodal model → no injection.
        messages2: list = []
        assert (
            _inject_browser_screenshots(messages2, message, tool_results, "some/text-model")
            == 0
        )
        assert messages2 == []
    finally:
        os.unlink(path)
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
    # The model was asked for a count, so the reply may be terse (e.g. "12" or
    # "0 matches"). Assert it produced a non-empty answer containing a number
    # rather than a fixed minimum length.
    assert text.strip(), f"Empty reply: {text[:200]}"
    assert any(ch.isdigit() for ch in text), f"Reply has no match count: {text[:200]}"
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
    # The completion protocol routes the visible reply through send_message
    # and signals end-of-turn with <DONE>.  The loop must assemble the
    # buffered bubble text into the visible reply and strip the marker —
    # so the test sees the actual message, not an empty string or "<DONE>".
    assert "hey there" in text.lower(), (
        f"Expected send_message text in reply, got: {text[:200]!r}"
    )
    assert "<done>" not in text.lower(), (
        f"<DONE> marker leaked into visible reply: {text[:200]!r}"
    )
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


async def test_graceful_finish_on_transient_error_after_tool_round() -> None:
    """After a tool round did real work, a transient parse error on the NEXT
    round must trigger a tools-stripped graceful finish instead of dumping the
    raw provider parser error at the user."""
    print("\nTest G1: Graceful finish on transient error after tool round")

    import agentic_loop as al

    calls = {"n": 0}

    async def fake_api_call(session, base_url, payload, headers, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            # Round 1: model issues a write tool call.
            return {
                "choices": [
                    {
                        "message": _write_call_message("<html>big page</html>"),
                    }
                ]
            }
        if calls["n"] == 2:
            # Round 2: provider chokes serializing the next tool call.
            return {
                "error": {
                    "message": "Unterminated string starting at: line 1 column 69 (char 67)"
                }
            }
        # Tools-stripped graceful-finish call: must have NO tools in payload.
        assert "tools" not in payload, "graceful finish must strip tools"
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Done — I created your page index.html.",
                    }
                }
            ]
        }

    async def fake_execute(message, allowed_tool_names=None):
        return [
            {
                "role": "tool",
                "tool_call_id": "call_write_1",
                "content": json.dumps({"success": True, "path": "index.html"}),
            }
        ]

    orig_api = al.api_call_with_retry
    orig_exec = al.execute_tool_calls
    al.api_call_with_retry = fake_api_call
    al.execute_tool_calls = fake_execute
    try:
        result = await run_agentic_loop(
            messages=[{"role": "user", "content": "make and publish a site"}],
            session=None,
            base_url="http://test",
            api_key="test",
            base_payload={"model": "test", "tools": [{"x": 1}], "temperature": 0.6},
        )
    finally:
        al.api_call_with_retry = orig_api
        al.execute_tool_calls = orig_exec

    parsed = parse_loop_result(result)
    text = parsed if isinstance(parsed, str) else parsed.get("text", "")
    assert not text.startswith("Error:"), f"Raw error leaked to user: {text!r}"
    assert "unterminated" not in text.lower(), f"Parser guts leaked: {text!r}"
    assert "index.html" in text, f"Expected summary of completed work, got: {text!r}"
    print("  PASS")


async def test_graceful_finish_on_rate_limit_after_tool_round() -> None:
    """After a tool round did real work, an exhausted rate limit on the NEXT
    round must trigger a tools-stripped graceful finish instead of dumping
    'Rate limit exceeded after retries' at the user."""
    print("\nTest G2: Graceful finish on rate-limit after tool round")

    import agentic_loop as al

    calls = {"n": 0}

    async def fake_api_call(session, base_url, payload, headers, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            # Round 1: model issues a write tool call (real work).
            return {
                "choices": [
                    {
                        "message": _write_call_message("<html>big page</html>"),
                    }
                ]
            }
        if calls["n"] == 2:
            # Round 2: the next round's retries exhaust on a 429.
            return {"error": {"message": "Rate limit exceeded after retries"}}
        # Tools-stripped graceful-finish call: must have NO tools in payload.
        assert "tools" not in payload, "graceful finish must strip tools"
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Done — I created your page index.html.",
                    }
                }
            ]
        }

    async def fake_execute(message, allowed_tool_names=None):
        return [
            {
                "role": "tool",
                "tool_call_id": "call_write_1",
                "content": json.dumps({"success": True, "path": "index.html"}),
            }
        ]

    orig_api = al.api_call_with_retry
    orig_exec = al.execute_tool_calls
    al.api_call_with_retry = fake_api_call
    al.execute_tool_calls = fake_execute
    try:
        result = await run_agentic_loop(
            messages=[{"role": "user", "content": "make and publish a site"}],
            session=None,
            base_url="http://test",
            api_key="test",
            base_payload={"model": "test", "tools": [{"x": 1}], "temperature": 0.6},
        )
    finally:
        al.api_call_with_retry = orig_api
        al.execute_tool_calls = orig_exec

    parsed = parse_loop_result(result)
    text = parsed if isinstance(parsed, str) else parsed.get("text", "")
    assert not text.startswith("Error:"), f"Raw error leaked to user: {text!r}"
    assert "rate limit" not in text.lower(), f"Rate-limit guts leaked: {text!r}"
    assert "index.html" in text, f"Expected summary of completed work, got: {text!r}"
    print("  PASS")


async def main() -> None:
    print("=" * 60)
    print("Testing agentic loop action-intent handling")
    print("=" * 60)

    test_compact_tool_args_elides_large_write_content()
    test_compact_tool_args_preserves_small_calls()
    test_compact_tool_args_does_not_mutate_input()
    test_compact_tool_args_idempotent()
    test_compact_tool_args_invalid_json_fallback()
    test_compact_tool_args_multi_call_round()
    test_strip_done_marker_removes_completion_signal()
    test_assemble_send_message_buffer_response_for_buffered_channel()
    test_assemble_send_message_buffer_response_for_channel_bound_delivery()
    test_inject_browser_screenshots()
    await test_graceful_finish_on_transient_error_after_tool_round()
    await test_graceful_finish_on_rate_limit_after_tool_round()

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
