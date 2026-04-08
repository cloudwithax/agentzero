#!/usr/bin/env python3
"""Regression tests for the agentic loop's action-intent handling."""

import asyncio
import json
from unittest.mock import AsyncMock, patch

from agentic_loop import (
    contains_action_intent_narration,
    is_bare_reaction_word,
    run_agentic_loop,
)
from handler import API_KEY, BASE_PAYLOAD, BASE_URL


def test_contains_action_intent_narration_matches_here_now_publish_language() -> None:
    """here.now-style narrated publish text should trigger an execution retry."""
    print("Test 1: Detect narrated here.now publish text")

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
    print("  ✓ Passed")


async def test_run_agentic_loop_retries_multiple_narrated_publish_attempts() -> None:
    """Action-intent retries should survive repeated narration before a tool call."""
    print("Test 2: Multiple narrated publish retries")

    initial_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "activating `here-now` skill now. stand by while i publish it.",
                }
            }
        ]
    }

    narrated_retry_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": (
                        "writing the corrected index.html now. then i'll push to both "
                        "crustyhub and here.now."
                    ),
                }
            }
        ]
    }

    tool_call_response = {
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
    }

    final_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "https://cedar-canopy-ytnp.here.now/",
                }
            }
        ]
    }

    followup_responses = [
        narrated_retry_response,
        tool_call_response,
        final_response,
    ]
    tool_result = {
        "tool_call_id": "call_publish",
        "role": "tool",
        "content": json.dumps(
            {"success": True, "stdout": "https://cedar-canopy-ytnp.here.now/\n"}
        ),
    }

    async def fake_api_call_with_retry(*args, **kwargs):
        assert followup_responses, "Unexpected extra model call"
        return followup_responses.pop(0)

    execute_tool_calls = AsyncMock(return_value=[tool_result])

    messages = [{"role": "user", "content": "put it on here.now"}]

    with (
        patch("agentic_loop.api_call_with_retry", side_effect=fake_api_call_with_retry),
        patch("agentic_loop.execute_tool_calls", execute_tool_calls),
    ):
        content = await run_agentic_loop(
            messages=messages,
            session=AsyncMock(),
            base_url=BASE_URL,
            api_key=API_KEY,
            base_payload=BASE_PAYLOAD.copy(),
            initial_response_data=initial_response,
            max_tool_leak_retries=1,
            max_action_intent_retries=3,
        )

    assert content == "https://cedar-canopy-ytnp.here.now/"
    assert execute_tool_calls.await_count == 1
    assert not followup_responses, f"Unused follow-up responses: {followup_responses}"

    user_nudges = [
        m["content"]
        for m in messages
        if m.get("role") == "user"
        and "did not make any tool calls" in m.get("content", "")
    ]
    assert len(user_nudges) == 2, (
        f"Expected two narration nudges, got {len(user_nudges)}"
    )
    print("  ✓ Passed")


async def test_run_agentic_loop_retries_when_user_explicitly_requires_tools() -> None:
    """If user says tool execution is mandatory, text-only replies should be retried."""
    print("Test 3: Explicit tool requirement retry")

    initial_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "grep_seen: yes\nbash: done-live3",
                }
            }
        ]
    }

    tool_call_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_grep",
                            "type": "function",
                            "function": {
                                "name": "grep",
                                "arguments": json.dumps(
                                    {
                                        "pattern": "Pitfall:",
                                        "path": ".",
                                        "include": "AGENTS.md",
                                        "max_matches": 1,
                                    }
                                ),
                            },
                        },
                        {
                            "id": "call_bash",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": json.dumps(
                                    {"command": "printf done-live3"}
                                ),
                            },
                        },
                    ],
                }
            }
        ]
    }

    final_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "grep_seen: yes\nbash: done-live3",
                }
            }
        ]
    }

    followup_responses = [tool_call_response, final_response]

    tool_results = [
        {
            "tool_call_id": "call_grep",
            "role": "tool",
            "content": json.dumps({"success": True, "matches": [{"line": 1}]}),
        },
        {
            "tool_call_id": "call_bash",
            "role": "tool",
            "content": json.dumps({"success": True, "stdout": "done-live3"}),
        },
    ]

    async def fake_api_call_with_retry(*args, **kwargs):
        assert followup_responses, "Unexpected extra model call"
        return followup_responses.pop(0)

    execute_tool_calls = AsyncMock(return_value=tool_results)
    messages = [
        {
            "role": "user",
            "content": (
                "Tool execution is mandatory. You must call grep and bash before "
                "answering this."
            ),
        }
    ]

    with (
        patch("agentic_loop.api_call_with_retry", side_effect=fake_api_call_with_retry),
        patch("agentic_loop.execute_tool_calls", execute_tool_calls),
    ):
        content = await run_agentic_loop(
            messages=messages,
            session=AsyncMock(),
            base_url=BASE_URL,
            api_key=API_KEY,
            base_payload=BASE_PAYLOAD.copy(),
            initial_response_data=initial_response,
            max_tool_leak_retries=1,
            max_action_intent_retries=3,
        )

    assert content == "grep_seen: yes\nbash: done-live3"
    assert execute_tool_calls.await_count == 1

    user_nudges = [
        m["content"]
        for m in messages
        if m.get("role") == "user"
        and "did not make any tool calls" in m.get("content", "")
    ]
    assert len(user_nudges) == 1, (
        f"Expected one tool-exec nudge, got {len(user_nudges)}"
    )
    print("  ✓ Passed")


async def test_run_agentic_loop_retries_for_normal_reply_after_tapback_ack() -> None:
    """Tapback-only acknowledgement text should trigger a normal-reply retry."""
    print("Test 4: Retry for normal reply after tapback acknowledgement")

    initial_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_tapback",
                            "type": "function",
                            "function": {
                                "name": "send_tapback",
                                "arguments": json.dumps(
                                    {
                                        "message_handle": "tapback-handle-1",
                                        "reaction": "like",
                                    }
                                ),
                            },
                        }
                    ],
                }
            }
        ]
    }

    tapback_ack_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "done",
                }
            }
        ]
    }

    followup_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "haha fair enough",
                }
            }
        ]
    }

    execute_tool_calls = AsyncMock(
        return_value=[
            {
                "tool_call_id": "call_tapback",
                "role": "tool",
                "content": json.dumps({"success": True, "status": 200}),
            }
        ]
    )

    api_call_with_retry = AsyncMock(
        side_effect=[tapback_ack_response, followup_response]
    )

    with (
        patch("agentic_loop.api_call_with_retry", new=api_call_with_retry),
        patch("agentic_loop.execute_tool_calls", execute_tool_calls),
    ):
        content = await run_agentic_loop(
            messages=[{"role": "user", "content": "just do the tapback"}],
            session=AsyncMock(),
            base_url=BASE_URL,
            api_key=API_KEY,
            base_payload=BASE_PAYLOAD.copy(),
            initial_response_data=initial_response,
        )

    assert content == "haha fair enough"
    assert execute_tool_calls.await_count == 1
    assert api_call_with_retry.await_count == 2
    print("  ✓ Passed")


def test_is_bare_reaction_word() -> None:
    """Bare reaction words should be detected but phrases should not."""
    print("Test 5: Bare reaction word detection")

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
    assert is_bare_reaction_word("that's a good question") is False
    assert is_bare_reaction_word("laugh out loud") is False
    assert is_bare_reaction_word("") is False
    assert is_bare_reaction_word("yes, I like it") is False

    print("  ✓ Passed")


async def test_run_agentic_loop_retries_bare_reaction_word() -> None:
    """Bare reaction word as the entire response should trigger a retry."""
    print("Test 6: Retry when model outputs bare reaction word")

    initial_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "like",
                }
            }
        ]
    }

    proper_reply_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "hey! yeah I'm here, what's up?",
                }
            }
        ]
    }

    api_call_with_retry = AsyncMock(side_effect=[proper_reply_response])

    messages = [{"role": "user", "content": "hey you there?"}]

    with (
        patch("agentic_loop.api_call_with_retry", new=api_call_with_retry),
        patch("agentic_loop.execute_tool_calls", AsyncMock(return_value=[])),
    ):
        content = await run_agentic_loop(
            messages=messages,
            session=AsyncMock(),
            base_url=BASE_URL,
            api_key=API_KEY,
            base_payload=BASE_PAYLOAD.copy(),
            initial_response_data=initial_response,
        )

    assert content == "hey! yeah I'm here, what's up?"
    assert api_call_with_retry.await_count == 1

    bare_reaction_nudges = [
        m["content"]
        for m in messages
        if m.get("role") == "user" and "bare reaction word" in m.get("content", "")
    ]
    assert len(bare_reaction_nudges) == 1, (
        f"Expected one bare-reaction nudge, got {len(bare_reaction_nudges)}"
    )
    print("  ✓ Passed")


async def main() -> None:
    print("=" * 60)
    print("Testing agentic loop action-intent handling")
    print("=" * 60)
    print()

    test_contains_action_intent_narration_matches_here_now_publish_language()
    await test_run_agentic_loop_retries_multiple_narrated_publish_attempts()
    await test_run_agentic_loop_retries_when_user_explicitly_requires_tools()
    await test_run_agentic_loop_retries_for_normal_reply_after_tapback_ack()
    test_is_bare_reaction_word()
    await test_run_agentic_loop_retries_bare_reaction_word()

    print()
    print("=" * 60)
    print("Agentic loop tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
