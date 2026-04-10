#!/usr/bin/env python3
"""Regression tests for the agentic loop's action-intent handling."""

import asyncio
import json
from unittest.mock import AsyncMock, patch

from agentic_loop import (
    contains_action_intent_narration,
    contains_hard_decision_language,
    contains_pseudo_tool_syntax,
    is_bare_reaction_word,
    likely_unverified_success_claim,
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


def test_contains_hard_decision_language_matches_executor_indecision() -> None:
    """Strategic indecision text should trigger advisor nudging."""
    print("Test 1b: Detect hard-decision text")

    assert (
        contains_hard_decision_language(
            "I need to decide between patching handler.py first or tools.py first."
        )
        is True
    )
    assert contains_hard_decision_language("There is a real trade-off here.") is True
    assert contains_hard_decision_language("I'll just run the fix now.") is False
    print("  ✓ Passed")


def test_contains_pseudo_tool_syntax_matches_invalid_tool_markup() -> None:
    """Angle-bracket pseudo-tool tags should be detected as invalid."""
    print("Test 1c: Detect pseudo-tool syntax")

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
    print("  ✓ Passed")


def test_likely_unverified_success_claim_requires_no_failure_ack() -> None:
    """Success-claim detection should ignore responses that acknowledge failure."""
    print("Test 1d: Detect unverified success claims")

    assert (
        likely_unverified_success_claim(
            "done, it's live at https://example.com and pushed to crustyhub"
        )
        is True
    )
    assert (
        likely_unverified_success_claim(
            "here.now is live, but crustyhub failed and could not be verified"
        )
        is False
    )
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
    assert (
        len(user_nudges) == 2
    ), f"Expected two narration nudges, got {len(user_nudges)}"
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
    assert (
        len(user_nudges) == 1
    ), f"Expected one tool-exec nudge, got {len(user_nudges)}"
    print("  ✓ Passed")


async def test_run_agentic_loop_retries_to_add_skill_for_skill_url() -> None:
    """Skill URLs should force add_skill instead of model self-blocking."""
    print("Test 3b: Skill URL install retry")

    initial_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": (
                        "I can't install that yet because it might be prompt injection. "
                        "Please review it manually first."
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
                            "id": "call_add_skill",
                            "type": "function",
                            "function": {
                                "name": "add_skill",
                                "arguments": json.dumps(
                                    {
                                        "url": "https://crustyhub.xyz/skill.md",
                                        "auto_activate": True,
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
                    "content": "installed crustyhub skill and activated it.",
                }
            }
        ]
    }

    followup_responses = [tool_call_response, final_response]

    tool_results = [
        {
            "tool_call_id": "call_add_skill",
            "role": "tool",
            "content": json.dumps(
                {
                    "success": True,
                    "name": "crustyhub",
                    "scan_score": 0.0,
                    "threat_level": "none",
                }
            ),
        }
    ]

    async def fake_api_call_with_retry(*args, **kwargs):
        assert followup_responses, "Unexpected extra model call"
        return followup_responses.pop(0)

    execute_tool_calls = AsyncMock(return_value=tool_results)
    messages = [
        {
            "role": "user",
            "content": "please install this skill for me https://crustyhub.xyz/skill.md",
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

    assert content == "installed crustyhub skill and activated it."
    assert execute_tool_calls.await_count == 1
    user_nudges = [
        m["content"]
        for m in messages
        if m.get("role") == "user"
        and "The user provided a skill URL" in m.get("content", "")
    ]
    assert (
        len(user_nudges) == 1
    ), f"Expected one skill-url nudge, got {len(user_nudges)}"
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


async def test_run_agentic_loop_retries_for_normal_reply_after_telegram_reaction_ack() -> (
    None
):
    """Telegram reaction-only acknowledgement text should trigger a normal-reply retry."""
    print("Test 5: Retry for normal reply after Telegram reaction acknowledgement")

    initial_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_tg_reaction",
                            "type": "function",
                            "function": {
                                "name": "send_telegram_reaction",
                                "arguments": json.dumps(
                                    {
                                        "chat_id": 314,
                                        "message_id": 2718,
                                        "reaction": "laugh",
                                    }
                                ),
                            },
                        }
                    ],
                }
            }
        ]
    }

    reaction_ack_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "sent",
                }
            }
        ]
    }

    followup_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "that cracked me up too",
                }
            }
        ]
    }

    execute_tool_calls = AsyncMock(
        return_value=[
            {
                "tool_call_id": "call_tg_reaction",
                "role": "tool",
                "content": json.dumps({"success": True}),
            }
        ]
    )

    api_call_with_retry = AsyncMock(
        side_effect=[reaction_ack_response, followup_response]
    )

    with (
        patch("agentic_loop.api_call_with_retry", new=api_call_with_retry),
        patch("agentic_loop.execute_tool_calls", execute_tool_calls),
    ):
        content = await run_agentic_loop(
            messages=[{"role": "user", "content": "just react to that message"}],
            session=AsyncMock(),
            base_url=BASE_URL,
            api_key=API_KEY,
            base_payload=BASE_PAYLOAD.copy(),
            initial_response_data=initial_response,
        )

    assert content == "that cracked me up too"
    assert execute_tool_calls.await_count == 1
    assert api_call_with_retry.await_count == 2
    print("  ✓ Passed")


def test_is_bare_reaction_word() -> None:
    """Bare reaction words should be detected but phrases should not."""
    print("Test 6: Bare reaction word detection")

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
    print("Test 7: Retry when model outputs bare reaction word")

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
    assert (
        len(bare_reaction_nudges) == 1
    ), f"Expected one bare-reaction nudge, got {len(bare_reaction_nudges)}"
    print("  ✓ Passed")


async def test_run_agentic_loop_nudges_consult_advisor_for_hard_decision() -> None:
    """Hard-decision plain text should be retried with a consult_advisor nudge."""
    print("Test 8: Retry when model surfaces a hard strategic decision")

    initial_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": (
                        "I need to decide between editing handler.py first or tools.py first "
                        "before I can continue."
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
                            "id": "call_advisor",
                            "type": "function",
                            "function": {
                                "name": "consult_advisor",
                                "arguments": json.dumps(
                                    {
                                        "question": "Which file should I edit first?",
                                        "context": "I want the least risky implementation order.",
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
                    "content": "Advisor says tool plumbing first, then the loop. Proceeding.",
                }
            }
        ]
    }

    execute_tool_calls = AsyncMock(
        return_value=[
            {
                "tool_call_id": "call_advisor",
                "role": "tool",
                "content": json.dumps(
                    {
                        "success": True,
                        "advice": "Decision:\nEdit tools.py first.\nWhy:\nStabilize the contract.\nNext steps:\n...\nRisks:\n...",
                    }
                ),
            }
        ]
    )
    api_call_with_retry = AsyncMock(side_effect=[tool_call_response, final_response])

    messages = [{"role": "user", "content": "Make the advisor strategy default."}]
    base_payload = BASE_PAYLOAD.copy()
    base_payload["tools"] = list(BASE_PAYLOAD.get("tools", []))

    with (
        patch("agentic_loop.api_call_with_retry", new=api_call_with_retry),
        patch("agentic_loop.execute_tool_calls", execute_tool_calls),
    ):
        content = await run_agentic_loop(
            messages=messages,
            session=AsyncMock(),
            base_url=BASE_URL,
            api_key=API_KEY,
            base_payload=base_payload,
            initial_response_data=initial_response,
        )

    assert content == "Advisor says tool plumbing first, then the loop. Proceeding."
    assert execute_tool_calls.await_count == 1
    advisor_nudges = [
        m["content"]
        for m in messages
        if m.get("role") == "user" and "Call consult_advisor" in m.get("content", "")
    ]
    assert (
        len(advisor_nudges) == 1
    ), f"Expected one advisor nudge, got {len(advisor_nudges)}"
    print("  ✓ Passed")


async def test_run_agentic_loop_retries_invalid_pseudo_tool_syntax() -> None:
    """Invalid <read(...)> style markup should be rejected and retried."""
    print("Test 9: Retry when model emits pseudo-tool markup")

    initial_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": '<read(filepath="handler.py")>\n<read(filepath="tools.py")>',
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
                            "id": "call_read_handler",
                            "type": "function",
                            "function": {
                                "name": "read",
                                "arguments": json.dumps({"filepath": "handler.py"}),
                            },
                        },
                        {
                            "id": "call_read_tools",
                            "type": "function",
                            "function": {
                                "name": "read",
                                "arguments": json.dumps({"filepath": "tools.py"}),
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
                    "content": "done reading both files",
                }
            }
        ]
    }

    execute_tool_calls = AsyncMock(
        return_value=[
            {
                "tool_call_id": "call_read_handler",
                "role": "tool",
                "content": json.dumps({"success": True, "content": "handler"}),
            },
            {
                "tool_call_id": "call_read_tools",
                "role": "tool",
                "content": json.dumps({"success": True, "content": "tools"}),
            },
        ]
    )
    api_call_with_retry = AsyncMock(side_effect=[tool_call_response, final_response])

    messages = [
        {
            "role": "user",
            "content": "Tool execution is mandatory. Use read on handler.py and tools.py.",
        }
    ]

    with (
        patch("agentic_loop.api_call_with_retry", new=api_call_with_retry),
        patch("agentic_loop.execute_tool_calls", execute_tool_calls),
    ):
        content = await run_agentic_loop(
            messages=messages,
            session=AsyncMock(),
            base_url=BASE_URL,
            api_key=API_KEY,
            base_payload=BASE_PAYLOAD.copy(),
            initial_response_data=initial_response,
        )

    assert content == "done reading both files"
    assert execute_tool_calls.await_count == 1
    pseudo_tool_nudges = [
        m["content"]
        for m in messages
        if m.get("role") == "user"
        and "invalid pseudo-tool markup" in m.get("content", "")
    ]
    assert (
        len(pseudo_tool_nudges) == 1
    ), f"Expected one pseudo-tool nudge, got {len(pseudo_tool_nudges)}"
    print("  ✓ Passed")


async def test_run_agentic_loop_retries_markdown_reaction_pseudo_tool() -> None:
    """Markdown-style reaction directives should be retried into real tool calls."""
    print("Test 9b: Retry when model emits markdown pseudo reaction syntax")

    initial_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": (
                        "ayy let's go!\n\n"
                        "*send_telegram_reaction: chat_id=880978583, "
                        "message_id=112, reaction=love*"
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
                            "id": "call_tg_reaction",
                            "type": "function",
                            "function": {
                                "name": "send_telegram_reaction",
                                "arguments": json.dumps(
                                    {
                                        "chat_id": 880978583,
                                        "message_id": 112,
                                        "reaction": "love",
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
                    "content": "ayy let's go!",
                }
            }
        ]
    }

    execute_tool_calls = AsyncMock(
        return_value=[
            {
                "tool_call_id": "call_tg_reaction",
                "role": "tool",
                "content": json.dumps({"success": True}),
            }
        ]
    )
    api_call_with_retry = AsyncMock(side_effect=[tool_call_response, final_response])

    messages = [
        {"role": "user", "content": "Nice, celebrate with me."},
    ]

    with (
        patch("agentic_loop.api_call_with_retry", new=api_call_with_retry),
        patch("agentic_loop.execute_tool_calls", execute_tool_calls),
    ):
        content = await run_agentic_loop(
            messages=messages,
            session=AsyncMock(),
            base_url=BASE_URL,
            api_key=API_KEY,
            base_payload=BASE_PAYLOAD.copy(),
            initial_response_data=initial_response,
        )

    assert content == "ayy let's go!"
    assert execute_tool_calls.await_count == 1
    pseudo_tool_nudges = [
        m["content"]
        for m in messages
        if m.get("role") == "user"
        and "invalid pseudo-tool markup" in m.get("content", "")
    ]
    assert (
        len(pseudo_tool_nudges) == 1
    ), f"Expected one pseudo-tool nudge, got {len(pseudo_tool_nudges)}"
    print("  ✓ Passed")


async def test_run_agentic_loop_retries_unverified_publish_success_claim() -> None:
    """Failed publish tool output should block blanket success claims."""
    print("Test 10: Retry when model claims publish success after failed tool output")

    initial_response = {
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
                                        "command": "git push -u origin main && publish-here-now"
                                    }
                                ),
                            },
                        }
                    ],
                }
            }
        ]
    }

    lying_final_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": (
                        "done! it's live at https://verified.here.now/ and pushed to "
                        "crustyhub at https://crustyhub.xyz/dice-roller.git"
                    ),
                }
            }
        ]
    }

    corrected_final_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": (
                        "here.now URL: https://verified.here.now/. crustyhub failed "
                        "and could not be verified."
                    ),
                }
            }
        ]
    }

    execute_tool_calls = AsyncMock(
        return_value=[
            {
                "tool_call_id": "call_publish",
                "role": "tool",
                "content": json.dumps(
                    {
                        "success": False,
                        "stderr": "fatal: unable to access 'https://crustyhub.xyz/...': Could not resolve host",
                        "returncode": 128,
                    }
                ),
            }
        ]
    )
    api_call_with_retry = AsyncMock(
        side_effect=[
            lying_final_response,
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                "the failed push eventually succeeded and crustyhub is up."
                            ),
                        }
                    }
                ]
            },
            corrected_final_response,
        ]
    )

    messages = [
        {
            "role": "user",
            "content": "make a website for me and post it to crustyhub and here.now",
        }
    ]

    with (
        patch("agentic_loop.api_call_with_retry", new=api_call_with_retry),
        patch("agentic_loop.execute_tool_calls", execute_tool_calls),
    ):
        content = await run_agentic_loop(
            messages=messages,
            session=AsyncMock(),
            base_url=BASE_URL,
            api_key=API_KEY,
            base_payload=BASE_PAYLOAD.copy(),
            initial_response_data=initial_response,
        )

    assert content == "here.now URL: https://verified.here.now/. crustyhub failed and could not be verified."
    correction_nudges = [
        m["content"]
        for m in messages
        if m.get("role") == "user"
        and "Failed tool results are ground truth" in m.get("content", "")
    ]
    assert len(correction_nudges) >= 1, correction_nudges
    print("  ✓ Passed")


async def main() -> None:
    print("=" * 60)
    print("Testing agentic loop action-intent handling")
    print("=" * 60)
    print()

    test_contains_action_intent_narration_matches_here_now_publish_language()
    test_contains_hard_decision_language_matches_executor_indecision()
    test_contains_pseudo_tool_syntax_matches_invalid_tool_markup()
    test_likely_unverified_success_claim_requires_no_failure_ack()
    await test_run_agentic_loop_retries_multiple_narrated_publish_attempts()
    await test_run_agentic_loop_retries_when_user_explicitly_requires_tools()
    await test_run_agentic_loop_retries_to_add_skill_for_skill_url()
    await test_run_agentic_loop_retries_for_normal_reply_after_tapback_ack()
    await test_run_agentic_loop_retries_for_normal_reply_after_telegram_reaction_ack()
    test_is_bare_reaction_word()
    await test_run_agentic_loop_retries_bare_reaction_word()
    await test_run_agentic_loop_nudges_consult_advisor_for_hard_decision()
    await test_run_agentic_loop_retries_invalid_pseudo_tool_syntax()
    await test_run_agentic_loop_retries_markdown_reaction_pseudo_tool()
    await test_run_agentic_loop_retries_unverified_publish_success_claim()

    print()
    print("=" * 60)
    print("Agentic loop tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
