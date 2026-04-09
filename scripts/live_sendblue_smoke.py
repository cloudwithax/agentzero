#!/usr/bin/env python3
"""Live in-process Sendblue smoke test against the real AgentZero handler."""

import argparse
import asyncio
import json
import os
import time
from typing import Any
from unittest.mock import patch

import aiohttp
from dotenv import load_dotenv

load_dotenv(".env")

import agentic_loop
import integrations
from integrations import process_imessage_and_reply
from main import initialize_agent


DEFAULT_PROMPT = (
    "Live Sendblue smoke test. You must use tools and actually execute them. "
    "First use the bash tool to run `printf sendblue-smoke`. "
    "Then use the read tool to read the first line of AGENTS.md. "
    "Reply in exactly two lines:\n"
    "status: <bash output>\n"
    "first_line: <first line>"
)


class _FakeSendblueResponse:
    def __init__(self, status: int, data: dict[str, Any]):
        self.status = status
        self._data = data

    async def __aenter__(self) -> "_FakeSendblueResponse":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def json(self, content_type=None) -> dict[str, Any]:
        return self._data


class _FakeSendblueSession:
    def __init__(self, sink: list[dict[str, Any]], real_session: aiohttp.ClientSession):
        self._sink = sink
        self._real_session = real_session

    async def __aenter__(self) -> "_FakeSendblueSession":
        await self._real_session.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return await self._real_session.__aexit__(exc_type, exc, tb)

    async def close(self) -> None:
        await self._real_session.close()

    def post(self, url: str, json=None, headers=None):
        if "api.sendblue." not in url:
            return self._real_session.post(url, json=json, headers=headers)

        event = {
            "url": url,
            "json": json,
            "headers_present": bool(headers),
            "timestamp": time.time(),
        }
        self._sink.append(event)

        if url.endswith("/api/send-message"):
            payload = {"status": "QUEUED", "message_handle": "shim-send-message"}
            return _FakeSendblueResponse(200, payload)
        if url.endswith("/api/send-typing-indicator"):
            return _FakeSendblueResponse(200, {"status": "OK"})
        if url.endswith("/api/mark-read"):
            return _FakeSendblueResponse(200, {"status": "OK"})
        if url.endswith("/api/send-reaction"):
            return _FakeSendblueResponse(200, {"status": "OK"})

        return _FakeSendblueResponse(200, {"status": "OK"})

    def get(self, url: str, *args, **kwargs):
        if "api.sendblue." not in url:
            return self._real_session.get(url, *args, **kwargs)
        return _FakeSendblueResponse(200, {"status": "OK"})


async def run_smoke(
    prompt: str,
    phone_number: str,
    message_handle: str,
    expected_substrings: list[str],
    required_tools: list[str],
    require_read_receipt: bool = True,
    require_typing_indicator: bool = True,
    require_outbound_message: bool = True,
) -> dict[str, Any]:
    handler, _acp_agent = initialize_agent()
    await handler.start_reminder_scheduler()

    sendblue_posts: list[dict[str, Any]] = []
    tool_rounds: list[list[str]] = []
    loop_response_snapshots: list[dict[str, Any]] = []
    started_at = time.time()

    original_execute_tool_calls = agentic_loop.execute_tool_calls
    original_api_call_with_retry = agentic_loop.api_call_with_retry

    async def wrapped_execute_tool_calls(message, allowed_tool_names=None):
        tool_rounds.append(
            [
                tool_call.get("function", {}).get("name", "")
                for tool_call in (message.get("tool_calls") or [])
            ]
        )
        return await original_execute_tool_calls(
            message,
            allowed_tool_names=allowed_tool_names,
        )

    async def wrapped_api_call_with_retry(*args, **kwargs):
        response_data = await original_api_call_with_retry(*args, **kwargs)
        choices = response_data.get("choices") or []
        if choices:
            message = choices[0].get("message", {}) or {}
            content_text = agentic_loop._message_content_to_text(
                message.get("content", "")
            )
            tool_calls = message.get("tool_calls") or []
            loop_response_snapshots.append(
                {
                    "content": content_text,
                    "has_tool_calls": bool(tool_calls),
                    "tool_call_names": [
                        tool_call.get("function", {}).get("name", "")
                        for tool_call in tool_calls
                    ],
                    "pseudo_tool_syntax": agentic_loop.contains_pseudo_tool_syntax(
                        content_text
                    ),
                    "action_intent_narration": (
                        agentic_loop.contains_action_intent_narration(content_text)
                    ),
                    "hard_decision_language": (
                        agentic_loop.contains_hard_decision_language(content_text)
                    ),
                }
            )
        return response_data

    try:
        real_client_session = aiohttp.ClientSession

        with (
            patch(
                "integrations.aiohttp.ClientSession",
                side_effect=lambda *args, **kwargs: _FakeSendblueSession(
                    sendblue_posts,
                    real_client_session(*args, **kwargs),
                ),
            ),
            patch(
                "integrations._maybe_send_random_sendblue_tapback",
                return_value=None,
            ),
            patch(
                "agentic_loop.api_call_with_retry",
                side_effect=wrapped_api_call_with_retry,
            ),
            patch(
                "agentic_loop.execute_tool_calls",
                side_effect=wrapped_execute_tool_calls,
            ),
        ):
            await process_imessage_and_reply(
                handler,
                phone_number,
                prompt,
                message_handle=message_handle,
                part_index=0,
            )
    finally:
        await handler.reminder_scheduler.stop()

    session_id = f"imessage_{phone_number.strip()}"
    conversation_history = handler.memory_store.get_conversation_history(
        session_id=session_id,
        limit=6,
    )
    outbound_messages = [
        event
        for event in sendblue_posts
        if event["url"].endswith("/api/send-message")
    ]
    typing_events = [
        event
        for event in sendblue_posts
        if event["url"].endswith("/api/send-typing-indicator")
    ]
    read_events = [
        event for event in sendblue_posts if event["url"].endswith("/api/mark-read")
    ]

    final_payload = outbound_messages[-1]["json"] if outbound_messages else {}
    final_content = str(final_payload.get("content", ""))
    flat_tool_names = [name for round_names in tool_rounds for name in round_names if name]
    text_only_before_tool = []
    for snapshot in loop_response_snapshots:
        if snapshot["has_tool_calls"]:
            break
        text_only_before_tool.append(snapshot)

    telemetry = {
        "loop_response_count": len(loop_response_snapshots),
        "text_only_rounds_before_first_tool": len(text_only_before_tool),
        "pseudo_tool_rounds_before_first_tool": sum(
            1 for snapshot in text_only_before_tool if snapshot["pseudo_tool_syntax"]
        ),
        "action_intent_rounds_before_first_tool": sum(
            1
            for snapshot in text_only_before_tool
            if snapshot["action_intent_narration"]
        ),
        "hard_decision_rounds_before_first_tool": sum(
            1
            for snapshot in text_only_before_tool
            if snapshot["hard_decision_language"]
        ),
        "needed_recovery_before_tools": bool(text_only_before_tool),
        "recovered_to_tool_execution": bool(text_only_before_tool and tool_rounds),
        "consult_advisor_seen": "consult_advisor" in set(flat_tool_names),
        "consult_reviewer_seen": "consult_reviewer" in set(flat_tool_names),
    }

    normalized_final_content = final_content.lower()
    normalized_tools = set(flat_tool_names)
    missing_substrings = [
        text for text in expected_substrings if text.lower() not in normalized_final_content
    ]
    missing_tools = [tool for tool in required_tools if tool not in normalized_tools]

    success = (
        (not require_read_receipt or bool(read_events))
        and (not require_typing_indicator or bool(typing_events))
        and (not require_outbound_message or bool(outbound_messages))
        and not missing_substrings
        and not missing_tools
    )

    return {
        "success": success,
        "elapsed_seconds": round(time.time() - started_at, 2),
        "session_id": session_id,
        "tool_rounds": tool_rounds,
        "tools_seen": sorted(normalized_tools),
        "required_tools": required_tools,
        "missing_tools": missing_tools,
        "expected_substrings": expected_substrings,
        "missing_substrings": missing_substrings,
        "read_receipt_count": len(read_events),
        "typing_indicator_count": len(typing_events),
        "outbound_message_count": len(outbound_messages),
        "final_outbound_content": final_content,
        "conversation_history": conversation_history,
        "agentic_metrics": telemetry,
        "loop_response_snapshots": loop_response_snapshots,
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run a live Sendblue-style smoke test.")
    parser.add_argument(
        "--phone-number",
        default=f"+1555{int(time.time()) % 10000000:07d}",
    )
    parser.add_argument("--message-handle", default="smoke-handle-001")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--expect-substring",
        action="append",
        dest="expected_substrings",
        help="Substring that must appear in outbound Sendblue content (repeatable).",
    )
    parser.add_argument(
        "--require-tool",
        action="append",
        dest="required_tools",
        help="Tool that must appear in at least one executed tool call (repeatable).",
    )
    parser.add_argument(
        "--no-require-read-receipt",
        action="store_true",
        help="Do not fail if mark-read is not called.",
    )
    parser.add_argument(
        "--no-require-typing-indicator",
        action="store_true",
        help="Do not fail if typing indicator is not called.",
    )
    parser.add_argument(
        "--no-require-outbound-message",
        action="store_true",
        help="Do not fail if send-message is not called.",
    )
    args = parser.parse_args()

    expected_substrings = (
        args.expected_substrings
        if args.expected_substrings
        else ["status:", "first_line:"]
    )
    required_tools = args.required_tools if args.required_tools else ["bash", "read"]

    result = await run_smoke(
        prompt=args.prompt,
        phone_number=args.phone_number,
        message_handle=args.message_handle,
        expected_substrings=expected_substrings,
        required_tools=required_tools,
        require_read_receipt=not args.no_require_read_receipt,
        require_typing_indicator=not args.no_require_typing_indicator,
        require_outbound_message=not args.no_require_outbound_message,
    )
    print(json.dumps(result, indent=2))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
