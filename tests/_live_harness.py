"""Shared harness for live-API agent tests.

All tests that use this harness call the real NVIDIA API. No mocking.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure dotenv is loaded before importing handler modules
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from handler import AgentHandler, API_KEY, BASE_PAYLOAD, BASE_URL, PRIMARY_MODEL_ID
from agentic_loop import run_agentic_loop
from api import api_call_with_retry, execute_tool_calls, process_response
from memory import EnhancedMemoryStore
from capabilities import Capability, CapabilityProfile, AdaptiveFormatter
from examples import AdaptiveFewShotManager, ExampleBank
from planning import TaskPlanner, TaskAnalyzer
from tools import (
    get_send_message_buffer,
    init_send_message_buffer,
    reset_send_message_buffer,
    init_declared_message_count,
    reset_declared_message_count,
    set_tool_runtime_session,
    reset_tool_runtime_session,
    set_memory_store,
)

TIMEOUT = 120  # seconds per test

LIVE = os.environ.get("AGENTZERO_LIVE_TESTS", "1").strip() in {"1", "true", "yes", "y", "on"}


def skip_if_not_live() -> None:
    """Fail-fast if live tests are disabled."""
    if not LIVE:
        raise RuntimeError(
            "Live API tests require AGENTZERO_LIVE_TESTS=1 and valid NVIDIA_API_KEY in .env"
        )


def _make_store(db_path: str = ":memory:") -> EnhancedMemoryStore:
    return EnhancedMemoryStore(db_path=db_path, api_key=API_KEY)


def _make_handler(
    store: EnhancedMemoryStore | None = None,
) -> AgentHandler:
    if store is None:
        store = EnhancedMemoryStore(db_path=":memory:", api_key=API_KEY)

    profile = CapabilityProfile(
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
        model_name=PRIMARY_MODEL_ID,
    )

    return AgentHandler(
        memory_store=store,
        capability_profile=profile,
        example_bank=AdaptiveFewShotManager(ExampleBank(exploration_rate=0.1)),
        task_planner=TaskPlanner(profile),
        task_analyzer=TaskAnalyzer(),
        adaptive_formatter=AdaptiveFormatter(profile),
    )


async def live_api_call(
    session,
    messages: list[dict[str, Any]],
    base_payload_override: dict[str, Any] | None = None,
    stream: bool = False,
) -> dict[str, Any]:
    """Call the real API and return the response dict."""
    payload = (base_payload_override or BASE_PAYLOAD).copy()
    payload["messages"] = messages
    payload["stream"] = stream

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Cache-Control": "no-cache, no-store, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
    }

    import aiohttp
    import uuid

    headers["X-Request-Id"] = f"test-{uuid.uuid4().hex[:12]}"

    async with aiohttp.ClientSession() as http_session:
        return await api_call_with_retry(
            http_session,
            BASE_URL,
            payload,
            headers,
            stream=stream,
        )


async def live_run_agentic_loop(
    messages: list[dict[str, Any]],
    session_id: str | None = None,
    max_iterations: int = 10,
    base_payload_override: dict[str, Any] | None = None,
) -> str:
    """Run the full agentic loop against the real API.

    Returns the final text assembled by the loop.
    """
    import aiohttp

    skip_if_not_live()

    payload = (base_payload_override or BASE_PAYLOAD).copy()

    if session_id:
        token = set_tool_runtime_session(session_id)

    async with aiohttp.ClientSession() as http_session:
        try:
            result = await run_agentic_loop(
                messages=messages,
                session=http_session,
                base_url=BASE_URL,
                api_key=API_KEY,
                base_payload=payload,
                max_iterations=max_iterations,
            )
            return result
        finally:
            if session_id:
                reset_tool_runtime_session(token)


async def live_agent_handle(
    handler: AgentHandler,
    user_text: str,
    session_id: str = "test_session",
    conversation_history: list[dict[str, Any]] | None = None,
) -> str:
    """Run a full handler.handle() call against the real API.

    Returns the assistant's plain-text response.
    """
    import aiohttp

    skip_if_not_live()

    messages: list[dict[str, Any]] = []
    if conversation_history:
        messages.extend(list(conversation_history))
    messages.append({"role": "user", "content": user_text})

    async with aiohttp.ClientSession() as http_session:
        return await handler.handle(
            messages=messages,
            session_id=session_id,
            session=http_session,
        )


def collect_tool_calls_from_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract all tool_call entries from message history."""
    tool_calls: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tool_calls.append(tc)
    return tool_calls


def collect_tool_names(messages: list[dict[str, Any]]) -> list[str]:
    """Return list of all tool names called across message history."""
    return [
        tc.get("function", {}).get("name", "")
        for tc in collect_tool_calls_from_messages(messages)
    ]


def parse_loop_result(result: str) -> dict[str, Any]:
    """Try to parse a JSON envelope from the loop result."""
    try:
        parsed = json.loads(result)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return {"text": result, "attachments": []}
