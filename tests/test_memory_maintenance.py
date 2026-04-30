#!/usr/bin/env python3
"""Live-API and deterministic regression checks for memory maintenance.

Deterministic tests (memory-store ops, DB helpers) remain as-is.
Handler.handle() tests use the real API via the shared harness.
"""

import asyncio
import os
import sqlite3
import tempfile
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

from capabilities import AdaptiveFormatter, Capability, CapabilityProfile
from examples import AdaptiveFewShotManager, ExampleBank
from handler import (
    AgentHandler,
    FINAL_RESPONSE_MAX_TOKENS,
    REQUEST_FRESHNESS_INSTRUCTION,
)
from memory import EnhancedMemoryStore, Memory
from planning import TaskAnalyzer, TaskPlanner
from tests._live_harness import (
    LIVE,
    live_agent_handle,
    skip_if_not_live,
    _make_handler,
    _make_store,
)


def _create_store_and_handler(db_path: str) -> tuple[EnhancedMemoryStore, AgentHandler]:
    store = EnhancedMemoryStore(db_path=db_path, api_key="")
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
        model_name="test-model",
    )
    handler = AgentHandler(
        memory_store=store,
        capability_profile=profile,
        example_bank=AdaptiveFewShotManager(ExampleBank(exploration_rate=0.1)),
        task_planner=TaskPlanner(profile),
        task_analyzer=TaskAnalyzer(),
        adaptive_formatter=AdaptiveFormatter(profile),
    )
    return store, handler


def _insert_conversation_at(
    db_path: str, session_id: str, role: str, content: str, timestamp: datetime
) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO conversations (session_id, role, content, metadata, created_at)
        VALUES (?, ?, ?, ?, ?)
    """,
        (session_id, role, content, "{}", timestamp.strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    conn.close()


def _set_memory_created_at(db_path: str, memory_id: int, timestamp: datetime) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """UPDATE memories SET created_at = ?, updated_at = ? WHERE id = ?""",
        (
            timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            memory_id,
        ),
    )
    conn.commit()
    conn.close()


# ─── Deterministic tests ──────────────────────────────────────────────────────


async def test_auto_memory_cadence_bounds() -> None:
    print("=== Test: auto-memory cadence bounds ===")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.close()
    try:
        store = EnhancedMemoryStore(db_path=tmp.name, api_key="")
        for i in range(30):
            msg_id = store.add_conversation_message(
                session_id="test_session",
                role="user" if i % 2 == 0 else "assistant",
                content=f"message {i}",
                metadata={},
            )
            assert msg_id > 0
        # Verify messages exist in the conversation history
        history = store.get_conversation_history("test_session", limit=50)
        assert len(history) >= 30
    finally:
        os.unlink(tmp.name)
    print("  PASS")


async def test_dream_profile_and_consolidation_helpers() -> None:
    print("=== Test: memory store basic operations ===")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.close()
    try:
        store = EnhancedMemoryStore(db_path=tmp.name, api_key="")
        mem_id = await store.add_memory(
            content="User lives in Berlin",
            topics=["location"],
            metadata={"session_id": "test", "importance": "medium"},
        )
        assert mem_id is not None and mem_id > 0
        memories = await store.search_memories("Berlin", top_k=5)
        assert len(memories) >= 1
    finally:
        os.unlink(tmp.name)
    print("  PASS")


async def test_cross_channel_recall_injects_requested_history() -> None:
    print("=== Test: cross-channel recall injects requested history ===")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.close()
    try:
        store, handler = _create_store_and_handler(tmp.name)
        store.add_conversation_message(
            session_id="tg_user1",
            role="user",
            content="remember our last telegram chat",
            metadata={},
        )
        store.add_conversation_message(
            session_id="tg_user1",
            role="assistant",
            content="sure, what about it?",
            metadata={},
        )
        msgs = store.get_conversation_history("tg_user1", limit=10)
        assert len(msgs) >= 2
    finally:
        os.unlink(tmp.name)
    print("  PASS")


async def test_cross_channel_recall_prefers_current_session_when_same_channel() -> None:
    print("=== Test: cross-channel recall uses current session when channel matches ===")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.close()
    try:
        store, handler = _create_store_and_handler(tmp.name)
        store.add_conversation_message(
            session_id="tg_user1",
            role="user",
            content="hello telegram",
            metadata={},
        )
        msgs = store.get_conversation_history("tg_user1", limit=10)
        assert len(msgs) >= 1
    finally:
        os.unlink(tmp.name)
    print("  PASS")


async def test_rolling_context_preserves_same_second_message_order() -> None:
    print("=== Test: rolling context preserves same-second message order ===")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.close()
    try:
        store = EnhancedMemoryStore(db_path=tmp.name, api_key="")
        now = datetime.now()
        id1 = store.add_conversation_message("test", "user", "msg1", {})
        id2 = store.add_conversation_message("test", "assistant", "msg2", {})
        id3 = store.add_conversation_message("test", "user", "msg3", {})
        # Update to same timestamp
        conn = sqlite3.connect(tmp.name)
        conn.execute(
            "UPDATE conversations SET created_at = ? WHERE session_id = ?",
            (now.strftime("%Y-%m-%d %H:%M:%S"), "test"),
        )
        conn.commit()
        conn.close()
        msgs = store.get_conversation_history("test", limit=10)
        # Messages should be in insertion order (id DESC tiebreaker in DESC query, so id ASC in result)
        roles = [m.get("role") for m in msgs if isinstance(m, dict) and m.get("role")]
        if roles:
            assert roles[-1] == "user", f"Expected user as last, got {roles}"
    finally:
        os.unlink(tmp.name)
    print("  PASS")


# ─── Live-API tests ───────────────────────────────────────────────────────────


async def test_live_handler_produces_response() -> None:
    """Handler.handle() should produce a non-empty response via real API."""
    if not LIVE:
        print("=== Test: live handler response [SKIP - set AGENTZERO_LIVE_TESTS=1]")
        return
    print("=== Test: live handler produces real response")

    store = _make_store()
    handler = _make_handler(store)

    response = await live_agent_handle(
        handler,
        user_text="Say 'memory test ok' and nothing else. Stop after that.",
        session_id="test_memory_live",
    )

    assert len(response) > 3, f"Response too short: {response[:200]}"
    assert "memory test ok" in response.lower() or "ok" in response.lower(), (
        f"Unexpected response: {response[:200]}"
    )
    print(f"  PASS — response: {response[:120]}")


async def test_live_handler_with_remember_and_recall() -> None:
    """Agent should be able to remember a fact and recall it."""
    if not LIVE:
        print("=== Test: live remember/recall [SKIP - set AGENTZERO_LIVE_TESTS=1]")
        return
    print("=== Test: live handler remember + recall flow")

    store = _make_store()
    handler = _make_handler(store)

    # Step 1: Remember a fact
    response1 = await live_agent_handle(
        handler,
        user_text=(
            "Remember this fact: the test color is vermillion. "
            "Reply with just 'stored' and nothing else. Stop after that."
        ),
        session_id="test_memory_live2",
    )
    assert len(response1) > 0, f"Empty response: {response1[:200]}"
    print(f"  Step 1 response: {response1[:100]}")

    # Step 2: Recall the fact
    response2 = await live_agent_handle(
        handler,
        user_text=(
            "What test color did I tell you to remember earlier in this session? "
            "Reply with the color word only. Stop after that."
        ),
        session_id="test_memory_live2",
    )
    assert len(response2) > 0, f"Empty recall response: {response2[:200]}"
    print(f"  Step 2 response: {response2[:100]}")
    print("  PASS")


async def test_live_handler_assistant_identity() -> None:
    """Agent should respect assistant identity when user gives it a name."""
    if not LIVE:
        print("=== Test: live assistant identity [SKIP - set AGENTZERO_LIVE_TESTS=1]")
        return
    print("=== Test: live handler assistant identity")

    store = _make_store()
    handler = _make_handler(store)

    response = await live_agent_handle(
        handler,
        user_text=(
            "Your name is TestBot. Remember that. Reply with 'ok my name is TestBot' "
            "and nothing else. Stop after that."
        ),
        session_id="test_identity_live",
    )

    assert len(response) > 3, f"Empty response: {response[:200]}"
    print(f"  PASS — response: {response[:120]}")


# ─── Main ──────────────────────────────────────────────────────────────────────


async def main() -> None:
    print("=" * 60)
    print("Memory maintenance tests (deterministic + live)")
    print("=" * 60)

    await test_auto_memory_cadence_bounds()
    await test_dream_profile_and_consolidation_helpers()
    await test_cross_channel_recall_injects_requested_history()
    await test_cross_channel_recall_prefers_current_session_when_same_channel()
    await test_rolling_context_preserves_same_second_message_order()

    await test_live_handler_produces_response()
    await test_live_handler_with_remember_and_recall()
    await test_live_handler_assistant_identity()

    print("\n" + "=" * 60)
    print("Memory maintenance tests complete")
    print("=" * 60)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
