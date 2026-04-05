#!/usr/bin/env python3
"""Regression checks for auto-memory cadence and dream consolidation helpers."""

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
    IMESSAGE_SYSTEM_PROMPT_SUFFIX,
    REQUEST_FRESHNESS_INSTRUCTION,
)
from memory import EnhancedMemoryStore
from planning import TaskAnalyzer, TaskPlanner


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
    db_path: str,
    session_id: str,
    role: str,
    content: str,
    timestamp: datetime,
) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO conversations (session_id, role, content, metadata, created_at)
        VALUES (?, ?, ?, ?, ?)
    """,
        (
            session_id,
            role,
            content,
            "{}",
            timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    conn.commit()
    conn.close()


def _set_memory_created_at(db_path: str, memory_id: int, timestamp: datetime) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE memories
        SET created_at = ?, updated_at = ?
        WHERE id = ?
    """,
        (
            timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            memory_id,
        ),
    )
    conn.commit()
    conn.close()


async def test_auto_memory_cadence_bounds() -> None:
    print("=== Test: auto-memory cadence bounds ===")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.close()

    try:
        store, handler = _create_store_and_handler(tmp.name)
        session_id = "memory-cadence-session"

        for i in range(9):
            store.add_conversation_message(
                role="user" if i % 2 == 0 else "assistant",
                content=f"msg-{i}",
                session_id=session_id,
            )

        should_capture, details = handler._should_capture_auto_memory(
            session_id=session_id,
            message_count=store.get_conversation_message_count(session_id),
        )
        assert should_capture is False
        assert details.get("reason") == "too_early"

        store.add_conversation_message(
            role="assistant",
            content="msg-9",
            session_id=session_id,
        )
        should_capture, details = handler._should_capture_auto_memory(
            session_id=session_id,
            message_count=store.get_conversation_message_count(session_id),
        )
        assert should_capture is True
        assert details.get("reason") == "bootstrap_memory"

        await store.add_memory(
            content="User prefers concise responses.",
            metadata={
                "type": "auto_memory",
                "session_id": session_id,
                "message_index": 10,
            },
            generate_embedding=False,
        )

        for i in range(10, 20):
            store.add_conversation_message(
                role="user" if i % 2 == 0 else "assistant",
                content=f"msg-{i}",
                session_id=session_id,
            )

        should_capture, details = handler._should_capture_auto_memory(
            session_id=session_id,
            message_count=store.get_conversation_message_count(session_id),
        )
        assert should_capture is False
        assert details.get("reason") == "cadence_ok"

        for i in range(20, 22):
            store.add_conversation_message(
                role="user" if i % 2 == 0 else "assistant",
                content=f"msg-{i}",
                session_id=session_id,
            )

        should_capture, details = handler._should_capture_auto_memory(
            session_id=session_id,
            message_count=store.get_conversation_message_count(session_id),
        )
        assert should_capture is True
        assert details.get("reason") == "ratio_above_max"

        await store.add_memory(
            content="User is building a memory-heavy agent project.",
            metadata={
                "type": "auto_memory",
                "session_id": session_id,
                "message_index": 20,
            },
            generate_embedding=False,
        )

        for i in range(22, 35):
            store.add_conversation_message(
                role="user" if i % 2 == 0 else "assistant",
                content=f"msg-{i}",
                session_id=session_id,
            )

        should_capture, details = handler._should_capture_auto_memory(
            session_id=session_id,
            message_count=store.get_conversation_message_count(session_id),
        )
        assert should_capture is True
        assert details.get("reason") == "stale_memory_interval"

        print("✓ cadence bounds behave as expected")
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)


async def test_dream_profile_and_consolidation_helpers() -> None:
    print("=== Test: dream-profile and consolidation helpers ===")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.close()

    try:
        store, _ = _create_store_and_handler(tmp.name)
        session_id = "dream-profile-session"

        now = datetime.now()
        for day_offset in range(14):
            day_anchor = now - timedelta(days=day_offset)
            for hour in (10, 11, 12, 13):
                _insert_conversation_at(
                    db_path=tmp.name,
                    session_id=session_id,
                    role="user",
                    content=f"busy-hour-{day_offset}-{hour}",
                    timestamp=day_anchor.replace(
                        hour=hour, minute=0, second=0, microsecond=0
                    ),
                )
            _insert_conversation_at(
                db_path=tmp.name,
                session_id=session_id,
                role="assistant",
                content=f"quiet-hour-{day_offset}",
                timestamp=day_anchor.replace(hour=3, minute=0, second=0, microsecond=0),
            )

        offpeak = store.infer_offpeak_hours(
            lookback_days=21,
            min_days=14,
            window_hours=6,
        )
        assert offpeak.get("reason") == "ok"
        hours = offpeak.get("hours", [])
        assert isinstance(hours, list) and len(hours) == 6
        assert all(isinstance(hour, int) and 0 <= hour <= 23 for hour in hours)

        auto_id = await store.add_memory(
            content="The user is migrating from chat logs to durable memory abstractions.",
            metadata={"type": "auto_memory", "importance": "medium"},
            generate_embedding=False,
        )
        long_term_id = await store.add_memory(
            content="Long-term memory entry",
            metadata={"type": "long_term_memory", "importance": "high"},
            generate_embedding=False,
        )
        system_id = await store.add_memory(
            content="Custom system prompt",
            metadata={"type": "system_prompt"},
            generate_embedding=False,
        )

        old_time = datetime.now() - timedelta(hours=2)
        _set_memory_created_at(tmp.name, int(auto_id), old_time)
        _set_memory_created_at(tmp.name, int(long_term_id), old_time)
        _set_memory_created_at(tmp.name, int(system_id), old_time)

        candidates = store.get_memories_for_consolidation(limit=10, min_age_hours=1)
        candidate_ids = {memory.id for memory in candidates}
        assert int(auto_id) in candidate_ids
        assert int(long_term_id) not in candidate_ids
        assert int(system_id) not in candidate_ids

        updated = store.mark_memories_dream_consolidated(
            [int(auto_id)],
            significance=0.82,
        )
        assert updated == 1

        recent = store.get_recent_memories(limit=10)
        auto_memory = next(
            (memory for memory in recent if memory.id == int(auto_id)), None
        )
        assert auto_memory is not None
        assert auto_memory.metadata.get("dream_consolidated") is True
        assert auto_memory.metadata.get("dream_significance") == 0.82

        print("✓ dream helper behavior validated")
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)


async def test_cross_channel_recall_injects_requested_history() -> None:
    print("=== Test: cross-channel recall injects requested history ===")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.close()

    try:
        _, handler = _create_store_and_handler(tmp.name)

        anchor = datetime.now() - timedelta(minutes=5)
        _insert_conversation_at(
            db_path=tmp.name,
            session_id="tg_123",
            role="user",
            content="Telegram oldest message",
            timestamp=anchor,
        )
        _insert_conversation_at(
            db_path=tmp.name,
            session_id="tg_123",
            role="assistant",
            content="Telegram middle message",
            timestamp=anchor + timedelta(minutes=1),
        )
        _insert_conversation_at(
            db_path=tmp.name,
            session_id="tg_123",
            role="user",
            content="Telegram latest message",
            timestamp=anchor + timedelta(minutes=2),
        )

        context = handler._build_cross_channel_context(
            user_query="hey remember what we we're talking about on telegram last 2 messages",
            session_id="imessage_+15550001111",
        )

        assert "Cross-channel context injected from telegram" in context
        assert "Telegram middle message" in context
        assert "Telegram latest message" in context
        assert "Telegram oldest message" not in context
        print("✓ requested channel history injected")
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)


async def test_cross_channel_recall_prefers_current_session_when_same_channel() -> None:
    print(
        "=== Test: cross-channel recall uses current session when channel matches ==="
    )
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.close()

    try:
        _, handler = _create_store_and_handler(tmp.name)

        anchor = datetime.now() - timedelta(minutes=5)
        _insert_conversation_at(
            db_path=tmp.name,
            session_id="tg_555",
            role="user",
            content="Current Telegram session message",
            timestamp=anchor,
        )
        _insert_conversation_at(
            db_path=tmp.name,
            session_id="tg_999",
            role="user",
            content="Other Telegram session message",
            timestamp=anchor + timedelta(minutes=1),
        )

        context = handler._build_cross_channel_context(
            user_query="remember what we were talking about on telegram last 1 message",
            session_id="tg_555",
        )

        assert "Current Telegram session message" in context
        assert "Other Telegram session message" not in context
        print("✓ current session is preferred for same-channel recall")
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)


async def test_cross_channel_recall_is_injected_into_handle_context() -> None:
    print("=== Test: cross-channel recall is injected into handle context ===")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.close()

    try:
        _, handler = _create_store_and_handler(tmp.name)

        anchor = datetime.now() - timedelta(minutes=5)
        _insert_conversation_at(
            db_path=tmp.name,
            session_id="tg_777",
            role="assistant",
            content="Telegram context payload message",
            timestamp=anchor,
        )

        captured_payload: dict[str, object] = {}

        async def _fake_api_call(_session, _url, payload, _headers, **_kwargs):
            captured_payload["messages"] = payload.get("messages", [])
            return {"id": "fake", "choices": [{"message": {"role": "assistant"}}]}

        with (
            patch.object(
                handler.memory_store, "search_memories", new=AsyncMock(return_value=[])
            ),
            patch(
                "handler.api_call_with_retry", new=AsyncMock(side_effect=_fake_api_call)
            ),
            patch("handler.process_response", new=AsyncMock(return_value="ok")),
        ):
            result = await handler.handle(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "hey remember what we we're talking about on telegram last 1 message",
                        }
                    ]
                },
                session_id="imessage_+15558889999",
            )

        assert result == "ok"
        assert captured_payload.get("messages")
        system_content = str(captured_payload["messages"][0]["content"])
        assert "Cross-channel context injected from telegram" in system_content
        assert "Telegram context payload message" in system_content
        print("✓ cross-channel context is injected into handler payload")
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)


async def test_imessage_session_injects_mobile_prompt_defaults() -> None:
    print("=== Test: iMessage sessions inject mobile prompt defaults ===")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.close()

    try:
        _, handler = _create_store_and_handler(tmp.name)

        captured_payload: dict[str, object] = {}

        async def _fake_api_call(_session, _url, payload, _headers, **_kwargs):
            captured_payload["messages"] = payload.get("messages", [])
            return {"id": "fake", "choices": [{"message": {"role": "assistant"}}]}

        with (
            patch.object(
                handler.memory_store, "search_memories", new=AsyncMock(return_value=[])
            ),
            patch(
                "handler.api_call_with_retry", new=AsyncMock(side_effect=_fake_api_call)
            ),
            patch("handler.process_response", new=AsyncMock(return_value="ok")),
        ):
            imessage_result = await handler.handle(
                {"messages": [{"role": "user", "content": "Quick status?"}]},
                session_id="imessage_+15550001111",
            )

            assert imessage_result == "ok"
            assert captured_payload.get("messages")
            imessage_system_content = str(captured_payload["messages"][0]["content"])
            assert IMESSAGE_SYSTEM_PROMPT_SUFFIX in imessage_system_content

            telegram_result = await handler.handle(
                {"messages": [{"role": "user", "content": "Quick status?"}]},
                session_id="tg_123",
            )

            assert telegram_result == "ok"
            assert captured_payload.get("messages")
            telegram_system_content = str(captured_payload["messages"][0]["content"])
            assert IMESSAGE_SYSTEM_PROMPT_SUFFIX not in telegram_system_content

        print("✓ iMessage session prompt defaults are injected conditionally")
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)


async def test_imessage_handle_ids_are_injected_and_persisted() -> None:
    print("=== Test: iMessage handle IDs are injected and persisted ===")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.close()

    try:
        store, handler = _create_store_and_handler(tmp.name)
        session_id = "imessage_+15550002222"

        store.add_conversation_message(
            role="user",
            content="Earlier inbound message",
            session_id=session_id,
            metadata={"message_handle": "older-handle", "part_index": 2},
        )

        captured_payload: dict[str, object] = {}

        async def _fake_api_call(_session, _url, payload, _headers, **_kwargs):
            captured_payload["messages"] = payload.get("messages", [])
            return {"id": "fake", "choices": [{"message": {"role": "assistant"}}]}

        with (
            patch.object(
                handler.memory_store, "search_memories", new=AsyncMock(return_value=[])
            ),
            patch(
                "handler.api_call_with_retry", new=AsyncMock(side_effect=_fake_api_call)
            ),
            patch("handler.process_response", new=AsyncMock(return_value="ok")),
        ):
            result = await handler.handle(
                {"messages": [{"role": "user", "content": "Tapback this one"}]},
                session_id=session_id,
                request_metadata={"message_handle": "current-handle", "part_index": 1},
            )

        assert result == "ok"
        assert captured_payload.get("messages")

        system_content = str(captured_payload["messages"][0]["content"])
        assert "Available iMessage tapback handles" in system_content
        assert "current-handle" in system_content
        assert "older-handle" in system_content
        assert "part_index=1" in system_content
        assert "part_index=2" in system_content

        history = store.get_conversation_history(session_id=session_id, limit=10)
        matching_user_rows = [
            message
            for message in history
            if message.get("role") == "user"
            and message.get("content") == "Tapback this one"
        ]
        assert matching_user_rows
        latest_user_row = matching_user_rows[0]
        assert (
            latest_user_row.get("metadata", {}).get("message_handle")
            == "current-handle"
        )
        assert latest_user_row.get("metadata", {}).get("part_index") == 1

        print("✓ iMessage handles are visible to the model and stored in history")
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)


async def test_handle_injects_request_freshness_token() -> None:
    print("=== Test: request freshness token is injected into system prompt ===")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.close()

    try:
        _, handler = _create_store_and_handler(tmp.name)

        captured_payload: dict[str, object] = {}

        async def _fake_api_call(_session, _url, payload, _headers, **_kwargs):
            captured_payload["messages"] = payload.get("messages", [])
            return {"id": "fake", "choices": [{"message": {"role": "assistant"}}]}

        with (
            patch.object(
                handler.memory_store, "search_memories", new=AsyncMock(return_value=[])
            ),
            patch(
                "handler.api_call_with_retry", new=AsyncMock(side_effect=_fake_api_call)
            ),
            patch("handler.process_response", new=AsyncMock(return_value="ok")),
        ):
            result = await handler.handle(
                {"messages": [{"role": "user", "content": "Say hello"}]},
                session_id="tg_456",
            )

        assert result == "ok"
        assert captured_payload.get("messages")

        system_content = str(captured_payload["messages"][0]["content"])
        assert REQUEST_FRESHNESS_INSTRUCTION in system_content
        assert "[Freshness Token]:" in system_content
        print("✓ request freshness token is included in the visible-response prompt")
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)


async def test_handle_strips_delivery_directives_from_response() -> None:
    print("=== Test: handle strips outbound delivery directives ===")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.close()

    try:
        _, handler = _create_store_and_handler(tmp.name)

        tagged_response = (
            '<typing seconds="1.2"/>'
            "<message>first reply</message>"
            "<message>second reply</message>"
        )

        with (
            patch.object(
                handler.memory_store, "search_memories", new=AsyncMock(return_value=[])
            ),
            patch(
                "handler.api_call_with_retry",
                new=AsyncMock(
                    return_value={
                        "id": "fake",
                        "choices": [{"message": {"role": "assistant"}}],
                    }
                ),
            ),
            patch(
                "handler.process_response", new=AsyncMock(return_value=tagged_response)
            ),
        ):
            result = await handler.handle(
                {"messages": [{"role": "user", "content": "Respond briefly"}]},
                session_id="tg_999",
            )

        assert result == "first reply\n\nsecond reply"

        history = handler.memory_store.get_conversation_history(
            session_id="tg_999",
            limit=5,
        )
        assistant_rows = [row for row in history if row.get("role") == "assistant"]
        assert assistant_rows
        assert assistant_rows[0].get("content") == "first reply\n\nsecond reply"
        print("✓ delivery directives are removed from visible responses and memory")
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)


async def main() -> int:
    await test_auto_memory_cadence_bounds()
    await test_dream_profile_and_consolidation_helpers()
    await test_cross_channel_recall_injects_requested_history()
    await test_cross_channel_recall_prefers_current_session_when_same_channel()
    await test_cross_channel_recall_is_injected_into_handle_context()
    await test_imessage_session_injects_mobile_prompt_defaults()
    await test_imessage_handle_ids_are_injected_and_persisted()
    await test_handle_injects_request_freshness_token()
    await test_handle_strips_delivery_directives_from_response()
    print("\nAll memory-maintenance tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
