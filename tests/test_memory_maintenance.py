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
    FINAL_RESPONSE_MAX_TOKENS,
    REQUEST_FRESHNESS_INSTRUCTION,
)
from memory import EnhancedMemoryStore, Memory
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


async def test_rolling_context_preserves_same_second_message_order() -> None:
    print("=== Test: rolling context preserves same-second message order ===")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.close()

    try:
        _, handler = _create_store_and_handler(tmp.name)
        session_id = "tg_222"
        timestamp = datetime.now().replace(microsecond=0)

        _insert_conversation_at(
            db_path=tmp.name,
            session_id=session_id,
            role="user",
            content="First user message",
            timestamp=timestamp,
        )
        _insert_conversation_at(
            db_path=tmp.name,
            session_id=session_id,
            role="assistant",
            content="First assistant reply",
            timestamp=timestamp,
        )

        messages = handler._build_rolling_context(
            system_message={"role": "system", "content": "sys"},
            current_messages=[{"role": "user", "content": "Current prompt"}],
            session_id=session_id,
            context_window=4000,
            buffer_tokens=100,
        )

        history_slice = messages[1:-1]
        assert [message["role"] for message in history_slice] == ["user", "assistant"]
        assert history_slice[0]["content"] == "First user message"
        assert history_slice[1]["content"] == "First assistant reply"
        print("✓ same-second history stays chronological inside rolling context")
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)


async def test_imessage_session_injects_mobile_prompt_defaults() -> None:
    print("=== Test: iMessage sessions inject mobile prompt defaults ===")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.close()

    try:
        _, handler = _create_store_and_handler(tmp.name)

        captured_payloads: list[dict[str, object]] = []

        async def _fake_api_call(_session, _url, payload, _headers, **_kwargs):
            captured_payloads.append(dict(payload))
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
            initial_payload = captured_payloads[0]
            assert initial_payload.get("max_tokens") != FINAL_RESPONSE_MAX_TOKENS, (
                f"Initial payload must NOT use the capped {FINAL_RESPONSE_MAX_TOKENS} limit "
                f"(which truncates tool-call responses for complex requests); "
                f"got max_tokens={initial_payload.get('max_tokens')}"
            )
            assert initial_payload.get("max_tokens") == 32768
            imessage_system_content = str(initial_payload["messages"][0]["content"])
            assert (
                "iMessage" not in imessage_system_content
                or "phone screen" not in imessage_system_content
            )
            assert "single ongoing conversations by design" in imessage_system_content

            telegram_result = await handler.handle(
                {"messages": [{"role": "user", "content": "Quick status?"}]},
                session_id="tg_123",
            )

            assert telegram_result == "ok"
            tg_initial_payload = captured_payloads[1]
            assert tg_initial_payload.get("max_tokens") != FINAL_RESPONSE_MAX_TOKENS
            assert tg_initial_payload.get("max_tokens") == 32768
            tg_system_content = str(tg_initial_payload["messages"][0]["content"])
            assert (
                "iMessage" not in tg_system_content
                or "phone screen" not in tg_system_content
            )
            assert "single ongoing conversations by design" in tg_system_content

        print(
            "✓ Initial responses use full max_tokens (4096) so tool calls are not truncated"
        )
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


async def test_telegram_reaction_targets_are_injected_and_persisted() -> None:
    print("=== Test: Telegram reaction targets are injected and persisted ===")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.close()

    try:
        store, handler = _create_store_and_handler(tmp.name)
        session_id = "tg_4242"

        store.add_conversation_message(
            role="user",
            content="Earlier Telegram inbound message",
            session_id=session_id,
            metadata={"telegram_chat_id": 4242, "telegram_message_id": 111},
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
                {"messages": [{"role": "user", "content": "React to this one"}]},
                session_id=session_id,
                request_metadata={
                    "telegram_chat_id": 4242,
                    "telegram_message_id": 222,
                },
            )

        assert result == "ok"
        assert captured_payload.get("messages")

        system_content = str(captured_payload["messages"][0]["content"])
        assert "Available Telegram reaction targets" in system_content
        assert "chat_id=4242; message_id=222" in system_content
        assert "chat_id=4242; message_id=111" in system_content

        history = store.get_conversation_history(session_id=session_id, limit=10)
        matching_user_rows = [
            message
            for message in history
            if message.get("role") == "user"
            and message.get("content") == "React to this one"
        ]
        assert matching_user_rows
        latest_user_row = matching_user_rows[0]
        assert latest_user_row.get("metadata", {}).get("telegram_chat_id") == 4242
        assert latest_user_row.get("metadata", {}).get("telegram_message_id") == 222

        print("✓ Telegram reaction targets are visible to the model and stored")
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)


async def test_handle_injects_session_continuity_brief() -> None:
    print("=== Test: session continuity brief is injected proactively ===")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.close()

    try:
        store, handler = _create_store_and_handler(tmp.name)
        session_id = "tg_2024"

        await store.add_memory(
            content="User goes by Alice and prefers blunt, concise replies.",
            metadata={
                "type": "explicit_memory",
                "importance": "high",
                "session_id": session_id,
            },
            generate_embedding=False,
        )
        await store.add_memory(
            content="We have an ongoing running joke about stranger-danger cold starts.",
            metadata={
                "type": "long_term_memory",
                "significance": 0.9,
                "session_id": session_id,
            },
            generate_embedding=False,
        )
        store.add_conversation_message(
            role="user",
            content="Last time we were redesigning the memory boot flow.",
            session_id=session_id,
        )
        store.add_conversation_message(
            role="assistant",
            content="We agreed the agent should act like it knows the user already.",
            session_id=session_id,
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
                {"messages": [{"role": "user", "content": "Pick up where we left off."}]},
                session_id=session_id,
            )

        assert result == "ok"
        system_content = str(captured_payload["messages"][0]["content"])
        assert "[Session continuity brief]:" in system_content
        assert "persistent conversation by design" in system_content
        assert "User goes by Alice and prefers blunt, concise replies." in system_content
        assert "stranger-danger cold starts" in system_content
        assert "Last time we were redesigning the memory boot flow." in system_content
        print("✓ session continuity is injected before the user has to ask for it")
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)


async def test_session_continuity_excludes_other_session_memories() -> None:
    print("=== Test: session continuity excludes other-session memories ===")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.close()

    try:
        store, handler = _create_store_and_handler(tmp.name)
        session_id = "tg_7070"

        await store.add_memory(
            content="Current Telegram user likes concise replies.",
            metadata={
                "type": "explicit_memory",
                "importance": "high",
                "session_id": session_id,
            },
            generate_embedding=False,
        )
        await store.add_memory(
            content="Other session user is building crustyhub all weekend.",
            metadata={
                "type": "explicit_memory",
                "importance": "high",
                "session_id": "tg_other",
            },
            generate_embedding=False,
        )
        await store.add_memory(
            content="Unscoped long-term memory that should not bleed into Telegram continuity.",
            metadata={
                "type": "long_term_memory",
                "significance": 0.9,
            },
            generate_embedding=False,
        )

        context = handler._build_session_continuity_context(session_id=session_id)

        assert "Current Telegram user likes concise replies." in context
        assert "Other session user is building crustyhub all weekend." not in context
        assert (
            "Unscoped long-term memory that should not bleed into Telegram continuity."
            not in context
        )
        print("✓ continuity brief is scoped to the active messaging session")
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)


async def test_handle_injects_continuity_fallback_when_lookup_fails() -> None:
    print("=== Test: continuity fallback is injected when lookup fails ===")
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
                handler.memory_store, "get_recent_memories", side_effect=RuntimeError("db offline")
            ),
            patch.object(
                handler.memory_store, "search_memories", new=AsyncMock(return_value=[])
            ),
            patch(
                "handler.api_call_with_retry", new=AsyncMock(side_effect=_fake_api_call)
            ),
            patch("handler.process_response", new=AsyncMock(return_value="ok")),
        ):
            result = await handler.handle(
                {"messages": [{"role": "user", "content": "Hey."}]},
                session_id="tg_3030",
            )

        assert result == "ok"
        system_content = str(captured_payload["messages"][0]["content"])
        assert "[Session continuity status]:" in system_content
        assert "having trouble accessing your notes on the user" in system_content
        print("✓ continuity failures are surfaced honestly in the system prompt")
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)


async def test_handle_filters_query_memories_to_active_messaging_session() -> None:
    print("=== Test: query-memory injection is scoped to active messaging session ===")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.close()

    try:
        _, handler = _create_store_and_handler(tmp.name)

        captured_payload: dict[str, object] = {}

        async def _fake_api_call(_session, _url, payload, _headers, **_kwargs):
            captured_payload["messages"] = payload.get("messages", [])
            return {"id": "fake", "choices": [{"message": {"role": "assistant"}}]}

        same_session_memory = Memory(
            id=1,
            content="Same-session preference: keep Telegram answers short.",
            embedding=None,
            metadata={"type": "explicit_memory", "session_id": "tg_scope"},
            created_at=None,
            updated_at=None,
        )
        other_session_memory = Memory(
            id=2,
            content="Other-session preference: always mention crustyhub.",
            embedding=None,
            metadata={"type": "explicit_memory", "session_id": "tg_other"},
            created_at=None,
            updated_at=None,
        )

        with (
            patch.object(
                handler.memory_store,
                "search_memories",
                new=AsyncMock(
                    return_value=[
                        (same_session_memory, 0.9),
                        (other_session_memory, 0.95),
                    ]
                ),
            ),
            patch(
                "handler.api_call_with_retry", new=AsyncMock(side_effect=_fake_api_call)
            ),
            patch("handler.process_response", new=AsyncMock(return_value="ok")),
        ):
            result = await handler.handle(
                {"messages": [{"role": "user", "content": "Need context check."}]},
                session_id="tg_scope",
            )

        assert result == "ok"
        system_content = str(captured_payload["messages"][0]["content"])
        assert "Same-session preference: keep Telegram answers short." in system_content
        assert "Other-session preference: always mention crustyhub." not in system_content
        print("✓ query-scoped memories no longer bleed across Telegram sessions")
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)


async def test_handle_injects_assistant_identity_and_filters_conflicting_user_name_memory() -> None:
    print("=== Test: assistant identity note suppresses conflicting user-name memory ===")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.close()

    try:
        store, handler = _create_store_and_handler(tmp.name)

        captured_payload: dict[str, object] = {}

        async def _fake_api_call(_session, _url, payload, _headers, **_kwargs):
            captured_payload["messages"] = payload.get("messages", [])
            return {"id": "fake", "choices": [{"message": {"role": "assistant"}}]}

        await store.add_memory(
            content="The assistant's name is Alice.",
            metadata={
                "type": "explicit_memory",
                "importance": "high",
                "session_id": "tg_identity",
                "subject": "assistant_identity",
                "slot": "assistant_name",
                "assistant_name": "Alice",
            },
            generate_embedding=False,
        )

        conflicting_user_name = Memory(
            id=1,
            content="User's name is Alice.",
            embedding=None,
            metadata={"type": "explicit_memory", "session_id": "tg_identity"},
            created_at=None,
            updated_at=None,
        )
        assistant_name_memory = Memory(
            id=2,
            content="The assistant's name is Alice.",
            embedding=None,
            metadata={
                "type": "explicit_memory",
                "session_id": "tg_identity",
                "subject": "assistant_identity",
                "slot": "assistant_name",
                "assistant_name": "Alice",
            },
            created_at=None,
            updated_at=None,
        )

        with (
            patch.object(
                handler.memory_store,
                "search_memories",
                new=AsyncMock(
                    return_value=[
                        (conflicting_user_name, 0.95),
                        (assistant_name_memory, 0.9),
                    ]
                ),
            ),
            patch(
                "handler.api_call_with_retry", new=AsyncMock(side_effect=_fake_api_call)
            ),
            patch("handler.process_response", new=AsyncMock(return_value="ok")),
        ):
            result = await handler.handle(
                {"messages": [{"role": "user", "content": "what's your name?"}]},
                session_id="tg_identity",
            )

        assert result == "ok"
        system_content = str(captured_payload["messages"][0]["content"])
        assert "[Assistant identity note]:" in system_content
        assert "Your configured name in this session is Alice." in system_content
        assert "The assistant's name is Alice." in system_content
        assert "User's name is Alice." not in system_content
        print("✓ assistant identity note is injected and conflicting user-name memory is filtered")
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)


async def test_handle_infers_assistant_identity_from_recent_history() -> None:
    print("=== Test: assistant identity can be inferred from recent user history ===")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.close()

    try:
        store, handler = _create_store_and_handler(tmp.name)

        store.add_conversation_message(
            role="user",
            content="your name is alice, remember that",
            session_id="tg_identity_history",
        )

        captured_payload: dict[str, object] = {}

        async def _fake_api_call(_session, _url, payload, _headers, **_kwargs):
            captured_payload["messages"] = payload.get("messages", [])
            return {"id": "fake", "choices": [{"message": {"role": "assistant"}}]}

        conflicting_user_name = Memory(
            id=3,
            content="User's name is Alice.",
            embedding=None,
            metadata={"type": "explicit_memory", "session_id": "tg_identity_history"},
            created_at=None,
            updated_at=None,
        )

        with (
            patch.object(
                handler.memory_store,
                "search_memories",
                new=AsyncMock(return_value=[(conflicting_user_name, 0.95)]),
            ),
            patch(
                "handler.api_call_with_retry", new=AsyncMock(side_effect=_fake_api_call)
            ),
            patch("handler.process_response", new=AsyncMock(return_value="ok")),
        ):
            result = await handler.handle(
                {"messages": [{"role": "user", "content": "what's your name?"}]},
                session_id="tg_identity_history",
            )

        assert result == "ok"
        system_content = str(captured_payload["messages"][0]["content"])
        assert "[Assistant identity note]:" in system_content
        assert "Your configured name in this session is alice." in system_content
        assert "User's name is Alice." not in system_content
        print("✓ recent user history can rescue assistant identity before memory catches up")
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


async def test_handle_runs_memory_maintenance_before_closing_http_session() -> None:
    print("=== Test: memory maintenance runs before HTTP session closes ===")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.close()

    try:
        _, handler = _create_store_and_handler(tmp.name)

        observed: dict[str, object] = {}

        async def _fake_api_call(_session, _url, _payload, _headers, **_kwargs):
            return {"id": "fake", "choices": [{"message": {"role": "assistant"}}]}

        async def _assert_open_session(
            session,
            session_id,
            user_query,
            assistant_response,
        ):
            observed["session_closed"] = session.closed
            observed["session_id"] = session_id
            observed["assistant_response"] = assistant_response
            assert session.closed is False

        with (
            patch.object(
                handler.memory_store, "search_memories", new=AsyncMock(return_value=[])
            ),
            patch(
                "handler.api_call_with_retry", new=AsyncMock(side_effect=_fake_api_call)
            ),
            patch("handler.process_response", new=AsyncMock(return_value="ok")),
            patch.object(
                handler,
                "_run_memory_maintenance",
                new=AsyncMock(side_effect=_assert_open_session),
            ),
        ):
            result = await handler.handle(
                {"messages": [{"role": "user", "content": "check session timing"}]},
                session_id="tg_live_session",
            )

        assert result == "ok"
        assert observed["session_closed"] is False
        assert observed["session_id"] == "tg_live_session"
        assert observed["assistant_response"] == "ok"

        history = handler.memory_store.get_conversation_history(
            session_id="tg_live_session",
            limit=5,
        )
        assert [row["role"] for row in history[:2]] == ["assistant", "user"], history
        print("✓ post-response maintenance sees a live HTTP session")
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)


async def test_persistent_messaging_sessions_without_notes_still_avoid_thread_resets() -> None:
    print("=== Test: messaging continuity fallback preserves one-thread assumption ===")
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
                {"messages": [{"role": "user", "content": "yo"}]},
                session_id="imessage_+15550007777",
            )

        assert result == "ok"
        system_content = str(captured_payload["messages"][0]["content"])
        assert "[Session continuity status]:" in system_content
        assert "one persistent conversation by design" in system_content
        assert "fresh thread" in system_content
        print("✓ messaging sessions keep persistent-thread guidance even without notes")
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


async def test_handle_strips_internal_prompt_residue_from_response() -> None:
    print("=== Test: handle strips internal prompt residue from response ===")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.close()

    try:
        _, handler = _create_store_and_handler(tmp.name)

        leaked_response = (
            "first reply. "
            "Tool execution is mandatory. "
            "No more loops, no more narration — just the reaction."
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
                "handler.process_response", new=AsyncMock(return_value=leaked_response)
            ),
        ):
            result = await handler.handle(
                {"messages": [{"role": "user", "content": "Respond briefly"}]},
                session_id="tg_1001",
            )

        assert result == "first reply."

        history = handler.memory_store.get_conversation_history(
            session_id="tg_1001",
            limit=5,
        )
        assistant_rows = [row for row in history if row.get("role") == "assistant"]
        assert assistant_rows
        assert assistant_rows[0].get("content") == "first reply."
        print("✓ internal prompt residue is removed from visible responses and memory")
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)


async def test_handle_strips_pseudo_reaction_directives_from_response() -> None:
    print("=== Test: handle strips leaked pseudo reaction directives ===")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.close()

    try:
        _, handler = _create_store_and_handler(tmp.name)

        leaked_response = (
            "ayy let's go!\n\n"
            "*send_telegram_reaction: chat_id=880978583, message_id=112, reaction=love*"
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
                "handler.process_response", new=AsyncMock(return_value=leaked_response)
            ),
        ):
            result = await handler.handle(
                {"messages": [{"role": "user", "content": "Celebrate"}]},
                session_id="tg_1002",
            )

        assert result == "ayy let's go!"

        history = handler.memory_store.get_conversation_history(
            session_id="tg_1002",
            limit=5,
        )
        assistant_rows = [row for row in history if row.get("role") == "assistant"]
        assert assistant_rows
        assert assistant_rows[0].get("content") == "ayy let's go!"
        print("✓ pseudo reaction directives are removed from visible responses")
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)


async def main() -> int:
    await test_auto_memory_cadence_bounds()
    await test_dream_profile_and_consolidation_helpers()
    await test_cross_channel_recall_injects_requested_history()
    await test_cross_channel_recall_prefers_current_session_when_same_channel()
    await test_cross_channel_recall_is_injected_into_handle_context()
    await test_rolling_context_preserves_same_second_message_order()
    await test_imessage_session_injects_mobile_prompt_defaults()
    await test_imessage_handle_ids_are_injected_and_persisted()
    await test_telegram_reaction_targets_are_injected_and_persisted()
    await test_handle_injects_session_continuity_brief()
    await test_session_continuity_excludes_other_session_memories()
    await test_handle_injects_continuity_fallback_when_lookup_fails()
    await test_handle_filters_query_memories_to_active_messaging_session()
    await test_handle_injects_assistant_identity_and_filters_conflicting_user_name_memory()
    await test_handle_infers_assistant_identity_from_recent_history()
    await test_handle_injects_request_freshness_token()
    await test_handle_runs_memory_maintenance_before_closing_http_session()
    await test_persistent_messaging_sessions_without_notes_still_avoid_thread_resets()
    await test_handle_strips_delivery_directives_from_response()
    await test_handle_strips_internal_prompt_residue_from_response()
    await test_handle_strips_pseudo_reaction_directives_from_response()
    print("\nAll memory-maintenance tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
