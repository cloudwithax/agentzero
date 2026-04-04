#!/usr/bin/env python3
"""Tests for Sendblue inbound debounce behavior."""

import asyncio
import os
import tempfile
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from integrations import _extract_sendblue_typing_state, _queue_sendblue_pending_message
from integrations import _extract_sendblue_sender_number
from integrations import _extract_webhook_type_urls, process_imessage_and_reply
from integrations import _format_sendblue_message_content, send_imessage
from integrations import _split_outbound_message_chunks
from integrations import _is_sendblue_message_unread
from integrations import handle_imessage, pending_prompt_phone_numbers
from integrations import (
    _replay_sendblue_startup_backlog,
    _replay_telegram_pending_updates,
)
from memory import MemoryStore


def test_documented_typing_payload_parsing() -> None:
    """Parse Sendblue documented `is_typing` webhook payload values."""
    assert _extract_sendblue_typing_state({"is_typing": True}) is True
    assert _extract_sendblue_typing_state({"is_typing": False}) is False


def test_typing_event_fallback_parsing() -> None:
    """Parse fallback start/stop event names when boolean is absent."""
    assert _extract_sendblue_typing_state({"event": "typing_started"}) is True
    assert _extract_sendblue_typing_state({"event": "typing_stopped"}) is False
    assert _extract_sendblue_typing_state({"event": "unrelated"}) is None


def test_sender_number_extraction_priority() -> None:
    """Prefer `number` for contact identity and fallback gracefully."""
    assert (
        _extract_sendblue_sender_number(
            {"number": "+14155551234", "from_number": "+19175550000"}
        )
        == "+14155551234"
    )
    assert (
        _extract_sendblue_sender_number({"phone_number": "+14155550001"})
        == "+14155550001"
    )
    assert (
        _extract_sendblue_sender_number({"from_number": "+19175550000"})
        == "+19175550000"
    )
    assert _extract_sendblue_sender_number({}) == ""


def test_extract_webhook_type_urls_for_typing_indicator() -> None:
    """typing_indicator extraction should include legacy typing alias entries."""
    hooks = {
        "typing_indicator": ["https://example.com/webhook/typing"],
        "typing": [{"url": "https://example.com/webhooks/typing"}],
    }
    urls = _extract_webhook_type_urls(hooks, "typing_indicator")
    assert urls == {
        "https://example.com/webhook/typing",
        "https://example.com/webhooks/typing",
    }


def test_extract_webhook_type_urls_for_receive() -> None:
    """receive extraction should return only receive webhook URLs."""
    hooks = {"receive": [{"url": "https://example.com/webhook/receive"}]}
    urls = _extract_webhook_type_urls(hooks, "receive")
    assert urls == {"https://example.com/webhook/receive"}


def test_sendblue_message_formatting_prefers_carriage_returns() -> None:
    """Dense key/value text should be split and emitted with carriage returns."""
    previous_force_cr = os.environ.get("SENDBLUE_FORCE_CARRIAGE_RETURNS")
    os.environ["SENDBLUE_FORCE_CARRIAGE_RETURNS"] = "1"

    try:
        formatted = _format_sendblue_message_content(
            "name: nicholas a order #: 900251 date: 04/02/2026 7:30 pm "
            "items: bf potato griller drinks: 0 sauces: 0"
        )
    finally:
        if previous_force_cr is None:
            os.environ.pop("SENDBLUE_FORCE_CARRIAGE_RETURNS", None)
        else:
            os.environ["SENDBLUE_FORCE_CARRIAGE_RETURNS"] = previous_force_cr

    assert "\r\r" in formatted
    assert "\n" not in formatted
    assert "name: nicholas a" in formatted
    assert "\r\rorder #: 900251" in formatted
    assert "\r\rdate: 04/02/2026 7:30 pm" in formatted


def test_split_outbound_message_chunks_prefers_explicit_blocks() -> None:
    """Explicit <message> blocks should define outbound chunk boundaries."""
    chunks = _split_outbound_message_chunks(
        "<message>first</message>\nrecalling memories...\n<message>second</message>"
    )
    assert chunks == ["first", "second"]


def test_split_outbound_message_chunks_ignores_typing_directives() -> None:
    """Typing directives should not be emitted as visible message chunks."""
    chunks = _split_outbound_message_chunks(
        '<message>first</message><typing seconds="1.5"/><message>second</message>'
    )
    assert chunks == ["first", "second"]


async def test_send_imessage_normalizes_content_before_send() -> None:
    """send_imessage should always submit carriage-return formatted content."""

    class _FakeResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, _exc_type, _exc, _tb):
            return False

        async def json(self):
            return {"status": "QUEUED"}

    class _FakeSession:
        def __init__(self):
            self.payload = None

        def post(self, _url: str, json=None, headers=None):
            self.payload = json
            return _FakeResponse()

    previous_api_key = os.environ.get("SENDBLUE_API_KEY")
    previous_api_secret = os.environ.get("SENDBLUE_API_SECRET")
    previous_number = os.environ.get("SENDBLUE_NUMBER")
    previous_force_cr = os.environ.get("SENDBLUE_FORCE_CARRIAGE_RETURNS")

    os.environ["SENDBLUE_API_KEY"] = "test-key"
    os.environ["SENDBLUE_API_SECRET"] = "test-secret"
    os.environ["SENDBLUE_NUMBER"] = "+15550001111"
    os.environ["SENDBLUE_FORCE_CARRIAGE_RETURNS"] = "1"

    fake_session = _FakeSession()
    try:
        result = await send_imessage(
            "+15551234567",
            "line one\nline two",
            session=fake_session,
        )
    finally:
        if previous_api_key is None:
            os.environ.pop("SENDBLUE_API_KEY", None)
        else:
            os.environ["SENDBLUE_API_KEY"] = previous_api_key

        if previous_api_secret is None:
            os.environ.pop("SENDBLUE_API_SECRET", None)
        else:
            os.environ["SENDBLUE_API_SECRET"] = previous_api_secret

        if previous_number is None:
            os.environ.pop("SENDBLUE_NUMBER", None)
        else:
            os.environ["SENDBLUE_NUMBER"] = previous_number

        if previous_force_cr is None:
            os.environ.pop("SENDBLUE_FORCE_CARRIAGE_RETURNS", None)
        else:
            os.environ["SENDBLUE_FORCE_CARRIAGE_RETURNS"] = previous_force_cr

    assert result.get("success") is True
    assert fake_session.payload is not None
    assert fake_session.payload.get("content") == "line one\r\rline two"


async def test_send_imessage_sends_explicit_message_blocks_separately() -> None:
    """Explicit message blocks should fan out to one Sendblue API call per chunk."""

    class _FakeResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, _exc_type, _exc, _tb):
            return False

        async def json(self):
            return {"status": "QUEUED"}

    class _FakeSession:
        def __init__(self):
            self.payloads = []

        def post(self, _url: str, json=None, headers=None):
            self.payloads.append(json)
            return _FakeResponse()

    previous_api_key = os.environ.get("SENDBLUE_API_KEY")
    previous_api_secret = os.environ.get("SENDBLUE_API_SECRET")
    previous_number = os.environ.get("SENDBLUE_NUMBER")
    previous_force_cr = os.environ.get("SENDBLUE_FORCE_CARRIAGE_RETURNS")

    os.environ["SENDBLUE_API_KEY"] = "test-key"
    os.environ["SENDBLUE_API_SECRET"] = "test-secret"
    os.environ["SENDBLUE_NUMBER"] = "+15550001111"
    os.environ["SENDBLUE_FORCE_CARRIAGE_RETURNS"] = "1"

    fake_session = _FakeSession()
    try:
        result = await send_imessage(
            "+15551234567",
            "<message>chunk one</message>\nrecalling memories...\n<message>chunk two</message>",
            media_urls=["https://img.example/final.png"],
            session=fake_session,
        )
    finally:
        if previous_api_key is None:
            os.environ.pop("SENDBLUE_API_KEY", None)
        else:
            os.environ["SENDBLUE_API_KEY"] = previous_api_key

        if previous_api_secret is None:
            os.environ.pop("SENDBLUE_API_SECRET", None)
        else:
            os.environ["SENDBLUE_API_SECRET"] = previous_api_secret

        if previous_number is None:
            os.environ.pop("SENDBLUE_NUMBER", None)
        else:
            os.environ["SENDBLUE_NUMBER"] = previous_number

        if previous_force_cr is None:
            os.environ.pop("SENDBLUE_FORCE_CARRIAGE_RETURNS", None)
        else:
            os.environ["SENDBLUE_FORCE_CARRIAGE_RETURNS"] = previous_force_cr

    assert result.get("success") is True
    assert result.get("chunks_sent") == 2
    assert len(fake_session.payloads) == 2
    assert fake_session.payloads[0].get("content") == "chunk one"
    assert "media_url" not in fake_session.payloads[0]
    assert fake_session.payloads[1].get("content") == "chunk two"
    assert fake_session.payloads[1].get("media_url") == "https://img.example/final.png"


async def test_send_imessage_typing_directive_triggers_indicator() -> None:
    """Typing directives should emit typing indicators between message chunks."""

    class _FakeResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, _exc_type, _exc, _tb):
            return False

        async def json(self):
            return {"status": "QUEUED"}

    class _FakeSession:
        def __init__(self):
            self.payloads = []

        def post(self, _url: str, json=None, headers=None):
            self.payloads.append(json)
            return _FakeResponse()

    previous_api_key = os.environ.get("SENDBLUE_API_KEY")
    previous_api_secret = os.environ.get("SENDBLUE_API_SECRET")
    previous_number = os.environ.get("SENDBLUE_NUMBER")
    previous_force_cr = os.environ.get("SENDBLUE_FORCE_CARRIAGE_RETURNS")

    os.environ["SENDBLUE_API_KEY"] = "test-key"
    os.environ["SENDBLUE_API_SECRET"] = "test-secret"
    os.environ["SENDBLUE_NUMBER"] = "+15550001111"
    os.environ["SENDBLUE_FORCE_CARRIAGE_RETURNS"] = "1"

    fake_session = _FakeSession()
    try:
        with (
            patch(
                "integrations.send_typing_indicator",
                new=AsyncMock(return_value={"success": True}),
            ) as typing_mock,
            patch("integrations.asyncio.sleep", new=AsyncMock()) as sleep_mock,
        ):
            result = await send_imessage(
                "+15551234567",
                '<message>chunk one</message><typing seconds="1.4"/><message>chunk two</message>',
                session=fake_session,
            )
    finally:
        if previous_api_key is None:
            os.environ.pop("SENDBLUE_API_KEY", None)
        else:
            os.environ["SENDBLUE_API_KEY"] = previous_api_key

        if previous_api_secret is None:
            os.environ.pop("SENDBLUE_API_SECRET", None)
        else:
            os.environ["SENDBLUE_API_SECRET"] = previous_api_secret

        if previous_number is None:
            os.environ.pop("SENDBLUE_NUMBER", None)
        else:
            os.environ["SENDBLUE_NUMBER"] = previous_number

        if previous_force_cr is None:
            os.environ.pop("SENDBLUE_FORCE_CARRIAGE_RETURNS", None)
        else:
            os.environ["SENDBLUE_FORCE_CARRIAGE_RETURNS"] = previous_force_cr

    assert result.get("success") is True
    assert result.get("chunks_sent") == 2
    assert len(fake_session.payloads) == 2
    assert typing_mock.await_count >= 1
    assert sleep_mock.await_count >= 1


async def test_imessage_start_and_setprompt_flow() -> None:
    """iMessage commands should support /start and /setprompt prompt-update flow."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    phone_number = "+15551230000"
    pending_prompt_phone_numbers.pop(phone_number, None)

    try:
        memory_store = MemoryStore(db_path=db_path, api_key="")
        handler = MagicMock()
        handler.memory_store = memory_store
        handler.handle = AsyncMock(return_value="fallback")

        start_response = await handle_imessage(handler, phone_number, "/start")
        assert "AgentZero" in start_response
        assert "/setprompt" in start_response

        setprompt_response = await handle_imessage(
            handler,
            phone_number,
            "/setprompt",
        )
        assert "Please send your new system prompt" in setprompt_response
        assert phone_number in pending_prompt_phone_numbers

        pending_warning = await handle_imessage(handler, phone_number, "/start")
        assert "currently updating the system prompt" in pending_warning
        assert phone_number in pending_prompt_phone_numbers

        prompt_response = await handle_imessage(
            handler,
            phone_number,
            "You are concise and action-oriented.",
        )
        assert "System prompt updated successfully" in prompt_response
        assert phone_number not in pending_prompt_phone_numbers
        assert (
            memory_store.get_system_prompt() == "You are concise and action-oriented."
        )

        assert handler.handle.await_count == 0
    finally:
        pending_prompt_phone_numbers.pop(phone_number, None)
        if os.path.exists(db_path):
            os.unlink(db_path)


async def test_imessage_memory_stats_command() -> None:
    """/memorystats should expose session cadence and dream-profile status."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    phone_number = "+15559870000"
    session_id = f"imessage_{phone_number}"

    try:
        memory_store = MemoryStore(db_path=db_path, api_key="")
        handler = MagicMock()
        handler.memory_store = memory_store
        handler.handle = AsyncMock(return_value="fallback")

        for i in range(12):
            memory_store.add_conversation_message(
                role="user" if i % 2 == 0 else "assistant",
                content=f"msg-{i}",
                session_id=session_id,
            )

        await memory_store.add_memory(
            content="User prefers terse, high-signal answers.",
            metadata={
                "type": "auto_memory",
                "session_id": session_id,
                "message_index": 10,
            },
            generate_embedding=False,
        )

        stats_response = await handle_imessage(handler, phone_number, "/memorystats")
        assert "Memory stats:" in stats_response
        assert "Messages per memory:" in stats_response
        assert "target 10-20" in stats_response
        assert "Session messages: 12" in stats_response
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_sendblue_unread_detection() -> None:
    """Detect unread state from common Sendblue read-status fields."""
    assert _is_sendblue_message_unread({"is_read": False}) is True
    assert _is_sendblue_message_unread({"is_read": True}) is False
    assert _is_sendblue_message_unread({"read": "0"}) is True
    assert _is_sendblue_message_unread({"read": "1"}) is False
    assert _is_sendblue_message_unread({"status": "READ"}) is False
    assert _is_sendblue_message_unread({"status": "delivered"}) is True
    assert _is_sendblue_message_unread({"status": "unknown"}) is None


async def test_sendblue_startup_replay_processes_unread_messages() -> None:
    """Startup replay should process only inbound unread messages by default."""
    previous_replay_enabled = os.environ.get("SENDBLUE_STARTUP_REPLAY_ENABLED")
    previous_lookback = os.environ.get("SENDBLUE_STARTUP_LOOKBACK_SECONDS")
    previous_unread_only = os.environ.get("SENDBLUE_STARTUP_UNREAD_ONLY")

    os.environ["SENDBLUE_STARTUP_REPLAY_ENABLED"] = "1"
    os.environ["SENDBLUE_STARTUP_LOOKBACK_SECONDS"] = "600"
    os.environ["SENDBLUE_STARTUP_UNREAD_ONLY"] = "1"

    checkpoint_time = datetime(2026, 4, 3, 12, 0, 0)
    processed_handles: set[str] = set()

    messages = [
        {
            "message_handle": "skip-outbound",
            "direction": "outgoing",
            "number": "+15550000001",
            "content": "ignore",
            "is_read": False,
        },
        {
            "message_handle": "process-unread",
            "direction": "incoming",
            "number": "+15550000002",
            "content": "hello from offline",
            "is_read": False,
        },
        {
            "message_handle": "skip-read",
            "direction": "incoming",
            "number": "+15550000003",
            "content": "already read",
            "is_read": True,
        },
    ]

    process_reply_mock = AsyncMock()

    async def _fake_build_content(text: str, attachment_urls: list[str]):
        return text, attachment_urls

    try:
        with (
            patch(
                "integrations.get_imessages",
                new=AsyncMock(
                    return_value={"success": True, "data": {"messages": messages}}
                ),
            ),
            patch("integrations._build_imessage_user_content", new=_fake_build_content),
            patch("integrations.process_imessage_and_reply", new=process_reply_mock),
        ):
            replayed_count = await _replay_sendblue_startup_backlog(
                object(),
                checkpoint_time=checkpoint_time,
                processed_handles=processed_handles,
            )
    finally:
        if previous_replay_enabled is None:
            os.environ.pop("SENDBLUE_STARTUP_REPLAY_ENABLED", None)
        else:
            os.environ["SENDBLUE_STARTUP_REPLAY_ENABLED"] = previous_replay_enabled

        if previous_lookback is None:
            os.environ.pop("SENDBLUE_STARTUP_LOOKBACK_SECONDS", None)
        else:
            os.environ["SENDBLUE_STARTUP_LOOKBACK_SECONDS"] = previous_lookback

        if previous_unread_only is None:
            os.environ.pop("SENDBLUE_STARTUP_UNREAD_ONLY", None)
        else:
            os.environ["SENDBLUE_STARTUP_UNREAD_ONLY"] = previous_unread_only

    assert replayed_count == 1
    assert process_reply_mock.await_count == 1
    assert process_reply_mock.await_args.args[1] == "+15550000002"
    assert "process-unread" in processed_handles
    assert "skip-outbound" not in processed_handles
    assert "skip-read" not in processed_handles


async def test_telegram_startup_replay_processes_pending_updates() -> None:
    """Telegram startup replay should process queued updates and ack offset."""
    previous_enabled = os.environ.get("TELEGRAM_REPLAY_PENDING_UPDATES_ON_STARTUP")
    previous_batch_size = os.environ.get("TELEGRAM_PENDING_UPDATES_BATCH_SIZE")
    previous_max_batches = os.environ.get("TELEGRAM_PENDING_UPDATES_MAX_BATCHES")

    os.environ["TELEGRAM_REPLAY_PENDING_UPDATES_ON_STARTUP"] = "1"
    os.environ["TELEGRAM_PENDING_UPDATES_BATCH_SIZE"] = "100"
    os.environ["TELEGRAM_PENDING_UPDATES_MAX_BATCHES"] = "3"

    class _FakeUpdate:
        def __init__(self, update_id: int):
            self.update_id = update_id

    app = MagicMock()
    app.bot = MagicMock()
    app.bot.get_updates = AsyncMock(
        side_effect=[[_FakeUpdate(1), _FakeUpdate(2)], [], []]
    )
    app.process_update = AsyncMock()

    try:
        processed_count = await _replay_telegram_pending_updates(app)
    finally:
        if previous_enabled is None:
            os.environ.pop("TELEGRAM_REPLAY_PENDING_UPDATES_ON_STARTUP", None)
        else:
            os.environ["TELEGRAM_REPLAY_PENDING_UPDATES_ON_STARTUP"] = previous_enabled

        if previous_batch_size is None:
            os.environ.pop("TELEGRAM_PENDING_UPDATES_BATCH_SIZE", None)
        else:
            os.environ["TELEGRAM_PENDING_UPDATES_BATCH_SIZE"] = previous_batch_size

        if previous_max_batches is None:
            os.environ.pop("TELEGRAM_PENDING_UPDATES_MAX_BATCHES", None)
        else:
            os.environ["TELEGRAM_PENDING_UPDATES_MAX_BATCHES"] = previous_max_batches

    assert processed_count == 2
    assert app.process_update.await_count == 2
    assert app.bot.get_updates.await_count == 3
    assert app.bot.get_updates.await_args_list[-1].kwargs.get("offset") == 3


async def test_attachment_then_text_are_coalesced() -> None:
    """Attachment webhook followed by text should flush once with merged content."""
    pending_messages: dict[str, dict] = {}
    pending_lock = asyncio.Lock()
    calls: list[tuple[str, str, list[str]]] = []

    async def process_callback(sender: str, text: str, attachments: list[str]) -> None:
        calls.append((sender, text, attachments))

    sender = "+15551234567"
    debounce_seconds = 0.05

    await _queue_sendblue_pending_message(
        pending_messages,
        pending_lock,
        sender,
        "",
        ["https://img.example/photo1.png"],
        debounce_seconds,
        process_callback,
    )
    await asyncio.sleep(0.02)
    await _queue_sendblue_pending_message(
        pending_messages,
        pending_lock,
        sender,
        "What do you think?",
        [],
        debounce_seconds,
        process_callback,
    )

    await asyncio.sleep(0.08)

    assert len(calls) == 1
    called_sender, called_text, called_attachments = calls[0]
    assert called_sender == sender
    assert called_text == "What do you think?"
    assert called_attachments == ["https://img.example/photo1.png"]


async def test_typing_event_extends_existing_debounce() -> None:
    """Typing updates should extend debounce only when sender already has pending data."""
    pending_messages: dict[str, dict] = {}
    pending_lock = asyncio.Lock()
    calls: list[tuple[str, str, list[str]]] = []

    async def process_callback(sender: str, text: str, attachments: list[str]) -> None:
        calls.append((sender, text, attachments))

    sender = "+15557654321"
    debounce_seconds = 0.05

    await _queue_sendblue_pending_message(
        pending_messages,
        pending_lock,
        sender,
        "",
        ["https://img.example/pending.png"],
        debounce_seconds,
        process_callback,
    )
    await asyncio.sleep(0.03)

    queued = await _queue_sendblue_pending_message(
        pending_messages,
        pending_lock,
        sender,
        "",
        [],
        debounce_seconds,
        process_callback,
        create_if_missing=False,
    )
    assert queued is True

    await asyncio.sleep(0.03)
    assert len(calls) == 0

    await asyncio.sleep(0.04)
    assert len(calls) == 1


async def test_typing_event_without_pending_message_is_ignored() -> None:
    """Typing updates should not create a new pending payload on their own."""
    pending_messages: dict[str, dict] = {}
    pending_lock = asyncio.Lock()

    async def process_callback(
        _sender: str, _text: str, _attachments: list[str]
    ) -> None:
        raise AssertionError("Callback should not run")

    queued = await _queue_sendblue_pending_message(
        pending_messages,
        pending_lock,
        "+15550000000",
        "",
        [],
        0.05,
        process_callback,
        create_if_missing=False,
    )

    assert queued is False


async def test_typing_indicator_stops_as_soon_as_reply_is_sent() -> None:
    """No typing events should be sent after the final reply dispatch starts."""

    class _FakeClientSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, _exc_type, _exc, _tb):
            return False

    loop = asyncio.get_running_loop()
    typing_call_times: list[float] = []
    reply_send_times: list[float] = []

    async def _fake_send_read_receipt(_phone: str, session=None) -> dict:
        return {"success": True}

    async def _fake_send_typing_indicator(_phone: str, session=None) -> dict:
        typing_call_times.append(loop.time())
        await asyncio.sleep(0.01)
        return {"success": True}

    async def _fake_handle_imessage(
        _handler,
        _phone: str,
        _content,
        interim_response_callback=None,
    ) -> str:
        if interim_response_callback is not None:
            await interim_response_callback("Working on it.")
        await asyncio.sleep(0.08)
        return "Ready"

    async def _fake_send_imessage(
        _phone: str,
        _message: str,
        media_urls=None,
        session=None,
    ) -> dict:
        reply_send_times.append(loop.time())
        return {"success": True}

    with (
        patch("integrations.aiohttp.ClientSession", return_value=_FakeClientSession()),
        patch("integrations.send_read_receipt", new=_fake_send_read_receipt),
        patch("integrations.send_typing_indicator", new=_fake_send_typing_indicator),
        patch("integrations.handle_imessage", new=_fake_handle_imessage),
        patch("integrations.send_imessage", new=_fake_send_imessage),
    ):
        await process_imessage_and_reply(object(), "+15550123456", "Hello")

    assert len(reply_send_times) == 2
    assert reply_send_times[0] <= reply_send_times[1]
    assert len(typing_call_times) >= 1

    reply_sent_at = reply_send_times[-1]
    assert all(call_time <= reply_sent_at for call_time in typing_call_times)


async def main() -> int:
    test_documented_typing_payload_parsing()
    test_typing_event_fallback_parsing()
    test_sender_number_extraction_priority()
    test_extract_webhook_type_urls_for_typing_indicator()
    test_extract_webhook_type_urls_for_receive()
    test_sendblue_message_formatting_prefers_carriage_returns()
    test_split_outbound_message_chunks_prefers_explicit_blocks()
    test_split_outbound_message_chunks_ignores_typing_directives()
    test_sendblue_unread_detection()
    await test_send_imessage_normalizes_content_before_send()
    await test_send_imessage_sends_explicit_message_blocks_separately()
    await test_send_imessage_typing_directive_triggers_indicator()
    await test_imessage_start_and_setprompt_flow()
    await test_imessage_memory_stats_command()
    await test_sendblue_startup_replay_processes_unread_messages()
    await test_telegram_startup_replay_processes_pending_updates()
    await test_attachment_then_text_are_coalesced()
    await test_typing_event_extends_existing_debounce()
    await test_typing_event_without_pending_message_is_ignored()
    await test_typing_indicator_stops_as_soon_as_reply_is_sent()
    print("All Sendblue debounce tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
