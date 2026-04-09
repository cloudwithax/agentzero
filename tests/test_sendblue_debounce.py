#!/usr/bin/env python3
"""Tests for Sendblue inbound debounce behavior."""

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

from integrations import _extract_sendblue_typing_state, _queue_sendblue_pending_message
from integrations import _choose_sendblue_tapback_reaction
from integrations import _is_sendblue_invalid_typing_webhook_type_error
from integrations import _extract_sendblue_sender_number
from integrations import _extract_webhook_type_urls, process_imessage_and_reply
from integrations import _format_sendblue_message_content, send_imessage
from integrations import (
    _maybe_send_random_sendblue_tapback,
    get_imessages,
    monitor_sendblue_typing_webhook,
    send_reaction,
)
from integrations import _split_outbound_message_chunks
from integrations import _is_sendblue_message_unread
from integrations import handle_imessage, pending_prompt_phone_numbers
from integrations import (
    _make_sendblue_content_dedup_key,
    _remember_processed_sendblue_content,
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


def test_choose_sendblue_tapback_reaction_is_contextual() -> None:
    """Tapback selection should map common intents to relevant reactions."""
    with patch("integrations.random.choice", side_effect=lambda options: options[0]):
        assert _choose_sendblue_tapback_reaction("lol that is hilarious") == "laugh"
        assert (
            _choose_sendblue_tapback_reaction("Can you send that over?") == "question"
        )
        assert _choose_sendblue_tapback_reaction("thanks, that works great") == "like"
        assert _choose_sendblue_tapback_reaction("/start") is None
        assert _choose_sendblue_tapback_reaction("plain status update") is None


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


def test_detect_invalid_sendblue_typing_webhook_type_error() -> None:
    """The monitor should recognize unrecoverable typing_indicator webhook errors."""
    assert _is_sendblue_invalid_typing_webhook_type_error(
        {
            "status": 400,
            "data": {
                "status": "ERROR",
                "message": ("Invalid webhook type. Must be one of: receive, call_log"),
            },
        }
    )
    assert _is_sendblue_invalid_typing_webhook_type_error(
        {
            "success": False,
            "data": {
                "status": "ERROR",
                "message": "Invalid webhook type. Must be one of: receive, call_log",
            },
        }
    )
    assert not _is_sendblue_invalid_typing_webhook_type_error(
        {"status": 500, "error": "temporary error"}
    )


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
    """Explicit <message> blocks should collapse into one sanitized outbound chunk."""
    chunks = _split_outbound_message_chunks(
        "<message>first</message>\nrecalling memories...\n<message>second</message>"
    )
    assert chunks == ["first\n\nsecond"]


def test_split_outbound_message_chunks_ignores_typing_directives() -> None:
    """Typing directives should be removed from the single outbound chunk."""
    chunks = _split_outbound_message_chunks(
        '<message>first</message><typing seconds="1.5"/><message>second</message>'
    )
    assert chunks == ["first\n\nsecond"]


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


async def test_send_imessage_skips_empty_payload() -> None:
    """send_imessage should not call Sendblue when there is nothing to send."""

    class _FakeSession:
        def __init__(self):
            self.post_calls = 0

        def post(self, _url: str, json=None, headers=None):
            self.post_calls += 1
            raise AssertionError("send_imessage should skip empty payloads")

    previous_api_key = os.environ.get("SENDBLUE_API_KEY")
    previous_api_secret = os.environ.get("SENDBLUE_API_SECRET")
    previous_number = os.environ.get("SENDBLUE_NUMBER")

    os.environ["SENDBLUE_API_KEY"] = "test-key"
    os.environ["SENDBLUE_API_SECRET"] = "test-secret"
    os.environ["SENDBLUE_NUMBER"] = "+15550001111"

    fake_session = _FakeSession()
    try:
        result = await send_imessage(
            "+15551234567",
            "",
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

    assert result == {"success": True, "skipped": True, "reason": "empty payload"}
    assert fake_session.post_calls == 0


async def test_send_reaction_posts_expected_payload() -> None:
    """send_reaction should call Sendblue reactions endpoint with message handle."""

    class _FakeResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, _exc_type, _exc, _tb):
            return False

        async def json(self, content_type=None):
            return {"status": "OK", "message": "Reaction request sent"}

    class _FakeSession:
        def __init__(self):
            self.url = None
            self.payload = None

        def post(self, url: str, json=None, headers=None):
            self.url = url
            self.payload = json
            return _FakeResponse()

    previous_api_key = os.environ.get("SENDBLUE_API_KEY")
    previous_api_secret = os.environ.get("SENDBLUE_API_SECRET")
    previous_number = os.environ.get("SENDBLUE_NUMBER")

    os.environ["SENDBLUE_API_KEY"] = "test-key"
    os.environ["SENDBLUE_API_SECRET"] = "test-secret"
    os.environ["SENDBLUE_NUMBER"] = "+15550001111"

    fake_session = _FakeSession()
    try:
        result = await send_reaction(
            message_handle="abc-123",
            reaction="laugh",
            part_index=1,
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

    assert result.get("success") is True
    assert fake_session.url == "https://api.sendblue.com/api/send-reaction"
    assert fake_session.payload == {
        "from_number": "+15550001111",
        "message_handle": "abc-123",
        "reaction": "laugh",
        "part_index": 1,
    }


async def test_get_imessages_uses_created_at_gte_filter() -> None:
    """Startup replay should query Sendblue with the documented created_at_gte filter."""

    class _FakeResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, _exc_type, _exc, _tb):
            return False

        async def json(self):
            return {"status": "OK", "data": []}

    class _FakeSession:
        def __init__(self):
            self.params = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, _exc_type, _exc, _tb):
            return False

        def get(self, _url: str, params=None, headers=None):
            self.params = params
            return _FakeResponse()

    previous_api_key = os.environ.get("SENDBLUE_API_KEY")
    previous_api_secret = os.environ.get("SENDBLUE_API_SECRET")
    os.environ["SENDBLUE_API_KEY"] = "test-key"
    os.environ["SENDBLUE_API_SECRET"] = "test-secret"

    fake_session = _FakeSession()
    try:
        with patch("integrations.aiohttp.ClientSession", return_value=fake_session):
            result = await get_imessages(
                phone_number="+15551234567",
                last_check=datetime(2026, 4, 4, 12, 0, 0),
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

    assert result.get("success") is True
    assert fake_session.params is not None
    assert fake_session.params.get("number") == "+15551234567"
    assert fake_session.params.get("created_at_gte") == "2026-04-04T12:00:00"
    assert "after" not in fake_session.params


async def test_typing_webhook_monitor_disables_on_invalid_type() -> None:
    """Typing webhook monitor should stop retrying when Sendblue rejects the type."""
    previous_url = os.environ.get("SENDBLUE_TYPING_WEBHOOK_URL")
    previous_interval = os.environ.get("SENDBLUE_WEBHOOK_CHECK_INTERVAL")

    os.environ["SENDBLUE_TYPING_WEBHOOK_URL"] = "https://example.com/webhook/typing"
    os.environ["SENDBLUE_WEBHOOK_CHECK_INTERVAL"] = "10"

    try:
        with (
            patch(
                "integrations.ensure_sendblue_typing_webhook",
                new=AsyncMock(
                    return_value={
                        "success": False,
                        "status": 400,
                        "data": {
                            "status": "ERROR",
                            "message": "Invalid webhook type. Must be one of: receive",
                        },
                    }
                ),
            ) as ensure_mock,
            patch("integrations.asyncio.sleep", new=AsyncMock()) as sleep_mock,
        ):
            await monitor_sendblue_typing_webhook()
    finally:
        if previous_url is None:
            os.environ.pop("SENDBLUE_TYPING_WEBHOOK_URL", None)
        else:
            os.environ["SENDBLUE_TYPING_WEBHOOK_URL"] = previous_url

        if previous_interval is None:
            os.environ.pop("SENDBLUE_WEBHOOK_CHECK_INTERVAL", None)
        else:
            os.environ["SENDBLUE_WEBHOOK_CHECK_INTERVAL"] = previous_interval

    assert ensure_mock.await_count == 1
    assert sleep_mock.await_count == 0


async def test_auto_tapback_sends_when_relevant_and_random_passes() -> None:
    """Auto tapback should react when the message is relevant and random gate passes."""
    previous_enabled = os.environ.get("SENDBLUE_AUTO_TAPBACK_ENABLED")
    previous_probability = os.environ.get("SENDBLUE_TAPBACK_PROBABILITY")

    os.environ["SENDBLUE_AUTO_TAPBACK_ENABLED"] = "1"
    os.environ["SENDBLUE_TAPBACK_PROBABILITY"] = "1"

    try:
        with (
            patch(
                "integrations.send_reaction",
                new=AsyncMock(return_value={"success": True}),
            ) as send_reaction_mock,
            patch("integrations.random.choice", side_effect=lambda options: options[0]),
            patch("integrations.random.random", return_value=0.0),
        ):
            result = await _maybe_send_random_sendblue_tapback(
                "+15551234567",
                "lol this made my day",
                "handle-42",
                0,
                session=object(),
            )
    finally:
        if previous_enabled is None:
            os.environ.pop("SENDBLUE_AUTO_TAPBACK_ENABLED", None)
        else:
            os.environ["SENDBLUE_AUTO_TAPBACK_ENABLED"] = previous_enabled

        if previous_probability is None:
            os.environ.pop("SENDBLUE_TAPBACK_PROBABILITY", None)
        else:
            os.environ["SENDBLUE_TAPBACK_PROBABILITY"] = previous_probability

    assert result is not None
    assert result.get("success") is True
    assert send_reaction_mock.await_count == 1
    assert send_reaction_mock.await_args.kwargs.get("message_handle") == "handle-42"
    assert send_reaction_mock.await_args.kwargs.get("reaction") == "laugh"


async def test_send_imessage_sends_explicit_message_blocks_separately() -> None:
    """Explicit message blocks should be collapsed into one Sendblue API call."""

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
    assert len(fake_session.payloads) == 1
    outbound_payload = fake_session.payloads[0]
    assert "<message>" not in str(outbound_payload.get("content", ""))
    assert "chunk one" in str(outbound_payload.get("content", ""))
    assert "chunk two" in str(outbound_payload.get("content", ""))
    assert outbound_payload.get("media_url") == "https://img.example/final.png"


async def test_send_imessage_typing_directive_is_ignored() -> None:
    """Typing directives should be ignored in favor of one plain outbound send."""

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
        with patch(
            "integrations.send_typing_indicator",
            new=AsyncMock(return_value={"success": True}),
        ) as typing_mock:
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
    assert len(fake_session.payloads) == 1
    assert "<typing" not in str(fake_session.payloads[0].get("content", ""))
    assert typing_mock.await_count == 0


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


async def test_handle_imessage_passes_request_metadata_to_handler() -> None:
    """Inbound iMessage metadata should be forwarded to the handler."""

    captured: dict[str, object] = {}

    class _FakeHandler:
        def __init__(self) -> None:
            self.memory_store = MagicMock()

        async def handle(
            self,
            request,
            session_id=None,
            interim_response_callback=None,
            request_metadata=None,
        ):
            captured["request"] = request
            captured["session_id"] = session_id
            captured["request_metadata"] = request_metadata
            captured["interim_response_callback"] = interim_response_callback
            return "ok"

    handler = _FakeHandler()
    result = await handle_imessage(
        handler,
        "+15550124444",
        "hello there",
        message_handle="handle-abc",
        part_index=3,
    )

    assert result == "ok"
    assert captured["session_id"] == "imessage_+15550124444"
    assert captured["request_metadata"] == {
        "message_handle": "handle-abc",
        "part_index": 3,
    }


def test_sendblue_unread_detection() -> None:
    """Detect unread state from common Sendblue read-status fields."""
    assert _is_sendblue_message_unread({"is_read": False}) is True
    assert _is_sendblue_message_unread({"is_read": True}) is False
    assert _is_sendblue_message_unread({"read": "0"}) is True
    assert _is_sendblue_message_unread({"read": "1"}) is False
    assert _is_sendblue_message_unread({"status": "READ"}) is False
    assert _is_sendblue_message_unread({"status": "delivered"}) is True
    assert _is_sendblue_message_unread({"status": "unknown"}) is None


async def test_attachment_then_text_are_coalesced() -> None:
    """Attachment webhook followed by text should flush once with merged content."""
    pending_messages: dict[str, dict] = {}
    pending_lock = asyncio.Lock()
    calls: list[tuple[str, str, list[str]]] = []

    async def process_callback(
        sender: str,
        text: str,
        attachments: list[str],
        _message_handle: str | None,
        _part_index: int | None,
    ) -> None:
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

    async def process_callback(
        sender: str,
        text: str,
        attachments: list[str],
        _message_handle: str | None,
        _part_index: int | None,
    ) -> None:
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
        _sender: str,
        _text: str,
        _attachments: list[str],
        _message_handle: str | None,
        _part_index: int | None,
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
        message_handle=None,
        part_index=None,
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


def test_content_dedup_key_is_stable_and_sender_scoped() -> None:
    """Same sender+content must always produce the same key; different senders must differ."""
    k1 = _make_sendblue_content_dedup_key("+14155551234", "hello")
    k2 = _make_sendblue_content_dedup_key("+14155551234", "hello")
    assert k1 == k2, "identical inputs must yield identical keys"

    k3 = _make_sendblue_content_dedup_key("+14155551234", "world")
    assert k1 != k3, "different content must differ"

    k4 = _make_sendblue_content_dedup_key("+19999999999", "hello")
    assert k1 != k4, "different sender must differ"


def test_content_dedup_key_tolerates_empty_inputs() -> None:
    """Empty content or sender should not crash; empty+empty is still a valid (if useless) key."""
    k_empty = _make_sendblue_content_dedup_key("", "")
    assert isinstance(k_empty, str) and len(k_empty) > 0

    k_no_sender = _make_sendblue_content_dedup_key("", "hi")
    k_no_content = _make_sendblue_content_dedup_key("+1", "")
    assert k_no_sender != k_no_content


async def test_content_dedup_set_prevents_reprocessing() -> None:
    """_remember_processed_sendblue_content + TTL expiry work correctly."""
    seen: set[str] = set()
    key = _make_sendblue_content_dedup_key("+14155551234", "test msg")

    _remember_processed_sendblue_content(
        seen, "+14155551234", "test msg", dedup_ttl_seconds=30
    )
    assert key in seen, "key must be present after registration"

    _remember_processed_sendblue_content(seen, "", "", dedup_ttl_seconds=30)
    assert len(seen) == 1, "empty inputs must not add spurious entries"


async def test_webhook_endpoint_content_dedup_blocks_duplicate_payloads() -> None:
    """Two webhook POSTs with same sender+content but different/missing handles
    must result in only ONE call to process_imessage_and_reply."""
    from integrations import (
        _make_sendblue_content_dedup_key,
        _remember_processed_sendblue_content,
    )

    processed_content_keys: set[str] = set()
    ttl = 30

    key_a = _make_sendblue_content_dedup_key(
        "+14155551234", "truly evil twin, i like it"
    )
    _remember_processed_sendblue_content(
        processed_content_keys,
        "+14155551234",
        "truly evil twin, i like it",
        ttl,
    )
    assert key_a in processed_content_keys

    key_b = _make_sendblue_content_dedup_key(
        "+14155551234", "truly evil twin, i like it"
    )
    assert key_b == key_a
    assert key_b in processed_content_keys, (
        "same sender+content with a different fake handle would be caught by content dedup"
    )

    key_c = _make_sendblue_content_dedup_key("+14155551234", "different message here")
    assert key_c not in processed_content_keys, "different content must not match"


async def main() -> int:
    test_documented_typing_payload_parsing()
    test_typing_event_fallback_parsing()
    test_sender_number_extraction_priority()
    test_choose_sendblue_tapback_reaction_is_contextual()
    test_extract_webhook_type_urls_for_typing_indicator()
    test_extract_webhook_type_urls_for_receive()
    test_sendblue_message_formatting_prefers_carriage_returns()
    test_split_outbound_message_chunks_prefers_explicit_blocks()
    test_split_outbound_message_chunks_ignores_typing_directives()
    test_sendblue_unread_detection()
    await test_send_imessage_normalizes_content_before_send()
    await test_send_imessage_skips_empty_payload()
    await test_send_reaction_posts_expected_payload()
    await test_auto_tapback_sends_when_relevant_and_random_passes()
    await test_send_imessage_sends_explicit_message_blocks_separately()
    await test_send_imessage_typing_directive_is_ignored()
    await test_imessage_start_and_setprompt_flow()
    await test_imessage_memory_stats_command()
    await test_handle_imessage_passes_request_metadata_to_handler()
    await test_attachment_then_text_are_coalesced()
    await test_typing_event_extends_existing_debounce()
    await test_typing_event_without_pending_message_is_ignored()
    await test_typing_indicator_stops_as_soon_as_reply_is_sent()
    test_content_dedup_key_is_stable_and_sender_scoped()
    test_content_dedup_key_tolerates_empty_inputs()
    await test_content_dedup_set_prevents_reprocessing()
    await test_webhook_endpoint_content_dedup_blocks_duplicate_payloads()
    print("All Sendblue debounce tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
