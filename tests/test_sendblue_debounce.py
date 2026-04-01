#!/usr/bin/env python3
"""Tests for Sendblue inbound debounce behavior."""

import asyncio

from integrations import _extract_sendblue_typing_state, _queue_sendblue_pending_message
from integrations import _extract_sendblue_sender_number
from integrations import _extract_webhook_type_urls


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


async def main() -> int:
    test_documented_typing_payload_parsing()
    test_typing_event_fallback_parsing()
    test_sender_number_extraction_priority()
    test_extract_webhook_type_urls_for_typing_indicator()
    test_extract_webhook_type_urls_for_receive()
    await test_attachment_then_text_are_coalesced()
    await test_typing_event_extends_existing_debounce()
    await test_typing_event_without_pending_message_is_ignored()
    print("All Sendblue debounce tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
