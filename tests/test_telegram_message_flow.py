#!/usr/bin/env python3
"""Regression tests for Telegram message delivery around the main agent loop."""

import asyncio
import json

from integrations import _process_telegram_message, send_telegram_reaction


class FakeBot:
    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.photos: list[dict] = []
        self.media_groups: list[dict] = []
        self.chat_actions: list[dict] = []
        self.reactions: list[dict] = []

    async def send_message(self, chat_id: int, text: str) -> None:
        self.messages.append({"chat_id": chat_id, "text": text})

    async def send_photo(self, chat_id: int, photo: str, caption: str | None = None) -> None:
        entry: dict = {"chat_id": chat_id, "photo": photo}
        if caption is not None:
            entry["caption"] = caption
        self.photos.append(entry)

    async def send_media_group(self, chat_id: int, media: list) -> None:
        self.media_groups.append({"chat_id": chat_id, "media": media})

    async def send_chat_action(self, chat_id: int, action: str) -> None:
        self.chat_actions.append({"chat_id": chat_id, "action": action})

    async def set_message_reaction(self, chat_id: int, message_id: int, reaction) -> bool:
        self.reactions.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "reaction": reaction,
            }
        )
        return True


class FakeHandler:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def handle(
        self,
        request,
        session_id=None,
        interim_response_callback=None,
        response_chunk_callback=None,
        request_metadata=None,
    ):
        self.calls.append(
            {
                "request": request,
                "session_id": session_id,
                "request_metadata": request_metadata,
            }
        )
        if interim_response_callback is not None:
            await interim_response_callback("Working on it.")
        return json.dumps(
            {
                "text": "Finished the investigation.",
                "attachments": ["https://example.com/result.png"],
            }
        )


async def test_process_telegram_message_sends_interim_and_final_payloads() -> None:
    """Telegram flow should deliver interim text, final text, and attachments."""
    print("Test: Telegram message flow delivers interim and final payloads")

    bot = FakeBot()
    handler = FakeHandler()

    await _process_telegram_message(
        handler,
        user_id=42,
        chat_id=314,
        text="Investigate the stuck reminder and publish the fix.",
        attachment_urls=[],
        bot=bot,
        request_metadata={"telegram_chat_id": 314, "telegram_message_id": 2718},
    )

    assert len(handler.calls) == 1
    assert handler.calls[0]["session_id"] == "tg_42"
    assert handler.calls[0]["request_metadata"] == {
        "telegram_chat_id": 314,
        "telegram_message_id": 2718,
    }
    request_message = handler.calls[0]["request"]["messages"][0]
    assert request_message["role"] == "user"
    assert request_message["content"] == "Investigate the stuck reminder and publish the fix."
    assert bot.chat_actions
    assert bot.chat_actions[0] == {"chat_id": 314, "action": "typing"}
    # Interim text arrives as a plain message; final text becomes the photo caption.
    assert [message["text"] for message in bot.messages] == [
        "Working on it.",
    ]
    assert bot.photos == [
        {
            "chat_id": 314,
            "photo": "https://example.com/result.png",
            "caption": "Finished the investigation.",
        }
    ]
    assert not bot.media_groups
    print("  ✓ Passed")


async def test_send_telegram_reaction_uses_expected_target_and_emoji() -> None:
    """Telegram reactions should translate semantic names into emoji targets."""
    print("Test: Telegram reaction tool targets the inbound Telegram message")

    bot = FakeBot()
    result = await send_telegram_reaction(
        chat_id=314,
        message_id=2718,
        reaction="laugh",
        bot=bot,
    )

    assert result["success"] is True
    assert result["reaction"] == "laugh"
    assert result["emoji"] == "🤣"
    assert len(bot.reactions) == 1
    reaction_entry = bot.reactions[0]
    assert reaction_entry["chat_id"] == 314
    assert reaction_entry["message_id"] == 2718
    assert len(reaction_entry["reaction"]) == 1
    assert getattr(reaction_entry["reaction"][0], "emoji", None) == "🤣"
    print("  ✓ Passed")


async def test_multipart_message_blocks_are_sent_separately() -> None:
    """Multiple <message> blocks should each become a separate Telegram message."""
    print("Test: Multiple <message> blocks arrive as separate Telegram messages")

    bot = FakeBot()
    handler = FakeHandler()
    # Override handle to return a response with explicit <message> blocks.
    handler.handle = _make_multipart_handler(
        "<message>Part one of the answer.</message>"
        "<message>Part two with more detail.</message>"
        "<message>Part three wrapping up.</message>"
    )

    await _process_telegram_message(
        handler,
        user_id=42,
        chat_id=314,
        text="Give me a multi-part answer.",
        attachment_urls=[],
        bot=bot,
        request_metadata={"telegram_chat_id": 314, "telegram_message_id": 3001},
    )

    # Extract only the non-typing messages (skip interim messages if any).
    texts = [m["text"] for m in bot.messages]
    assert "Part one of the answer." in texts, f"Expected part one in {texts}"
    assert "Part two with more detail." in texts, f"Expected part two in {texts}"
    assert "Part three wrapping up." in texts, f"Expected part three in {texts}"
    print("  ✓ Passed")


async def test_long_message_is_split_at_telegram_limit() -> None:
    """A single message exceeding 4096 characters should be split into multiple sends."""
    print("Test: Long message (>4096 chars) is split into multiple Telegram messages")

    bot = FakeBot()
    handler = FakeHandler()
    # Build a message that's ~6000 chars — two paragraphs of ~3000 each.
    paragraph_a = "Alpha paragraph. " * 180  # ~3060 chars
    paragraph_b = "Beta paragraph. " * 180   # ~3060 chars
    long_text = paragraph_a.strip() + "\n\n" + paragraph_b.strip()
    handler.handle = _make_multipart_handler(long_text)

    await _process_telegram_message(
        handler,
        user_id=42,
        chat_id=314,
        text="Tell me something very long.",
        attachment_urls=[],
        bot=bot,
        request_metadata={"telegram_chat_id": 314, "telegram_message_id": 3002},
    )

    texts = [m["text"] for m in bot.messages]
    assert len(texts) >= 2, f"Expected >=2 messages but got {len(texts)}: {[len(t) for t in texts]}"
    for t in texts:
        assert len(t) <= 4096, f"Message chunk exceeds 4096 chars: {len(t)}"
    # Verify all content was delivered.
    combined = " ".join(texts)
    assert "Alpha paragraph" in combined
    assert "Beta paragraph" in combined
    print("  ✓ Passed")


async def test_multipart_blocks_with_attachments() -> None:
    """Last text part becomes the photo caption; earlier parts are plain messages."""
    print("Test: Photo caption uses last text part, earlier parts sent separately")

    bot = FakeBot()
    handler = FakeHandler()
    handler.handle = _make_multipart_handler(
        json.dumps({
            "text": "<message>First part.</message><message>Second part.</message>",
            "attachments": ["https://example.com/img1.png"],
        })
    )

    await _process_telegram_message(
        handler,
        user_id=42,
        chat_id=314,
        text="Show me results.",
        attachment_urls=[],
        bot=bot,
        request_metadata={"telegram_chat_id": 314, "telegram_message_id": 3003},
    )

    texts = [m["text"] for m in bot.messages]
    assert texts == ["First part."], f"Expected only first part as message, got {texts}"
    assert bot.photos == [
        {
            "chat_id": 314,
            "photo": "https://example.com/img1.png",
            "caption": "Second part.",
        }
    ]
    print("  ✓ Passed")


async def test_long_caption_falls_back_to_separate_message() -> None:
    """When the last text part exceeds 1024 chars it stays a message; photo has no caption."""
    print("Test: Long text (>1024 chars) sent as message, photo has no caption")

    bot = FakeBot()
    handler = FakeHandler()
    long_text = "A" * 1100  # exceeds 1024 caption limit
    handler.handle = _make_multipart_handler(
        json.dumps({
            "text": long_text,
            "attachments": ["https://example.com/big.png"],
        })
    )

    await _process_telegram_message(
        handler,
        user_id=42,
        chat_id=314,
        text="Show me.",
        attachment_urls=[],
        bot=bot,
        request_metadata={"telegram_chat_id": 314, "telegram_message_id": 3004},
    )

    texts = [m["text"] for m in bot.messages]
    assert len(texts) == 1, f"Expected 1 plain message, got {len(texts)}"
    assert texts[0] == long_text
    # Photo sent without caption.
    assert bot.photos == [{"chat_id": 314, "photo": "https://example.com/big.png"}]
    print("  ✓ Passed")


def _make_multipart_handler(response_text: str):
    """Return an async handle callable that returns *response_text* directly."""

    async def _handle(request, session_id=None, interim_response_callback=None,
                      response_chunk_callback=None, request_metadata=None):
        return response_text

    return _handle


async def main() -> None:
    failures = 0
    for test_fn in (
        test_process_telegram_message_sends_interim_and_final_payloads,
        test_send_telegram_reaction_uses_expected_target_and_emoji,
        test_multipart_message_blocks_are_sent_separately,
        test_long_message_is_split_at_telegram_limit,
        test_multipart_blocks_with_attachments,
        test_long_caption_falls_back_to_separate_message,
    ):
        try:
            await test_fn()
        except Exception as exc:
            print(f"  ✗ FAILED: {exc}")
            failures += 1
    if failures:
        print(f"\n{failures} test(s) failed")
    else:
        print("\nAll Telegram message flow tests passed!")
    raise SystemExit(failures)


if __name__ == "__main__":
    asyncio.run(main())
