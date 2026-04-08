#!/usr/bin/env python3
"""Regression tests for Telegram message delivery around the main agent loop."""

import asyncio
import json

from integrations import _process_telegram_message


class FakeBot:
    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.photos: list[dict] = []
        self.media_groups: list[dict] = []

    async def send_message(self, chat_id: int, text: str) -> None:
        self.messages.append({"chat_id": chat_id, "text": text})

    async def send_photo(self, chat_id: int, photo: str) -> None:
        self.photos.append({"chat_id": chat_id, "photo": photo})

    async def send_media_group(self, chat_id: int, media: list) -> None:
        self.media_groups.append({"chat_id": chat_id, "media": media})


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
        self.calls.append({"request": request, "session_id": session_id})
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
    )

    assert len(handler.calls) == 1
    assert handler.calls[0]["session_id"] == "tg_42"
    request_message = handler.calls[0]["request"]["messages"][0]
    assert request_message["role"] == "user"
    assert request_message["content"] == "Investigate the stuck reminder and publish the fix."
    assert [message["text"] for message in bot.messages] == [
        "Working on it.",
        "Finished the investigation.",
    ]
    assert bot.photos == [{"chat_id": 314, "photo": "https://example.com/result.png"}]
    assert not bot.media_groups
    print("  ✓ Passed")


async def main() -> None:
    await test_process_telegram_message_sends_interim_and_final_payloads()
    print("All Telegram message flow tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
