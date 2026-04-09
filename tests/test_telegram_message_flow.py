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

    async def send_photo(self, chat_id: int, photo: str) -> None:
        self.photos.append({"chat_id": chat_id, "photo": photo})

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
    assert [message["text"] for message in bot.messages] == [
        "Working on it.",
        "Finished the investigation.",
    ]
    assert bot.photos == [{"chat_id": 314, "photo": "https://example.com/result.png"}]
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
    assert result["emoji"] == "😂"
    assert len(bot.reactions) == 1
    reaction_entry = bot.reactions[0]
    assert reaction_entry["chat_id"] == 314
    assert reaction_entry["message_id"] == 2718
    assert len(reaction_entry["reaction"]) == 1
    assert getattr(reaction_entry["reaction"][0], "emoji", None) == "😂"
    print("  ✓ Passed")


async def main() -> None:
    await test_process_telegram_message_sends_interim_and_final_payloads()
    await test_send_telegram_reaction_uses_expected_target_and_emoji()
    print("All Telegram message flow tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
