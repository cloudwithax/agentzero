#!/usr/bin/env python3
"""Live in-process Telegram smoke test against the real AgentZero handler."""

import argparse
import asyncio
import base64
import json
import os
import subprocess
import tempfile
import time
from typing import Any
from unittest.mock import patch

from dotenv import load_dotenv

load_dotenv(".env")

import agentic_loop
import handler as handler_module
from integrations import _process_telegram_message
from main import initialize_agent


DEFAULT_FIRST_PROMPT = (
    "Remember this exact token for our Telegram conversation: cobalt-ferret-731. "
    "Reply only with: stored cobalt-ferret-731"
)
DEFAULT_SECOND_PROMPT = (
    "What token did I tell you in my previous Telegram message? "
    "Reply with the token only."
)
DEFAULT_IMAGE_PROMPT = (
    "The attached image contains a single lowercase word. "
    "Reply with that word only."
)


class FakeTelegramBot:
    """Capture outbound Telegram sends without calling the real Telegram API."""

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.photos: list[dict[str, Any]] = []
        self.media_groups: list[dict[str, Any]] = []
        self.chat_actions: list[dict[str, Any]] = []
        self.reactions: list[dict[str, Any]] = []

    async def send_message(self, chat_id: int, text: str) -> None:
        self.messages.append({"chat_id": chat_id, "text": text, "ts": time.time()})

    async def send_photo(self, chat_id: int, photo: str) -> None:
        self.photos.append({"chat_id": chat_id, "photo": photo, "ts": time.time()})

    async def send_media_group(self, chat_id: int, media: list[Any]) -> None:
        self.media_groups.append(
            {"chat_id": chat_id, "media": media, "ts": time.time()}
        )

    async def send_chat_action(self, chat_id: int, action: str) -> None:
        self.chat_actions.append(
            {"chat_id": chat_id, "action": action, "ts": time.time()}
        )

    async def set_message_reaction(
        self, chat_id: int, message_id: int, reaction: Any
    ) -> bool:
        self.reactions.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "reaction": reaction,
                "ts": time.time(),
            }
        )
        return True


def _extract_outbound_text(bot: FakeTelegramBot, start_index: int) -> str:
    """Return the last outbound Telegram text emitted after a turn starts."""
    new_messages = bot.messages[start_index:]
    if not new_messages:
        return ""
    return str(new_messages[-1].get("text", "")).strip()


def _message_content_has_image_block(content: Any) -> bool:
    """Check whether one user content payload contains at least one image block."""
    if not isinstance(content, list):
        return False
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "image_url":
            continue
        image_url = item.get("image_url")
        if isinstance(image_url, dict) and str(image_url.get("url", "")).strip():
            return True
    return False


def _message_content_to_text(content: Any) -> str:
    """Extract text from multimodal content payloads."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)

    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text" and isinstance(item.get("text"), str):
            parts.append(item["text"])
        elif isinstance(item.get("content"), str):
            parts.append(item["content"])
    return "\n".join(part for part in parts if part).strip()


def _build_test_image_data_url(word: str) -> str:
    """Generate a simple PNG with one centered word and return it as a data URL."""
    normalized_word = str(word or "").strip() or "telegram"
    with tempfile.TemporaryDirectory(prefix="agentzero-telegram-smoke-") as tmp_dir:
        image_path = os.path.join(tmp_dir, "telegram-smoke.png")
        command = [
            "magick",
            "-size",
            "900x320",
            "xc:white",
            "-gravity",
            "center",
            "-fill",
            "black",
            "-pointsize",
            "96",
            "-annotate",
            "0",
            normalized_word,
            image_path,
        ]
        subprocess.run(command, check=True, capture_output=True)
        with open(image_path, "rb") as handle:
            encoded = base64.b64encode(handle.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


async def run_smoke(
    user_id: int,
    chat_id: int,
    first_prompt: str,
    second_prompt: str,
    image_prompt: str,
    image_word: str,
) -> dict[str, Any]:
    """Run two Telegram text turns plus one Telegram image turn through the real handler."""
    handler, _acp_agent = initialize_agent()
    await handler.start_reminder_scheduler()

    session_id = f"tg_{user_id}"
    bot = FakeTelegramBot()
    request_payloads: list[dict[str, Any]] = []
    loop_response_snapshots: list[dict[str, Any]] = []
    tool_rounds: list[list[str]] = []
    started_at = time.time()

    original_handler_api_call = handler_module.api_call_with_retry
    original_loop_api_call = agentic_loop.api_call_with_retry
    original_execute_tool_calls = agentic_loop.execute_tool_calls

    async def _capture_handler_api_call(*args, **kwargs):
        if len(args) >= 3 and isinstance(args[2], dict):
            payload = args[2]
            request_payloads.append(
                {
                    "source": "handler",
                    "model": payload.get("model"),
                    "messages": payload.get("messages", []),
                    "ts": time.time(),
                }
            )
        return await original_handler_api_call(*args, **kwargs)

    async def _capture_loop_api_call(*args, **kwargs):
        if len(args) >= 3 and isinstance(args[2], dict):
            payload = args[2]
            request_payloads.append(
                {
                    "source": "agentic_loop",
                    "model": payload.get("model"),
                    "messages": payload.get("messages", []),
                    "ts": time.time(),
                }
            )
        response_data = await original_loop_api_call(*args, **kwargs)
        choices = response_data.get("choices") or []
        if choices:
            message = choices[0].get("message", {}) or {}
            content_text = agentic_loop._message_content_to_text(
                message.get("content", "")
            )
            loop_response_snapshots.append(
                {
                    "content": content_text,
                    "has_tool_calls": bool(message.get("tool_calls")),
                    "tool_call_names": [
                        tool_call.get("function", {}).get("name", "")
                        for tool_call in (message.get("tool_calls") or [])
                    ],
                }
            )
        return response_data

    async def _capture_execute_tool_calls(message, allowed_tool_names=None):
        tool_rounds.append(
            [
                tool_call.get("function", {}).get("name", "")
                for tool_call in (message.get("tool_calls") or [])
            ]
        )
        return await original_execute_tool_calls(
            message,
            allowed_tool_names=allowed_tool_names,
        )

    first_text = ""
    second_text = ""
    image_text = ""
    image_data_url = _build_test_image_data_url(image_word)

    try:
        with (
            patch(
                "integrations._maybe_send_random_telegram_reaction",
                return_value=None,
            ),
            patch(
                "handler.api_call_with_retry",
                side_effect=_capture_handler_api_call,
            ),
            patch(
                "agentic_loop.api_call_with_retry",
                side_effect=_capture_loop_api_call,
            ),
            patch(
                "agentic_loop.execute_tool_calls",
                side_effect=_capture_execute_tool_calls,
            ),
        ):
            first_msg_index = len(bot.messages)
            await _process_telegram_message(
                handler,
                user_id=user_id,
                chat_id=chat_id,
                text=first_prompt,
                attachment_urls=[],
                bot=bot,
                request_metadata={
                    "telegram_chat_id": chat_id,
                    "telegram_message_id": 1001,
                },
            )
            first_text = _extract_outbound_text(bot, first_msg_index)

            second_msg_index = len(bot.messages)
            await _process_telegram_message(
                handler,
                user_id=user_id,
                chat_id=chat_id,
                text=second_prompt,
                attachment_urls=[],
                bot=bot,
                request_metadata={
                    "telegram_chat_id": chat_id,
                    "telegram_message_id": 1002,
                },
            )
            second_text = _extract_outbound_text(bot, second_msg_index)

            image_msg_index = len(bot.messages)
            await _process_telegram_message(
                handler,
                user_id=user_id,
                chat_id=chat_id,
                text=image_prompt,
                attachment_urls=[image_data_url],
                bot=bot,
                request_metadata={
                    "telegram_chat_id": chat_id,
                    "telegram_message_id": 1003,
                },
            )
            image_text = _extract_outbound_text(bot, image_msg_index)
    finally:
        await handler.reminder_scheduler.stop()

    conversation_history = handler.memory_store.get_conversation_history(
        session_id=session_id,
        limit=10,
    )

    second_turn_payload = None
    image_turn_payload = None
    for payload in request_payloads:
        messages = payload.get("messages", [])
        if not isinstance(messages, list) or not messages:
            continue
        last_message = messages[-1]
        if not isinstance(last_message, dict):
            continue
        last_content = last_message.get("content")
        last_text = _message_content_to_text(last_content)
        if second_prompt in last_text:
            second_turn_payload = payload
        if image_prompt in last_text or _message_content_has_image_block(last_content):
            image_turn_payload = payload

    second_turn_messages = second_turn_payload.get("messages", []) if second_turn_payload else []
    second_turn_history_visible = any(
        isinstance(message, dict)
        and message.get("role") == "user"
        and first_prompt in _message_content_to_text(message.get("content"))
        for message in second_turn_messages[:-1]
    ) and any(
        isinstance(message, dict)
        and message.get("role") == "assistant"
        and "stored cobalt-ferret-731" in _message_content_to_text(message.get("content"))
        for message in second_turn_messages[:-1]
    )

    image_turn_messages = image_turn_payload.get("messages", []) if image_turn_payload else []
    image_user_content = image_turn_messages[-1].get("content") if image_turn_messages else None
    image_payload_has_blocks = _message_content_has_image_block(image_user_content)

    return {
        "success": bool(second_turn_history_visible and image_payload_has_blocks),
        "elapsed_seconds": round(time.time() - started_at, 2),
        "session_id": session_id,
        "first_reply": first_text,
        "second_reply": second_text,
        "image_reply": image_text,
        "second_turn_history_visible": second_turn_history_visible,
        "image_payload_has_blocks": image_payload_has_blocks,
        "request_payload_count": len(request_payloads),
        "tool_rounds": tool_rounds,
        "loop_response_snapshots": loop_response_snapshots,
        "conversation_history": conversation_history,
        "bot_messages": bot.messages,
        "bot_photos": bot.photos,
        "bot_media_groups": bot.media_groups,
        "bot_chat_actions": bot.chat_actions,
        "second_turn_payload_preview": (
            second_turn_messages[-4:] if second_turn_messages else []
        ),
        "image_turn_payload_preview": (
            image_turn_messages[-2:] if image_turn_messages else []
        ),
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run a live Telegram smoke test.")
    default_user_id = int(time.time()) % 1000000000
    default_chat_id = default_user_id + 1000
    parser.add_argument("--user-id", type=int, default=default_user_id)
    parser.add_argument("--chat-id", type=int, default=default_chat_id)
    parser.add_argument("--first-prompt", default=DEFAULT_FIRST_PROMPT)
    parser.add_argument("--second-prompt", default=DEFAULT_SECOND_PROMPT)
    parser.add_argument("--image-prompt", default=DEFAULT_IMAGE_PROMPT)
    parser.add_argument("--image-word", default="telegram")
    args = parser.parse_args()

    result = await run_smoke(
        user_id=args.user_id,
        chat_id=args.chat_id,
        first_prompt=args.first_prompt,
        second_prompt=args.second_prompt,
        image_prompt=args.image_prompt,
        image_word=args.image_word,
    )
    print(json.dumps(result, indent=2))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
