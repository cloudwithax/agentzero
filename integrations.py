"""Bot integrations for iMessage and Telegram."""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Optional

import aiohttp
from aiohttp import web

from handler import AgentHandler

logger = logging.getLogger(__name__)

# Track users who are in the process of setting a system prompt
pending_prompt_users: Dict[int, bool] = {}
pending_telegram_media_groups: Dict[str, dict[str, Any]] = {}
telegram_media_group_lock = asyncio.Lock()

# Valid multimodal models for image inputs.
MULTIMODAL_MODEL_IDS = {
    "black-forest-labs/flux.1-kontext-dev",
    "google/paligemma",
    "meta/llama-3.2-11b-vision-instruct",
    "meta/llama-3.2-90b-vision-instruct",
    "meta/llama-4-maverick-17b-128e-instruct",
    "meta/llama-4-scout-17b-16e-instruct",
    "microsoft/phi-3.5-vision-instruct",
    "moonshotai/kimi-k2.5",
    "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
    "qwen/qwen3.5-397b-a17b",
}

# Optional telegram imports
try:
    from telegram import Update
    from telegram.ext import (  # noqa: F401
        Application,
        CommandHandler,
        MessageHandler,
        filters,
        ContextTypes,
    )

    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False


# iMessage (Sendblue) Integration


def _to_bool(value) -> bool:
    """Parse common webhook boolean encodings safely."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _normalize_attachment_urls(value: Any) -> list[str]:
    """Normalize mixed attachment payloads into a de-duplicated URL list."""
    if value is None:
        return []

    raw_values = value if isinstance(value, list) else [value]
    urls: list[str] = []
    seen = set()

    for item in raw_values:
        candidate = ""
        if isinstance(item, str):
            candidate = item.strip()
        elif isinstance(item, dict):
            for key in ("url", "media_url", "image_url", "src"):
                if isinstance(item.get(key), str):
                    candidate = item[key].strip()
                    break

        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        urls.append(candidate)

    return urls


def _extract_sendblue_attachment_urls(payload: dict[str, Any]) -> list[str]:
    """Extract Sendblue attachment URLs from known payload fields."""
    attachments: list[str] = []
    for key in ("media_url", "media_urls", "attachments"):
        attachments.extend(_normalize_attachment_urls(payload.get(key)))
    return _normalize_attachment_urls(attachments)


def _extract_sendblue_sender_number(payload: dict[str, Any]) -> str:
    """Extract the contact number from Sendblue webhook payloads."""
    value = (
        payload.get("number")
        or payload.get("phone_number")
        or payload.get("from_number")
    )
    if not isinstance(value, str):
        return ""
    return value.strip()


def _extract_sendblue_typing_state(payload: dict[str, Any]) -> bool | None:
    """Return typing state if payload appears to be a typing-indicator event."""
    if "is_typing" in payload:
        return _to_bool(payload.get("is_typing"))

    if "isTyping" in payload:
        return _to_bool(payload.get("isTyping"))

    for key in ("event", "type", "status", "action"):
        value = payload.get(key)
        if not isinstance(value, str):
            continue
        normalized = value.strip().lower()
        if normalized in {
            "typing",
            "typing_start",
            "typing_started",
            "started_typing",
            "typing-indicator",
            "typing_indicator",
        }:
            return True
        if normalized in {
            "typing_stop",
            "typing_stopped",
            "stopped_typing",
        }:
            return False

    if "typing" in payload:
        return _to_bool(payload.get("typing"))

    return None


def _schedule_sendblue_pending_flush_locked(
    pending_sendblue_messages: dict[str, dict[str, Any]],
    sender_number: str,
    pending_sendblue_lock: asyncio.Lock,
    debounce_seconds: float,
    process_callback: Callable[[str, str, list[str]], Awaitable[None]],
) -> None:
    """Reset and schedule delayed processing for a queued sender payload."""
    pending_payload = pending_sendblue_messages.get(sender_number)
    if pending_payload is None:
        return

    existing_task = pending_payload.get("task")
    if isinstance(existing_task, asyncio.Task):
        existing_task.cancel()

    async def _flush_after_delay() -> None:
        try:
            await asyncio.sleep(debounce_seconds)
            async with pending_sendblue_lock:
                payload = pending_sendblue_messages.pop(sender_number, None)

            if not payload:
                return

            text_parts = payload.get("text_parts", [])
            text = "\n".join(
                part for part in text_parts if isinstance(part, str) and part.strip()
            ).strip()
            attachments = _normalize_attachment_urls(payload.get("attachments", []))

            if not text and not attachments:
                return

            await process_callback(sender_number, text, attachments)
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.error("Error flushing Sendblue debounce queue: %s", e)

    pending_payload["task"] = asyncio.create_task(_flush_after_delay())


async def _queue_sendblue_pending_message(
    pending_sendblue_messages: dict[str, dict[str, Any]],
    pending_sendblue_lock: asyncio.Lock,
    sender_number: str,
    text: str,
    attachments: list[str],
    debounce_seconds: float,
    process_callback: Callable[[str, str, list[str]], Awaitable[None]],
    *,
    create_if_missing: bool = True,
) -> bool:
    """Queue or update a sender payload, then debounce-send it to the agent."""
    normalized_text = (text or "").strip()
    normalized_attachments = _normalize_attachment_urls(attachments)

    async with pending_sendblue_lock:
        payload = pending_sendblue_messages.get(sender_number)
        if payload is None:
            if not create_if_missing:
                return False
            payload = {"text_parts": [], "attachments": [], "task": None}
            pending_sendblue_messages[sender_number] = payload

        if normalized_text:
            payload["text_parts"].append(normalized_text)
        if normalized_attachments:
            payload["attachments"].extend(normalized_attachments)

        _schedule_sendblue_pending_flush_locked(
            pending_sendblue_messages,
            sender_number,
            pending_sendblue_lock,
            debounce_seconds,
            process_callback,
        )

    return True


async def _has_pending_sendblue_message(
    pending_sendblue_messages: dict[str, dict[str, Any]],
    pending_sendblue_lock: asyncio.Lock,
    sender_number: str,
) -> bool:
    """Check whether a sender currently has a queued Sendblue payload."""
    async with pending_sendblue_lock:
        return sender_number in pending_sendblue_messages


def _model_supports_multimodal() -> bool:
    model_id = os.environ.get("MODEL_ID", "").strip().lower()
    return model_id in MULTIMODAL_MODEL_IDS


def _build_user_message_content(
    text: str, attachment_urls: list[str]
) -> str | list[dict[str, Any]]:
    """Build model input content, using image blocks for multimodal models."""
    normalized_text = (text or "").strip()
    normalized_attachments = _normalize_attachment_urls(attachment_urls)

    if not normalized_attachments:
        return normalized_text

    if _model_supports_multimodal():
        content: list[dict[str, Any]] = [
            {"type": "text", "text": normalized_text or "Analyze these images."}
        ]
        for url in normalized_attachments:
            content.append({"type": "image_url", "image_url": {"url": url}})
        return content

    attachment_lines = "\n".join(f"- {url}" for url in normalized_attachments)
    if normalized_text:
        return f"{normalized_text}\n\n[Image attachments]\n{attachment_lines}"
    return f"[Image attachments]\n{attachment_lines}"


def _extract_agent_response_payload(response: str) -> tuple[str, list[str]]:
    """Parse optional structured attachment payload from assistant text output."""
    if not response:
        return "", []

    try:
        parsed = json.loads(response)
    except Exception:
        return response, []

    if not isinstance(parsed, dict):
        return response, []

    attachments = []
    for key in ("attachments", "images", "media_urls", "media_url"):
        attachments.extend(_normalize_attachment_urls(parsed.get(key)))

    text = ""
    for key in ("text", "message", "content", "reply"):
        value = parsed.get(key)
        if isinstance(value, str):
            text = value
            break

    if text or attachments:
        return text, _normalize_attachment_urls(attachments)
    return response, []


async def _telegram_file_url(bot: Any, file_id: str) -> str | None:
    """Build direct Telegram file download URL for a file_id."""
    if not file_id:
        return None
    try:
        tg_file = await bot.get_file(file_id)
        if not tg_file or not tg_file.file_path:
            return None
        return f"https://api.telegram.org/file/bot{bot.token}/{tg_file.file_path}"
    except Exception as e:
        logger.warning("Failed to resolve Telegram file URL: %s", e)
        return None


async def _extract_telegram_attachment_urls(message: Any, bot: Any) -> list[str]:
    """Extract image attachment URLs from a Telegram message."""
    urls: list[str] = []

    if message.photo:
        photo_url = await _telegram_file_url(bot, message.photo[-1].file_id)
        if photo_url:
            urls.append(photo_url)

    if (
        message.document
        and isinstance(message.document.mime_type, str)
        and message.document.mime_type.startswith("image/")
    ):
        doc_url = await _telegram_file_url(bot, message.document.file_id)
        if doc_url:
            urls.append(doc_url)

    return _normalize_attachment_urls(urls)


async def _send_telegram_response(
    bot: Any, chat_id: int, text: str, attachment_urls: list[str]
) -> None:
    """Send text and attachments to Telegram, batching images when possible."""
    normalized_text = (text or "").strip()
    attachments = _normalize_attachment_urls(attachment_urls)

    if normalized_text:
        await bot.send_message(chat_id=chat_id, text=normalized_text)

    if not attachments:
        if not normalized_text:
            await bot.send_message(chat_id=chat_id, text="No response.")
        return

    if len(attachments) == 1:
        await bot.send_photo(chat_id=chat_id, photo=attachments[0])
        return

    # Import lazily so optional telegram dependency stays type-safe.
    from telegram import InputMediaPhoto as _InputMediaPhoto

    for i in range(0, len(attachments), 10):
        chunk = attachments[i : i + 10]
        media_group = [_InputMediaPhoto(media=url) for url in chunk]
        await bot.send_media_group(chat_id=chat_id, media=media_group)


async def _process_telegram_message(
    handler: AgentHandler,
    user_id: int,
    chat_id: int,
    text: str,
    attachment_urls: list[str],
    bot: Any,
) -> None:
    """Process a Telegram user message and send text/media reply."""
    user_content = _build_user_message_content(text, attachment_urls)
    response = await handler.handle(
        {"messages": [{"role": "user", "content": user_content}]},
        session_id=f"tg_{user_id}",
    )
    response_text, response_attachments = _extract_agent_response_payload(response)
    await _send_telegram_response(bot, chat_id, response_text, response_attachments)


async def _queue_telegram_media_group(
    handler: AgentHandler, update: "Update", context: "ContextTypes.DEFAULT_TYPE"
) -> None:
    """Collect media-group updates briefly, then process them as one request."""
    if (
        update.message is None
        or update.effective_user is None
        or update.effective_chat is None
    ):
        return

    media_group_id = update.message.media_group_id
    if not media_group_id:
        return

    group_key = f"{update.effective_chat.id}:{media_group_id}"

    async with telegram_media_group_lock:
        group = pending_telegram_media_groups.get(group_key)
        if group is None:
            group = {
                "messages": [],
                "user_id": update.effective_user.id,
                "chat_id": update.effective_chat.id,
                "task": None,
            }
            pending_telegram_media_groups[group_key] = group

        group["messages"].append(update.message)

        if group["task"] is None:

            async def _flush_group() -> None:
                await asyncio.sleep(1.0)
                async with telegram_media_group_lock:
                    payload = pending_telegram_media_groups.pop(group_key, None)

                if not payload:
                    return

                text = ""
                attachments: list[str] = []
                for msg in payload["messages"]:
                    if not text:
                        text = (msg.caption or msg.text or "").strip()
                    attachments.extend(
                        await _extract_telegram_attachment_urls(msg, context.bot)
                    )

                if not text and not attachments:
                    return

                await context.bot.send_chat_action(
                    chat_id=payload["chat_id"], action="typing"
                )
                await _process_telegram_message(
                    handler,
                    user_id=payload["user_id"],
                    chat_id=payload["chat_id"],
                    text=text,
                    attachment_urls=attachments,
                    bot=context.bot,
                )

            group["task"] = asyncio.create_task(_flush_group())


async def send_imessage(
    phone_number: str,
    message: str,
    media_urls: Optional[list[str]] = None,
    session: Optional[aiohttp.ClientSession] = None,
) -> dict:
    """Send an iMessage via Sendblue API."""
    api_key = os.environ.get("SENDBLUE_API_KEY")
    api_secret = os.environ.get("SENDBLUE_API_SECRET")
    from_number = os.environ.get("SENDBLUE_NUMBER")
    if not api_key or not api_secret or not from_number:
        return {"success": False, "error": "Credentials or SENDBLUE_NUMBER missing"}

    headers = {
        "Content-Type": "application/json",
        "sb-api-key-id": api_key,
        "sb-api-secret-key": api_secret,
    }
    payload = {
        "number": phone_number,
        "from_number": from_number,
        "content": message,
        "send_style": "regular",
    }
    payload: dict[str, Any] = payload
    normalized_media_urls = _normalize_attachment_urls(media_urls)
    if normalized_media_urls:
        payload["media_url"] = (
            normalized_media_urls
            if len(normalized_media_urls) > 1
            else normalized_media_urls[0]
        )

    close_session = False
    if session is None:
        session = aiohttp.ClientSession()
        close_session = True
    try:
        async with session.post(
            "https://api.sendblue.co/api/send-message", json=payload, headers=headers
        ) as resp:
            data = await resp.json()
            return {"success": resp.status == 200, "data": data}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if close_session:
            await session.close()


async def send_typing_indicator(
    phone_number: str, session: Optional[aiohttp.ClientSession] = None
) -> dict:
    """Send a typing indicator via Sendblue API."""
    api_key = os.environ.get("SENDBLUE_API_KEY")
    api_secret = os.environ.get("SENDBLUE_API_SECRET")
    from_number = os.environ.get("SENDBLUE_NUMBER")
    if not api_key or not api_secret:
        return {"success": False, "error": "Credentials missing"}

    headers = {
        "Content-Type": "application/json",
        "sb-api-key-id": api_key,
        "sb-api-secret-key": api_secret,
    }
    payload = {"number": phone_number}
    if from_number:
        payload["from_number"] = from_number

    close_session = False
    if session is None:
        session = aiohttp.ClientSession()
        close_session = True
    try:
        async with session.post(
            "https://api.sendblue.co/api/send-typing-indicator",
            json=payload,
            headers=headers,
        ) as resp:
            data = await resp.json()
            return {"success": resp.status == 200, "data": data}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if close_session:
            await session.close()


async def send_read_receipt(
    phone_number: str, session: Optional[aiohttp.ClientSession] = None
) -> dict:
    """Send a read receipt via Sendblue API."""
    api_key = os.environ.get("SENDBLUE_API_KEY")
    api_secret = os.environ.get("SENDBLUE_API_SECRET")
    from_number = os.environ.get("SENDBLUE_NUMBER")
    if not api_key or not api_secret:
        return {"success": False, "error": "Credentials missing"}

    headers = {
        "Content-Type": "application/json",
        "sb-api-key-id": api_key,
        "sb-api-secret-key": api_secret,
    }
    payload = {"number": phone_number}
    if from_number:
        payload["from_number"] = from_number

    close_session = False
    if session is None:
        session = aiohttp.ClientSession()
        close_session = True
    try:
        async with session.post(
            "https://api.sendblue.co/api/mark-read", json=payload, headers=headers
        ) as resp:
            data = await resp.json()
            return {"success": resp.status == 200, "data": data}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if close_session:
            await session.close()


async def get_imessages(
    phone_number: Optional[str] = None, last_check: Optional[datetime] = None
) -> dict:
    """Get messages from Sendblue API."""
    api_key = os.environ.get("SENDBLUE_API_KEY")
    api_secret = os.environ.get("SENDBLUE_API_SECRET")
    if not api_key or not api_secret:
        return {"success": False, "error": "Credentials missing"}
    headers = {"sb-api-key-id": api_key, "sb-api-secret-key": api_secret}
    params = {}
    if phone_number:
        params["number"] = phone_number
    if last_check:
        params["after"] = last_check.isoformat()

    async with aiohttp.ClientSession() as session:
        async with session.get(
            "https://api.sendblue.co/api/v2/messages", params=params, headers=headers
        ) as resp:
            data = await resp.json()
            return {"success": resp.status == 200, "data": data}


async def list_sendblue_webhooks(
    session: Optional[aiohttp.ClientSession] = None,
) -> dict:
    """List webhook configuration from Sendblue account settings."""
    api_key = os.environ.get("SENDBLUE_API_KEY")
    api_secret = os.environ.get("SENDBLUE_API_SECRET")
    if not api_key or not api_secret:
        return {"success": False, "error": "Credentials missing"}

    headers = {
        "Content-Type": "application/json",
        "sb-api-key-id": api_key,
        "sb-api-secret-key": api_secret,
    }

    close_session = False
    if session is None:
        session = aiohttp.ClientSession()
        close_session = True

    try:
        async with session.get(
            "https://api.sendblue.co/api/account/webhooks", headers=headers
        ) as resp:
            data = await resp.json()
            return {"success": resp.status == 200, "data": data, "status": resp.status}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if close_session:
            await session.close()


async def add_sendblue_receive_webhook(
    webhook_url: str, session: Optional[aiohttp.ClientSession] = None
) -> dict:
    """Append a receive webhook using Sendblue POST /account/webhooks."""
    api_key = os.environ.get("SENDBLUE_API_KEY")
    api_secret = os.environ.get("SENDBLUE_API_SECRET")
    if not api_key or not api_secret:
        return {"success": False, "error": "Credentials missing"}

    headers = {
        "Content-Type": "application/json",
        "sb-api-key-id": api_key,
        "sb-api-secret-key": api_secret,
    }
    payload = {"webhooks": [webhook_url], "type": "receive"}

    close_session = False
    if session is None:
        session = aiohttp.ClientSession()
        close_session = True

    try:
        async with session.post(
            "https://api.sendblue.co/api/account/webhooks",
            json=payload,
            headers=headers,
        ) as resp:
            data = await resp.json()
            return {"success": resp.status == 200, "data": data, "status": resp.status}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if close_session:
            await session.close()


async def add_sendblue_typing_webhook(
    webhook_url: str, session: Optional[aiohttp.ClientSession] = None
) -> dict:
    """Append a typing_indicator webhook using Sendblue POST /account/webhooks."""
    api_key = os.environ.get("SENDBLUE_API_KEY")
    api_secret = os.environ.get("SENDBLUE_API_SECRET")
    if not api_key or not api_secret:
        return {"success": False, "error": "Credentials missing"}

    headers = {
        "Content-Type": "application/json",
        "sb-api-key-id": api_key,
        "sb-api-secret-key": api_secret,
    }
    payload = {"webhooks": [webhook_url], "type": "typing_indicator"}

    close_session = False
    if session is None:
        session = aiohttp.ClientSession()
        close_session = True

    try:
        async with session.post(
            "https://api.sendblue.co/api/account/webhooks",
            json=payload,
            headers=headers,
        ) as resp:
            data = await resp.json()
            return {"success": resp.status == 200, "data": data, "status": resp.status}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if close_session:
            await session.close()


def _extract_webhook_urls(values) -> set:
    """Extract URL strings from mixed webhook entries (string/object)."""
    urls = set()
    if not isinstance(values, list):
        return urls
    for entry in values:
        if isinstance(entry, str):
            urls.add(entry.strip())
        elif isinstance(entry, dict):
            url = entry.get("url")
            if isinstance(url, str):
                urls.add(url.strip())
    return urls


def _extract_webhook_type_urls(hooks: Any, webhook_type: str) -> set:
    """Extract webhook URLs for a specific Sendblue webhook type."""
    if not isinstance(hooks, dict):
        return set()

    urls = _extract_webhook_urls(hooks.get(webhook_type, []))

    # Backward-compatible alias support for typing webhooks.
    if webhook_type == "typing_indicator":
        urls.update(_extract_webhook_urls(hooks.get("typing", [])))

    return urls


async def ensure_sendblue_receive_webhook(
    webhook_url: str, session: Optional[aiohttp.ClientSession] = None
) -> dict:
    """Ensure the receive webhook exists; add it via POST if missing."""
    target = webhook_url.strip()
    if not target:
        return {"success": False, "error": "Empty webhook URL"}

    close_session = False
    if session is None:
        session = aiohttp.ClientSession()
        close_session = True

    try:
        listed = await list_sendblue_webhooks(session=session)
        if not listed.get("success"):
            return listed

        data = listed.get("data", {})
        hooks = data.get("webhooks", {}) if isinstance(data, dict) else {}
        receive_urls = _extract_webhook_type_urls(hooks, "receive")
        if target in receive_urls:
            return {"success": True, "already_present": True}

        added = await add_sendblue_receive_webhook(target, session=session)
        if added.get("success"):
            logger.warning("Re-added missing Sendblue receive webhook: %s", target)
        return added
    finally:
        if close_session:
            await session.close()


async def ensure_sendblue_typing_webhook(
    webhook_url: str, session: Optional[aiohttp.ClientSession] = None
) -> dict:
    """Ensure the typing_indicator webhook exists; add it via POST if missing."""
    target = webhook_url.strip()
    if not target:
        return {"success": False, "error": "Empty webhook URL"}

    close_session = False
    if session is None:
        session = aiohttp.ClientSession()
        close_session = True

    try:
        listed = await list_sendblue_webhooks(session=session)
        if not listed.get("success"):
            return listed

        data = listed.get("data", {})
        hooks = data.get("webhooks", {}) if isinstance(data, dict) else {}
        typing_urls = _extract_webhook_type_urls(hooks, "typing_indicator")
        if target in typing_urls:
            return {"success": True, "already_present": True}

        added = await add_sendblue_typing_webhook(target, session=session)
        if added.get("success"):
            logger.warning("Re-added missing Sendblue typing webhook: %s", target)
        return added
    finally:
        if close_session:
            await session.close()


async def monitor_sendblue_receive_webhook() -> None:
    """Periodically re-add missing receive webhook using append-only API."""
    webhook_url = os.environ.get("SENDBLUE_RECEIVE_WEBHOOK_URL", "").strip()
    if not webhook_url:
        return

    interval = int(os.environ.get("SENDBLUE_WEBHOOK_CHECK_INTERVAL", "60"))
    interval = max(interval, 10)

    logger.info("Sendblue receive webhook monitor enabled (interval: %ss)", interval)
    while True:
        try:
            result = await ensure_sendblue_receive_webhook(webhook_url)
            if not result.get("success"):
                logger.error(
                    "Sendblue receive webhook check failed: %s",
                    result.get("error") or result,
                )
        except Exception as e:
            logger.error("Sendblue receive webhook monitor error: %s", e)
        await asyncio.sleep(interval)


async def monitor_sendblue_typing_webhook(webhook_url: Optional[str] = None) -> None:
    """Periodically re-add missing typing_indicator webhook using append-only API."""
    target_url = webhook_url or os.environ.get("SENDBLUE_TYPING_WEBHOOK_URL", "")
    target_url = target_url.strip()
    if not target_url:
        return

    interval = int(os.environ.get("SENDBLUE_WEBHOOK_CHECK_INTERVAL", "60"))
    interval = max(interval, 10)

    logger.info("Sendblue typing webhook monitor enabled (interval: %ss)", interval)
    while True:
        try:
            result = await ensure_sendblue_typing_webhook(target_url)
            if not result.get("success"):
                logger.error(
                    "Sendblue typing webhook check failed: %s",
                    result.get("error") or result,
                )
        except Exception as e:
            logger.error("Sendblue typing webhook monitor error: %s", e)
        await asyncio.sleep(interval)


async def handle_imessage(
    handler: AgentHandler, phone_number: str, user_content: str | list[dict[str, Any]]
) -> str:
    """Process an incoming iMessage."""
    session_id = f"imessage_{phone_number}"
    text = user_content if isinstance(user_content, str) else ""

    # Check for /clear command
    if text.strip().lower() == "/clear":
        try:
            deleted_count = handler.memory_store.clear_conversation_history(session_id)
            return f"✅ Conversation cleared! Started fresh. ({deleted_count} messages removed)"
        except Exception as e:
            logger.error(f"Failed to clear conversation: {e}")
            return f"❌ Failed to clear conversation: {str(e)}"

    try:
        return await handler.handle(
            {"messages": [{"role": "user", "content": user_content}]},
            session_id=session_id,
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        return "Sorry, an error occurred."


async def process_imessage_and_reply(
    handler: AgentHandler,
    phone_number: str,
    user_content: str | list[dict[str, Any]],
) -> None:
    """Send read receipt/typing signals, run agent, then send response."""

    async with aiohttp.ClientSession() as session:
        read_res = await send_read_receipt(phone_number, session=session)
        if not read_res.get("success"):
            logger.warning("Failed to send read receipt: %s", read_res)

        async def _typing_loop():
            while True:
                typing_res = await send_typing_indicator(phone_number, session=session)
                if not typing_res.get("success"):
                    logger.debug("Typing indicator failed: %s", typing_res)
                await asyncio.sleep(4)

        typing_task = asyncio.create_task(_typing_loop())
        try:
            resp = await handle_imessage(handler, phone_number, user_content)
        finally:
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass

        reply_text, reply_attachments = _extract_agent_response_payload(resp)
        send_res = await send_imessage(
            phone_number,
            reply_text,
            media_urls=reply_attachments,
            session=session,
        )
        if not send_res.get("success"):
            logger.error("Failed to send Sendblue reply: %s", send_res)


async def start_sendblue_webhook_server(handler: AgentHandler, port: int):
    """Start a webhook server for Sendblue."""
    app = web.Application()
    own_number = os.environ.get("SENDBLUE_NUMBER")
    processed_handles = set()
    dedup_ttl_seconds = 60
    pending_sendblue_messages: dict[str, dict[str, Any]] = {}
    pending_sendblue_lock = asyncio.Lock()

    attachment_debounce_seconds = 2.0
    try:
        attachment_debounce_seconds = max(
            0.1,
            float(os.environ.get("SENDBLUE_ATTACHMENT_DEBOUNCE_SECONDS", "2.0")),
        )
    except (TypeError, ValueError):
        logger.warning(
            "Invalid SENDBLUE_ATTACHMENT_DEBOUNCE_SECONDS, defaulting to %.1fs",
            attachment_debounce_seconds,
        )

    typing_debounce_seconds = attachment_debounce_seconds
    try:
        typing_debounce_seconds = max(
            0.1,
            float(
                os.environ.get(
                    "SENDBLUE_TYPING_DEBOUNCE_SECONDS",
                    str(attachment_debounce_seconds),
                )
            ),
        )
    except (TypeError, ValueError):
        logger.warning(
            "Invalid SENDBLUE_TYPING_DEBOUNCE_SECONDS, defaulting to %.1fs",
            typing_debounce_seconds,
        )

    async def _process_queued_sender_payload(
        sender_number: str, text: str, attachments: list[str]
    ) -> None:
        user_content = _build_user_message_content(text, attachments)
        await process_imessage_and_reply(handler, sender_number, user_content)

    async def webhook_endpoint(request):
        try:
            try:
                data = await request.json()
            except Exception:
                # Some webhook providers may send form-encoded payloads.
                form_data = await request.post()
                data = dict(form_data)

            sender_number = _extract_sendblue_sender_number(data)
            content = data.get("content") or data.get("message") or data.get("text", "")
            attachments = _extract_sendblue_attachment_urls(data)
            direction = str(data.get("direction", "")).lower()
            is_outbound = _to_bool(data.get("is_outbound"))
            message_handle = data.get("message_handle")

            if message_handle:
                if message_handle in processed_handles:
                    logger.info(
                        "Ignoring duplicate Sendblue webhook: %s", message_handle
                    )
                    return web.Response(status=200, text="OK")
                processed_handles.add(message_handle)
                asyncio.get_running_loop().call_later(
                    dedup_ttl_seconds,
                    lambda: processed_handles.discard(message_handle),
                )

            # Process asynchronously so we can quickly return 200 OK
            if sender_number:
                # Ignore outbound webhook events and self-originated messages.
                if (
                    direction == "outgoing"
                    or is_outbound
                    or (own_number and sender_number == own_number)
                ):
                    return web.Response(status=200, text="OK")

                typing_state = _extract_sendblue_typing_state(data)
                if typing_state is not None:
                    if typing_state:
                        queued = await _queue_sendblue_pending_message(
                            pending_sendblue_messages,
                            pending_sendblue_lock,
                            sender_number,
                            "",
                            [],
                            typing_debounce_seconds,
                            _process_queued_sender_payload,
                            create_if_missing=False,
                        )
                        if queued:
                            logger.debug(
                                "Extended Sendblue debounce for typing event from %s",
                                sender_number,
                            )
                    return web.Response(status=200, text="OK")

                if not content.strip() and not attachments:
                    logger.info(
                        "Ignoring webhook event with empty content and no attachments: %s",
                        data,
                    )
                    return web.Response(status=200, text="OK")

                has_pending = await _has_pending_sendblue_message(
                    pending_sendblue_messages,
                    pending_sendblue_lock,
                    sender_number,
                )
                should_debounce = bool(
                    (attachments and not content.strip()) or has_pending
                )

                if should_debounce:
                    await _queue_sendblue_pending_message(
                        pending_sendblue_messages,
                        pending_sendblue_lock,
                        sender_number,
                        content,
                        attachments,
                        attachment_debounce_seconds,
                        _process_queued_sender_payload,
                    )
                    return web.Response(status=200, text="OK")

                user_content = _build_user_message_content(content, attachments)

                async def _process_and_reply():
                    try:
                        await process_imessage_and_reply(
                            handler, sender_number, user_content
                        )
                    except Exception as e:
                        logger.error(f"Error processing webhook message: {e}")

                task = asyncio.create_task(_process_and_reply())

                def _log_task_error(done_task: asyncio.Task):
                    if done_task.cancelled():
                        return
                    exc = done_task.exception()
                    if exc:
                        logger.error(f"Webhook background task failed: {exc}")

                task.add_done_callback(_log_task_error)
            else:
                logger.info("Ignoring webhook event with no sender number: %s", data)

            return web.Response(status=200, text="OK")
        except Exception as e:
            # Never return non-2xx to provider webhooks or they may disable callbacks.
            logger.error(f"Webhook error: {e}")
            return web.Response(status=200, text="OK")

    async def webhook_healthcheck(_request):
        """Allow provider verification probes that may use GET/HEAD."""
        return web.Response(status=200, text="OK")

    app.router.add_post("/webhook", webhook_endpoint)
    app.router.add_post("/webhook/receive", webhook_endpoint)
    app.router.add_post("/webhook/typing", webhook_endpoint)
    app.router.add_post("/webhooks/typing", webhook_endpoint)
    app.router.add_post("/webhook/typing-indicator", webhook_endpoint)
    app.router.add_post("/", webhook_endpoint)
    app.router.add_get("/webhook", webhook_healthcheck)
    app.router.add_get("/webhook/receive", webhook_healthcheck)
    app.router.add_get("/webhook/typing", webhook_healthcheck)
    app.router.add_get("/webhooks/typing", webhook_healthcheck)
    app.router.add_get("/webhook/typing-indicator", webhook_healthcheck)
    app.router.add_get("/", webhook_healthcheck)
    app.router.add_head("/webhook", webhook_healthcheck)
    app.router.add_head("/webhook/receive", webhook_healthcheck)
    app.router.add_head("/webhook/typing", webhook_healthcheck)
    app.router.add_head("/webhooks/typing", webhook_healthcheck)
    app.router.add_head("/webhook/typing-indicator", webhook_healthcheck)
    app.router.add_head("/", webhook_healthcheck)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logger.info(f"iMessage webhook server started on port {port}")

    # Keep the task alive
    while True:
        await asyncio.sleep(3600)


async def start_sendblue_bot(handler: AgentHandler):
    """Start Sendblue bot either via webhook or polling based on env config."""
    monitor_tasks: list[asyncio.Task] = []
    receive_webhook_url = os.environ.get("SENDBLUE_RECEIVE_WEBHOOK_URL", "").strip()
    typing_webhook_url = os.environ.get("SENDBLUE_TYPING_WEBHOOK_URL", "").strip()

    if receive_webhook_url:
        monitor_tasks.append(asyncio.create_task(monitor_sendblue_receive_webhook()))

    # If a dedicated typing webhook URL is not provided, reuse receive webhook URL.
    typing_target = typing_webhook_url or receive_webhook_url
    if typing_target:
        monitor_tasks.append(
            asyncio.create_task(monitor_sendblue_typing_webhook(typing_target))
        )

    webhook_port = os.environ.get("SENDBLUE_WEBHOOK_PORT")
    if webhook_port:
        try:
            await start_sendblue_webhook_server(handler, int(webhook_port))
        finally:
            for monitor_task in monitor_tasks:
                monitor_task.cancel()
        return

    # Polling fallback
    interval = int(os.environ.get("SENDBLUE_POLL_INTERVAL", "10"))
    last_check = datetime.utcnow()
    logger.info(f"iMessage bot started (interval: {interval}s)")
    while True:
        try:
            poll_started_at = datetime.utcnow()
            res = await get_imessages(last_check=last_check)
            if not res["success"]:
                logger.error(
                    f"Failed to get messages: {res.get('error', 'Unknown error')}"
                )
            else:
                messages = (
                    res["data"]
                    if isinstance(res["data"], list)
                    else res["data"].get("messages", [])
                )
                for msg in messages:
                    if msg.get("direction") == "outgoing" or msg.get("is_outbound"):
                        continue
                    num = msg.get("from_number") or msg.get("number")
                    if not num:
                        logger.warning(f"Skipping message with no sender number: {msg}")
                        continue
                    content = (
                        msg.get("content") or msg.get("message") or msg.get("text", "")
                    )
                    attachments = _extract_sendblue_attachment_urls(msg)
                    if not content.strip() and not attachments:
                        continue
                    user_content = _build_user_message_content(content, attachments)
                    await process_imessage_and_reply(handler, num, user_content)
            # Move checkpoint to poll start time to avoid gaps where messages
            # arrive while the previous batch is being processed.
            last_check = poll_started_at
        except Exception as e:
            logger.error(f"Polling error: {e}")
        await asyncio.sleep(interval)


# Telegram Integration


async def telegram_start(update: "Update", context: "ContextTypes.DEFAULT_TYPE"):
    """Handle /start command."""
    if (
        update.message is None
        or update.effective_user is None
        or update.effective_chat is None
    ):
        return
    await update.message.reply_text(
        f"Hello {update.effective_user.first_name}! I'm AgentZero."
    )


async def telegram_setprompt(update: "Update", context: "ContextTypes.DEFAULT_TYPE"):
    """Handle /setprompt command - initiate system prompt change."""
    if update.message is None or update.effective_user is None:
        return
    user_id = update.effective_user.id
    pending_prompt_users[user_id] = True
    await update.message.reply_text(
        "Please send your new system prompt in the next message. "
        "It will replace the current system prompt and be used for all future conversations."
    )


async def telegram_clear(
    handler: AgentHandler, update: "Update", context: "ContextTypes.DEFAULT_TYPE"
):
    """Handle /clear command - clear conversation history."""
    if update.message is None or update.effective_user is None:
        return
    user_id = update.effective_user.id
    session_id = f"tg_{user_id}"

    try:
        deleted_count = handler.memory_store.clear_conversation_history(session_id)
        await update.message.reply_text(
            f"✅ Conversation cleared! Started fresh. ({deleted_count} messages removed)"
        )
    except Exception as e:
        logger.error(f"Failed to clear conversation: {e}")
        await update.message.reply_text(f"❌ Failed to clear conversation: {str(e)}")


async def telegram_handle_msg(
    handler: AgentHandler, update: "Update", context: "ContextTypes.DEFAULT_TYPE"
):
    """Handle incoming Telegram messages."""
    if (
        update.message is None
        or update.effective_user is None
        or update.effective_chat is None
    ):
        return
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    # Check if user is in prompt-setting mode
    if user_id in pending_prompt_users:
        # Remove from pending state
        del pending_prompt_users[user_id]

        # Get the new prompt from the message
        new_prompt = (update.message.text or "").strip()

        if not new_prompt:
            await update.message.reply_text("Prompt cannot be empty. Cancelled.")
            return

        # Store the new system prompt in memory
        try:
            handler.memory_store.set_system_prompt(new_prompt)
            await update.message.reply_text(
                "✅ System prompt updated successfully! The new prompt will be used immediately."
            )
        except Exception as e:
            logger.error(f"Failed to set system prompt: {e}")
            await update.message.reply_text(
                f"❌ Failed to update system prompt: {str(e)}"
            )
        return

    if update.message.media_group_id:
        await _queue_telegram_media_group(handler, update, context)
        return

    text = (update.message.text or update.message.caption or "").strip()
    attachments = await _extract_telegram_attachment_urls(update.message, context.bot)
    if not text and not attachments:
        return

    # Normal message handling
    await update.message.chat.send_action(action="typing")
    await _process_telegram_message(
        handler,
        user_id=user_id,
        chat_id=chat_id,
        text=text,
        attachment_urls=attachments,
        bot=context.bot,
    )


async def telegram_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Handle errors in the Telegram bot."""
    logger.error(f"Telegram bot error: {context.error}", exc_info=context.error)


def telegram_polling_error_handler(exc: Exception) -> None:
    """Handle polling errors specifically."""
    logger.warning(f"Telegram polling error (continuing): {exc}")
    # Don't re-raise the exception to keep polling alive


def run_telegram_bot(handler: AgentHandler):
    """Run the telegram bot."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token or not TELEGRAM_AVAILABLE:
        logger.error("Telegram token missing or library not installed.")
        return

    # Type assertions for Pylance since we checked TELEGRAM_AVAILABLE
    from telegram.ext import Application, CommandHandler, MessageHandler, filters

    assert Application is not None
    assert CommandHandler is not None
    assert MessageHandler is not None
    assert filters is not None

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", telegram_start))
    app.add_handler(CommandHandler("setprompt", telegram_setprompt))
    app.add_handler(
        CommandHandler(
            "clear", lambda update, context: telegram_clear(handler, update, context)
        )
    )
    app.add_handler(
        MessageHandler(
            (filters.TEXT | filters.PHOTO | filters.Document.IMAGE) & ~filters.COMMAND,
            lambda update, context: telegram_handle_msg(handler, update, context),
        )
    )
    # Add error handler to catch polling errors gracefully
    app.add_error_handler(telegram_error_handler)
    logger.info("Telegram bot starting...")
    app.run_polling()


async def run_telegram_bot_async(handler: AgentHandler):
    """Run the telegram bot asynchronously."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token or not TELEGRAM_AVAILABLE:
        logger.error("Telegram token missing or library not installed.")
        return

    # Type assertions for Pylance since we checked TELEGRAM_AVAILABLE
    from telegram.ext import Application, CommandHandler, MessageHandler, filters

    assert Application is not None
    assert CommandHandler is not None
    assert MessageHandler is not None
    assert filters is not None

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", telegram_start))
    app.add_handler(CommandHandler("setprompt", telegram_setprompt))
    app.add_handler(
        CommandHandler(
            "clear", lambda update, context: telegram_clear(handler, update, context)
        )
    )
    app.add_handler(
        MessageHandler(
            (filters.TEXT | filters.PHOTO | filters.Document.IMAGE) & ~filters.COMMAND,
            lambda update, context: telegram_handle_msg(handler, update, context),
        )
    )
    # Add error handler to catch polling errors gracefully
    app.add_error_handler(telegram_error_handler)
    logger.info("Telegram bot starting (async)...")
    assert app.updater is not None, "Updater should not be None"
    async with app:
        await app.start()
        await app.updater.start_polling(error_callback=telegram_polling_error_handler)
        # Keep running until cancelled
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            await app.updater.stop()
            await app.stop()
