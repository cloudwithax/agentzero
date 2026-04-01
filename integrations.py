"""Bot integrations for iMessage and Telegram."""

import asyncio
import logging
import os
from datetime import datetime
from typing import Optional, Dict

import aiohttp
from aiohttp import web

from handler import AgentHandler

logger = logging.getLogger(__name__)

# Track users who are in the process of setting a system prompt
pending_prompt_users: Dict[int, bool] = {}

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


async def send_imessage(
    phone_number: str, message: str, session: Optional[aiohttp.ClientSession] = None
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


async def handle_imessage(handler: AgentHandler, phone_number: str, text: str) -> str:
    """Process an incoming iMessage."""
    session_id = f"imessage_{phone_number}"

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
            {"messages": [{"role": "user", "content": text}]}, session_id=session_id
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        return "Sorry, an error occurred."


async def start_sendblue_webhook_server(handler: AgentHandler, port: int):
    """Start a webhook server for Sendblue."""
    app = web.Application()
    own_number = os.environ.get("SENDBLUE_NUMBER")
    processed_handles = set()
    dedup_ttl_seconds = 60

    async def webhook_endpoint(request):
        try:
            try:
                data = await request.json()
            except Exception:
                # Some webhook providers may send form-encoded payloads.
                form_data = await request.post()
                data = dict(form_data)

            sender_number = (
                data.get("from_number")
                or data.get("number")
                or data.get("phone_number")
            )
            content = data.get("content") or data.get("message") or data.get("text", "")
            direction = str(data.get("direction", "")).lower()
            is_outbound = bool(data.get("is_outbound"))
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

                if not content.strip():
                    logger.info("Ignoring webhook event with empty content: %s", data)
                    return web.Response(status=200, text="OK")

                async def _process_and_reply():
                    try:
                        resp = await handle_imessage(handler, sender_number, content)
                        send_res = await send_imessage(sender_number, resp)
                        if not send_res.get("success"):
                            logger.error("Failed to send Sendblue reply: %s", send_res)
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
            logger.error(f"Webhook error: {e}")
            return web.Response(status=500, text="Error")

    app.router.add_post("/webhook", webhook_endpoint)
    app.router.add_post("/webhook/receive", webhook_endpoint)
    app.router.add_post("/", webhook_endpoint)

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
    webhook_port = os.environ.get("SENDBLUE_WEBHOOK_PORT")
    if webhook_port:
        await start_sendblue_webhook_server(handler, int(webhook_port))
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
                    if msg.get("direction") == "outgoing":
                        continue
                    num = msg.get("from_number") or msg.get("number")
                    if not num:
                        logger.warning(f"Skipping message with no sender number: {msg}")
                        continue
                    resp = await handle_imessage(handler, num, msg.get("content", ""))
                    await send_imessage(num, resp)
            # Move checkpoint to poll start time to avoid gaps where messages
            # arrive while the previous batch is being processed.
            last_check = poll_started_at
        except Exception as e:
            logger.error(f"Polling error: {e}")
        await asyncio.sleep(interval)


# Telegram Integration


async def telegram_start(update: "Update", context: "ContextTypes.DEFAULT_TYPE"):
    """Handle /start command."""
    if update.message is None or update.effective_user is None:
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
        or update.message.text is None
    ):
        return
    user_id = update.effective_user.id

    # Check if user is in prompt-setting mode
    if user_id in pending_prompt_users:
        # Remove from pending state
        del pending_prompt_users[user_id]

        # Get the new prompt from the message
        new_prompt = update.message.text.strip()

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

    # Normal message handling
    await update.message.chat.send_action(action="typing")
    resp = await handler.handle(
        {"messages": [{"role": "user", "content": update.message.text}]},
        session_id=f"tg_{update.effective_user.id}",
    )
    await update.message.reply_text(resp or "No response.")


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
            filters.TEXT & ~filters.COMMAND,
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
            filters.TEXT & ~filters.COMMAND,
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
