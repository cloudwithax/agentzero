"""Centralized logging configuration and turn/step log helpers.

Call ``configure_logging()`` once at process startup (main.py).  Everything else
uses the ``preview()`` and ``log_user_turn()`` / ``log_assistant_turn()`` helpers
so user/assistant messages render in a consistent, scannable one-line format.
"""

import logging
import os
from typing import Any, Optional

# Logger dedicated to user-facing turns so they stand out in the stream.
turn_logger = logging.getLogger("agentzero.turn")

# Third-party loggers that are noisy at INFO/DEBUG and rarely useful.
_NOISY_LOGGERS = (
    "aiohttp",
    "aiohttp.access",
    "urllib3",
    "httpx",
    "httpcore",
    "telegram",
    "telegram.ext",
    "apscheduler",
    "asyncio",
)

_LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: Optional[str] = None) -> int:
    """Install a single timestamped stream handler on the root logger.

    Idempotent: re-running replaces existing handlers rather than stacking them.
    Returns the resolved numeric log level.
    """
    requested = (level or os.environ.get("LOG_LEVEL", "INFO")).strip().upper()
    resolved = getattr(logging, requested, None)
    if not isinstance(resolved, int):
        resolved = logging.INFO

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(resolved)

    root = logging.getLogger()
    root.setLevel(resolved)
    root.handlers.clear()
    root.addHandler(handler)

    # Keep third-party chatter out of the way; our own loggers follow root.
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    return resolved


def preview(text: Any, limit: int = 300) -> str:
    """Collapse text to a single trimmed line for log readability."""
    if text is None:
        return ""
    s = text if isinstance(text, str) else str(text)
    s = " ".join(s.split())
    if len(s) > limit:
        return f"{s[:limit]}… (+{len(s) - limit} chars)"
    return s


def _sid(session_id: Optional[str]) -> str:
    return session_id or "-"


def log_user_turn(session_id: Optional[str], text: Any) -> None:
    """Log an inbound user message."""
    turn_logger.info("USER ▶ [%s] %s", _sid(session_id), preview(text))


def log_assistant_turn(
    session_id: Optional[str], text: Any, *, is_error: bool = False
) -> None:
    """Log the outbound assistant reply (or an error reply)."""
    if is_error:
        turn_logger.error("ASSISTANT ✖ [%s] %s", _sid(session_id), preview(text))
    else:
        turn_logger.info("ASSISTANT ◀ [%s] %s", _sid(session_id), preview(text))
