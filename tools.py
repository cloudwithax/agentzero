"""Tool functions and registry for the agent."""

import asyncio
import contextvars
import fnmatch
import glob
import json
import os
import re
import subprocess
import time
from datetime import datetime, timedelta
from typing import Any, Optional

import aiohttp

# Web search cache to avoid redundant API calls per query
_web_search_cache: dict[str, tuple[dict[str, Any], float]] = {}

from memory import EnhancedMemoryStore

# ---------------------------------------------------------------------------
# Workspace enforcement
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.normpath(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_WORKSPACE = os.path.join(_PROJECT_ROOT, "workspace")
AGENT_WORKSPACE = os.path.normpath(
    os.environ.get("AGENT_WORKSPACE", _DEFAULT_WORKSPACE).strip() or _DEFAULT_WORKSPACE
)
# Paths the agent is explicitly allowed to write to even outside the workspace
# (the project root itself for self-modification tasks).
_ALWAYS_ALLOWED_PREFIXES = (AGENT_WORKSPACE, _PROJECT_ROOT)


def _resolve_write_path(filepath: str) -> tuple[str, str | None]:
    """Return (resolved_path, warning_or_None).

    Relative paths are resolved relative to AGENT_WORKSPACE so that bare
    filenames like ``meeting.ics`` land in the workspace rather than in the
    project root.  Absolute paths that fall outside every allowed prefix are
    redirected to ``workspace/scratch/<basename>``.
    """
    if os.path.isabs(filepath):
        resolved = os.path.normpath(filepath)
    else:
        # Strip leading "workspace/" prefix so that "workspace/foo.txt" and
        # "foo.txt" both resolve to AGENT_WORKSPACE/foo.txt, preventing the
        # model from creating a nested workspace/workspace/ directory.
        workspace_name = os.path.basename(AGENT_WORKSPACE)
        parts = filepath.replace("\\", "/").split("/")
        if parts and parts[0] == workspace_name:
            filepath = "/".join(parts[1:]) or "."
        resolved = os.path.normpath(os.path.join(AGENT_WORKSPACE, filepath))

    for allowed in _ALWAYS_ALLOWED_PREFIXES:
        if resolved.startswith(allowed + os.sep) or resolved == allowed:
            return resolved, None

    # Outside every allowed prefix — redirect to workspace/scratch/
    basename = os.path.basename(resolved) or "file"
    redirected = os.path.join(AGENT_WORKSPACE, "scratch", basename)
    warning = (
        f"Path '{filepath}' is outside the workspace. "
        f"Redirected to '{redirected}'. "
        f"Always write files inside {AGENT_WORKSPACE}."
    )
    return redirected, warning


# Initialize memory store (will be set from main module)
memory_store: Optional[EnhancedMemoryStore] = None
consortium_controller: Any = None
reminder_controller: Any = None

skill_registry: Any = None
acp_agent: Any = None
_runtime_session_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "runtime_session_id",
    default=None,
)
_runtime_messages: contextvars.ContextVar[Optional[list[dict[str, Any]]]] = (
    contextvars.ContextVar(
        "runtime_messages",
        default=None,
    )
)
_send_message_buffer: contextvars.ContextVar[Optional[list[dict[str, Any]]]] = (
    contextvars.ContextVar(
        "send_message_buffer",
        default=None,
    )
)
_declared_message_count: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "declared_message_count",
    default=None,
)

_ASSISTANT_NAME_USER_TEXT_PATTERNS = (
    re.compile(
        r"""(?ix)
        \b(?:your|ur)\s+name\s+(?:is|=|should\s+be)\s+
        ['"]?(?P<name>[A-Za-z][A-Za-z0-9'_-]{0,39})['"]?
        """
    ),
    re.compile(
        r"""(?ix)
        \b(?:i(?:'m| am)\s+calling\s+you|call\s+yourself|you(?:'re| are)\s+(?:called|named))\s+
        ['"]?(?P<name>[A-Za-z][A-Za-z0-9'_-]{0,39})['"]?
        """
    ),
)
_ASSISTANT_NAME_MEMORY_PATTERNS = (
    re.compile(
        r"""(?ix)
        \b(?:the\s+)?assistant(?:'s)?\s+name\s+(?:is|=)\s+
        ['"]?(?P<name>[A-Za-z][A-Za-z0-9'_-]{0,39})['"]?
        """
    ),
    re.compile(
        r"""(?ix)
        \b(?:your|bot)\s+name\s+(?:is|=)\s+
        ['"]?(?P<name>[A-Za-z][A-Za-z0-9'_-]{0,39})['"]?
        """
    ),
)
_USER_NAME_MEMORY_PATTERNS = (
    re.compile(
        r"""(?ix)
        \b(?:the\s+)?user(?:'s)?\s+name\s+(?:is|=)\s+
        ['"]?(?P<name>[A-Za-z][A-Za-z0-9'_-]{0,39})['"]?
        """
    ),
    re.compile(
        r"""(?ix)
        \buser\s+goes\s+by\s+['"]?(?P<name>[A-Za-z][A-Za-z0-9'_-]{0,39})['"]?
        """
    ),
)


def set_agent_workspace(path: str) -> None:
    """Override the workspace path at runtime (called from main/handler)."""
    global AGENT_WORKSPACE
    AGENT_WORKSPACE = os.path.normpath(path)


def set_memory_store(store: EnhancedMemoryStore):
    """Set the memory store instance for tools to use."""
    global memory_store
    memory_store = store


def set_consortium_controller(controller: Any):
    """Set the consortium task controller used by consortium tools."""
    global consortium_controller
    consortium_controller = controller


def set_advisor_controller(controller: Any):
    """Set the advisor controller (no-op placeholder)."""
    pass


def set_reviewer_controller(controller: Any):
    """Set the reviewer controller (no-op placeholder)."""
    pass


def set_reminder_controller(controller: Any):
    """Set the reminder task controller used by reminder tools."""
    global reminder_controller
    reminder_controller = controller


def set_skill_registry(registry: Any):
    """Set the skill registry used by skill activation tools."""
    global skill_registry
    skill_registry = registry


def set_acp_agent(agent: Any):
    """Set the ACP agent instance for tools to use."""
    global acp_agent
    acp_agent = agent


def set_tool_runtime_session(session_id: Optional[str]):
    """Bind session context for tool execution during one model turn."""
    normalized = str(session_id or "").strip() or None
    return _runtime_session_id.set(normalized)


def reset_tool_runtime_session(token: contextvars.Token):
    """Restore previous tool runtime session context."""
    _runtime_session_id.reset(token)


def set_tool_runtime_messages(messages: Optional[list[dict[str, Any]]]):
    """Bind the current turn's shared messages for tools that need live context."""
    normalized: Optional[list[dict[str, Any]]] = None
    if messages:
        normalized = [
            dict(message) for message in messages if isinstance(message, dict)
        ]
    return _runtime_messages.set(normalized)


def reset_tool_runtime_messages(token: contextvars.Token):
    """Restore previous shared-message runtime context."""
    _runtime_messages.reset(token)


def init_send_message_buffer():
    """Begin tracking send_message tool deliveries for the current turn."""
    return _send_message_buffer.set([])


def reset_send_message_buffer(token: contextvars.Token):
    """Restore previous send_message buffer context."""
    _send_message_buffer.reset(token)


def init_declared_message_count():
    """Begin tracking declared message count for the current turn."""
    return _declared_message_count.set(None)


def reset_declared_message_count(token: contextvars.Token):
    """Restore previous declared message count context."""
    _declared_message_count.reset(token)


def get_declared_message_count() -> Optional[int]:
    """Return the declared message count for the current turn, or None."""
    return _declared_message_count.get()


def set_declared_message_count(count: int):
    """Set the declared message count for the current turn."""
    return _declared_message_count.set(int(count))


def get_send_message_buffer() -> list[dict[str, Any]]:
    """Return a copy of the messages dispatched via send_message this turn."""
    current = _send_message_buffer.get()
    return list(current or [])


def _message_content_to_text(content: Any) -> str:
    """Extract plain text from content that may include multimodal blocks."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item.get("content"), str):
                parts.append(item["content"])
        return "\n".join(part for part in parts if part).strip()

    return "" if content is None else str(content)


def _normalize_name_key(value: str) -> str:
    """Normalize a short name-like token for equality checks."""
    lowered = str(value or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "", lowered)


def _format_name_display(value: str) -> str:
    """Preserve mixed-case names, but title-case all-lower/all-upper tokens."""
    normalized = str(value or "").strip()
    if not normalized:
        return ""
    if normalized.islower() or normalized.isupper():
        return normalized[:1].upper() + normalized[1:].lower()
    return normalized


def _latest_real_user_text(messages: Optional[list[dict[str, Any]]]) -> str:
    """Return the latest non-system user text from shared runtime messages."""
    if not messages:
        return ""

    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        candidate = _message_content_to_text(message.get("content", ""))
        if candidate.strip().startswith("[System:"):
            continue
        return candidate

    return ""


def _extract_name_with_patterns(
    text: str, patterns: tuple[re.Pattern[str], ...]
) -> str:
    """Extract the first matching name token from text."""
    normalized = str(text or "").strip()
    if not normalized:
        return ""

    for pattern in patterns:
        match = pattern.search(normalized)
        if not match:
            continue
        name = str(match.group("name") or "").strip(" \t\r\n'\".,!?")
        if name:
            return name

    return ""


def extract_assistant_name_from_user_text(text: str) -> str:
    """Extract an assistant-name assignment from the latest user turn."""
    return _extract_name_with_patterns(text, _ASSISTANT_NAME_USER_TEXT_PATTERNS)


def extract_assistant_name_from_memory_content(content: str) -> str:
    """Extract an assistant-name memory from stored memory text."""
    return _extract_name_with_patterns(content, _ASSISTANT_NAME_MEMORY_PATTERNS)


def extract_user_name_from_memory_content(content: str) -> str:
    """Extract a user-name memory from stored memory text."""
    return _extract_name_with_patterns(content, _USER_NAME_MEMORY_PATTERNS)


def normalize_memory_candidate_from_user_text(
    content: str,
    user_text: str,
    topics: Optional[list[str]] = None,
) -> tuple[str, list[str], dict[str, Any]]:
    """Correct obvious subject/perspective mistakes using the live user turn."""
    normalized_content = str(content or "").strip()
    normalized_topics = [
        str(topic).strip() for topic in (topics or []) if str(topic).strip()
    ]
    metadata_updates: dict[str, Any] = {}

    assistant_name = extract_assistant_name_from_user_text(user_text)
    if not assistant_name:
        return normalized_content, normalized_topics, metadata_updates

    assistant_name_key = _normalize_name_key(assistant_name)
    remembered_user_name = extract_user_name_from_memory_content(normalized_content)
    remembered_assistant_name = extract_assistant_name_from_memory_content(
        normalized_content
    )

    if assistant_name_key and (
        _normalize_name_key(remembered_user_name) == assistant_name_key
        or _normalize_name_key(remembered_assistant_name) == assistant_name_key
    ):
        assistant_name = _format_name_display(assistant_name)
        normalized_content = f"The assistant's name is {assistant_name}."
        metadata_updates = {
            "subject": "assistant_identity",
            "slot": "assistant_name",
            "assistant_name": assistant_name,
        }
        for topic in ("assistant_identity", "assistant_name"):
            if topic not in normalized_topics:
                normalized_topics.append(topic)

    return normalized_content, normalized_topics, metadata_updates


# File tools
async def read_file_tool(filepath, offset: int | None = None, limit: int | None = None):
    """Read the contents of a file, with optional line slicing.

    ``offset`` and ``limit`` are interpreted as line-oriented values to match
    common model tool-call behavior. ``offset`` is human-friendly: ``0`` and
    ``1`` both start from the first line.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        if offset is None and limit is None:
            return {"success": True, "content": content}

        offset_value = int(offset or 0)
        limit_value = None if limit is None else int(limit)
        if offset_value < 0:
            return {"success": False, "error": "offset must be non-negative"}
        if limit_value is not None and limit_value < 0:
            return {"success": False, "error": "limit must be non-negative"}

        lines = content.splitlines(keepends=True)
        start_index = 0 if offset_value <= 1 else offset_value - 1
        end_index = None if limit_value is None else start_index + limit_value
        sliced = "".join(lines[start_index:end_index])
        return {
            "success": True,
            "content": sliced,
            "offset": offset_value,
            "limit": limit_value,
            "truncated": end_index is not None and end_index < len(lines),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def write_file_tool(filepath, content):
    """Write content to a file (overwrites existing)."""
    try:
        resolved, warning = _resolve_write_path(filepath)
        os.makedirs(os.path.dirname(resolved) or ".", exist_ok=True)
        with open(resolved, "w", encoding="utf-8") as f:
            f.write(content)
        result: dict[str, Any] = {"success": True, "message": f"Written to {resolved}"}
        if warning:
            result["warning"] = warning
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


async def edit_file_tool(filepath, old_str, new_str):
    """Replace old_str with new_str in file. Requires exact match."""
    try:
        resolved, warning = _resolve_write_path(filepath)
        with open(resolved, "r", encoding="utf-8") as f:
            content = f.read()

        if old_str not in content:
            return {"success": False, "error": "old_str not found in file"}

        new_content = content.replace(old_str, new_str, 1)
        with open(resolved, "w", encoding="utf-8") as f:
            f.write(new_content)
        result: dict[str, Any] = {"success": True, "message": f"Edited {resolved}"}
        if warning:
            result["warning"] = warning
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


async def glob_tool(pattern):
    """Find files matching a glob pattern."""
    try:
        matches = glob.glob(pattern, recursive=True)
        return {"success": True, "matches": matches}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def grep_tool(
    pattern,
    path=".",
    include: str | None = None,
    max_matches: int | None = None,
    **_ignored_kwargs,
):
    """Search for pattern in files. Returns matching lines with filenames.

    Supports optional ``include`` filename patterns and ``max_matches`` limits
    to align with common model-generated grep arguments.
    """
    try:
        compiled_pattern = re.compile(pattern)
        matches = []
        include_patterns = []
        if include:
            include_patterns = [
                item.strip() for item in str(include).split(",") if item.strip()
            ]

        max_match_count = None
        if max_matches is not None:
            max_match_count = int(max_matches)
            if max_match_count < 0:
                return {"success": False, "error": "max_matches must be non-negative"}

        for root, dirs, files in os.walk(path):
            for file in files:
                filepath = os.path.join(root, file)
                if include_patterns:
                    if not any(
                        fnmatch.fnmatch(file, pattern_item)
                        or fnmatch.fnmatch(filepath, pattern_item)
                        for pattern_item in include_patterns
                    ):
                        continue
                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        for i, line in enumerate(f, 1):
                            if compiled_pattern.search(line):
                                matches.append(
                                    {
                                        "file": filepath,
                                        "line": i,
                                        "content": line.rstrip(),
                                    }
                                )
                                if (
                                    max_match_count is not None
                                    and len(matches) >= max_match_count
                                ):
                                    return {"success": True, "matches": matches}
                except Exception:
                    continue
        return {"success": True, "matches": matches}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def bash_tool(command):
    """Execute a shell command and return output."""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30
        )
        return {
            "success": True,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Command timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# Calendar and date tools
async def get_current_date_tool():
    """Get the current date and time."""
    now = datetime.now()
    return {
        "success": True,
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "datetime": now.isoformat(),
        "day_of_week": now.strftime("%A"),
        "weekday": now.weekday(),  # 0=Monday, 6=Sunday
    }


async def get_next_weekday_tool(weekday_name: str):
    """Get the date of the next occurrence of a specific weekday.

    Args:
        weekday_name: Name of weekday (e.g., 'Monday', 'Tuesday', 'next Tuesday')
    """
    try:
        # Normalize weekday name
        weekday_name = weekday_name.lower().replace("next ", "").strip()

        # Map weekday names to numbers (0=Monday, 6=Sunday)
        weekday_map = {
            "monday": 0,
            "mon": 0,
            "tuesday": 1,
            "tue": 1,
            "tues": 1,
            "wednesday": 2,
            "wed": 2,
            "weds": 2,
            "thursday": 3,
            "thu": 3,
            "thur": 3,
            "thurs": 3,
            "friday": 4,
            "fri": 4,
            "saturday": 5,
            "sat": 5,
            "sunday": 6,
            "sun": 6,
        }

        if weekday_name not in weekday_map:
            return {"success": False, "error": f"Unknown weekday: {weekday_name}"}

        target_weekday = weekday_map[weekday_name]
        today = datetime.now()
        current_weekday = today.weekday()

        # Calculate days until next occurrence
        days_ahead = (target_weekday - current_weekday) % 7
        if days_ahead == 0:
            # If today is the target day, get next week
            days_ahead = 7

        next_date = today + timedelta(days=days_ahead)

        return {
            "success": True,
            "next_date": next_date.strftime("%Y-%m-%d"),
            "date_formatted": next_date.strftime("%B %d, %Y"),
            "weekday": next_date.strftime("%A"),
            "days_from_now": days_ahead,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def format_date_tool(
    date_str: str, input_format: str = "%Y-%m-%d", output_format: str = "%B %d, %Y"
):
    """Format a date string from one format to another.

    Args:
        date_str: The date string to format
        input_format: Format of input date (default: YYYY-MM-DD)
        output_format: Desired output format (default: Month DD, YYYY)
    """
    try:
        date_obj = datetime.strptime(date_str, input_format)
        return {
            "success": True,
            "formatted": date_obj.strftime(output_format),
            "iso": date_obj.isoformat(),
            "date": date_obj.strftime("%Y-%m-%d"),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# PDF tool
async def read_pdf_tool(filepath: str):
    """Extract text content from a PDF file.

    Args:
        filepath: Path to the PDF file to read
    """
    try:
        # Check if file exists
        if not os.path.exists(filepath):
            return {"success": False, "error": f"File not found: {filepath}"}

        # Check if it's a PDF
        if not filepath.lower().endswith(".pdf"):
            return {"success": False, "error": "File is not a PDF"}

        # Use pdftotext to extract text
        result = subprocess.run(
            ["pdftotext", "-layout", filepath, "-"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            return {
                "success": False,
                "error": f"Failed to extract PDF: {result.stderr}",
            }

        text = result.stdout

        return {
            "success": True,
            "content": text,
            "length": len(text),
            "lines": text.split("\n"),
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "PDF extraction timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# Memory tools
async def remember_tool(
    content: str, topics: Optional[list] = None, importance: str = "medium"
):
    """Store important information in persistent memory.

    Args:
        content: The information to remember
        topics: Optional topics/tags for categorization
        importance: Importance level (low/medium/high)
    """
    try:
        if memory_store is None:
            return {"success": False, "error": "Memory store not initialized"}

        normalized_content, normalized_topics, inferred_metadata = (
            normalize_memory_candidate_from_user_text(
                content=content,
                user_text=_latest_real_user_text(_runtime_messages.get()),
                topics=topics,
            )
        )
        metadata = {"importance": importance, "type": "explicit_memory"}
        metadata.update(inferred_metadata)
        session_id = _runtime_session_id.get()
        if session_id:
            metadata["session_id"] = session_id
        memory_id = await memory_store.add_memory(
            content=normalized_content,
            metadata=metadata,
            topics=normalized_topics,
            generate_embedding=True,
        )
        return {
            "success": True,
            "memory_id": memory_id,
            "message": f"Successfully stored memory #{memory_id}",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def recall_tool(query: str, top_k: int = 5, topic: Optional[str] = None):
    """Search and retrieve information from persistent memory.

    Args:
        query: What to search for
        top_k: Number of memories to retrieve
        topic: Optional topic filter
    """
    try:
        if memory_store is None:
            return {"success": False, "error": "Memory store not initialized"}

        results = await memory_store.search_memories(
            query=query, top_k=top_k, topic=topic
        )
        memories = []
        for memory, score in results:
            memories.append(
                {
                    "id": memory.id,
                    "content": memory.content,
                    "similarity": round(score, 4),
                    "metadata": memory.metadata,
                    "created_at": memory.created_at,
                }
            )
        return {
            "success": True,
            "count": len(memories),
            "memories": memories,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def get_recent_memories_tool(limit: int = 10):
    """Get the most recent memories.

    Args:
        limit: Number of memories to retrieve
    """
    try:
        if memory_store is None:
            return {"success": False, "error": "Memory store not initialized"}

        memories = memory_store.get_recent_memories(limit=limit)
        return {
            "success": True,
            "count": len(memories),
            "memories": [
                {
                    "id": m.id,
                    "content": m.content,
                    "metadata": m.metadata,
                    "created_at": m.created_at,
                }
                for m in memories
            ],
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def forget_tool(memory_id: int):
    """Delete a specific memory.

    Args:
        memory_id: The ID of the memory to delete
    """
    try:
        if memory_store is None:
            return {"success": False, "error": "Memory store not initialized"}

        deleted = memory_store.delete_memory(memory_id)
        if deleted:
            return {
                "success": True,
                "message": f"Memory #{memory_id} deleted successfully",
            }
        else:
            return {"success": False, "error": f"Memory #{memory_id} not found"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def memory_stats_tool():
    """Get statistics about the memory system."""
    try:
        if memory_store is None:
            return {"success": False, "error": "Memory store not initialized"}

        stats = memory_store.get_memory_stats()
        return {"success": True, "stats": stats}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def consortium_start_tool(task: str, task_id: Optional[str] = None):
    """Start a consortium task in the background."""
    try:
        if consortium_controller is None:
            return {"success": False, "error": "Consortium controller not initialized"}

        return await consortium_controller.start_consortium_task(
            task=task,
            task_id=task_id,
        )
    except Exception as e:
        return {"success": False, "error": str(e)}


async def consortium_stop_tool(task_id: str, reason: str = ""):
    """Stop a running consortium task."""
    try:
        if consortium_controller is None:
            return {"success": False, "error": "Consortium controller not initialized"}

        return await consortium_controller.stop_consortium_task(
            task_id=task_id,
            reason=reason,
        )
    except Exception as e:
        return {"success": False, "error": str(e)}


async def consortium_status_tool(task_id: Optional[str] = None):
    """Get status for one consortium task or all consortium tasks."""
    try:
        if consortium_controller is None:
            return {"success": False, "error": "Consortium controller not initialized"}

        return await consortium_controller.get_consortium_task_status(task_id=task_id)
    except Exception as e:
        return {"success": False, "error": str(e)}


async def reminder_create_tool(
    cron: str,
    message: str = "",
    session_id: Optional[str] = None,
    one_off: bool = False,
    run_ai: bool = False,
    ai_prompt: str = "",
    task_id: Optional[str] = None,
    name: str = "",
):
    """Create a cron-based reminder task (one-off or recurring)."""
    try:
        if reminder_controller is None:
            return {"success": False, "error": "Reminder controller not initialized"}

        bound_session_id = str(session_id or "").strip() or _runtime_session_id.get()

        return await reminder_controller.create_reminder_task(
            cron=cron,
            message=message,
            session_id=bound_session_id,
            one_off=one_off,
            run_ai=run_ai,
            ai_prompt=ai_prompt,
            task_id=task_id,
            name=name,
        )
    except Exception as e:
        return {"success": False, "error": str(e)}


async def reminder_list_tool(include_disabled: bool = True):
    """List reminder tasks."""
    try:
        if reminder_controller is None:
            return {"success": False, "error": "Reminder controller not initialized"}

        return await reminder_controller.list_reminder_tasks(
            include_disabled=include_disabled,
        )
    except Exception as e:
        return {"success": False, "error": str(e)}


async def reminder_status_tool(task_id: str):
    """Get status for one reminder task."""
    try:
        if reminder_controller is None:
            return {"success": False, "error": "Reminder controller not initialized"}

        return await reminder_controller.get_reminder_task_status(task_id=task_id)
    except Exception as e:
        return {"success": False, "error": str(e)}


async def reminder_cancel_tool(task_id: str, reason: str = ""):
    """Cancel a reminder task."""
    try:
        if reminder_controller is None:
            return {"success": False, "error": "Reminder controller not initialized"}

        return await reminder_controller.cancel_reminder_task(
            task_id=task_id,
            reason=reason,
        )
    except Exception as e:
        return {"success": False, "error": str(e)}


async def reminder_run_now_tool(task_id: str):
    """Run a reminder task immediately."""
    try:
        if reminder_controller is None:
            return {"success": False, "error": "Reminder controller not initialized"}

        return await reminder_controller.run_reminder_task_now(task_id=task_id)
    except Exception as e:
        return {"success": False, "error": str(e)}


async def consortium_agree_tool(
    verdict: str = "",
    rationale: str = "",
    confidence: float = 1.0,
    key_points: Optional[list] = None,
):
    """Signal that a consortium member agrees on a final verdict."""
    try:
        normalized_confidence = float(confidence)
    except (TypeError, ValueError):
        normalized_confidence = 1.0

    return {
        "success": True,
        "agreed": True,
        "verdict": verdict.strip() if isinstance(verdict, str) else "",
        "rationale": rationale.strip() if isinstance(rationale, str) else "",
        "confidence": max(0.0, min(1.0, normalized_confidence)),
        "key_points": key_points if isinstance(key_points, list) else [],
    }


async def parse_mcp_response(resp):
    """Parse MCP response handling both JSON and SSE formats."""
    content_type = resp.headers.get("Content-Type", "")

    if "text/event-stream" in content_type:
        # Read SSE format
        body = await resp.text()
        # Parse SSE data lines
        for line in body.split("\n"):
            if line.startswith("data: "):
                data = line[6:]  # Remove "data: " prefix
                if data == "[DONE]":
                    continue
                try:
                    return json.loads(data)
                except json.JSONDecodeError:
                    continue
        return {}
    else:
        # Regular JSON response
        return await resp.json()


def parse_search_results(text: str) -> list:
    """Parse the formatted search results text into structured data."""
    results = []
    current = {}

    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("Title:"):
            if current:
                results.append(current)
            current = {"title": line[6:].strip()}
        elif line.startswith("URL:") and current:
            current["url"] = line[4:].strip()
        elif line.startswith("Published:") and current:
            current["publishedDate"] = line[10:].strip()
        elif line.startswith("Author:") and current:
            current["author"] = line[7:].strip()
        elif line.startswith("Highlights:") and current:
            # Collect highlights until next separator or Title
            highlights = []
            i += 1
            while i < len(lines):
                hl_line = lines[i].strip()
                if hl_line.startswith("---") or hl_line.startswith("Title:"):
                    i -= 1  # Go back one line
                    break
                if (
                    hl_line
                    and not hl_line.startswith("URL:")
                    and not hl_line.startswith("Published:")
                ):
                    highlights.append(hl_line)
                i += 1
            current["highlights"] = " ".join(highlights)
        elif line == "---" and current:
            if current:
                results.append(current)
                current = {}
        i += 1

    if current:
        results.append(current)

    return results


# Web search tool using MCP protocol (stateless - no API key required)
async def web_search_tool(
    query: str, numResults: int = 5, category: Optional[str] = None, type: str = "auto"
):
    """Search the web using Exa MCP server (no API key required, has rate limits).

    Args:
        query: The search query string
        numResults: Number of results to return (1-100, default: 5)
        category: Optional category filter (company, research paper, news, people)
        type: Search type (auto, neural, fast, deep, deep-reasoning, instant)
    """
    try:
        # Quick cache check — avoid redundant API calls
        cache_key = f"{query.lower().strip()}:{numResults}:{category or ''}"
        if cache_key in _web_search_cache:
            cached_result, cached_at = _web_search_cache[cache_key]
            if time.time() - cached_at < 300:  # 5 min cache
                return cached_result

        arguments = {"query": query, "numResults": min(max(numResults, 1), 100)}
        if category:
            arguments["category"] = category

        tool_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "web_search_exa", "arguments": arguments},
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://mcp.exa.ai/mcp",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
                json=tool_payload,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    return {
                        "success": False,
                        "error": f"MCP error (status {resp.status}): {error_text}",
                    }

                data = await parse_mcp_response(resp)

                # Check for JSON-RPC error
                if "error" in data:
                    return {
                        "success": False,
                        "error": data["error"].get("message", "Unknown MCP error"),
                    }

                # Extract results from MCP response
                result = data.get("result", {})
                content = result.get("content", [])

                # Get the text content
                full_text = ""
                for item in content:
                    if item.get("type") == "text":
                        full_text = item.get("text", "")
                        break

                # Parse structured results
                search_results = parse_search_results(full_text)

                result_data = {
                    "success": True,
                    "query": query,
                    "resultCount": len(search_results),
                    "results": search_results[:numResults],
                    "formatted_text": full_text[:5000] if full_text else "",
                }
                _web_search_cache[cache_key] = (result_data, time.time())
                return result_data

    except aiohttp.ClientError as e:
        return {"success": False, "error": f"Network error: {str(e)}"}
    except asyncio.TimeoutError:
        return {"success": False, "error": "Request timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def generate_image_tool(
    prompt: str,
    filename: str,
    width: int = 1024,
    height: int = 1024,
    model: str = "flux",
):
    """Generate an image from a text prompt and save it to the workspace.

    Uses Pollinations.ai (free, no API key required).

    Args:
        prompt: Text description of the image to generate
        filename: Output filename (e.g. robot_cafe.png). Saved to workspace.
        width: Image width in pixels (default 1024)
        height: Image height in pixels (default 1024)
        model: Model to use — 'flux' (default, high quality) or 'turbo' (faster)
    """
    try:
        import urllib.parse

        encoded_prompt = urllib.parse.quote(prompt)
        url = (
            f"https://image.pollinations.ai/prompt/{encoded_prompt}"
            f"?width={width}&height={height}&model={model}&nologo=true&enhance=true"
        )

        resolved, warning = _resolve_write_path(filename)
        os.makedirs(os.path.dirname(resolved) or ".", exist_ok=True)

        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=120)
            ) as resp:
                if resp.status != 200:
                    return {
                        "success": False,
                        "error": f"Image generation failed (HTTP {resp.status})",
                    }
                image_bytes = await resp.read()

        with open(resolved, "wb") as f:
            f.write(image_bytes)

        result: dict[str, Any] = {
            "success": True,
            "message": f"Image saved to {resolved} ({len(image_bytes)} bytes)",
            "path": resolved,
            "url": url,
        }
        if warning:
            result["warning"] = warning
        return result

    except asyncio.TimeoutError:
        return {"success": False, "error": "Image generation timed out (>120s)"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def activate_skill_tool(name: str):
    """Activate a discovered skill and return its wrapped instructions."""
    if skill_registry is None:
        return {"success": False, "error": "Skill registry is not configured"}

    try:
        session_id = _runtime_session_id.get()
        return skill_registry.activate_skill(
            name=str(name or "").strip(),
            session_id=session_id,
            source="model",
        )
    except Exception as e:
        return {"success": False, "error": str(e)}


async def add_skill_tool(url: str, auto_activate: bool = True):
    """Fetch a skill from a URL, scan for injection attacks, and install it.

    The skill content is validated against the prompt-injection-defense scanner
    before being written to disk.  If the content passes validation it is
    persisted in the user skills directory and automatically activated for the
    current session.
    """
    if skill_registry is None:
        return {"success": False, "error": "Skill registry is not configured"}

    try:
        session_id = _runtime_session_id.get()
        return await skill_registry.add_skill_from_url(
            url=str(url or "").strip(),
            session_id=session_id,
            auto_activate=bool(auto_activate),
        )
    except Exception as e:
        return {"success": False, "error": str(e)}


async def declare_message_count_tool(count: int):
    """Declare how many send_message calls will be made this turn.

    Call this BEFORE any send_message calls. After sending all declared
    messages, output <DONE> on its own line to signal completion.
    """
    try:
        value = int(count)
        if value < 0:
            return {"success": False, "error": "count must be non-negative"}
        set_declared_message_count(value)
        return {"success": True, "declared_count": value}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def send_message_tool(
    text: str,
    attachments: Optional[list[str]] = None,
):
    """Deliver one user-facing message to the active conversation channel.

    Each call sends exactly one message to the user. Call this multiple
    times in the same turn to send several short messages back-to-back —
    use that pattern for natural pacing (e.g. setup then punchline,
    answer then aside, question then context). Do not split for the
    sake of splitting; only when the message has clearly distinct beats.

    All user-facing replies must go through this tool. Plain text
    responses are not delivered.

    Args:
        text: The message body for this single bubble.
        attachments: Optional list of media URLs to attach to this message.
    """
    body = "" if text is None else str(text)
    body_stripped = body.strip()
    normalized_attachments: list[str] = []
    if isinstance(attachments, list):
        normalized_attachments = [
            str(item).strip() for item in attachments if str(item or "").strip()
        ]

    if not body_stripped and not normalized_attachments:
        return {"success": False, "error": "text or attachments required"}

    session_id = (_runtime_session_id.get() or "").strip()
    record: dict[str, Any] = {
        "text": body_stripped,
        "attachments": list(normalized_attachments),
        "channel": "buffered",
    }

    try:
        if session_id.startswith("imessage_"):
            phone_number = session_id[len("imessage_") :]
            try:
                from integrations import send_imessage
            except Exception as import_error:
                return {
                    "success": False,
                    "error": f"iMessage integration unavailable: {import_error}",
                }
            delivery = await send_imessage(
                phone_number=phone_number,
                message=body_stripped,
                media_urls=normalized_attachments or None,
            )
            success = bool(delivery.get("success", False))
            record["channel"] = "imessage"
            record["delivery"] = delivery
            buffer = _send_message_buffer.get()
            if buffer is not None:
                buffer.append(record)
            return {
                "success": success,
                "channel": "imessage",
                "delivery": delivery,
            }

        if session_id.startswith("tg_"):
            try:
                from integrations import (
                    _send_telegram_response,
                    session_delivery_targets,
                )
            except Exception as import_error:
                return {
                    "success": False,
                    "error": f"Telegram integration unavailable: {import_error}",
                }
            target = session_delivery_targets.get(session_id) or {}
            chat_id = target.get("chat_id")
            bot = target.get("bot")
            if chat_id is None or bot is None:
                return {
                    "success": False,
                    "error": "Telegram delivery target not registered for this session",
                }
            try:
                await _send_telegram_response(
                    bot,
                    int(chat_id),
                    body_stripped,
                    list(normalized_attachments),
                )
            except Exception as send_error:
                return {
                    "success": False,
                    "error": f"Telegram send failed: {send_error}",
                }
            record["channel"] = "telegram"
            buffer = _send_message_buffer.get()
            if buffer is not None:
                buffer.append(record)
            return {"success": True, "channel": "telegram"}

        # Non-messaging channel (CLI, OpenAI-compat, scheduled run, etc.).
        # Buffer the message; the loop assembles the final response from
        # the buffer when no live channel is bound.
        buffer = _send_message_buffer.get()
        if buffer is not None:
            buffer.append(record)
        return {"success": True, "channel": "buffered", "queued": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def send_tapback_tool(
    message_handle: str,
    reaction: str,
    part_index: Optional[int] = None,
):
    """Send an iMessage tapback reaction for a specific Sendblue message handle.

    Args:
        message_handle: Sendblue message handle/GUID from inbound webhook payloads
        reaction: One of love, like, dislike, laugh, emphasize, question
        part_index: Optional non-negative part index for multi-part messages
    """
    try:
        try:
            from integrations import send_reaction
        except Exception as import_error:
            return {
                "success": False,
                "error": f"Tapback integration unavailable: {import_error}",
            }

        normalized_part_index = None
        if part_index is not None:
            try:
                normalized_part_index = int(part_index)
            except (TypeError, ValueError):
                return {
                    "success": False,
                    "error": "part_index must be an integer when provided",
                }

        return await send_reaction(
            message_handle=str(message_handle),
            reaction=str(reaction),
            part_index=normalized_part_index,
        )
    except Exception as e:
        return {"success": False, "error": str(e)}


async def send_telegram_reaction_tool(
    chat_id: int,
    message_id: int,
    reaction: str,
):
    """Send a Telegram emoji reaction to a specific message.

    Args:
        chat_id: Telegram chat ID where the message was sent
        message_id: Telegram message ID to react to
        reaction: Reaction name. Core: like (👍), love (❤), dislike (👎), laugh (🤣), emphasize (🔥), question (🤔). Extended: party (🎉), clap (👏), cry (😢), sob (😭), scream (😱), mindblown (🤯), pray (🙏), cool (😎), 100 (💯), hearts (🥰), starry (🤩), angry (😡), devil (😈), ghost (👻), clown (🤡), shrug (🤷), eyes (👀), kiss (😘), hug (🤗), salute (🫡), nerd (🤓), trophy (🏆), and more
    """
    try:
        try:
            from integrations import send_telegram_reaction
        except Exception as import_error:
            return {
                "success": False,
                "error": f"Telegram reaction integration unavailable: {import_error}",
            }

        try:
            normalized_chat_id = int(chat_id)
        except (TypeError, ValueError):
            return {"success": False, "error": "chat_id must be an integer"}

        try:
            normalized_message_id = int(message_id)
        except (TypeError, ValueError):
            return {"success": False, "error": "message_id must be an integer"}

        return await send_telegram_reaction(
            chat_id=normalized_chat_id,
            message_id=normalized_message_id,
            reaction=str(reaction),
        )
    except Exception as e:
        return {"success": False, "error": str(e)}


async def acp_register_service_tool(
    service_name: str,
    capabilities: str,
    description: str,
    endpoint: str,
) -> dict[str, Any]:
    """Register capabilities with the ACP network."""
    if not acp_agent:
        return {"success": False, "error": "ACP agent not initialized"}

    try:
        capabilities_list = [c.strip() for c in capabilities.split(",") if c.strip()]
        await acp_agent.register_capabilities(capabilities_list)

        return {
            "success": True,
            "message": f"Service '{service_name}' registered with capabilities: {capabilities_list}",
            "service_name": service_name,
            "capabilities": capabilities_list,
            "endpoint": endpoint,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def store_credential_tool(key: str, value: str, description: str = ""):
    """Store a credential (API key, token, password, etc.) in the encrypted vault.

    Values are encrypted at rest using Fernet symmetric encryption. The key is
    a human-readable name you can use to retrieve the credential later.

    Args:
        key: A unique name for this credential (e.g. 'github_token', 'aws_secret_key')
        value: The secret value to store
        description: Optional description of what this credential is for
    """
    try:
        if memory_store is None:
            return {"success": False, "error": "Memory store not initialized"}
        metadata = {}
        if description:
            metadata["description"] = str(description).strip()
        return memory_store.store_credential(key=key, value=value, metadata=metadata)
    except Exception as e:
        return {"success": False, "error": str(e)}


async def get_credential_tool(key: str):
    """Retrieve a stored credential from the encrypted vault by its key name.

    Args:
        key: The name of the credential to retrieve
    """
    try:
        if memory_store is None:
            return {"success": False, "error": "Memory store not initialized"}
        return memory_store.get_credential(key=key)
    except Exception as e:
        return {"success": False, "error": str(e)}


async def delete_credential_tool(key: str):
    """Delete a stored credential from the encrypted vault.

    Args:
        key: The name of the credential to delete
    """
    try:
        if memory_store is None:
            return {"success": False, "error": "Memory store not initialized"}
        return memory_store.delete_credential(key=key)
    except Exception as e:
        return {"success": False, "error": str(e)}


async def list_credentials_tool():
    """List all stored credential keys (values are never exposed in listings)."""
    try:
        if memory_store is None:
            return {"success": False, "error": "Memory store not initialized"}
        return memory_store.list_credentials()
    except Exception as e:
        return {"success": False, "error": str(e)}


async def acp_discover_peers_tool(
    query_type: str = "all",
    query_value: Optional[str] = None,
) -> dict[str, Any]:
    """Discover peers in the ACP network."""
    if not acp_agent:
        return {"success": False, "error": "ACP agent not initialized"}

    try:
        peers = await acp_agent.discover_peers(
            query_type=query_type,
            query_value=query_value,
        )

        profiles = []
        for peer in peers:
            profiles.append(
                {
                    "agent_id": peer.agent_id,
                    "agent_name": peer.agent_name,
                    "capabilities": peer.capabilities,
                    "endpoints": peer.endpoints,
                    "supported_protocols": peer.supported_protocols,
                }
            )

        return {
            "success": True,
            "peers_found": len(peers),
            "peers": profiles,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def acp_send_message_tool(
    recipient_id: str,
    message: str,
    payload: Optional[str] = None,
    secure: bool = True,
) -> dict[str, Any]:
    """Send a message to another agent via ACP."""
    if not acp_agent:
        return {"success": False, "error": "ACP agent not initialized"}

    try:
        message_data = {
            "text": message,
        }
        if payload:
            message_data["payload"] = payload

        success = await acp_agent.send_message(
            recipient_id=recipient_id,
            payload=message_data,
            secure=secure,
        )

        if success:
            return {
                "success": True,
                "message": f"Message sent to {recipient_id}",
                "recipient_id": recipient_id,
            }
        else:
            return {"success": False, "error": "Failed to send message"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def acp_get_registry_tool() -> dict[str, Any]:
    """Get the current ACP registry status."""
    if not acp_agent:
        return {"success": False, "error": "ACP agent not initialized"}

    try:
        status = await acp_agent.get_registry_status()
        return {
            "success": True,
            **status,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def acp_list_peers_tool() -> dict[str, Any]:
    """List all known peers in the ACP network."""
    if not acp_agent:
        return {"success": False, "error": "ACP agent not initialized"}

    try:
        peers = await acp_agent.discover_peers()
        return {
            "success": True,
            "peers_found": len(peers),
            "peers": [
                {
                    "agent_id": peer.agent_id,
                    "agent_name": peer.agent_name,
                    "capabilities": peer.capabilities,
                    "endpoints": peer.endpoints,
                }
                for peer in peers
            ],
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def self_heal_status_tool() -> dict[str, Any]:
    """Get current status of the self-healing subsystem."""
    try:
        from self_heal import _SELF_HEAL_ENABLED, _SELF_HEAL_CLAUDE_MODEL
        from self_heal import (
            _SELF_HEAL_COOLDOWN_SECONDS,
            _SELF_HEAL_TIMEOUT_SECONDS,
            _SELF_HEAL_AUTO_MERGE,
            _SELF_HEAL_MAX_ATTEMPTS_PER_SESSION,
        )
    except ImportError:
        return {"success": False, "error": "self_heal module not available"}

    from handler import AgentHandler

    manager = getattr(AgentHandler, "_self_heal_manager_instance", None)
    if not manager:
        for frame_info in inspect.stack():
            local_vars = (
                frame_info.frame.f_locals if hasattr(frame_info, "frame") else {}
            )
            handler_instance = local_vars.get("self", None)
            if isinstance(handler_instance, AgentHandler):
                manager = getattr(handler_instance, "_self_heal_manager", None)
                break

    status = {
        "enabled": _SELF_HEAL_ENABLED,
        "claude_model": _SELF_HEAL_CLAUDE_MODEL or "(default)",
        "cooldown_seconds": _SELF_HEAL_COOLDOWN_SECONDS,
        "timeout_seconds": _SELF_HEAL_TIMEOUT_SECONDS,
        "auto_merge": _SELF_HEAL_AUTO_MERGE,
        "max_attempts_per_session": _SELF_HEAL_MAX_ATTEMPTS_PER_SESSION,
        "manager_active": manager is not None,
    }
    if manager:
        status.update(manager.get_status())
    return {"success": True, **status}


import inspect


# Tool registry for easy access - defined before handle() to avoid NameError
# Includes aliases for compatibility with benchmark grading
TOOLS = {
    "read": read_file_tool,
    "read_file": read_file_tool,  # Alias for benchmark compatibility
    "readFile": read_file_tool,  # Another alias
    "write": write_file_tool,
    "write_file": write_file_tool,  # Alias
    "edit": edit_file_tool,
    "edit_file": edit_file_tool,  # Alias
    "glob": glob_tool,
    "grep": grep_tool,
    "bash": bash_tool,
    "get_current_date": get_current_date_tool,
    "get_next_weekday": get_next_weekday_tool,
    "format_date": format_date_tool,
    "read_pdf": read_pdf_tool,
    "remember": remember_tool,
    "recall": recall_tool,
    "get_recent_memories": get_recent_memories_tool,
    "forget": forget_tool,
    "memory_stats": memory_stats_tool,
    "web_search": web_search_tool,
    "generate_image": generate_image_tool,
    "create_image": generate_image_tool,  # Alias
    "activate_skill": activate_skill_tool,
    "add_skill": add_skill_tool,
    "send_message": send_message_tool,
    "send_tapback": send_tapback_tool,
    "send_reaction": send_tapback_tool,
    "send_telegram_reaction": send_telegram_reaction_tool,
    "consortium_start": consortium_start_tool,
    "consortium_stop": consortium_stop_tool,
    "consortium_status": consortium_status_tool,
    "reminder_create": reminder_create_tool,
    "reminder_list": reminder_list_tool,
    "reminder_status": reminder_status_tool,
    "reminder_cancel": reminder_cancel_tool,
    "reminder_run_now": reminder_run_now_tool,
    "consortium_agree": consortium_agree_tool,
    # ACP tools
    "acp_register_service": acp_register_service_tool,
    "acp_discover_peers": acp_discover_peers_tool,
    "acp_send_message": acp_send_message_tool,
    "acp_get_registry": acp_get_registry_tool,
    "acp_list_peers": acp_list_peers_tool,
    # Credential vault tools
    "store_credential": store_credential_tool,
    "get_credential": get_credential_tool,
    "delete_credential": delete_credential_tool,
    "list_credentials": list_credentials_tool,
    # Self-healing tools
    "self_heal_status": self_heal_status_tool,
}


def validate_tool_args(func_name: str, func_args: dict) -> tuple:
    """
    Validate tool call arguments before execution.
    Returns (is_valid, error_message).
    """
    # Required parameter validation for each tool
    required_params = {
        "read": ["filepath"],
        "read_file": ["filepath"],
        "readFile": ["filepath"],
        "write": ["filepath", "content"],
        "write_file": ["filepath", "content"],
        "edit": ["filepath", "old_str", "new_str"],
        "edit_file": ["filepath", "old_str", "new_str"],
        "glob": ["pattern"],
        "grep": ["pattern"],
        "bash": ["command"],
        "get_next_weekday": ["weekday_name"],
        "format_date": ["date_str"],
        "read_pdf": ["filepath"],
        "remember": ["content"],
        "recall": ["query"],
        "forget": ["memory_id"],
        "web_search": ["query"],
        "generate_image": ["prompt", "filename"],
        "create_image": ["prompt", "filename"],
        "activate_skill": ["name"],
        "add_skill": ["url"],
        "send_message": ["text"],
        "send_tapback": ["message_handle", "reaction"],
        "send_reaction": ["message_handle", "reaction"],
        "send_telegram_reaction": ["chat_id", "message_id", "reaction"],
        "consortium_start": ["task"],
        "consortium_stop": ["task_id"],
        "consortium_status": [],
        "reminder_create": ["cron"],
        "reminder_list": [],
        "reminder_status": ["task_id"],
        "reminder_cancel": ["task_id"],
        "reminder_run_now": ["task_id"],
        "consortium_agree": [],
        "acp_register_service": [
            "service_name",
            "capabilities",
            "description",
            "endpoint",
        ],
        "acp_discover_peers": [],
        "acp_send_message": ["recipient_id", "message"],
        "acp_get_registry": [],
        "acp_list_peers": [],
        "store_credential": ["key", "value"],
        "get_credential": ["key"],
        "delete_credential": ["key"],
        "list_credentials": [],
        "self_heal_status": [],
    }

    if func_name in required_params:
        missing = [p for p in required_params[func_name] if p not in func_args]
        if missing:
            return False, f"Missing required parameters: {missing}"

    return True, None
