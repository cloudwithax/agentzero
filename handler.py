"""Main handler and chat loop for the agent."""

import asyncio
import aiohttp
import datetime
import json
import logging
import os
import re
import uuid
from difflib import SequenceMatcher
from typing import Any, Awaitable, Callable, Optional

from memory import EnhancedMemoryStore
from capabilities import CapabilityProfile, AdaptiveFormatter
from examples import AdaptiveFewShotManager
from planning import TaskType, TaskPlanner, TaskAnalyzer
from api import api_call_with_retry, process_response
from skills import SkillRegistry
from tools import (
    reset_tool_runtime_session,
    set_consortium_controller,
    set_tool_runtime_session,
)

logger = logging.getLogger(__name__)

IMESSAGE_HANDLE_CONTEXT_LIMIT = 50
REQUEST_FRESHNESS_INSTRUCTION = (
    "[Request Freshness]: This turn includes a one-time freshness token to discourage "
    "cache reuse and repeated phrasing. Treat the request as new and answer independently."
)


def _content_to_text(content: Any) -> str:
    """Extract plain text from message content that may include multimodal blocks."""
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


def _build_consortium_agree_tool_schema() -> dict[str, Any]:
    """Build internal-only tool schema used by consortium members."""
    return {
        "type": "function",
        "function": {
            "name": "consortium_agree",
            "description": "Signal that a consortium member agrees on the final verdict",
            "parameters": {
                "type": "object",
                "properties": {
                    "verdict": {
                        "type": "string",
                        "description": "Final verdict proposed by this member",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Short reasoning for the verdict",
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence from 0 to 1",
                    },
                    "key_points": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional bullet points supporting agreement",
                    },
                },
            },
        },
    }


# Configuration
BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
API_KEY = os.environ.get(
    "NVIDIA_API_KEY",
    "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",
)
MODEL_ID = os.environ.get("MODEL_ID", "moonshotai/kimi-k2-instruct-0905")
CONSORTIUM_MODEL_ID = os.environ.get("CONSORTIUM_MODEL", MODEL_ID).strip() or MODEL_ID

# Base payload template (do not mutate globally)
BASE_PAYLOAD = {
    "model": MODEL_ID,
    "temperature": 0.6,
    "top_p": 0.9,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "max_tokens": 4096,
    "stream": False,
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read the contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "Path to the file to read",
                        }
                    },
                    "required": ["filepath"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write",
                "description": "Write content to a file (overwrites existing)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "Path to the file to write",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write",
                        },
                    },
                    "required": ["filepath", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit",
                "description": "Replace old_str with new_str in file. Requires exact match.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "Path to the file to edit",
                        },
                        "old_str": {
                            "type": "string",
                            "description": "Exact string to replace",
                        },
                        "new_str": {
                            "type": "string",
                            "description": "New string to insert",
                        },
                    },
                    "required": ["filepath", "old_str", "new_str"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "glob",
                "description": "Find files matching a glob pattern",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Glob pattern (e.g., '**/*.py')",
                        }
                    },
                    "required": ["pattern"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "grep",
                "description": "Search for pattern in files. Returns matching lines with filenames.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Regex pattern to search for",
                        },
                        "path": {
                            "type": "string",
                            "description": "Directory to search in (default: current directory)",
                        },
                    },
                    "required": ["pattern"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a shell command and return output",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Shell command to execute",
                        }
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_date",
                "description": "Get the current date and time information",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_next_weekday",
                "description": "Calculate the date of the next occurrence of a specific weekday (e.g., 'next Tuesday')",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "weekday_name": {
                            "type": "string",
                            "description": "Name of the weekday (e.g., 'Tuesday', 'next Tuesday', 'Monday')",
                        }
                    },
                    "required": ["weekday_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "format_date",
                "description": "Format a date string from one format to another",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date_str": {
                            "type": "string",
                            "description": "The date string to format",
                        },
                        "input_format": {
                            "type": "string",
                            "description": "Format of input date (default: %Y-%m-%d)",
                        },
                        "output_format": {
                            "type": "string",
                            "description": "Desired output format (default: %B %d, %Y)",
                        },
                    },
                    "required": ["date_str"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_pdf",
                "description": "Extract and read text content from a PDF file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "Path to the PDF file to read",
                        }
                    },
                    "required": ["filepath"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "remember",
                "description": "Store important information in persistent memory for future reference",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The information to remember",
                        },
                        "topics": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional topics/tags for categorization",
                        },
                        "importance": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                            "description": "Importance level of this memory",
                        },
                    },
                    "required": ["content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "recall",
                "description": "Search and retrieve information from persistent memory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for in memory",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of memories to retrieve (default: 5)",
                        },
                        "topic": {
                            "type": "string",
                            "description": "Optional topic filter",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_recent_memories",
                "description": "Get the most recent memories added to the system",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Number of recent memories to retrieve (default: 10)",
                        }
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "forget",
                "description": "Delete a specific memory by ID",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "integer",
                            "description": "The ID of the memory to delete",
                        }
                    },
                    "required": ["memory_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "memory_stats",
                "description": "Get statistics about the memory system",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for any topic and get clean, ready-to-use content from top results",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query string",
                        },
                        "numResults": {
                            "type": "integer",
                            "description": "Number of results to return (1-100, default: 10)",
                        },
                        "category": {
                            "type": "string",
                            "enum": ["company", "research paper", "news", "people"],
                            "description": "Optional category filter for search results",
                        },
                        "type": {
                            "type": "string",
                            "enum": [
                                "neural",
                                "fast",
                                "auto",
                                "deep",
                                "deep-reasoning",
                                "instant",
                            ],
                            "description": "Search type: auto (default), neural, fast, deep, deep-reasoning, instant",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "send_tapback",
                "description": "Send an iMessage tapback reaction to a specific inbound Sendblue message",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message_handle": {
                            "type": "string",
                            "description": "Sendblue message handle/GUID from inbound webhook payload",
                        },
                        "reaction": {
                            "type": "string",
                            "enum": [
                                "love",
                                "like",
                                "dislike",
                                "laugh",
                                "emphasize",
                                "question",
                            ],
                            "description": "Tapback reaction type",
                        },
                        "part_index": {
                            "type": "integer",
                            "description": "Optional non-negative part index for multi-part messages",
                        },
                    },
                    "required": ["message_handle", "reaction"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "consortium_start",
                "description": "Start a consortium-mode task in the background",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "The user request or task prompt to send to consortium mode",
                        },
                        "task_id": {
                            "type": "string",
                            "description": "Optional task identifier. If omitted, one is generated",
                        },
                    },
                    "required": ["task"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "consortium_stop",
                "description": "Stop a running consortium-mode task",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "The consortium task identifier",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Optional reason for stopping the task",
                        },
                    },
                    "required": ["task_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "consortium_status",
                "description": "Check status for one consortium task or all consortium tasks",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "Optional task identifier. If omitted, returns all tasks",
                        }
                    },
                },
            },
        },
    ],
}

CONSORTIUM_CONTACT_MESSAGE = "contacting the consortium to decide your verdict"
CONSORTIUM_COMPLETION_MESSAGE = "the consortium has reached an agreement."
CONSORTIUM_MAX_ROUNDS = 4
IMESSAGE_SYSTEM_PROMPT_SUFFIX = (
    '"The person is using iMessage. A phone screen shows about 6-8 sentences at a time. '
    "For simple questions, you will answer in 1-2 sentences. For how-to questions, a short "
    "list with no intro. For substantive topics, 2-3 short paragraphs - roughly one screenful. "
    "For complex questions, you should keep it under two screenfuls. You will always lead with "
    "the answer. No preamble, no restating the question, no filler. If the answer is naturally "
    "list-shaped - benefits and precautions, a checklist, a comparison - keep it as a short "
    "list. Lists scan faster than prose on a small screen. These are defaults - if the person "
    'asks to go deeper or explain fully, you will respond at whatever length the topic needs."'
)

AUTO_MEMORY_ENABLED = os.environ.get("AUTO_MEMORY_ENABLED", "1").strip() != "0"
AUTO_MEMORY_MIN_MESSAGES_PER_MEMORY = max(
    2, int(os.environ.get("AUTO_MEMORY_MIN_MESSAGES_PER_MEMORY", "10"))
)
AUTO_MEMORY_MAX_MESSAGES_PER_MEMORY = max(
    AUTO_MEMORY_MIN_MESSAGES_PER_MEMORY,
    int(os.environ.get("AUTO_MEMORY_MAX_MESSAGES_PER_MEMORY", "20")),
)
AUTO_MEMORY_TARGET_MESSAGES_PER_MEMORY = min(
    AUTO_MEMORY_MAX_MESSAGES_PER_MEMORY,
    max(
        AUTO_MEMORY_MIN_MESSAGES_PER_MEMORY,
        int(os.environ.get("AUTO_MEMORY_TARGET_MESSAGES_PER_MEMORY", "15")),
    ),
)
AUTO_MEMORY_DEDUPE_THRESHOLD = max(
    0.7, min(0.99, float(os.environ.get("AUTO_MEMORY_DEDUPE_THRESHOLD", "0.9")))
)

CROSS_CHANNEL_CONTEXT_ENABLED = (
    os.environ.get("CROSS_CHANNEL_CONTEXT_ENABLED", "1").strip() != "0"
)
CROSS_CHANNEL_CONTEXT_DEFAULT_MESSAGES = max(
    2,
    int(os.environ.get("CROSS_CHANNEL_CONTEXT_DEFAULT_MESSAGES", "12")),
)
CROSS_CHANNEL_CONTEXT_MAX_MESSAGES = max(
    CROSS_CHANNEL_CONTEXT_DEFAULT_MESSAGES,
    int(os.environ.get("CROSS_CHANNEL_CONTEXT_MAX_MESSAGES", "40")),
)
CROSS_CHANNEL_SESSION_LOOKUP_LIMIT = max(
    1,
    int(os.environ.get("CROSS_CHANNEL_SESSION_LOOKUP_LIMIT", "10")),
)

DREAM_MODE_ENABLED = os.environ.get("MEMORY_DREAM_ENABLED", "1").strip() != "0"
DREAM_LOOKBACK_DAYS = max(7, int(os.environ.get("MEMORY_DREAM_LOOKBACK_DAYS", "21")))
DREAM_MIN_DAYS_FOR_PROFILE = max(7, int(os.environ.get("MEMORY_DREAM_MIN_DAYS", "14")))
DREAM_OFFPEAK_WINDOW_HOURS = max(
    2, min(12, int(os.environ.get("MEMORY_DREAM_OFFPEAK_WINDOW_HOURS", "6")))
)
DREAM_MIN_INTERVAL_HOURS = max(
    6, int(os.environ.get("MEMORY_DREAM_MIN_INTERVAL_HOURS", "24"))
)
DREAM_MIN_CANDIDATES = max(1, int(os.environ.get("MEMORY_DREAM_MIN_CANDIDATES", "4")))
DREAM_CANDIDATE_LIMIT = max(
    DREAM_MIN_CANDIDATES,
    int(os.environ.get("MEMORY_DREAM_CANDIDATE_LIMIT", "24")),
)
DREAM_MIN_AGE_HOURS = max(1, int(os.environ.get("MEMORY_DREAM_MIN_AGE_HOURS", "24")))

CONSORTIUM_MEMBERS = [
    {
        "member_id": "forensic_skeptic",
        "name": "Forensic Skeptic",
        "temperature": 0.25,
        "stance": "Evidence-first, aggressively tests claims and assumptions.",
        "persona": (
            "You are precise, surgical, and skeptical. You trust verifiable signals over hype "
            "and always pressure-test weak claims."
        ),
    },
    {
        "member_id": "practical_operator",
        "name": "Practical Operator",
        "temperature": 0.45,
        "stance": "Execution-first, prioritizes speed, simplicity, and real-world usability.",
        "persona": (
            "You are concise, tactical, and practical. You optimize for decisions users can act "
            "on today with low complexity."
        ),
    },
    {
        "member_id": "risk_sentinel",
        "name": "Risk Sentinel",
        "temperature": 0.35,
        "stance": "Risk-aware, focuses on failure modes, tradeoffs, and user downside.",
        "persona": (
            "You are vigilant, cautious, and analytical. You surface hidden risks, edge cases, "
            "and second-order consequences."
        ),
    },
    {
        "member_id": "strategic_synthesizer",
        "name": "Strategic Synthesizer",
        "temperature": 0.65,
        "stance": "Long-horizon thinker, unifies competing arguments into coherent decisions.",
        "persona": (
            "You are measured, integrative, and strategic. You reconcile disagreement and "
            "seek robust consensus under uncertainty."
        ),
    },
]


class AgentHandler:
    """Main handler for processing requests."""

    def __init__(
        self,
        memory_store: EnhancedMemoryStore,
        capability_profile: CapabilityProfile,
        example_bank: AdaptiveFewShotManager,
        task_planner: TaskPlanner,
        task_analyzer: TaskAnalyzer,
        adaptive_formatter: AdaptiveFormatter,
        skill_registry: Optional[SkillRegistry] = None,
    ):
        self.memory_store = memory_store
        self.capability_profile = capability_profile
        self.example_bank = example_bank
        self.task_planner = task_planner
        self.task_analyzer = task_analyzer
        self.adaptive_formatter = adaptive_formatter
        self.skill_registry = skill_registry
        self._consortium_tasks: dict[str, dict[str, Any]] = {}
        self._consortium_tasks_lock = asyncio.Lock()
        set_consortium_controller(self)

    def _build_request_payload_template(self) -> dict[str, Any]:
        """Build base payload with dynamic tool registration (including skills)."""
        payload = BASE_PAYLOAD.copy()
        payload["tools"] = list(BASE_PAYLOAD.get("tools", []))

        if self.skill_registry:
            try:
                tool_schema = self.skill_registry.build_activation_tool_schema()
                if tool_schema:
                    payload["tools"].append(tool_schema)
            except Exception:
                logger.exception("Failed to append activate_skill tool schema")

        return payload

    def get_available_skills_summary(self) -> list[dict[str, str]]:
        """Return discovered skills for user-facing listings."""
        if not self.skill_registry:
            return []

        try:
            self.skill_registry.refresh_if_due()
            return [
                {"name": skill.name, "description": skill.description}
                for skill in self.skill_registry.list_skills(include_model_hidden=True)
            ]
        except Exception:
            logger.exception("Failed to list available skills")
            return []

    def activate_skill_for_session(
        self,
        session_id: Optional[str],
        skill_name: str,
        source: str = "user",
    ) -> dict[str, Any]:
        """Activate a skill explicitly for the current session."""
        if not self.skill_registry:
            return {"success": False, "error": "Skill support is not configured"}

        try:
            return self.skill_registry.activate_skill(
                name=skill_name,
                session_id=session_id,
                source=source,
            )
        except Exception as exc:
            logger.exception("Failed to activate skill %s", skill_name)
            return {"success": False, "error": str(exc)}

    def clear_session_skills(self, session_id: Optional[str]) -> None:
        """Clear per-session activated skill cache."""
        if not self.skill_registry:
            return
        try:
            self.skill_registry.clear_session_active_skills(session_id)
        except Exception:
            logger.exception("Failed clearing session skills for %s", session_id)

    def _utc_timestamp(self) -> str:
        """Return UTC timestamp with second precision."""
        return datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        )

    def _normalize_consortium_task_id(self, task_id: Optional[str]) -> str:
        """Normalize task IDs to safe characters used in API/tool results."""
        if task_id is None:
            return ""

        normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(task_id).strip())
        normalized = normalized.strip("_")
        return normalized[:80]

    def _generate_consortium_task_id(self) -> str:
        """Generate a unique consortium task ID."""
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"consortium_task_{stamp}_{uuid.uuid4().hex[:8]}"

    def _consortium_task_snapshot(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Return a public-safe consortium task snapshot."""
        runner_task = entry.get("runner_task")
        is_running = isinstance(runner_task, asyncio.Task) and not runner_task.done()

        return {
            "task_id": entry.get("task_id", ""),
            "task": entry.get("task", ""),
            "status": entry.get("status", "unknown"),
            "created_at": entry.get("created_at"),
            "updated_at": entry.get("updated_at"),
            "completed_at": entry.get("completed_at"),
            "running": is_running,
            "cancel_reason": entry.get("cancel_reason", ""),
            "error": entry.get("error", ""),
            "result": entry.get("result", ""),
        }

    async def _update_consortium_task(self, task_id: str, **updates: Any) -> None:
        """Apply updates to a consortium task if it still exists."""
        async with self._consortium_tasks_lock:
            task = self._consortium_tasks.get(task_id)
            if task is None:
                return

            task.update(updates)
            task["updated_at"] = self._utc_timestamp()

    async def _execute_consortium_task(self, task_id: str, user_query: str) -> None:
        """Run one consortium task to completion in the background."""
        try:
            custom_prompt = self.memory_store.get_system_prompt() or ""
            async with aiohttp.ClientSession() as session:
                result = await self._run_consortium_mode(
                    user_query=user_query,
                    session=session,
                    custom_prompt=custom_prompt,
                    memory_context="",
                )

            completion_time = self._utc_timestamp()
            await self._update_consortium_task(
                task_id,
                status="completed",
                completed_at=completion_time,
                result=result,
                error="",
            )
        except asyncio.CancelledError:
            cancel_reason = "Stopped by request"
            async with self._consortium_tasks_lock:
                task = self._consortium_tasks.get(task_id)
                if task and str(task.get("cancel_reason", "")).strip():
                    cancel_reason = str(task.get("cancel_reason")).strip()

            completion_time = self._utc_timestamp()
            await self._update_consortium_task(
                task_id,
                status="stopped",
                completed_at=completion_time,
                cancel_reason=cancel_reason,
                error="",
            )
            raise
        except Exception as exc:
            logger.exception("Consortium task %s failed", task_id)
            completion_time = self._utc_timestamp()
            await self._update_consortium_task(
                task_id,
                status="failed",
                completed_at=completion_time,
                error=str(exc),
            )

    async def start_consortium_task(
        self, task: str, task_id: Optional[str] = None
    ) -> dict[str, Any]:
        """Start a new consortium task and return its initial status."""
        task_text = str(task or "").strip()
        if not task_text:
            return {"success": False, "error": "task must be a non-empty string"}

        normalized_task_id = self._normalize_consortium_task_id(task_id)
        if task_id and not normalized_task_id:
            return {
                "success": False,
                "error": "task_id must include letters, numbers, '-' or '_'",
            }

        if not normalized_task_id:
            normalized_task_id = self._generate_consortium_task_id()

        async with self._consortium_tasks_lock:
            if normalized_task_id in self._consortium_tasks:
                return {
                    "success": False,
                    "error": f"Task already exists: {normalized_task_id}",
                }

            now = self._utc_timestamp()
            runner_task = asyncio.create_task(
                self._execute_consortium_task(
                    task_id=normalized_task_id,
                    user_query=task_text,
                )
            )

            self._consortium_tasks[normalized_task_id] = {
                "task_id": normalized_task_id,
                "task": task_text,
                "status": "running",
                "created_at": now,
                "updated_at": now,
                "completed_at": None,
                "cancel_reason": "",
                "error": "",
                "result": "",
                "runner_task": runner_task,
            }
            snapshot = self._consortium_task_snapshot(
                self._consortium_tasks[normalized_task_id]
            )

        return {
            "success": True,
            "message": f"Started consortium task {normalized_task_id}",
            "task": snapshot,
        }

    async def stop_consortium_task(
        self, task_id: str, reason: str = ""
    ) -> dict[str, Any]:
        """Stop a running consortium task."""
        normalized_task_id = self._normalize_consortium_task_id(task_id)
        if not normalized_task_id:
            return {"success": False, "error": "task_id is required"}

        async with self._consortium_tasks_lock:
            task = self._consortium_tasks.get(normalized_task_id)
            if task is None:
                return {
                    "success": False,
                    "error": f"Task not found: {normalized_task_id}",
                }

            runner_task = task.get("runner_task")
            status = str(task.get("status", ""))
            if status in {"completed", "failed", "stopped"} or not isinstance(
                runner_task, asyncio.Task
            ):
                return {
                    "success": True,
                    "message": f"Task {normalized_task_id} is already {status or 'finished'}",
                    "task": self._consortium_task_snapshot(task),
                }

            task["status"] = "stopping"
            task["cancel_reason"] = (
                str(reason).strip() if str(reason).strip() else "Stopped by request"
            )
            task["updated_at"] = self._utc_timestamp()
            snapshot = self._consortium_task_snapshot(task)

        if isinstance(runner_task, asyncio.Task) and not runner_task.done():
            runner_task.cancel()

        return {
            "success": True,
            "message": f"Stopping consortium task {normalized_task_id}",
            "task": snapshot,
        }

    async def get_consortium_task_status(
        self, task_id: Optional[str] = None
    ) -> dict[str, Any]:
        """Get status for one consortium task or all consortium tasks."""
        normalized_task_id = self._normalize_consortium_task_id(task_id)

        async with self._consortium_tasks_lock:
            if normalized_task_id:
                task = self._consortium_tasks.get(normalized_task_id)
                if task is None:
                    return {
                        "success": False,
                        "error": f"Task not found: {normalized_task_id}",
                    }

                return {
                    "success": True,
                    "task": self._consortium_task_snapshot(task),
                }

            tasks = [
                self._consortium_task_snapshot(task)
                for task in self._consortium_tasks.values()
            ]

        tasks.sort(key=lambda item: item.get("created_at") or "", reverse=True)
        running_count = sum(
            1 for task in tasks if task.get("status") in {"running", "stopping"}
        )

        return {
            "success": True,
            "count": len(tasks),
            "running_count": running_count,
            "tasks": tasks,
        }

    def _detect_consortium_contact_intent(self, user_query: str) -> tuple[bool, bool]:
        """Return (should_contact_consortium, is_confident) for routing decisions."""
        if not user_query:
            return False, False

        query = re.sub(r"\s+", " ", user_query.strip().lower())
        if not query:
            return False, False

        hard_no_contact_patterns = [
            r"\btell\b[^.?!]*\bconsortium\b[^.?!]*\b(?:fuck|screw|shove|damn)\b",
            r"\bconsortium\b[^.?!]*\b(?:fuck|screw|shove|damn)\b",
        ]
        if any(re.search(pattern, query) for pattern in hard_no_contact_patterns):
            return False, True

        intent_score = 0
        ambiguity_score = 0

        direct_request_patterns = [
            r"^\s*(?:please\s+)?(?:ask|consult|contact|call|use|run|invoke|loop in|bring in)\b[^.?!]*\b(?:the\s+)?consortium\b",
            r"\b(?:can|could|would|will)\s+you\b[^.?!]*\b(?:ask|consult|contact|call|use|run|invoke)\b[^.?!]*\b(?:the\s+)?consortium\b",
            r"\b(?:please|i want you to|let's|let us)\b[^.?!]*\b(?:ask|consult|contact|call|use|run|invoke)\b[^.?!]*\b(?:the\s+)?consortium\b",
        ]
        if any(re.search(pattern, query) for pattern in direct_request_patterns):
            intent_score += 4

        mode_activation_patterns = [
            r"\b(?:use|enable|activate|switch to|turn on|run)\b[^.?!]*\bconsortium mode\b",
            r"\bconsortium mode\b[^.?!]*\b(?:please|now|for this|for this decision)\b",
        ]
        if any(re.search(pattern, query) for pattern in mode_activation_patterns):
            intent_score += 3

        if "consortium" in query and re.search(
            r"\b(ask|consult|contact|call|use|run|invoke|loop in|bring in)\b", query
        ):
            intent_score += 1

        informational_patterns = [
            r"^\s*(?:what|who|why|how|when|where)\b[^.?!]*\bconsortium\b\??$",
            r"\b(?:what is|define|definition of|meaning of|explain|describe|history of|example of)\b[^.?!]*\bconsortium\b",
            r"\b(?:the word|the term|the phrase)\b[^.?!]*\bconsortium\b",
            r"\b(?:did you know|fyi|for your information)\b[^.?!]*\b(?:contact|call|ask|consult|invoke|use)\b[^.?!]*\bconsortium\b",
            r"\b(?:you can|one can|people can)\b[^.?!]*\b(?:contact|call|ask|consult|invoke|use)\b[^.?!]*\bconsortium\b",
        ]
        if any(re.search(pattern, query) for pattern in informational_patterns):
            ambiguity_score += 4

        sarcastic_patterns = [
            r"\b(?:yeah|sure|right)\b[^.?!]*\b(?:totally|obviously)\b[^.?!]*\bconsortium\b",
            r"\bconsortium\b[^.?!]*\bwould\b[^.?!]*\blove to hear that\b",
            r"\b(?:lol|lmao|as if)\b[^.?!]*\bconsortium\b",
        ]
        if any(re.search(pattern, query) for pattern in sarcastic_patterns):
            ambiguity_score += 5

        hypothetical_patterns = [
            r"\b(?:should i|when should i|would it make sense to|if needed)\b[^.?!]*\b(?:contact|ask|consult|use|invoke)\b[^.?!]*\bconsortium\b",
            r"\b(?:should we|would we)\b[^.?!]*\b(?:contact|ask|consult|use|invoke)\b[^.?!]*\bconsortium\b",
        ]
        if any(re.search(pattern, query) for pattern in hypothetical_patterns):
            ambiguity_score += 2

        negative_intent_patterns = [
            r"\b(?:do not|don't|dont|without|no need to|avoid)\b[^.?!]*\bconsortium\b",
        ]
        if any(re.search(pattern, query) for pattern in negative_intent_patterns):
            ambiguity_score += 5

        should_contact = intent_score >= 3 and intent_score > ambiguity_score
        is_confident = abs(intent_score - ambiguity_score) >= 2 and (
            intent_score >= 3 or ambiguity_score >= 3
        )
        return should_contact, is_confident

    def _should_use_consortium_mode(self, user_query: str) -> bool:
        """Backward-compatible consortium router gate."""
        should_contact, is_confident = self._detect_consortium_contact_intent(
            user_query
        )
        return should_contact and is_confident

    @staticmethod
    def _coerce_json_dict(raw_text: str) -> dict[str, Any]:
        """Parse a JSON object from model output, tolerating wrapper text."""
        if not raw_text:
            return {}

        candidate = raw_text.strip()
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass

        match = re.search(r"\{[\s\S]*\}", candidate)
        if not match:
            return {}

        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _coerce_float(value: Any, default: float = 0.0) -> float:
        """Convert arbitrary values to float with a safe default."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _parse_state_timestamp(value: Any) -> Optional[datetime.datetime]:
        """Parse a timestamp from persisted state fields."""
        if not value:
            return None
        candidate = str(value).strip()
        if not candidate:
            return None

        try:
            return datetime.datetime.fromisoformat(candidate.replace("Z", "+00:00"))
        except ValueError:
            pass

        try:
            return datetime.datetime.fromisoformat(candidate.replace(" ", "T"))
        except ValueError:
            return None

    @staticmethod
    def _normalize_channel_name(raw_channel: str) -> str:
        """Normalize channel aliases to canonical values."""
        normalized = re.sub(r"[^a-z]", "", (raw_channel or "").lower())
        if normalized == "telegram":
            return "telegram"
        if normalized in {"imessage", "imsg"}:
            return "imessage"
        return ""

    @staticmethod
    def _channel_session_prefix(channel_name: str) -> str:
        """Map canonical channel name to session ID prefix."""
        if channel_name == "telegram":
            return "tg_"
        if channel_name == "imessage":
            return "imessage_"
        return ""

    def _parse_cross_channel_recall_request(
        self,
        user_query: str,
    ) -> Optional[dict[str, Any]]:
        """Detect explicit cross-channel recall requests and parse options."""
        if not CROSS_CHANNEL_CONTEXT_ENABLED or not user_query:
            return None

        normalized_query = re.sub(r"\s+", " ", user_query.strip().lower())
        normalized_query = normalized_query.replace("’", "'")
        if not normalized_query:
            return None

        match = re.search(
            r"\b(?:remember|recall)\b[^.?!\n]{0,180}"
            r"\bwhat\s+we(?:\s+[a-z']+){0,3}\s+"
            r"(?:talk(?:ing|ed)|discuss(?:ing|ed))\s+about\b"
            r"[^.?!\n]{0,120}\b(?:on|from)\s+(?:the\s+)?"
            r"(?P<channel>i[\s-]?message|telegram)\b",
            normalized_query,
        )
        if not match:
            return None

        channel_name = self._normalize_channel_name(match.group("channel"))
        if not channel_name:
            return None

        message_limit = CROSS_CHANNEL_CONTEXT_DEFAULT_MESSAGES
        limit_match = re.search(
            r"\blast\s+(?P<count>\d{1,3})(?:\s*(?:messages?|msgs?))?\b",
            normalized_query,
        )
        if not limit_match:
            limit_match = re.search(
                r"\b(?P<count>\d{1,3})\s*(?:messages?|msgs?)\b",
                normalized_query,
            )

        if limit_match:
            message_limit = int(limit_match.group("count"))

        message_limit = max(1, min(CROSS_CHANNEL_CONTEXT_MAX_MESSAGES, message_limit))

        return {
            "channel": channel_name,
            "limit": message_limit,
        }

    def _resolve_cross_channel_session_id(
        self,
        current_session_id: Optional[str],
        channel_name: str,
    ) -> Optional[str]:
        """Resolve which session to use for a requested channel recall."""
        session_prefix = self._channel_session_prefix(channel_name)
        if not session_prefix:
            return None

        if current_session_id and current_session_id.startswith(session_prefix):
            return current_session_id

        candidates = self.memory_store.get_recent_session_ids_by_prefix(
            session_prefix,
            limit=CROSS_CHANNEL_SESSION_LOOKUP_LIMIT,
        )
        if current_session_id:
            candidates = [sid for sid in candidates if sid != current_session_id]

        if not candidates:
            return None

        return candidates[0]

    def _build_cross_channel_context(
        self,
        user_query: str,
        session_id: Optional[str],
    ) -> str:
        """Build context block from recent channel-specific conversation messages."""
        recall_request = self._parse_cross_channel_recall_request(user_query)
        if not recall_request:
            return ""

        channel_name = str(recall_request.get("channel", "")).strip().lower()
        message_limit = int(
            recall_request.get("limit", CROSS_CHANNEL_CONTEXT_DEFAULT_MESSAGES)
        )
        session_prefix = self._channel_session_prefix(channel_name)
        if not session_prefix:
            return ""

        source_session_id = self._resolve_cross_channel_session_id(
            current_session_id=session_id,
            channel_name=channel_name,
        )
        if not source_session_id:
            return (
                "\n\n"
                f"[Cross-channel recall requested for {channel_name}, but no prior "
                f"{channel_name} conversation history is available.]\n"
            )

        history = self.memory_store.get_recent_conversation_messages_for_prefix(
            session_prefix=session_prefix,
            limit=message_limit,
            session_id=source_session_id,
        )
        if not history:
            return (
                "\n\n"
                f"[Cross-channel recall requested for {channel_name}, but no messages "
                "were found in the selected channel session.]\n"
            )

        lines: list[str] = []
        for index, message in enumerate(reversed(history), start=1):
            content = re.sub(
                r"\s+",
                " ",
                _content_to_text(message.get("content", "")).strip(),
            )
            if not content:
                continue

            if len(content) > 320:
                content = f"{content[:317]}..."

            role = str(message.get("role", "assistant")).strip().lower()
            if role not in {"user", "assistant", "system"}:
                role = "assistant"

            lines.append(f"{index}. {role}: {content}")

        if not lines:
            return (
                "\n\n"
                f"[Cross-channel recall requested for {channel_name}, but matching "
                "messages were empty after normalization.]\n"
            )

        return (
            "\n\n"
            f"[Cross-channel context injected from {channel_name} "
            f"({source_session_id}), last {len(lines)} messages]:\n"
            + "\n".join(lines)
            + "\n"
        )

    @staticmethod
    def _normalize_request_metadata(
        request_metadata: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Keep only normalized request metadata fields that matter to the model."""
        if not isinstance(request_metadata, dict):
            return {}

        normalized: dict[str, Any] = {}

        message_handle = request_metadata.get("message_handle")
        if isinstance(message_handle, str) and message_handle.strip():
            normalized["message_handle"] = message_handle.strip()

        part_index = request_metadata.get("part_index")
        if isinstance(part_index, str) and part_index.strip().isdigit():
            part_index = int(part_index.strip())
        if isinstance(part_index, int) and part_index >= 0:
            normalized["part_index"] = part_index

        return normalized

    @staticmethod
    def _build_imessage_handle_context_line(
        entry_index: int,
        message_handle: str,
        part_index: Optional[int],
        content: str,
        label: str,
    ) -> str:
        """Format one iMessage tapback target line for the system prompt."""
        normalized_content = re.sub(r"\s+", " ", (content or "").strip())
        if len(normalized_content) > 160:
            normalized_content = f"{normalized_content[:157]}..."

        part_suffix = ""
        if isinstance(part_index, int) and part_index >= 0:
            part_suffix = f"; part_index={part_index}"

        if normalized_content:
            return (
                f"{entry_index}. {label}: message_handle={message_handle}{part_suffix}; "
                f'message="{normalized_content}"'
            )

        return f"{entry_index}. {label}: message_handle={message_handle}{part_suffix}"

    def _build_imessage_handle_context(
        self,
        user_query: str,
        session_id: Optional[str],
        request_metadata: Optional[dict[str, Any]],
    ) -> str:
        """Expose recent iMessage handle IDs so send_tapback can be called concretely."""
        if not session_id or not session_id.startswith("imessage_"):
            return ""

        normalized_request_metadata = self._normalize_request_metadata(request_metadata)
        history = self.memory_store.get_conversation_history(
            session_id=session_id,
            limit=IMESSAGE_HANDLE_CONTEXT_LIMIT,
        )

        lines: list[str] = []
        seen_targets: set[tuple[str, Optional[int]]] = set()

        current_handle = normalized_request_metadata.get("message_handle")
        current_part_index = normalized_request_metadata.get("part_index")
        if isinstance(current_handle, str) and current_handle:
            target = (current_handle, current_part_index)
            seen_targets.add(target)
            lines.append(
                self._build_imessage_handle_context_line(
                    entry_index=len(lines) + 1,
                    message_handle=current_handle,
                    part_index=current_part_index,
                    content=user_query,
                    label="current inbound message",
                )
            )

        for message in history:
            if str(message.get("role", "")).strip().lower() != "user":
                continue

            metadata = self._normalize_request_metadata(message.get("metadata"))
            message_handle = metadata.get("message_handle")
            if not isinstance(message_handle, str) or not message_handle:
                continue

            part_index = metadata.get("part_index")
            target = (message_handle, part_index)
            if target in seen_targets:
                continue

            seen_targets.add(target)
            lines.append(
                self._build_imessage_handle_context_line(
                    entry_index=len(lines) + 1,
                    message_handle=message_handle,
                    part_index=part_index,
                    content=_content_to_text(message.get("content", "")),
                    label="recent inbound message",
                )
            )

        if not lines:
            return ""

        return (
            "\n\n"
            "[Available iMessage tapback handles from current and recent messages]:\n"
            "Use these exact message_handle values when calling send_tapback. Include part_index when present.\n"
            + "\n".join(lines)
            + "\n"
        )

    @staticmethod
    def _normalize_memory_text(content: str) -> str:
        """Normalize memory text for robust duplicate checks."""
        normalized = re.sub(r"\s+", " ", (content or "")).strip().lower()
        normalized = re.sub(r"[^a-z0-9\s]", "", normalized)
        return normalized

    def _is_duplicate_memory(self, content: str, session_id: Optional[str]) -> bool:
        """Check whether a memory candidate is near-duplicate of recent memories."""
        candidate = self._normalize_memory_text(content)
        if not candidate:
            return True

        recent_memories = self.memory_store.get_recent_memories(limit=80)
        for memory in recent_memories:
            metadata = memory.metadata or {}
            if metadata.get("type") == "system_prompt":
                continue

            # Prefer session-scoped dedupe, but still avoid obvious global duplicates.
            if session_id and metadata.get("session_id") not in (None, session_id):
                continue

            existing = self._normalize_memory_text(memory.content)
            if not existing:
                continue

            if existing == candidate:
                return True
            if candidate in existing or existing in candidate:
                return True

            if (
                SequenceMatcher(None, candidate, existing).ratio()
                >= AUTO_MEMORY_DEDUPE_THRESHOLD
            ):
                return True

        return False

    def _should_capture_auto_memory(
        self,
        session_id: Optional[str],
        message_count: int,
    ) -> tuple[bool, dict[str, Any]]:
        """Decide if automatic memory capture should run for this turn."""
        if not AUTO_MEMORY_ENABLED:
            return False, {"reason": "auto_memory_disabled"}

        if not session_id:
            return False, {"reason": "missing_session_id"}

        if message_count < AUTO_MEMORY_MIN_MESSAGES_PER_MEMORY:
            return False, {
                "reason": "too_early",
                "message_count": message_count,
            }

        cadence_stats = self.memory_store.get_session_memory_cadence(
            session_id=session_id,
            memory_types={"explicit_memory", "auto_memory", "long_term_memory"},
        )
        memory_count = int(cadence_stats.get("memory_count", 0) or 0)
        latest_message_index = cadence_stats.get("latest_message_index")

        ratio = float("inf") if memory_count == 0 else (message_count / memory_count)
        messages_since_last = (
            message_count - latest_message_index
            if isinstance(latest_message_index, int)
            else message_count
        )

        should_capture = False
        reason = "cadence_ok"

        if memory_count == 0 and message_count >= AUTO_MEMORY_MIN_MESSAGES_PER_MEMORY:
            should_capture = True
            reason = "bootstrap_memory"
        elif ratio > AUTO_MEMORY_MAX_MESSAGES_PER_MEMORY:
            should_capture = True
            reason = "ratio_above_max"
        elif messages_since_last >= AUTO_MEMORY_TARGET_MESSAGES_PER_MEMORY:
            should_capture = True
            reason = "stale_memory_interval"

        return should_capture, {
            "reason": reason,
            "message_count": message_count,
            "memory_count": memory_count,
            "ratio": ratio,
            "messages_since_last": messages_since_last,
        }

    async def _extract_auto_memory_candidate(
        self,
        session: aiohttp.ClientSession,
        user_query: str,
        assistant_response: str,
    ) -> dict[str, Any]:
        """Ask the model for one high-value memory candidate from the latest exchange."""
        payload = BASE_PAYLOAD.copy()
        payload["model"] = MODEL_ID
        payload["temperature"] = 0.15
        payload["max_tokens"] = 320
        payload["tools"] = []
        payload["messages"] = [
            {
                "role": "system",
                "content": (
                    "Extract at most one durable memory from the latest user/assistant exchange. "
                    "Only store stable preferences, long-term projects, personal facts, recurring constraints, "
                    "or high-impact commitments. Ignore one-off requests and generic chatter. "
                    "Respond with JSON only using this schema: "
                    '{"store": boolean, "content": string, "importance": "low"|"medium"|"high", '
                    '"topics": string[], "significance": number, "reason": string}.'
                ),
            },
            {
                "role": "user",
                "content": (
                    "User message:\n"
                    f"{user_query}\n\n"
                    "Assistant response:\n"
                    f"{assistant_response}"
                ),
            },
        ]

        response_data = await api_call_with_retry(
            session,
            BASE_URL,
            payload,
            {"Authorization": f"Bearer {API_KEY}"},
        )
        raw_output = await process_response(
            response_data,
            payload["messages"],
            session,
            BASE_URL,
            API_KEY,
            payload,
        )

        parsed = self._coerce_json_dict(raw_output)
        should_store = parsed.get("store", False)
        if isinstance(should_store, str):
            should_store = should_store.strip().lower() in {"true", "1", "yes"}
        else:
            should_store = bool(should_store)

        content = str(parsed.get("content", "")).strip()
        if len(content) < 20:
            should_store = False

        importance = str(parsed.get("importance", "medium")).strip().lower()
        if importance not in {"low", "medium", "high"}:
            importance = "medium"

        topics_raw = parsed.get("topics", [])
        topics: list[str] = []
        if isinstance(topics_raw, list):
            for item in topics_raw:
                topic = str(item).strip().lower()
                if not topic or topic in topics:
                    continue
                topics.append(topic)
                if len(topics) >= 6:
                    break

        significance = max(
            0.0,
            min(1.0, self._coerce_float(parsed.get("significance"), default=0.5)),
        )

        return {
            "store": should_store,
            "content": content,
            "importance": importance,
            "topics": topics,
            "significance": significance,
            "reason": str(parsed.get("reason", "")).strip(),
        }

    async def _auto_capture_memory(
        self,
        session: aiohttp.ClientSession,
        session_id: Optional[str],
        user_query: str,
        assistant_response: str,
    ):
        """Automatically capture memory while keeping a bounded messages-to-memory ratio."""
        if not session_id or not user_query or not assistant_response:
            return

        message_count = self.memory_store.get_conversation_message_count(session_id)
        should_capture, cadence = self._should_capture_auto_memory(
            session_id=session_id,
            message_count=message_count,
        )
        if not should_capture:
            return

        candidate = await self._extract_auto_memory_candidate(
            session=session,
            user_query=user_query,
            assistant_response=assistant_response,
        )
        if not candidate.get("store"):
            logger.info("Auto-memory skipped by extractor for session %s", session_id)
            return

        content = str(candidate.get("content", "")).strip()
        if self._is_duplicate_memory(content, session_id=session_id):
            logger.info("Auto-memory deduped for session %s", session_id)
            return

        metadata = {
            "type": "auto_memory",
            "source": "auto_capture",
            "session_id": session_id,
            "importance": candidate.get("importance", "medium"),
            "significance": candidate.get("significance", 0.5),
            "message_index": message_count,
            "cadence_reason": cadence.get("reason"),
            "cadence_ratio": cadence.get("ratio"),
        }
        if candidate.get("reason"):
            metadata["extractor_reason"] = candidate["reason"]

        memory_id = await self.memory_store.add_memory(
            content=content,
            metadata=metadata,
            topics=candidate.get("topics", []),
            generate_embedding=True,
        )
        logger.info(
            "Auto-memory stored: id=%s session=%s reason=%s ratio=%s",
            memory_id,
            session_id,
            cadence.get("reason"),
            cadence.get("ratio"),
        )

    def _get_or_refresh_dream_profile(self) -> dict[str, Any]:
        """Compute or refresh the learned off-peak profile used for dream mode."""
        now = datetime.datetime.now()
        profile = self.memory_store.get_agent_state("dream.profile", default={})
        if not isinstance(profile, dict):
            profile = {}

        profiled_at = self._parse_state_timestamp(profile.get("profiled_at"))
        should_refresh = (
            not profile
            or not profile.get("hours")
            or not profiled_at
            or (now - profiled_at).total_seconds() >= 24 * 3600
        )

        if not should_refresh:
            return profile

        inferred = self.memory_store.infer_offpeak_hours(
            lookback_days=DREAM_LOOKBACK_DAYS,
            min_days=DREAM_MIN_DAYS_FOR_PROFILE,
            window_hours=DREAM_OFFPEAK_WINDOW_HOURS,
        )
        refreshed_profile = {
            "hours": inferred.get("hours", []),
            "reason": inferred.get("reason", "unknown"),
            "stats": inferred.get("stats", {}),
            "window_score": inferred.get("window_score", 0),
            "profiled_at": now.isoformat(timespec="seconds"),
        }
        self.memory_store.set_agent_state("dream.profile", refreshed_profile)
        return refreshed_profile

    async def _extract_dream_consolidation_payload(
        self,
        session: aiohttp.ClientSession,
        candidates_payload: str,
        candidate_ids: set[int],
    ) -> list[dict[str, Any]]:
        """Generate long-term consolidated memories from short-term memory candidates."""
        payload = BASE_PAYLOAD.copy()
        payload["model"] = MODEL_ID
        payload["temperature"] = 0.2
        payload["max_tokens"] = 700
        payload["tools"] = []
        payload["messages"] = [
            {
                "role": "system",
                "content": (
                    "You are running an offline memory consolidation pass. "
                    "Merge related short-term memories into a small set of durable long-term memories. "
                    "Only keep information likely to matter in future conversations. "
                    "Output JSON only with schema: "
                    '{"consolidated": [{"content": string, "significance": number, '
                    '"topics": string[], "source_memory_ids": number[]}]}'
                ),
            },
            {
                "role": "user",
                "content": (
                    "Candidate memories:\n"
                    f"{candidates_payload}\n\n"
                    "Return at most 6 consolidated memories."
                ),
            },
        ]

        response_data = await api_call_with_retry(
            session,
            BASE_URL,
            payload,
            {"Authorization": f"Bearer {API_KEY}"},
        )
        raw_output = await process_response(
            response_data,
            payload["messages"],
            session,
            BASE_URL,
            API_KEY,
            payload,
        )

        parsed = self._coerce_json_dict(raw_output)
        entries = parsed.get("consolidated", [])
        if not isinstance(entries, list):
            return []

        consolidated: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue

            content = str(entry.get("content", "")).strip()
            if len(content) < 25:
                continue

            significance = max(
                0.0,
                min(1.0, self._coerce_float(entry.get("significance"), default=0.5)),
            )

            topics_raw = entry.get("topics", [])
            topics: list[str] = []
            if isinstance(topics_raw, list):
                for topic_item in topics_raw:
                    topic = str(topic_item).strip().lower()
                    if topic and topic not in topics:
                        topics.append(topic)
                    if len(topics) >= 6:
                        break

            source_ids_raw = entry.get("source_memory_ids", [])
            source_ids: list[int] = []
            if isinstance(source_ids_raw, list):
                for item in source_ids_raw:
                    try:
                        source_id = int(item)
                    except (TypeError, ValueError):
                        continue
                    if source_id in candidate_ids and source_id not in source_ids:
                        source_ids.append(source_id)

            if not source_ids:
                continue

            consolidated.append(
                {
                    "content": content,
                    "significance": significance,
                    "topics": topics,
                    "source_memory_ids": source_ids,
                }
            )

            if len(consolidated) >= 6:
                break

        return consolidated

    async def _run_dream_cycle_if_due(self, session: aiohttp.ClientSession):
        """Run periodic memory consolidation during inferred off-peak hours."""
        if not DREAM_MODE_ENABLED:
            return

        profile = self._get_or_refresh_dream_profile()
        hours = profile.get("hours", [])
        if not isinstance(hours, list) or not hours:
            return

        now = datetime.datetime.now()
        if now.hour not in {int(hour) for hour in hours if isinstance(hour, int)}:
            return

        last_run_at = self._parse_state_timestamp(
            self.memory_store.get_agent_state("dream.last_run_at")
        )
        if (
            last_run_at
            and (now - last_run_at).total_seconds() < DREAM_MIN_INTERVAL_HOURS * 3600
        ):
            return

        candidates = self.memory_store.get_memories_for_consolidation(
            limit=DREAM_CANDIDATE_LIMIT,
            min_age_hours=DREAM_MIN_AGE_HOURS,
        )
        if len(candidates) < DREAM_MIN_CANDIDATES:
            return

        candidate_ids = {memory.id for memory in candidates if memory.id is not None}
        candidate_lines: list[str] = []
        for memory in candidates:
            if memory.id is None:
                continue
            importance = str((memory.metadata or {}).get("importance", "medium"))
            candidate_lines.append(
                f"id={memory.id} | importance={importance} | content={memory.content}"
            )

        consolidated = await self._extract_dream_consolidation_payload(
            session=session,
            candidates_payload="\n".join(candidate_lines),
            candidate_ids=candidate_ids,
        )

        created_ids: list[int] = []
        source_significance: dict[int, float] = {}
        for entry in consolidated:
            content = entry["content"]
            if self._is_duplicate_memory(content, session_id=None):
                continue

            significance = max(0.0, min(1.0, float(entry["significance"])))
            metadata = {
                "type": "long_term_memory",
                "source": "dream_cycle",
                "long_term": True,
                "significance": significance,
                "importance": "high" if significance >= 0.8 else "medium",
                "source_memory_ids": entry["source_memory_ids"],
                "dream_hours": hours,
            }

            new_memory_id = await self.memory_store.add_memory(
                content=content,
                metadata=metadata,
                topics=entry.get("topics", []),
                generate_embedding=True,
            )
            if new_memory_id is not None:
                created_ids.append(int(new_memory_id))

            for source_id in entry["source_memory_ids"]:
                source_significance[source_id] = max(
                    source_significance.get(source_id, 0.0),
                    significance,
                )

        for source_id, significance in source_significance.items():
            self.memory_store.mark_memories_dream_consolidated(
                [source_id], significance=significance
            )

        run_timestamp = now.isoformat(timespec="seconds")
        self.memory_store.set_agent_state("dream.last_run_at", run_timestamp)
        self.memory_store.set_agent_state(
            "dream.last_result",
            {
                "run_at": run_timestamp,
                "candidate_count": len(candidates),
                "consolidated_count": len(created_ids),
                "created_memory_ids": created_ids,
            },
        )

        if created_ids:
            logger.info(
                "Dream cycle consolidated %d long-term memories", len(created_ids)
            )

    async def _run_memory_maintenance(
        self,
        session: aiohttp.ClientSession,
        session_id: Optional[str],
        user_query: str,
        assistant_response: str,
    ):
        """Run post-response memory upkeep: cadence capture + dream consolidation."""
        try:
            await self._auto_capture_memory(
                session=session,
                session_id=session_id,
                user_query=user_query,
                assistant_response=assistant_response,
            )
        except Exception as exc:
            logger.warning("Auto-memory capture failed: %s", exc)

        try:
            await self._run_dream_cycle_if_due(session=session)
        except Exception as exc:
            logger.warning("Dream-cycle maintenance failed: %s", exc)

    def _extract_consortium_vote(
        self, messages: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Extract the latest consortium_agree tool arguments from a model turn."""
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue

            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue

            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                function_obj = tool_call.get("function", {})
                if not isinstance(function_obj, dict):
                    continue
                if function_obj.get("name") != "consortium_agree":
                    continue

                arguments = function_obj.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except Exception:
                        arguments = {}

                if not isinstance(arguments, dict):
                    arguments = {}

                return arguments

        return None

    def _normalize_consortium_vote(self, raw_vote: dict[str, Any]) -> dict[str, Any]:
        """Normalize vote payload returned via consortium_agree."""
        confidence = raw_vote.get("confidence", 1.0)
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            confidence_value = 1.0

        key_points = raw_vote.get("key_points", [])
        if not isinstance(key_points, list):
            key_points = [str(key_points)]

        return {
            "verdict": str(raw_vote.get("verdict", "")).strip(),
            "rationale": str(raw_vote.get("rationale", "")).strip(),
            "confidence": max(0.0, min(1.0, confidence_value)),
            "key_points": [
                str(point).strip() for point in key_points if str(point).strip()
            ],
        }

    def _majority_verdict(self, votes: dict[str, dict[str, Any]]) -> str:
        """Return the most common non-empty verdict from current consortium votes."""
        verdict_counts: dict[str, dict[str, Any]] = {}

        for vote in votes.values():
            verdict = str(vote.get("verdict", "")).strip()
            if not verdict:
                continue

            normalized = verdict.lower()
            if normalized not in verdict_counts:
                verdict_counts[normalized] = {"verdict": verdict, "count": 0}
            verdict_counts[normalized]["count"] += 1

        if not verdict_counts:
            return ""

        return max(verdict_counts.values(), key=lambda item: item["count"])["verdict"]

    def _build_system_content(
        self,
        custom_prompt: str,
        memory_context: str,
        plan_context: str,
        example_context: str,
        skills_catalog_context: str = "",
        active_skills_context: str = "",
        session_prompt_suffix: str = "",
        request_freshness_token: Optional[str] = None,
    ) -> str:
        """Build the canonical system prompt used by the primary agent."""
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        universal_instructions = (
            f"\n\n[Current Date and Time]: {current_time}\n"
            "\n\n[System Instructions & Tools]:\n"
            "You have access to tools via tool_calls. Use the remember() tool to store important facts about the user or session.\n"
            "If the user asks you to remember or store something, you MUST use the remember() tool.\n"
            "Use recall() to retrieve past memories.\n"
            "If you need access to current information not available to you, use the web_search() tool.\n"
            "Use consortium_start() to launch a consortium task, consortium_status() to check progress/results, and consortium_stop() to cancel an active consortium task.\n"
            "If skills are listed in <available_skills>, call activate_skill(name) before executing specialized workflows.\n"
            "Users can also explicitly activate skills with /skill-name or $skill-name. Treat that as a harness-level activation signal.\n"
            'For readability, you may split a reply into multiple chunks using <message>...</message> blocks. You may also insert <typing seconds="1.2"/> between message blocks to add brief pacing pauses. If you use this format, keep all user-visible text inside those message blocks.\n'
            "IMPORTANT: Do not use markdown formatting, code blocks, or emojis in your responses. Respond in plain text only.\n"
        )
        if request_freshness_token:
            universal_instructions += (
                f"{REQUEST_FRESHNESS_INSTRUCTION}\n"
                f"[Freshness Token]: {request_freshness_token}\n"
            )
        session_suffix = (
            f"\n\n{session_prompt_suffix}\n" if session_prompt_suffix else ""
        )

        if custom_prompt:
            return (
                custom_prompt
                + universal_instructions
                + session_suffix
                + memory_context
                + plan_context
                + example_context
                + (f"\n\n{skills_catalog_context}" if skills_catalog_context else "")
                + (f"\n\n{active_skills_context}" if active_skills_context else "")
            )

        skills_catalog_suffix = f"\n\n{skills_catalog_context}" if skills_catalog_context else ""
        active_skills_suffix = f"\n\n{active_skills_context}" if active_skills_context else ""
        return (
            "You are a helpful AI assistant with persistent memory."
            f"{universal_instructions}{session_suffix}{memory_context}{plan_context}{example_context}"
            f"{skills_catalog_suffix}{active_skills_suffix}"
        )

    def _build_request_freshness_token(self) -> str:
        """Return a unique token for the current visible response generation."""
        return uuid.uuid4().hex

    def _get_session_prompt_suffix(self, session_id: Optional[str]) -> str:
        """Return channel-specific prompt guidance for the active session."""
        if session_id and session_id.startswith("imessage_"):
            return IMESSAGE_SYSTEM_PROMPT_SUFFIX
        return ""

    async def _generate_consortium_acknowledgement(
        self,
        session: aiohttp.ClientSession,
        user_query: str,
        system_content: str,
    ) -> str:
        """Generate a short acknowledgement before consortium deliberation."""
        payload = BASE_PAYLOAD.copy()
        payload["model"] = MODEL_ID
        payload["temperature"] = 0.3
        payload["max_tokens"] = 120
        payload["tools"] = []
        payload["messages"] = [
            {"role": "system", "content": system_content},
            {
                "role": "user",
                "content": (
                    "The user requested consortium mode. Reply with one short acknowledgement "
                    "that you are consulting the consortium now and will return with a final "
                    "answer next. Do not provide analysis or a verdict yet.\n\n"
                    f"User request:\n{user_query}"
                ),
            },
        ]

        try:
            response_data = await api_call_with_retry(
                session,
                BASE_URL,
                payload,
                {"Authorization": f"Bearer {API_KEY}"},
            )
            acknowledgement = await process_response(
                response_data,
                payload["messages"],
                session,
                BASE_URL,
                API_KEY,
                payload,
            )
            return (acknowledgement or "").strip() or CONSORTIUM_CONTACT_MESSAGE
        except Exception:
            logger.exception("Failed to generate consortium acknowledgement")
            return CONSORTIUM_CONTACT_MESSAGE

    def _format_consortium_transcript(
        self, transcript: list[dict[str, Any]], max_entries: int = 24
    ) -> str:
        """Format consortium transcript for model consumption."""
        if not transcript:
            return "No debate turns yet."

        lines: list[str] = []
        for entry in transcript[-max_entries:]:
            agreed_suffix = " [AGREED]" if entry.get("agreed") else ""
            lines.append(
                f"Round {entry.get('round')} | {entry.get('name')} "
                f"({entry.get('member_id')} via {entry.get('model')}){agreed_suffix}: "
                f"{str(entry.get('content', '')).strip()}"
            )

        return "\n".join(lines)

    def _format_anonymized_panel_notes(
        self, transcript: list[dict[str, Any]], max_entries: int = 16
    ) -> str:
        """Format prior turns without member identity to reduce context bleed."""
        if not transcript:
            return "No prior panel notes available."

        lines: list[str] = []
        for index, entry in enumerate(transcript[-max_entries:], start=1):
            note_type = "agreement" if entry.get("agreed") else "debate"
            lines.append(
                f"Panel note {index} ({note_type}): {str(entry.get('content', '')).strip()}"
            )

        return "\n".join(lines)

    async def _run_consortium_turn(
        self,
        session: aiohttp.ClientSession,
        user_query: str,
        transcript: list[dict[str, Any]],
        agreed_member_ids: set[str],
        vote_by_member: dict[str, dict[str, Any]],
        member: dict[str, Any],
        round_number: int,
        custom_prompt: str,
        memory_context: str,
        skills_catalog_context: str,
        active_skills_context: str,
        session_prompt_suffix: str,
        request_freshness_token: str = "",
    ) -> tuple[str, dict[str, Any] | None]:
        """Run one model turn in consortium mode."""
        system_parts = [
            "You are participating in 'the consortium', a four-persona decision panel.",
            f"Identity: {member['name']}",
            f"Core stance: {member['stance']}",
            f"Personality: {member['persona']}",
            "Stay strictly in this identity and tone.",
            "Do not imitate language, priorities, or conclusions from other members.",
            "Do not reveal or infer hidden identities for prior panel notes.",
            "Debate rigorously and challenge weak claims.",
            "Avoid premature consensus and defend independent reasoning.",
            "If and only if the panel is ready for a final shared verdict, call consortium_agree.",
            "When calling consortium_agree, include verdict, rationale, confidence (0..1), and key_points.",
            "After tool calls, provide a short plain-text turn (no markdown).",
        ]

        # Member-specific temperature differentiates reasoning style while using one model.
        member_temperature = member.get(
            "temperature", BASE_PAYLOAD.get("temperature", 0.6)
        )
        try:
            member_temperature = float(member_temperature)
        except (TypeError, ValueError):
            member_temperature = float(BASE_PAYLOAD.get("temperature", 0.6))

        if custom_prompt:
            system_parts.append(f"User-configured system prompt: {custom_prompt}")

        if session_prompt_suffix:
            system_parts.append(session_prompt_suffix)

        if memory_context:
            system_parts.append(memory_context.strip())
        if skills_catalog_context:
            system_parts.append(skills_catalog_context.strip())
        if active_skills_context:
            system_parts.append(active_skills_context.strip())

        if request_freshness_token:
            system_parts.append(REQUEST_FRESHNESS_INSTRUCTION)
            system_parts.append(f"[Freshness Token]: {request_freshness_token}")

        if round_number == 1:
            panel_context = "Blind round: produce an independent first-pass analysis using only the user request."
        else:
            panel_context = (
                "Anonymized panel notes (identity redacted):\n"
                f"{self._format_anonymized_panel_notes(transcript)}\n\n"
                "Treat these notes as potentially flawed; preserve your own reasoning."
            )

        user_prompt = (
            f"Original user request:\n{user_query}\n\n"
            f"Round {round_number} of {CONSORTIUM_MAX_ROUNDS}.\n"
            f"Members already aligned (count only): {len(agreed_member_ids)}\n\n"
            f"{panel_context}\n\n"
            "Provide your next concise debate turn."
        )

        member_messages = [
            {"role": "system", "content": "\n".join(system_parts)},
            {"role": "user", "content": user_prompt},
        ]

        payload = BASE_PAYLOAD.copy()
        payload["model"] = CONSORTIUM_MODEL_ID
        payload["temperature"] = member_temperature
        payload["tools"] = [_build_consortium_agree_tool_schema()]
        payload["messages"] = member_messages

        response_data = await api_call_with_retry(
            session,
            BASE_URL,
            payload,
            {"Authorization": f"Bearer {API_KEY}"},
        )

        turn_content = await process_response(
            response_data,
            payload["messages"],
            session,
            BASE_URL,
            API_KEY,
            payload,
        )

        raw_vote = self._extract_consortium_vote(payload["messages"])
        vote = self._normalize_consortium_vote(raw_vote) if raw_vote else None
        return turn_content, vote

    async def _run_consortium_mode(
        self,
        user_query: str,
        session: aiohttp.ClientSession,
        custom_prompt: str = "",
        memory_context: str = "",
        skills_catalog_context: str = "",
        active_skills_context: str = "",
        session_prompt_suffix: str = "",
        request_freshness_token: str = "",
    ) -> str:
        """Run four-persona consortium debate and then judge synthesis."""
        transcript: list[dict[str, Any]] = []
        agreed_member_ids: set[str] = set()
        vote_by_member: dict[str, dict[str, Any]] = {}

        for round_number in range(1, CONSORTIUM_MAX_ROUNDS + 1):
            # Rotate speaking order each round to reduce first-mover effects.
            round_offset = (round_number - 1) % len(CONSORTIUM_MEMBERS)
            ordered_members = (
                CONSORTIUM_MEMBERS[round_offset:] + CONSORTIUM_MEMBERS[:round_offset]
            )

            for member in ordered_members:
                if member["member_id"] in agreed_member_ids:
                    continue

                try:
                    turn_content, vote = await self._run_consortium_turn(
                        session=session,
                        user_query=user_query,
                        transcript=transcript,
                        agreed_member_ids=agreed_member_ids,
                        vote_by_member=vote_by_member,
                        member=member,
                        round_number=round_number,
                        custom_prompt=custom_prompt,
                        memory_context=memory_context,
                        skills_catalog_context=skills_catalog_context,
                        active_skills_context=active_skills_context,
                        session_prompt_suffix=session_prompt_suffix,
                        request_freshness_token=request_freshness_token,
                    )
                except Exception as exc:
                    logger.exception(
                        "Consortium turn failed for %s", member["member_id"]
                    )
                    turn_content = f"I encountered an internal issue this round: {exc}"
                    vote = None

                transcript.append(
                    {
                        "round": round_number,
                        "name": member["name"],
                        "member_id": member["member_id"],
                        "model": CONSORTIUM_MODEL_ID,
                        "content": turn_content,
                        "agreed": vote is not None,
                    }
                )

                if vote is not None:
                    agreed_member_ids.add(member["member_id"])
                    vote_by_member[member["member_id"]] = vote

            if len(agreed_member_ids) == len(CONSORTIUM_MEMBERS):
                break

        consensus_reached = len(agreed_member_ids) == len(CONSORTIUM_MEMBERS)
        aligned_member_names = [
            member["name"]
            for member in CONSORTIUM_MEMBERS
            if member["member_id"] in agreed_member_ids
        ]

        judge_system_parts = [
            "You are the judge model for 'the consortium'.",
            "You receive a transcript from four debating specialist personas.",
            "Synthesize their arguments into one final plain-text answer for the user.",
            "Be clear, decisive, and actionable.",
        ]
        if custom_prompt:
            judge_system_parts.append(f"User-configured system prompt: {custom_prompt}")
        if session_prompt_suffix:
            judge_system_parts.append(session_prompt_suffix)
        if memory_context:
            judge_system_parts.append(memory_context.strip())
        if skills_catalog_context:
            judge_system_parts.append(skills_catalog_context.strip())
        if active_skills_context:
            judge_system_parts.append(active_skills_context.strip())
        if request_freshness_token:
            judge_system_parts.append(REQUEST_FRESHNESS_INSTRUCTION)
            judge_system_parts.append(f"[Freshness Token]: {request_freshness_token}")

        judge_user_prompt = (
            f"Original user request:\n{user_query}\n\n"
            "Consortium status:\n"
            f"- Full consensus via consortium_agree: {'yes' if consensus_reached else 'no'}\n"
            f"- Aligned members: {', '.join(aligned_member_names) if aligned_member_names else 'none'}\n\n"
            "Agreement payloads:\n"
            f"{json.dumps(vote_by_member, ensure_ascii=True, indent=2)}\n\n"
            "Debate transcript:\n"
            f"{self._format_consortium_transcript(transcript, max_entries=80)}\n\n"
            "Return the final answer to the user."
        )

        judge_payload = BASE_PAYLOAD.copy()
        judge_payload["model"] = CONSORTIUM_MODEL_ID
        judge_payload["tools"] = []
        judge_payload["messages"] = [
            {"role": "system", "content": "\n".join(judge_system_parts)},
            {"role": "user", "content": judge_user_prompt},
        ]

        judge_response_data = await api_call_with_retry(
            session,
            BASE_URL,
            judge_payload,
            {"Authorization": f"Bearer {API_KEY}"},
        )

        judge_content = await process_response(
            judge_response_data,
            judge_payload["messages"],
            session,
            BASE_URL,
            API_KEY,
            judge_payload,
        )

        if not judge_content:
            fallback_verdict = self._majority_verdict(vote_by_member)
            judge_content = (
                fallback_verdict
                if fallback_verdict
                else "The consortium debated the request but could not finalize a clear verdict."
            )

        status_line = (
            CONSORTIUM_COMPLETION_MESSAGE
            if consensus_reached
            else "the consortium reached a partial consensus and escalated to the judge model."
        )

        return f"{status_line}\n{judge_content}"

    async def analyze_and_plan_task(self, user_query: str) -> Optional[Any]:
        """Analyze user query and create a task plan if it's complex enough."""
        if not user_query:
            return None

        # Analyze the task
        task = self.task_analyzer.analyze(user_query)

        # Only plan if task is complex enough
        if task.type in [TaskType.GENERIC.value] and len(user_query) < 100:
            return None

        return task

    async def handle(
        self,
        request,
        session_id: Optional[str] = None,
        interim_response_callback: Optional[Callable[[str], Awaitable[None]]] = None,
        request_metadata: Optional[dict[str, Any]] = None,
    ):
        """Handle a request and return the response content."""
        data = request
        normalized_request_metadata = self._normalize_request_metadata(request_metadata)

        # Get the user's message for memory retrieval
        user_messages = [m for m in data["messages"] if m.get("role") == "user"]
        user_query = (
            _content_to_text(user_messages[-1].get("content")) if user_messages else ""
        )

        # Analyze task and create plan for complex queries
        task_plan = None
        task = None
        if user_query:
            task = await self.analyze_and_plan_task(user_query)
            if task:
                task_plan = self.task_planner.plan(task)
                logger.info(f"Task type: {task.type}, Steps: {len(task_plan.steps)}")

        # Retrieve relevant memories for context
        memory_context = ""
        if user_query:
            try:
                relevant_memories = await self.memory_store.search_memories(
                    query=user_query, top_k=8, threshold=0.15
                )
                if relevant_memories:
                    weighted_memories: list[tuple[Any, float, float, float]] = []
                    for memory, similarity in relevant_memories:
                        metadata = memory.metadata or {}
                        significance = max(
                            0.0,
                            min(
                                1.0,
                                self._coerce_float(
                                    metadata.get("significance"), default=0.0
                                ),
                            ),
                        )
                        if metadata.get("type") == "long_term_memory":
                            significance = max(significance, 0.6)

                        effective_score = similarity + (0.25 * significance)
                        weighted_memories.append(
                            (memory, similarity, significance, effective_score)
                        )

                    weighted_memories.sort(key=lambda item: item[3], reverse=True)
                    memory_context = (
                        "\n\n[Relevant memories from past conversations]:\n"
                    )
                    for i, (memory, _, significance, _) in enumerate(
                        weighted_memories[:3], 1
                    ):
                        if significance >= 0.75:
                            memory_context += (
                                f"{i}. [high significance {significance:.2f}] "
                                f"{memory.content}\n"
                            )
                        else:
                            memory_context += f"{i}. {memory.content}\n"
            except Exception as e:
                logger.warning(f"Failed to retrieve memories: {e}")

        if user_query:
            try:
                cross_channel_context = self._build_cross_channel_context(
                    user_query=user_query,
                    session_id=session_id,
                )
                if cross_channel_context:
                    memory_context += cross_channel_context
            except Exception as e:
                logger.warning(f"Failed to build cross-channel context: {e}")

        if user_query:
            try:
                imessage_handle_context = self._build_imessage_handle_context(
                    user_query=user_query,
                    session_id=session_id,
                    request_metadata=normalized_request_metadata,
                )
                if imessage_handle_context:
                    memory_context += imessage_handle_context
            except Exception as e:
                logger.warning(f"Failed to build iMessage handle context: {e}")

        # Get few-shot examples for the task type
        few_shot_examples = []
        if task:
            few_shot_examples = self.example_bank.get_examples_for_task(
                task_type=task.type,
                query=user_query,
                max_examples=self.capability_profile.get_max_examples(),
            )

        # Build system prompt with adaptive formatting
        custom_prompt = self.memory_store.get_system_prompt() or ""
        session_prompt_suffix = self._get_session_prompt_suffix(session_id)
        skills_catalog_context = ""
        active_skills_context = ""

        if self.skill_registry:
            try:
                self.skill_registry.refresh_if_due()
                skills_catalog_context = (
                    self.skill_registry.build_available_skills_catalog()
                )
                active_skills_context = self.skill_registry.build_active_skills_context(
                    session_id
                )
            except Exception:
                logger.exception("Failed building skills context")

        # Add task plan context if available
        plan_context = ""
        if task_plan and len(task_plan.steps) > 1:
            task_type_label = task.type if task else "generic"
            plan_context = f"\n\n[Task Plan - {task_type_label}]:\n"
            for i, step in enumerate(task_plan.steps, 1):
                plan_context += f"{i}. {step.description}\n"

        # Add few-shot examples if available
        example_context = ""
        if few_shot_examples:
            example_context = "\n\n[Examples]:\n"
            for ex in few_shot_examples:
                example_context += f"Input: {ex['input']}\n"
                example_context += f"Output: {ex['output']}\n\n"

        request_freshness_token = self._build_request_freshness_token()

        system_content = self._build_system_content(
            custom_prompt=custom_prompt,
            memory_context=memory_context,
            plan_context=plan_context,
            example_context=example_context,
            skills_catalog_context=skills_catalog_context,
            active_skills_context=active_skills_context,
            session_prompt_suffix=session_prompt_suffix,
            request_freshness_token=request_freshness_token,
        )

        should_contact_consortium = False
        confident_consortium_intent = False
        if user_query:
            (
                should_contact_consortium,
                confident_consortium_intent,
            ) = self._detect_consortium_contact_intent(user_query)

        if should_contact_consortium and confident_consortium_intent:
            acknowledgement = ""
            acknowledgement_delivered = False

            async with aiohttp.ClientSession() as session:
                acknowledgement = await self._generate_consortium_acknowledgement(
                    session=session,
                    user_query=user_query,
                    system_content=system_content,
                )

                if interim_response_callback and acknowledgement:
                    try:
                        await interim_response_callback(acknowledgement)
                        acknowledgement_delivered = True
                    except Exception:
                        logger.exception(
                            "Failed to deliver interim consortium acknowledgement"
                        )

                session_token = set_tool_runtime_session(session_id)
                try:
                    content = await self._run_consortium_mode(
                        user_query=user_query,
                        session=session,
                        custom_prompt=custom_prompt,
                        memory_context=memory_context,
                        skills_catalog_context=skills_catalog_context,
                        active_skills_context=active_skills_context,
                        session_prompt_suffix=session_prompt_suffix,
                        request_freshness_token=request_freshness_token,
                    )
                finally:
                    reset_tool_runtime_session(session_token)

                if acknowledgement and not acknowledgement_delivered:
                    content = f"{acknowledgement}\n\n{content}"

                if session_id and user_query:
                    self.memory_store.add_conversation_message(
                        role="user",
                        content=user_query,
                        session_id=session_id,
                        metadata=normalized_request_metadata or None,
                    )
                if session_id and acknowledgement_delivered and acknowledgement:
                    self.memory_store.add_conversation_message(
                        role="assistant", content=acknowledgement, session_id=session_id
                    )
                if session_id and content:
                    self.memory_store.add_conversation_message(
                        role="assistant", content=content, session_id=session_id
                    )

                await self._run_memory_maintenance(
                    session=session,
                    session_id=session_id,
                    user_query=user_query,
                    assistant_response=content,
                )

            if task and content and not content.startswith("Error:"):
                self.example_bank.auto_feedback(task.type, success=True, efficiency=1.0)

            print(content)
            return content

        system_message = {
            "role": "system",
            "content": system_content,
        }

        # Build rolling context window
        messages = self._build_rolling_context(
            system_message=system_message,
            current_messages=data.get("messages", []),
            session_id=session_id,
            context_window=128000,  # Kimi K2 context window
            buffer_tokens=2000,
        )

        # Create a fresh payload for this request (avoid global mutation)
        payload_template = self._build_request_payload_template()
        current_payload = payload_template.copy()
        current_payload["messages"] = messages

        async with aiohttp.ClientSession() as session:
            # Make initial API call with retry logic
            response_data = await api_call_with_retry(
                session,
                BASE_URL,
                current_payload,
                {"Authorization": f"Bearer {API_KEY}"},
            )

            session_token = set_tool_runtime_session(session_id)
            try:
                content = await process_response(
                    response_data,
                    current_payload["messages"],
                    session,
                    BASE_URL,
                    API_KEY,
                    payload_template,
                )
            finally:
                reset_tool_runtime_session(session_token)

            # Store conversation in memory
            if session_id and user_query:
                self.memory_store.add_conversation_message(
                    role="user",
                    content=user_query,
                    session_id=session_id,
                    metadata=normalized_request_metadata or None,
                )
            if session_id and content:
                self.memory_store.add_conversation_message(
                    role="assistant", content=content, session_id=session_id
                )

            await self._run_memory_maintenance(
                session=session,
                session_id=session_id,
                user_query=user_query,
                assistant_response=content,
            )

            # Provide feedback on examples if task was completed
            if task and content and not content.startswith("Error:"):
                self.example_bank.auto_feedback(task.type, success=True, efficiency=1.0)

            print(content)
            return content

    def _build_rolling_context(
        self,
        system_message: dict,
        current_messages: list,
        session_id: Optional[str],
        context_window: int = 128000,
        buffer_tokens: int = 2000,
    ) -> list:
        """Build a rolling context window from persistent memory.

        Formula: context_window - system_prompt - current_input - buffer = available_history
        """

        # Rough token estimation (4 chars per token)
        def estimate_tokens(text: str) -> int:
            return len(text) // 4

        # Calculate tokens for system message
        system_tokens = estimate_tokens(system_message.get("content", ""))

        # Calculate tokens for current user input
        current_input_tokens = 0
        for msg in current_messages:
            current_input_tokens += estimate_tokens(
                _content_to_text(msg.get("content", ""))
            )

        # Calculate available tokens for history
        available_tokens = (
            context_window - system_tokens - current_input_tokens - buffer_tokens
        )

        if available_tokens <= 0:
            logger.warning("Context window full, only using system + current messages")
            return [system_message] + current_messages

        # Retrieve conversation history from persistent memory
        history = []
        if session_id:
            try:
                # Get recent conversation history (will be sorted DESC by created_at)
                history = self.memory_store.get_conversation_history(
                    session_id=session_id,
                    limit=100,  # Get more than we need, then filter by tokens
                )
                # Reverse to get chronological order (oldest first)
                history = list(reversed(history))
            except Exception as e:
                logger.warning(f"Failed to retrieve conversation history: {e}")

        # Build rolling window from history until we hit token limit
        selected_history = []
        current_tokens = 0

        for msg in history:
            msg_tokens = estimate_tokens(_content_to_text(msg.get("content", "")))
            if current_tokens + msg_tokens > available_tokens:
                break

            selected_history.append(
                {
                    "role": msg["role"],
                    "content": msg["content"],
                }
            )
            current_tokens += msg_tokens

        # Combine: system + history + current messages
        final_messages = [system_message] + selected_history + current_messages

        total_tokens = system_tokens + current_tokens + current_input_tokens
        logger.info(
            f"Context window: {total_tokens} tokens used ({len(selected_history)} history messages)"
        )

        return final_messages
