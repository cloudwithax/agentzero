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
from api import api_call_with_retry, process_response, safe_strip_markdown
from skills import SkillRegistry
from reminder_tasks import ReminderScheduler
from tools import (
    extract_assistant_name_from_user_text,
    extract_assistant_name_from_memory_content,
    extract_user_name_from_memory_content,
    normalize_memory_candidate_from_user_text,
    reset_tool_runtime_session,
    set_advisor_controller,
    set_consortium_controller,
    set_reviewer_controller,
    set_reminder_controller,
    set_tool_runtime_session,
    set_acp_agent,
)
from acp import ACPAgent
from self_heal import SelfHealManager
from prompt_templates import get_template

logger = logging.getLogger(__name__)

IMESSAGE_HANDLE_CONTEXT_LIMIT = 50
TELEGRAM_REACTION_CONTEXT_LIMIT = 50
REQUEST_FRESHNESS_INSTRUCTION = (
    "[Request Freshness]: This turn includes a one-time freshness token to discourage "
    "cache reuse and repeated phrasing. Treat the request as new and answer independently."
)
DELIVERY_MESSAGE_BLOCK_PATTERN = re.compile(
    r"<message>\s*(?P<message>.*?)\s*</message>",
    re.IGNORECASE | re.DOTALL,
)
DELIVERY_TYPING_DIRECTIVE_PATTERN = re.compile(
    r"<typing(?:\s+seconds\s*=\s*['\"]?\d+(?:\.\d+)?['\"]?)?\s*/>",
    re.IGNORECASE,
)
DELIVERY_MESSAGE_TAG_PATTERN = re.compile(r"</?message>", re.IGNORECASE)
INTERNAL_PROMPT_RESIDUE_PATTERNS = (
    re.compile(
        r"(?i)(?:^|[\s\n])no more loops,\s*no more narration(?:\s*[—-]\s*just the reaction)?[.!]?"
    ),
    re.compile(r"(?i)(?:^|[\s\n])tool execution is mandatory[.!]?"),
    re.compile(r"(?i)(?:^|[\s\n])act,\s*don't narrate[.!]?"),
    re.compile(r"(?i)(?:^|[\s\n])do not narrate(?: what you will do)?[.!]?"),
    re.compile(
        r"(?im)^[ \t>*_`~-]*(?:send_tapback|send_telegram_reaction|send_reaction)\s*:[^\n]*$"
    ),
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


def _strip_delivery_directives(content: Any) -> str:
    """Remove integration-only delivery wrappers from plain assistant text."""
    normalized = "" if content is None else str(content)
    if not normalized.strip():
        return ""

    message_chunks = [
        match.group("message").strip()
        for match in DELIVERY_MESSAGE_BLOCK_PATTERN.finditer(normalized)
        if isinstance(match.group("message"), str) and match.group("message").strip()
    ]
    if message_chunks:
        return "\n\n".join(message_chunks).strip()

    sanitized = DELIVERY_TYPING_DIRECTIVE_PATTERN.sub("", normalized)
    sanitized = DELIVERY_MESSAGE_TAG_PATTERN.sub("", sanitized)
    return sanitized.strip()


def _strip_internal_prompt_residue(content: str) -> str:
    """Remove obvious leaked internal control phrases from visible assistant text."""
    cleaned = "" if content is None else str(content)
    if not cleaned.strip():
        return ""

    for pattern in INTERNAL_PROMPT_RESIDUE_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)

    cleaned = re.sub(r"\s+([.,!?])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip(" \t\r\n-—")


def _extract_iteration_from_error(error_string: str) -> int:
    match = re.search(r"iteration=(\d+)", error_string or "")
    return int(match.group(1)) if match else 0


def _extract_tools_from_error(error_string: str) -> list[str]:
    match = re.search(r"tools_executed=\[([^\]]*)\]", error_string or "")
    if not match:
        return []
    raw = match.group(1).strip()
    return [t.strip().strip("'\"") for t in raw.split(",") if t.strip()]


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


# Configuration — OpenAI-compatible client endpoint takes priority over NVIDIA
_OPENAI_CLIENT_BASE_URL = os.environ.get("OPENAI_CLIENT_BASE_URL", "").strip()
_OPENAI_CLIENT_API_KEY = os.environ.get("OPENAI_CLIENT_API_KEY", "").strip()
_OPENAI_CLIENT_MODEL = os.environ.get("OPENAI_CLIENT_MODEL", "").strip()

if _OPENAI_CLIENT_BASE_URL and _OPENAI_CLIENT_API_KEY:
    BASE_URL = _OPENAI_CLIENT_BASE_URL.rstrip("/")
    if not BASE_URL.endswith("/chat/completions"):
        BASE_URL = BASE_URL + "/chat/completions"
    API_KEY = _OPENAI_CLIENT_API_KEY
    _DEFAULT_MODEL_ID = _OPENAI_CLIENT_MODEL or os.environ.get(
        "MODEL_ID", "moonshotai/kimi-k2-instruct-0905"
    )
else:
    BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
    API_KEY = os.environ.get(
        "NVIDIA_API_KEY",
        "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",
    )
    _DEFAULT_MODEL_ID = os.environ.get("MODEL_ID", "moonshotai/kimi-k2-instruct-0905")

EXECUTOR_MODEL_ID = (
    os.environ.get("EXECUTOR_MODEL", "").strip()
    or os.environ.get("PREFLIGHT_MODEL", "").strip()
    or _DEFAULT_MODEL_ID
)
ADVISOR_MODEL_ID = (
    os.environ.get("ADVISOR_MODEL", "").strip()
    or os.environ.get("MAIN_MODEL", "").strip()
    or _DEFAULT_MODEL_ID
)
REVIEWER_MODEL_ID = (
    os.environ.get("REVIEWER_MODEL", "").strip()
    or os.environ.get("ADVISOR_MODEL", "").strip()
    or os.environ.get("MAIN_MODEL", "").strip()
    or _DEFAULT_MODEL_ID
)
PRIMARY_MODEL_ID = ADVISOR_MODEL_ID
MODEL_ID = PRIMARY_MODEL_ID
CONSORTIUM_MODEL_ID = os.environ.get("CONSORTIUM_MODEL", MODEL_ID).strip() or MODEL_ID

# Workspace — all agent-created files must live here.
_DEFAULT_WORKSPACE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "workspace"
)
AGENT_WORKSPACE = os.path.normpath(
    os.environ.get("AGENT_WORKSPACE", _DEFAULT_WORKSPACE).strip() or _DEFAULT_WORKSPACE
)
ADVISOR_RESPONSE_MAX_TOKENS = int(os.environ.get("ADVISOR_RESPONSE_MAX_TOKENS", "1200"))
REVIEWER_RESPONSE_MAX_TOKENS = int(
    os.environ.get("REVIEWER_RESPONSE_MAX_TOKENS", "1400")
)

# Base payload template (do not mutate globally)
BASE_PAYLOAD = {
    "model": MODEL_ID,
    "temperature": 0.6,
    "top_p": 0.9,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "max_tokens": 32768,
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
                "description": "Store important information in persistent memory for future reference. Keep the subject correct: if the user names or renames the assistant, remember that as assistant identity, not as a user fact.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The information to remember, written from the correct subject perspective",
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
                "name": "generate_image",
                "description": "Generate an image from a text prompt and save it as a PNG/JPG file in the workspace. Use this for any image creation, illustration, or visual generation task.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Detailed text description of the image to generate",
                        },
                        "filename": {
                            "type": "string",
                            "description": "Output filename (e.g. robot_cafe.png). Saved to workspace.",
                        },
                        "width": {
                            "type": "integer",
                            "description": "Image width in pixels (default 1024)",
                        },
                        "height": {
                            "type": "integer",
                            "description": "Image height in pixels (default 1024)",
                        },
                        "model": {
                            "type": "string",
                            "enum": ["flux", "turbo"],
                            "description": "Model: flux (high quality, default) or turbo (faster)",
                        },
                    },
                    "required": ["prompt", "filename"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "add_skill",
                "description": (
                    "Fetch a skill from a URL, scan it for prompt-injection attacks, "
                    "and install it for the current and future sessions. Use when the user "
                    "mentions a URL to a SKILL.md or asks you to add/install a skill from a link."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "HTTPS or HTTP URL pointing to a SKILL.md file",
                        },
                        "auto_activate": {
                            "type": "boolean",
                            "description": "Automatically activate the skill for this session after install (default: true)",
                        },
                    },
                    "required": ["url"],
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
                "name": "send_telegram_reaction",
                "description": "Send a Telegram emoji reaction to a specific inbound Telegram message",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "chat_id": {
                            "type": "integer",
                            "description": "Telegram chat ID where the inbound message was received",
                        },
                        "message_id": {
                            "type": "integer",
                            "description": "Telegram message ID to react to",
                        },
                        "reaction": {
                            "type": "string",
                            "enum": [
                                "like",
                                "love",
                                "dislike",
                                "laugh",
                                "emphasize",
                                "question",
                                "party",
                                "clap",
                                "cry",
                                "sob",
                                "scream",
                                "mindblown",
                                "pray",
                                "cool",
                                "100",
                                "hearts",
                                "starry",
                                "angry",
                                "devil",
                                "ghost",
                                "clown",
                                "shrug",
                                "eyes",
                                "kiss",
                                "hug",
                                "salute",
                                "nerd",
                                "trophy",
                                "heartbreak",
                                "heartonfire",
                                "vomit",
                                "poo",
                                "ok",
                                "whale",
                                "dove",
                                "unicorn",
                                "moai",
                                "banana",
                                "strawberry",
                                "champagne",
                                "hotdog",
                                "yawn",
                                "woozy",
                                "sleep",
                                "scared",
                                "handshake",
                                "halo",
                                "grin",
                                "alien",
                                "lightning",
                                "moon",
                                "cursing",
                                "zany",
                                "lipstick",
                                "nailpolish",
                                "middlefinger",
                                "coder",
                                "pill",
                                "pumpkin",
                                "cupid",
                                "hearteyes",
                                "writing",
                                "santa",
                                "christmas",
                                "snowman",
                                "seenoevil",
                            ],
                            "description": "Reaction name: like (👍), love (❤), dislike (👎), laugh (🤣), emphasize (🔥), question (🤔), party (🎉), clap (👏), cool (😎), 100 (💯), pray (🙏), eyes (👀), etc.",
                        },
                    },
                    "required": ["chat_id", "message_id", "reaction"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "consult_advisor",
                "description": (
                    "Consult the advisor model for a concise strategy when you hit a hard "
                    "decision, branching choice, architecture tradeoff, or repeated failure "
                    "mid-run. The advisor reads the same shared context and returns a plan; "
                    "after the tool result, continue executing yourself."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The exact decision or blocker you need the advisor to resolve",
                        },
                        "context": {
                            "type": "string",
                            "description": "Optional extra context, options under consideration, or recent failed attempts",
                        },
                    },
                    "required": ["question"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "consult_reviewer",
                "description": (
                    "Consult the reviewer model for a concise implementation review focused "
                    "on bugs, regressions, missing validation, and residual risks. Use this "
                    "after inspection or implementation work when you want a fast risk pass "
                    "before finalizing."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The exact thing you want the reviewer to evaluate",
                        },
                        "context": {
                            "type": "string",
                            "description": "Optional extra context, change summary, known risks, or open questions",
                        },
                    },
                    "required": ["question"],
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
                "name": "reminder_create",
                "description": "Create a cron-based reminder task (one-off or recurring), with optional direct AI output generation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cron": {
                            "type": "string",
                            "description": "Cron schedule with 5 fields: minute hour day month weekday",
                        },
                        "message": {
                            "type": "string",
                            "description": "Reminder text to deliver when run_ai is false",
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Optional session ID to write scheduled outputs into conversation history",
                        },
                        "one_off": {
                            "type": "boolean",
                            "description": "If true, task executes once at the next cron match",
                        },
                        "run_ai": {
                            "type": "boolean",
                            "description": "If true, run direct AI inference using ai_prompt",
                        },
                        "ai_prompt": {
                            "type": "string",
                            "description": "Prompt used for direct AI inference when run_ai is true",
                        },
                        "task_id": {
                            "type": "string",
                            "description": "Optional custom task ID",
                        },
                        "name": {
                            "type": "string",
                            "description": "Optional human-friendly task name",
                        },
                    },
                    "required": ["cron"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "reminder_list",
                "description": "List scheduled reminder tasks",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "include_disabled": {
                            "type": "boolean",
                            "description": "Include completed/cancelled tasks",
                        }
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "reminder_status",
                "description": "Get status for one reminder task",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "Reminder task identifier",
                        }
                    },
                    "required": ["task_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "reminder_cancel",
                "description": "Cancel a reminder task",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "Reminder task identifier",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Optional cancellation reason",
                        },
                    },
                    "required": ["task_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "reminder_run_now",
                "description": "Run a reminder task immediately",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "Reminder task identifier",
                        }
                    },
                    "required": ["task_id"],
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
        {
            "type": "function",
            "function": {
                "name": "acp_register_service",
                "description": "Register capabilities with the ACP network for service discovery",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "service_name": {
                            "type": "string",
                            "description": "Name of the service to register",
                        },
                        "capabilities": {
                            "type": "string",
                            "description": "Comma-separated list of capabilities (e.g., 'memory_access,web_search,tool_execution')",
                        },
                        "description": {
                            "type": "string",
                            "description": "Description of the service",
                        },
                        "endpoint": {
                            "type": "string",
                            "description": "Service endpoint address",
                        },
                    },
                    "required": [
                        "service_name",
                        "capabilities",
                        "description",
                        "endpoint",
                    ],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "acp_discover_peers",
                "description": "Discover peers in the ACP network by capability or name",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query_type": {
                            "type": "string",
                            "enum": ["all", "capability", "name"],
                            "description": "Type of query: 'all', 'capability', or 'name'",
                        },
                        "query_value": {
                            "type": "string",
                            "description": "Optional capability or name pattern to search for",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "acp_send_message",
                "description": "Send a message to another agent via ACP",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "recipient_id": {
                            "type": "string",
                            "description": "Target agent ID",
                        },
                        "message": {
                            "type": "string",
                            "description": "Message content",
                        },
                        "payload": {
                            "type": "string",
                            "description": "Optional structured payload data",
                        },
                        "secure": {
                            "type": "boolean",
                            "description": "Whether to encrypt and sign the message (default: true)",
                        },
                    },
                    "required": ["recipient_id", "message"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "acp_get_registry",
                "description": "Get the current ACP registry status and all registered services",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "acp_list_peers",
                "description": "List all known peers in the ACP network",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "store_credential",
                "description": "Store a credential (API key, token, password, etc.) in the encrypted vault. Values are encrypted at rest.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "A unique name for this credential (e.g. 'github_token', 'aws_secret_key')",
                        },
                        "value": {
                            "type": "string",
                            "description": "The secret value to store",
                        },
                        "description": {
                            "type": "string",
                            "description": "Optional description of what this credential is for",
                        },
                    },
                    "required": ["key", "value"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_credential",
                "description": "Retrieve a stored credential from the encrypted vault by its key name",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "The name of the credential to retrieve",
                        }
                    },
                    "required": ["key"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delete_credential",
                "description": "Delete a stored credential from the encrypted vault",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "The name of the credential to delete",
                        }
                    },
                    "required": ["key"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_credentials",
                "description": "List all stored credential keys (values are never exposed in listings)",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "self_heal_status",
                "description": "Get current status of the self-healing subsystem including heal history, worktree state, and configuration",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
    ],
}

CONSORTIUM_CONTACT_MESSAGE = "contacting the consortium to decide your verdict"
CONSORTIUM_COMPLETION_MESSAGE = "the consortium has reached an agreement."
CONSORTIUM_MAX_ROUNDS = 4

FINAL_RESPONSE_MAX_TOKENS = int(os.environ.get("FINAL_RESPONSE_MAX_TOKENS", "150"))

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
SESSION_CONTINUITY_ENABLED = (
    os.environ.get("SESSION_CONTINUITY_ENABLED", "1").strip() != "0"
)
SESSION_CONTINUITY_MEMORY_SCAN_LIMIT = max(
    12,
    int(os.environ.get("SESSION_CONTINUITY_MEMORY_SCAN_LIMIT", "60")),
)
SESSION_CONTINUITY_HISTORY_LOOKBACK = max(
    2,
    int(os.environ.get("SESSION_CONTINUITY_HISTORY_LOOKBACK", "8")),
)
SESSION_CONTINUITY_MAX_PROFILE_LINES = max(
    1,
    int(os.environ.get("SESSION_CONTINUITY_MAX_PROFILE_LINES", "4")),
)
SESSION_CONTINUITY_MAX_HISTORY_LINES = max(
    1,
    int(os.environ.get("SESSION_CONTINUITY_MAX_HISTORY_LINES", "4")),
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
        acp_agent: Optional[ACPAgent] = None,
    ):
        self.memory_store = memory_store
        self.capability_profile = capability_profile
        self.example_bank = example_bank
        self.task_planner = task_planner
        self.task_analyzer = task_analyzer
        self.adaptive_formatter = adaptive_formatter
        self.skill_registry = skill_registry
        self.acp_agent = acp_agent
        self._self_heal_manager: Optional[SelfHealManager] = None
        self._consortium_tasks: dict[str, dict[str, Any]] = {}
        self._consortium_tasks_lock = asyncio.Lock()
        self.reminder_scheduler = ReminderScheduler(
            memory_store=self.memory_store,
            ai_runner=self._run_direct_ai_inference,
            delivery_callback=self._deliver_reminder_output,
        )
        set_advisor_controller(self)
        set_reviewer_controller(self)
        set_consortium_controller(self)
        set_reminder_controller(self)
        set_acp_agent(acp_agent)

    async def start_reminder_scheduler(self) -> None:
        """Start the reminder scheduler background loop."""
        await self.reminder_scheduler.start()

    def set_self_heal_manager(self, manager: SelfHealManager) -> None:
        self._self_heal_manager = manager

    async def create_reminder_task(
        self,
        cron: str,
        message: str = "",
        session_id: Optional[str] = None,
        one_off: bool = False,
        run_ai: bool = False,
        ai_prompt: str = "",
        task_id: Optional[str] = None,
        name: str = "",
    ) -> dict[str, Any]:
        """Create one reminder task."""
        return await self.reminder_scheduler.create_task(
            cron=cron,
            message=message,
            session_id=session_id,
            one_off=one_off,
            run_ai=run_ai,
            ai_prompt=ai_prompt,
            task_id=task_id,
            name=name,
        )

    async def list_reminder_tasks(
        self, include_disabled: bool = True
    ) -> dict[str, Any]:
        """List reminder tasks."""
        return await self.reminder_scheduler.list_tasks(
            include_disabled=include_disabled,
        )

    async def get_reminder_task_status(self, task_id: str) -> dict[str, Any]:
        """Get reminder task status."""
        return await self.reminder_scheduler.get_task_status(task_id=task_id)

    async def cancel_reminder_task(
        self, task_id: str, reason: str = ""
    ) -> dict[str, Any]:
        """Cancel a reminder task."""
        return await self.reminder_scheduler.cancel_task(
            task_id=task_id,
            reason=reason,
        )

    async def run_reminder_task_now(self, task_id: str) -> dict[str, Any]:
        """Run one reminder task immediately."""
        return await self.reminder_scheduler.run_task_now(task_id=task_id)

    async def _run_direct_ai_inference(self, prompt: str, task_id: str) -> str:
        """Run direct AI inference for reminder tasks with tools disabled."""
        prompt_text = str(prompt or "").strip()
        if not prompt_text:
            return ""

        system_content = get_template("direct_inference", {})

        payload = BASE_PAYLOAD.copy()
        payload["model"] = MODEL_ID
        payload["tools"] = []
        payload["messages"] = [
            {"role": "system", "content": system_content},
            {
                "role": "user",
                "content": f"Scheduled task {task_id}:\n{prompt_text}",
            },
        ]

        async with aiohttp.ClientSession() as session:
            response_data = await api_call_with_retry(
                session,
                BASE_URL,
                payload,
                {"Authorization": f"Bearer {API_KEY}"},
            )
            output = await process_response(
                response_data,
                payload["messages"],
                session,
                BASE_URL,
                API_KEY,
                payload,
            )

        return (output or "").strip()

    async def consult_advisor(
        self,
        question: str,
        context: str = "",
        session_id: Optional[str] = None,
        shared_messages: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """Consult the advisor model using the current turn's shared context."""
        normalized_question = str(question or "").strip()
        if not normalized_question:
            return {"success": False, "error": "question is required"}

        runtime_messages = [
            dict(message)
            for message in (shared_messages or [])
            if isinstance(message, dict)
        ]
        base_system_content = ""
        non_system_messages: list[dict[str, Any]] = []

        for message in runtime_messages:
            if message.get("role") == "system" and not base_system_content:
                base_system_content = _content_to_text(message.get("content", ""))
                continue
            non_system_messages.append(message)

        advisor_system_content = get_template(
            "advisor_consultation",
            {
                "base_system_content": base_system_content,
                "session_id": session_id or "",
            },
        )
        consultation_request = (
            "[Executor consultation request]\n"
            f"Decision needed: {normalized_question}\n"
            f"Additional context: {str(context or '').strip() or 'None provided.'}\n\n"
            "Return a concise operational memo for the executor with these exact sections:\n"
            "Decision:\nWhy:\nNext steps:\nRisks:\n"
        )

        payload = BASE_PAYLOAD.copy()
        payload["model"] = ADVISOR_MODEL_ID
        payload["tools"] = []
        payload["max_tokens"] = ADVISOR_RESPONSE_MAX_TOKENS
        payload["messages"] = [
            {"role": "system", "content": advisor_system_content},
            *non_system_messages,
            {"role": "user", "content": consultation_request},
        ]

        async with aiohttp.ClientSession() as session:
            response_data = await api_call_with_retry(
                session,
                BASE_URL,
                payload,
                {"Authorization": f"Bearer {API_KEY}"},
            )

        if "error" in response_data:
            error_msg = response_data["error"].get("message", "Unknown API error")
            return {"success": False, "error": error_msg}

        choices = response_data.get("choices") or []
        if not choices:
            return {"success": False, "error": "No response from advisor model"}

        advice = safe_strip_markdown(
            _content_to_text(choices[0].get("message", {}).get("content", ""))
        ).strip()
        if not advice:
            return {"success": False, "error": "Advisor returned empty advice"}

        return {
            "success": True,
            "advisor_model": ADVISOR_MODEL_ID,
            "question": normalized_question,
            "advice": advice,
        }

    async def consult_reviewer(
        self,
        question: str,
        context: str = "",
        session_id: Optional[str] = None,
        shared_messages: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """Consult the reviewer model using the current turn's shared context."""
        normalized_question = str(question or "").strip()
        if not normalized_question:
            return {"success": False, "error": "question is required"}

        runtime_messages = [
            dict(message)
            for message in (shared_messages or [])
            if isinstance(message, dict)
        ]
        base_system_content = ""
        non_system_messages: list[dict[str, Any]] = []

        for message in runtime_messages:
            if message.get("role") == "system" and not base_system_content:
                base_system_content = _content_to_text(message.get("content", ""))
                continue
            non_system_messages.append(message)

        reviewer_system_content = get_template(
            "reviewer_consultation",
            {
                "base_system_content": base_system_content,
                "session_id": session_id or "",
            },
        )
        review_request = (
            "[Executor review request]\n"
            f"Review target: {normalized_question}\n"
            f"Additional context: {str(context or '').strip() or 'None provided.'}\n\n"
            "Return a concise operational memo for the executor with these exact sections:\n"
            "Verdict:\nFindings:\nFixes:\nResidual risks:\n"
        )

        payload = BASE_PAYLOAD.copy()
        payload["model"] = REVIEWER_MODEL_ID
        payload["tools"] = []
        payload["max_tokens"] = REVIEWER_RESPONSE_MAX_TOKENS
        payload["messages"] = [
            {"role": "system", "content": reviewer_system_content},
            *non_system_messages,
            {"role": "user", "content": review_request},
        ]

        async with aiohttp.ClientSession() as session:
            response_data = await api_call_with_retry(
                session,
                BASE_URL,
                payload,
                {"Authorization": f"Bearer {API_KEY}"},
            )

        if "error" in response_data:
            error_msg = response_data["error"].get("message", "Unknown API error")
            return {"success": False, "error": error_msg}

        choices = response_data.get("choices") or []
        if not choices:
            return {"success": False, "error": "No response from reviewer model"}

        review = safe_strip_markdown(
            _content_to_text(choices[0].get("message", {}).get("content", ""))
        ).strip()
        if not review:
            return {"success": False, "error": "Reviewer returned empty review"}

        return {
            "success": True,
            "reviewer_model": REVIEWER_MODEL_ID,
            "question": normalized_question,
            "review": review,
        }

    async def _deliver_reminder_output(
        self,
        session_id: str,
        output: str,
    ) -> dict[str, Any]:
        """Deliver scheduled task output back through the active integration channel."""
        try:
            from integrations import deliver_scheduled_session_output
        except Exception as exc:
            return {
                "success": False,
                "error": f"Scheduled delivery integration unavailable: {exc}",
            }

        return await deliver_scheduled_session_output(
            session_id=session_id,
            output=output,
        )

    _BARE_REACTION_RE = re.compile(
        r"^(?:like|love|dislike|laugh|emphasize|question)[.!]?\s*$",
        re.IGNORECASE,
    )

    def _finalize_visible_response(
        self, content: Any, session_id: Optional[str]
    ) -> str:
        """Return text safe for visible responses with delivery directives removed.

        If *content* is a JSON payload carrying ``attachments`` alongside
        ``text`` (emitted by the agentic loop when tools produce images),
        only the text portion is sanitized and the JSON envelope is
        preserved so downstream callers can extract the attachments.
        """
        normalized = "" if content is None else str(content)

        # Detect JSON payloads with attachments and sanitize only the text.
        _parsed_payload: dict | None = None
        try:
            _parsed_payload = json.loads(normalized)
        except Exception:
            pass
        if (
            isinstance(_parsed_payload, dict)
            and "attachments" in _parsed_payload
            and isinstance(_parsed_payload.get("text"), str)
        ):
            inner_text = _strip_delivery_directives(_parsed_payload["text"])
            inner_text = _strip_internal_prompt_residue(inner_text)
            if (
                session_id
                and session_id.startswith(("imessage_", "tg_"))
                and self._BARE_REACTION_RE.match(inner_text.strip())
            ):
                logger.warning(
                    "Suppressing bare reaction word '%s' from outbound mobile "
                    "delivery — reactions must use send_tapback or "
                    "send_telegram_reaction.",
                    inner_text.strip(),
                )
                inner_text = ""
            _parsed_payload["text"] = inner_text
            return json.dumps(_parsed_payload)

        visible_text = _strip_delivery_directives(normalized)
        visible_text = _strip_internal_prompt_residue(visible_text)
        if (
            session_id
            and session_id.startswith(("imessage_", "tg_"))
            and self._BARE_REACTION_RE.match(visible_text.strip())
        ):
            logger.warning(
                "Suppressing bare reaction word '%s' from outbound mobile "
                "delivery — reactions must use send_tapback or "
                "send_telegram_reaction.",
                visible_text.strip(),
            )
            return ""
        return visible_text

    @staticmethod
    def _visible_text_from_response(content: str) -> str:
        """Extract the human-visible text from a response.

        If *content* is a JSON envelope with ``text`` and ``attachments``
        (produced by the agentic loop when tools generate images), return
        only the ``text`` portion.  Otherwise return *content* as-is.
        """
        if not content:
            return content
        try:
            parsed = json.loads(content)
        except Exception:
            return content
        if isinstance(parsed, dict) and "text" in parsed and "attachments" in parsed:
            return str(parsed["text"])
        return content

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
    def _is_persistent_messaging_session(session_id: Optional[str]) -> bool:
        """Messaging channels behave like one ongoing thread per user identity."""
        if not session_id:
            return False
        return session_id.startswith("imessage_") or session_id.startswith("tg_")

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
    def _truncate_context_snippet(content: str, max_chars: int = 180) -> str:
        """Normalize one memory/history snippet for system-prompt injection."""
        normalized = re.sub(r"\s+", " ", (content or "").strip())
        if len(normalized) > max_chars:
            return f"{normalized[: max_chars - 3]}..."
        return normalized

    def _score_session_continuity_memory(
        self,
        memory: Any,
        session_id: Optional[str],
    ) -> float:
        """Rank durable memories for continuity-first prompt injection."""
        metadata = memory.metadata or {}
        memory_type = str(metadata.get("type", "")).strip()
        if memory_type not in {"explicit_memory", "auto_memory", "long_term_memory"}:
            return float("-inf")
        if not self._memory_matches_session_scope(memory, session_id=session_id):
            return float("-inf")

        score = {
            "explicit_memory": 2.5,
            "auto_memory": 1.5,
            "long_term_memory": 3.5,
        }.get(memory_type, 0.0)

        memory_session_id = str(metadata.get("session_id", "")).strip()
        if session_id and memory_session_id == session_id:
            score += 4.0
        elif session_id and memory_session_id:
            score -= 2.0

        importance = str(metadata.get("importance", "medium")).strip().lower()
        score += {"high": 2.0, "medium": 1.0, "low": 0.25}.get(importance, 0.5)

        significance = max(
            0.0,
            min(1.0, self._coerce_float(metadata.get("significance"), default=0.0)),
        )
        if memory_type == "long_term_memory":
            significance = max(significance, 0.6)
        score += significance * 2.5

        normalized_content = (memory.content or "").strip().lower()
        if any(
            token in normalized_content
            for token in (
                "name",
                "call me",
                "prefer",
                "likes",
                "dislike",
                "running joke",
                "inside joke",
                "project",
                "working on",
                "building",
            )
        ):
            score += 0.75

        created_at = self._parse_state_timestamp(getattr(memory, "created_at", None))
        if created_at is not None:
            now = (
                datetime.datetime.now(created_at.tzinfo)
                if created_at.tzinfo
                else datetime.datetime.now()
            )
            age_days = max(0.0, (now - created_at).total_seconds() / 86400.0)
            score += max(0.0, 1.5 - (age_days / 14.0))

        return score

    @staticmethod
    def _extract_assistant_name_from_memory(memory: Any) -> str:
        """Return configured assistant name from memory metadata/content when present."""
        metadata = getattr(memory, "metadata", {}) or {}
        if (
            str(metadata.get("subject", "")).strip() == "assistant_identity"
            and str(metadata.get("slot", "")).strip() == "assistant_name"
        ):
            candidate = str(metadata.get("assistant_name", "")).strip()
            if candidate:
                return candidate
        return extract_assistant_name_from_memory_content(
            str(getattr(memory, "content", "") or "")
        )

    def _collect_assistant_identity_names(
        self,
        memories: list[Any],
        session_id: Optional[str],
    ) -> dict[str, str]:
        """Collect known assistant-name assignments keyed by normalized name."""
        names: dict[str, str] = {}
        for memory in memories:
            if not self._memory_matches_session_scope(memory, session_id=session_id):
                continue
            assistant_name = self._extract_assistant_name_from_memory(memory)
            normalized = self._normalize_memory_text(assistant_name)
            if normalized and normalized not in names:
                names[normalized] = assistant_name
        return names

    def _collect_assistant_identity_names_from_history(
        self,
        history: list[dict[str, Any]],
    ) -> dict[str, str]:
        """Infer assistant-name assignments directly from recent user turns."""
        names: dict[str, str] = {}
        for message in history:
            if str(message.get("role", "")).strip().lower() != "user":
                continue
            assistant_name = extract_assistant_name_from_user_text(
                _content_to_text(message.get("content", ""))
            )
            normalized = self._normalize_memory_text(assistant_name)
            if normalized and normalized not in names:
                names[normalized] = assistant_name
        return names

    def _merge_assistant_identity_names(
        self,
        memory_names: dict[str, str],
        history_names: dict[str, str],
    ) -> dict[str, str]:
        """Prefer explicit memory-backed assistant identity, then recent history."""
        merged = dict(memory_names)
        for key, value in history_names.items():
            merged.setdefault(key, value)
        return merged

    def _memory_conflicts_with_assistant_identity(
        self,
        memory: Any,
        assistant_identity_names: dict[str, str],
    ) -> bool:
        """Skip stale user-name memories that contradict a known assistant name."""
        if not assistant_identity_names:
            return False
        user_name = extract_user_name_from_memory_content(
            str(getattr(memory, "content", "") or "")
        )
        if not user_name:
            return False
        normalized = self._normalize_memory_text(user_name)
        return bool(normalized and normalized in assistant_identity_names)

    def _build_assistant_identity_context(
        self,
        session_id: Optional[str],
    ) -> str:
        """Inject a direct assistant-identity note when a session has one."""
        if not session_id:
            return ""

        try:
            recent_memories = self.memory_store.get_recent_memories(
                limit=SESSION_CONTINUITY_MEMORY_SCAN_LIMIT
            )
            recent_history = self.memory_store.get_conversation_history(
                session_id=session_id,
                limit=SESSION_CONTINUITY_HISTORY_LOOKBACK,
            )
        except Exception:
            return ""

        assistant_identity_names = self._merge_assistant_identity_names(
            self._collect_assistant_identity_names(
                recent_memories,
                session_id=session_id,
            ),
            self._collect_assistant_identity_names_from_history(recent_history),
        )
        if not assistant_identity_names:
            return ""

        assistant_name = next(iter(assistant_identity_names.values()))
        return (
            "\n\n[Assistant identity note]:\n"
            f"Your configured name in this session is {assistant_name}. "
            'If the user says "your name" or addresses that name, they mean you, '
            "not the user.\n"
        )

    def _memory_matches_session_scope(
        self,
        memory: Any,
        session_id: Optional[str],
    ) -> bool:
        """Scope memory injection to the active messaging session to avoid user bleed."""
        if not session_id or not self._is_persistent_messaging_session(session_id):
            return True

        metadata = getattr(memory, "metadata", {}) or {}
        memory_session_id = str(metadata.get("session_id", "")).strip()
        if not memory_session_id:
            return False
        return memory_session_id == session_id

    def _build_session_continuity_context(
        self,
        session_id: Optional[str],
    ) -> str:
        """Build a proactive continuity brief from durable memories + recent history."""
        if not SESSION_CONTINUITY_ENABLED or not session_id:
            return ""

        is_persistent_messaging = self._is_persistent_messaging_session(session_id)

        try:
            recent_history = self.memory_store.get_conversation_history(
                session_id=session_id,
                limit=SESSION_CONTINUITY_HISTORY_LOOKBACK,
            )
            recent_memories = self.memory_store.get_recent_memories(
                limit=SESSION_CONTINUITY_MEMORY_SCAN_LIMIT
            )
        except Exception as e:
            logger.warning("Failed to build session continuity context: %s", e)
            return (
                "\n\n[Session continuity status]:\n"
                "Continuity lookup failed this turn. Do not invent familiarity. "
                "If continuity matters to the reply, say you are having trouble "
                "accessing your notes on the user and want to rebuild them quickly.\n"
            )

        assistant_identity_names = self._merge_assistant_identity_names(
            self._collect_assistant_identity_names(
                recent_memories,
                session_id=session_id,
            ),
            self._collect_assistant_identity_names_from_history(recent_history),
        )
        scored_memories: list[tuple[float, Any]] = []
        seen_memory_texts: set[str] = set()
        for memory in recent_memories:
            if self._memory_conflicts_with_assistant_identity(
                memory,
                assistant_identity_names,
            ):
                continue
            score = self._score_session_continuity_memory(memory, session_id=session_id)
            if score == float("-inf"):
                continue

            normalized_text = self._normalize_memory_text(memory.content)
            if not normalized_text or normalized_text in seen_memory_texts:
                continue

            seen_memory_texts.add(normalized_text)
            scored_memories.append((score, memory))

        scored_memories.sort(key=lambda item: item[0], reverse=True)

        profile_lines: list[str] = []
        for _, memory in scored_memories[:SESSION_CONTINUITY_MAX_PROFILE_LINES]:
            snippet = self._truncate_context_snippet(memory.content, max_chars=180)
            if not snippet:
                continue
            profile_lines.append(f"{len(profile_lines) + 1}. {snippet}")

        history_lines: list[str] = []
        for message in reversed(recent_history[:SESSION_CONTINUITY_HISTORY_LOOKBACK]):
            role = str(message.get("role", "")).strip().lower()
            if role not in {"user", "assistant"}:
                continue

            snippet = self._truncate_context_snippet(
                _content_to_text(message.get("content", "")),
                max_chars=180,
            )
            if not snippet:
                continue

            history_lines.append(f"{len(history_lines) + 1}. {role}: {snippet}")
            if len(history_lines) >= SESSION_CONTINUITY_MAX_HISTORY_LINES:
                break

        if not profile_lines and not history_lines:
            if is_persistent_messaging:
                return (
                    "\n\n[Session continuity status]:\n"
                    "This messaging channel is one persistent conversation by design. "
                    "Do not treat this turn like a fresh thread or re-introduce yourself. "
                    "If continuity notes are missing, acknowledge that plainly and continue "
                    "without making the user re-establish the relationship from scratch.\n"
                )
            return (
                "\n\n[Session continuity status]:\n"
                "No continuity notes were found yet. Do not fake familiarity. "
                "If continuity matters, acknowledge the gap plainly and warmly, then "
                "rebuild context without making the user re-teach basic facts more than necessary.\n"
            )

        lines = [
            "",
            "",
            "[Session continuity brief]:",
            "Default stance: this is an ongoing relationship, not a cold start.",
            "Use relevant continuity naturally. If the user is resuming prior work or conversation, lead with it instead of asking whether you remember.",
            "In direct task mode, silently use this context unless saying it aloud genuinely helps.",
        ]

        if is_persistent_messaging:
            lines.append(
                "This iMessage/Telegram session is one persistent conversation by design. Do not act like each inbound message starts a new thread."
            )

        if profile_lines:
            lines.append("Stable user/profile notes:")
            lines.extend(profile_lines)

        if history_lines:
            lines.append("Recent conversation thread:")
            lines.extend(history_lines)

        return "\n".join(lines) + "\n"

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

        telegram_chat_id = request_metadata.get("telegram_chat_id")
        if isinstance(telegram_chat_id, str):
            candidate = telegram_chat_id.strip()
            if candidate.lstrip("-").isdigit():
                telegram_chat_id = int(candidate)
        if isinstance(telegram_chat_id, int):
            normalized["telegram_chat_id"] = telegram_chat_id

        telegram_message_id = request_metadata.get("telegram_message_id")
        if (
            isinstance(telegram_message_id, str)
            and telegram_message_id.strip().isdigit()
        ):
            telegram_message_id = int(telegram_message_id.strip())
        if isinstance(telegram_message_id, int) and telegram_message_id > 0:
            normalized["telegram_message_id"] = telegram_message_id

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
    def _build_telegram_reaction_context_line(
        entry_index: int,
        chat_id: int,
        message_id: int,
        content: str,
        label: str,
    ) -> str:
        """Format one Telegram reaction target line for the system prompt."""
        normalized_content = re.sub(r"\s+", " ", (content or "").strip())
        if len(normalized_content) > 160:
            normalized_content = f"{normalized_content[:157]}..."

        prefix = f"{entry_index}. {label}: chat_id={chat_id}; message_id={message_id}"
        if normalized_content:
            return f'{prefix}; message="{normalized_content}"'
        return prefix

    def _build_telegram_reaction_context(
        self,
        user_query: str,
        session_id: Optional[str],
        request_metadata: Optional[dict[str, Any]],
    ) -> str:
        """Expose Telegram message IDs so send_telegram_reaction can be called concretely."""
        if not session_id or not session_id.startswith("tg_"):
            return ""

        normalized_request_metadata = self._normalize_request_metadata(request_metadata)
        history = self.memory_store.get_conversation_history(
            session_id=session_id,
            limit=TELEGRAM_REACTION_CONTEXT_LIMIT,
        )

        lines: list[str] = []
        seen_targets: set[tuple[int, int]] = set()

        current_chat_id = normalized_request_metadata.get("telegram_chat_id")
        current_message_id = normalized_request_metadata.get("telegram_message_id")
        if isinstance(current_chat_id, int) and isinstance(current_message_id, int):
            target = (current_chat_id, current_message_id)
            seen_targets.add(target)
            lines.append(
                self._build_telegram_reaction_context_line(
                    entry_index=len(lines) + 1,
                    chat_id=current_chat_id,
                    message_id=current_message_id,
                    content=user_query,
                    label="current inbound message",
                )
            )

        for message in history:
            if str(message.get("role", "")).strip().lower() != "user":
                continue

            metadata = self._normalize_request_metadata(message.get("metadata"))
            chat_id = metadata.get("telegram_chat_id")
            message_id = metadata.get("telegram_message_id")
            if not isinstance(chat_id, int) or not isinstance(message_id, int):
                continue

            target = (chat_id, message_id)
            if target in seen_targets:
                continue

            seen_targets.add(target)
            lines.append(
                self._build_telegram_reaction_context_line(
                    entry_index=len(lines) + 1,
                    chat_id=chat_id,
                    message_id=message_id,
                    content=_content_to_text(message.get("content", "")),
                    label="recent inbound message",
                )
            )

        if not lines:
            return ""

        return (
            "\n\n"
            "[Available Telegram reaction targets from current and recent messages]:\n"
            "Use these exact chat_id/message_id values when calling send_telegram_reaction.\n"
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
                    "Only store durable identity/context such as preferred name, pronouns, tone or style preferences, "
                    "running jokes, ongoing projects, personal facts, recurring constraints, relationship context, "
                    "or high-impact commitments. Store facts from the correct subject perspective: "
                    "if the user says something about the assistant like 'your name is Alice', "
                    "store that as assistant identity, not as a user fact. Ignore one-off requests and generic chatter. "
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

        content, normalized_topics, inferred_metadata = (
            normalize_memory_candidate_from_user_text(
                content=str(candidate.get("content", "")).strip(),
                user_text=user_query,
                topics=candidate.get("topics", []),
            )
        )
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
        metadata.update(inferred_metadata)
        if candidate.get("reason"):
            metadata["extractor_reason"] = candidate["reason"]

        memory_id = await self.memory_store.add_memory(
            content=content,
            metadata=metadata,
            topics=normalized_topics,
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
        consultation_context: str,
        example_context: str,
        skills_catalog_context: str = "",
        active_skills_context: str = "",
        session_prompt_suffix: str = "",
        request_freshness_token: Optional[str] = None,
    ) -> str:
        """Build the canonical system prompt used by the primary agent."""
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        context = {
            "current_time": current_time,
            "workspace_path": AGENT_WORKSPACE,
            "memory_context": memory_context,
            "skills_catalog_context": skills_catalog_context,
            "active_skills_context": active_skills_context,
            "request_freshness_token": request_freshness_token,
            "session_prompt_suffix": session_prompt_suffix,
            "identity": (custom_prompt if custom_prompt else ""),
            "plan_context": plan_context,
            "consultation_context": consultation_context,
            "example_context": example_context,
        }

        return get_template("system_prompt", context)

    def _build_consultation_context(
        self,
        user_query: str,
        task: Optional[Any],
        task_plan: Optional[Any],
    ) -> str:
        """Describe primary-model routing and when consultation is appropriate."""
        if not user_query:
            return ""

        step_count = (
            len(task_plan.steps)
            if task_plan and getattr(task_plan, "steps", None)
            else 0
        )
        query_length = len(user_query.strip())
        task_type = str(getattr(task, "type", "generic") or "generic").lower()

        complex_task = (
            step_count >= 3
            or query_length >= 180
            or task_type in {"coding", "research", "analysis", "comparison"}
        )

        lines = [
            "\n\n[Execution Architecture]:",
            f"Primary model for this turn: advisor ({PRIMARY_MODEL_ID}).",
            "All user-facing channels should stay on the advisor-capable primary model by default.",
        ]

        if complex_task:
            lines.extend(
                [
                    "This request appears complex enough to use the primary-model-plus-review pattern.",
                    "Stay on the advisor-capable primary model for the actual work.",
                    "Use consult_advisor() only if you want a second strategic pass.",
                    "Use consult_reviewer() after non-trivial implementation or before finalizing risky work.",
                ]
            )
        else:
            lines.extend(
                [
                    "This request appears low-complexity. Prefer staying entirely on the advisor-capable primary model unless you hit a real blocker.",
                    "Do not escalate to consultation tools unless strategy or risk genuinely requires it.",
                ]
            )

        return "\n".join(lines)

    def _build_request_freshness_token(self) -> str:
        """Return a unique token for the current visible response generation."""
        return uuid.uuid4().hex

    _SKILL_URL_PATTERN = re.compile(
        r"https?://\S*?skill\S*?\.md(?:\b|[?#]|$)",
        re.IGNORECASE,
    )
    _GENERAL_MD_URL_PATTERN = re.compile(
        r"https?://\S+?\.md(?:\b|[?#]|$)",
        re.IGNORECASE,
    )

    def _detect_skill_url_hint(self, user_query: str) -> str:
        """Return a prompt hint when the user message contains a skill-related URL."""
        if not user_query:
            return ""

        has_skill_url = bool(self._SKILL_URL_PATTERN.search(user_query))
        has_md_url = bool(self._GENERAL_MD_URL_PATTERN.search(user_query))

        if not has_skill_url and not has_md_url:
            return ""

        lines = [
            "\n\n[Skill URL Detected]:",
            "The user's message contains a URL that may point to a skill definition.",
            "Use the `add_skill` tool with that URL to fetch, validate, and install the skill.",
            "The tool will automatically scan the content for prompt-injection attacks before installing.",
            "After a successful install, you can also `activate_skill` if you need the instructions immediately.",
        ]

        if has_skill_url:
            match = self._SKILL_URL_PATTERN.search(user_query)
            if match:
                lines.append(f"Detected URL: {match.group(0)}")

        if has_md_url and not has_skill_url:
            match = self._GENERAL_MD_URL_PATTERN.search(user_query)
            if match:
                lines.append(f"Detected URL: {match.group(0)}")

        return "\n".join(lines)

    def _get_session_prompt_suffix(self, session_id: Optional[str]) -> str:
        """Return channel-specific prompt guidance for the active session."""
        if not session_id:
            return ""

        is_messaging = self._is_persistent_messaging_session(session_id)

        if not is_messaging:
            return ""

        return (
            "\n\nCONVERSATIONAL STYLE — MESSAGING SESSIONS:\n"
            "You are in a live text conversation, not writing an email or documentation. "
            "Follow these rules strictly:\n\n"
            "1. EMOTION FIRST, LOGIC SECOND. When a message carries emotional weight "
            "(frustration, excitement, sadness, anger, fear, surprise, joy), acknowledge "
            "the emotion before offering any practical advice. A human reacts to how "
            "something feels before they solve what happened.\n\n"
            "2. KEEP REPLIES SHORT. One to three sentences per response. Never dump a "
            "wall of text. If you have multiple things to say, pick the most emotionally "
            "relevant one first and let the rest come out naturally across follow-up "
            "turns.\n\n"
            "3. SOFT PROBES OVER INFO DUMPS. Instead of listing everything at once, ask "
            "one gentle follow-up question that lets the user share more. Practical "
            "details (insurance, next steps, logistics) should come up organically after "
            "the emotional moment has been addressed — not before.\n\n"
            "4. MATCH ENERGY. Mirror the user's tone and intensity. If they're casual "
            "and low-key, be casual. If they're fired up, match that energy with "
            "sympathy. If they're being funny, lean into it. Never respond to a "
            "distressed message with clinical detachment.\n\n"
            '5. NO AI-FLAVORED FILLER. Do not start with "I\'m sorry to hear that" or '
            '"That sounds difficult." Use real conversational language: "damn," '
            '"that sucks," "no way," "oh man," "wait really?" — whatever fits '
            "the moment naturally.\n\n"
            "6. TASK MODE EXCEPTION. If the user is clearly in task mode (asking you to "
            "write code, search the web, execute tools, look something up), switch to "
            "efficient execution mode. Get the job done concisely. The conversational "
            "style above applies to casual, emotional, or storytelling exchanges — not "
            "to directed work requests.\n\n"
            "7. PERSISTENT THREAD ASSUMPTION. iMessage and Telegram sessions here are "
            "single ongoing conversations by design. Do not greet the user like a stranger, "
            "do not re-introduce yourself, and do not treat each new inbound message as a "
            "fresh thread unless the user explicitly resets the context."
        )

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
        system_content = get_template(
            "consortium_member",
            {
                "member_name": member["name"],
                "member_stance": member["stance"],
                "member_persona": member["persona"],
                "custom_prompt": custom_prompt,
                "session_prompt_suffix": session_prompt_suffix,
                "memory_context": memory_context.strip() if memory_context else "",
                "skills_catalog_context": (
                    skills_catalog_context.strip() if skills_catalog_context else ""
                ),
                "active_skills_context": (
                    active_skills_context.strip() if active_skills_context else ""
                ),
                "request_freshness_token": request_freshness_token,
            },
        )

        member_temperature = member.get(
            "temperature", BASE_PAYLOAD.get("temperature", 0.6)
        )
        try:
            member_temperature = float(member_temperature)
        except (TypeError, ValueError):
            member_temperature = float(BASE_PAYLOAD.get("temperature", 0.6))

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
            {"role": "system", "content": system_content},
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

        judge_system_content = get_template(
            "consortium_judge",
            {
                "custom_prompt": custom_prompt,
                "session_prompt_suffix": session_prompt_suffix,
                "memory_context": memory_context.strip() if memory_context else "",
                "skills_catalog_context": (
                    skills_catalog_context.strip() if skills_catalog_context else ""
                ),
                "active_skills_context": (
                    active_skills_context.strip() if active_skills_context else ""
                ),
                "request_freshness_token": request_freshness_token,
            },
        )

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
            {"role": "system", "content": judge_system_content},
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
        response_chunk_callback: Optional[Callable[[str], Awaitable[None]]] = None,
        request_metadata: Optional[dict[str, Any]] = None,
    ):
        """Handle a request and return the response content."""
        data = request
        normalized_request_metadata = self._normalize_request_metadata(request_metadata)

        await self.start_reminder_scheduler()

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
        if session_id:
            assistant_identity_context = self._build_assistant_identity_context(
                session_id=session_id
            )
            if assistant_identity_context:
                memory_context += assistant_identity_context

            continuity_context = self._build_session_continuity_context(
                session_id=session_id
            )
            if continuity_context:
                memory_context += continuity_context

        if user_query:
            try:
                assistant_identity_names = self._merge_assistant_identity_names(
                    self._collect_assistant_identity_names(
                        self.memory_store.get_recent_memories(
                            limit=SESSION_CONTINUITY_MEMORY_SCAN_LIMIT
                        ),
                        session_id=session_id,
                    ),
                    self._collect_assistant_identity_names_from_history(
                        self.memory_store.get_conversation_history(
                            session_id=session_id,
                            limit=SESSION_CONTINUITY_HISTORY_LOOKBACK,
                        )
                    ),
                )
                relevant_memories = await self.memory_store.search_memories(
                    query=user_query, top_k=8, threshold=0.15
                )
                if relevant_memories:
                    weighted_memories: list[tuple[Any, float, float, float]] = []
                    for memory, similarity in relevant_memories:
                        if not self._memory_matches_session_scope(
                            memory, session_id=session_id
                        ):
                            continue
                        if self._memory_conflicts_with_assistant_identity(
                            memory,
                            assistant_identity_names,
                        ):
                            continue
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
                    memory_context += (
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

        if user_query:
            try:
                telegram_reaction_context = self._build_telegram_reaction_context(
                    user_query=user_query,
                    session_id=session_id,
                    request_metadata=normalized_request_metadata,
                )
                if telegram_reaction_context:
                    memory_context += telegram_reaction_context
            except Exception as e:
                logger.warning(f"Failed to build Telegram reaction context: {e}")

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

        if user_query:
            skill_url_hint = self._detect_skill_url_hint(user_query)
            if skill_url_hint:
                skills_catalog_context = (skills_catalog_context or "") + skill_url_hint

        # Add task plan context if available
        plan_context = ""
        if task_plan and len(task_plan.steps) > 1:
            task_type_label = task.type if task else "generic"
            plan_context = f"\n\n[Task Plan - {task_type_label}]:\n"
            for i, step in enumerate(task_plan.steps, 1):
                plan_context += f"{i}. {step.description}\n"

        consultation_context = self._build_consultation_context(
            user_query=user_query,
            task=task,
            task_plan=task_plan,
        )

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
            consultation_context=consultation_context,
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

                content = self._finalize_visible_response(content, session_id)

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
        messages = self._build_rolling_context(
            system_message=system_message,
            current_messages=data.get("messages", []),
            session_id=session_id,
            context_window=128000,
            buffer_tokens=2000,
        )
        payload_template = self._build_request_payload_template()
        current_payload = payload_template.copy()
        current_payload["messages"] = messages

        async with aiohttp.ClientSession() as session:
            response_data = await api_call_with_retry(
                session,
                BASE_URL,
                current_payload,
                {"Authorization": f"Bearer {API_KEY}"},
                stream=response_chunk_callback is not None,
                stream_chunk_callback=response_chunk_callback,
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
                    stream_chunk_callback=response_chunk_callback,
                )
                if self._self_heal_manager and content and content.startswith("Error:"):
                    heal_result = await self._self_heal_manager.try_heal(
                        error_string=content,
                        context={
                            "session_id": session_id,
                            "user_query": user_query,
                            "iteration": _extract_iteration_from_error(content),
                            "tools_executed": _extract_tools_from_error(content),
                        },
                    )
                    if heal_result.success and heal_result.applied:
                        content = f"{content}\n\n[Self-healed: {heal_result.summary}]"
                        logger.info("Self-heal applied: %s", heal_result.summary)
            finally:
                reset_tool_runtime_session(session_token)

            content = self._finalize_visible_response(content, session_id)

            # Extract the visible text for history / memory (strip the
            # JSON attachment envelope if present so history stays clean).
            visible_text = self._visible_text_from_response(content)

            # Persist the visible turn before post-response maintenance so
            # cadence calculations see the latest exchange.
            if session_id and user_query:
                self.memory_store.add_conversation_message(
                    role="user",
                    content=user_query,
                    session_id=session_id,
                    metadata=normalized_request_metadata or None,
                )
            if session_id and visible_text:
                self.memory_store.add_conversation_message(
                    role="assistant", content=visible_text, session_id=session_id
                )

            await self._run_memory_maintenance(
                session=session,
                session_id=session_id,
                user_query=user_query,
                assistant_response=visible_text,
            )

        # Provide feedback on examples if task was completed
        visible_text = self._visible_text_from_response(content)
        if task and visible_text and not visible_text.startswith("Error:"):
            self.example_bank.auto_feedback(task.type, success=True, efficiency=1.0)

        print(visible_text)
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

            base_msg = {
                "role": msg["role"],
                "content": msg["content"],
            }
            # Preserve tool call and tool result fields
            if "tool_calls" in msg:
                base_msg["tool_calls"] = msg["tool_calls"]
            if "tool_call_id" in msg:
                base_msg["tool_call_id"] = msg["tool_call_id"]
            selected_history.append(base_msg)
            current_tokens += msg_tokens

        # Combine: system + history + current messages
        final_messages = [system_message] + selected_history + current_messages

        total_tokens = system_tokens + current_tokens + current_input_tokens
        logger.info(
            f"Context window: {total_tokens} tokens used ({len(selected_history)} history messages)"
        )

        return final_messages
