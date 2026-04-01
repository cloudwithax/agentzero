"""Main handler and chat loop for the agent."""

import aiohttp
import datetime
import logging
import os
from typing import Any, Optional

from memory import EnhancedMemoryStore
from capabilities import CapabilityProfile, AdaptiveFormatter
from examples import AdaptiveFewShotManager
from planning import TaskType, TaskPlanner, TaskAnalyzer
from api import api_call_with_retry, process_response

logger = logging.getLogger(__name__)


# Configuration
BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
API_KEY = os.environ.get(
    "NVIDIA_API_KEY",
    "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",
)
MODEL_ID = os.environ.get("MODEL_ID", "moonshotai/kimi-k2-instruct-0905")

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
    ],
}


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
    ):
        self.memory_store = memory_store
        self.capability_profile = capability_profile
        self.example_bank = example_bank
        self.task_planner = task_planner
        self.task_analyzer = task_analyzer
        self.adaptive_formatter = adaptive_formatter

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

    async def handle(self, request, session_id: Optional[str] = None):
        """Handle a request and return the response content."""
        data = request

        # Get the user's message for memory retrieval
        user_messages = [m for m in data["messages"] if m.get("role") == "user"]
        user_query = user_messages[-1]["content"] if user_messages else ""

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
                    query=user_query, top_k=3, threshold=0.15
                )
                if relevant_memories:
                    memory_context = (
                        "\n\n[Relevant memories from past conversations]:\n"
                    )
                    for i, (memory, score) in enumerate(relevant_memories, 1):
                        memory_context += f"{i}. {memory.content}\n"
            except Exception as e:
                logger.warning(f"Failed to retrieve memories: {e}")

        # Get few-shot examples for the task type
        few_shot_examples = []
        if task:
            few_shot_examples = self.example_bank.get_examples_for_task(
                task_type=task.type,
                query=user_query,
                max_examples=self.capability_profile.get_max_examples(),
            )

        # Build system prompt with adaptive formatting
        custom_prompt = self.memory_store.get_system_prompt()

        # Add task plan context if available
        plan_context = ""
        if task_plan and len(task_plan.steps) > 1:
            plan_context = f"\n\n[Task Plan - {task.type}]:\n"
            for i, step in enumerate(task_plan.steps, 1):
                plan_context += f"{i}. {step.description}\n"

        # Add few-shot examples if available
        example_context = ""
        if few_shot_examples:
            example_context = "\n\n[Examples]:\n"
            for ex in few_shot_examples:
                example_context += f"Input: {ex['input']}\n"
                example_context += f"Output: {ex['output']}\n\n"

        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        universal_instructions = (
            f"\n\n[Current Date and Time]: {current_time}\n"
            "\n\n[System Instructions & Tools]:\n"
            "You have access to tools via tool_calls. Use the remember() tool to store important facts about the user or session.\n"
            "If the user asks you to remember or store something, you MUST use the remember() tool.\n"
            "Use recall() to retrieve past memories.\n"
            "If you need access to current information not available to you, use the web_search() tool.\n"
            "IMPORTANT: Do not use markdown formatting, code blocks, or emojis in your responses. Respond in plain text only.\n"
        )

        if custom_prompt:
            system_content = (
                custom_prompt
                + universal_instructions
                + memory_context
                + plan_context
                + example_context
            )
        else:
            system_content = (
                "You are a helpful AI assistant with persistent memory."
                f"{universal_instructions}{memory_context}{plan_context}{example_context}"
            )

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
        current_payload = BASE_PAYLOAD.copy()
        current_payload["messages"] = messages

        async with aiohttp.ClientSession() as session:
            # Make initial API call with retry logic
            response_data = await api_call_with_retry(
                session,
                BASE_URL,
                current_payload,
                {"Authorization": f"Bearer {API_KEY}"},
            )

            content = await process_response(
                response_data,
                current_payload["messages"],
                session,
                BASE_URL,
                API_KEY,
                BASE_PAYLOAD,
            )

            # Store conversation in memory
            if session_id and user_query:
                self.memory_store.add_conversation_message(
                    role="user", content=user_query, session_id=session_id
                )
            if session_id and content:
                self.memory_store.add_conversation_message(
                    role="assistant", content=content, session_id=session_id
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
            current_input_tokens += estimate_tokens(msg.get("content", ""))

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
            msg_tokens = estimate_tokens(msg.get("content", ""))
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
