"""Tool functions and registry for the agent."""

import asyncio
import glob
import json
import os
import re
import subprocess
from datetime import datetime, timedelta
from typing import Any, Optional

import aiohttp

from memory import EnhancedMemoryStore


# Initialize memory store (will be set from main module)
memory_store: Optional[EnhancedMemoryStore] = None
consortium_controller: Any = None


def set_memory_store(store: EnhancedMemoryStore):
    """Set the memory store instance for tools to use."""
    global memory_store
    memory_store = store


def set_consortium_controller(controller: Any):
    """Set the consortium task controller used by consortium tools."""
    global consortium_controller
    consortium_controller = controller


# File tools
async def read_file_tool(filepath):
    """Read the contents of a file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return {"success": True, "content": content}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def write_file_tool(filepath, content):
    """Write content to a file (overwrites existing)."""
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return {"success": True, "message": f"Written to {filepath}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def edit_file_tool(filepath, old_str, new_str):
    """Replace old_str with new_str in file. Requires exact match."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        if old_str not in content:
            return {"success": False, "error": "old_str not found in file"}

        new_content = content.replace(old_str, new_str, 1)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)
        return {"success": True, "message": f"Edited {filepath}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def glob_tool(pattern):
    """Find files matching a glob pattern."""
    try:
        matches = glob.glob(pattern, recursive=True)
        return {"success": True, "matches": matches}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def grep_tool(pattern, path="."):
    """Search for pattern in files. Returns matching lines with filenames."""
    try:
        matches = []
        for root, dirs, files in os.walk(path):
            for file in files:
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        for i, line in enumerate(f, 1):
                            if re.search(pattern, line):
                                matches.append(
                                    {
                                        "file": filepath,
                                        "line": i,
                                        "content": line.rstrip(),
                                    }
                                )
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

        metadata = {"importance": importance, "type": "explicit_memory"}
        memory_id = await memory_store.add_memory(
            content=content,
            metadata=metadata,
            topics=topics or [],
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
    query: str, numResults: int = 10, category: Optional[str] = None, type: str = "auto"
):
    """Search the web using Exa MCP server (no API key required, has rate limits).

    Args:
        query: The search query string
        numResults: Number of results to return (1-100, default: 10)
        category: Optional category filter (company, research paper, news, people)
        type: Search type (auto, neural, fast, deep, deep-reasoning, instant)
    """
    try:
        # Build tool arguments
        arguments = {"query": query, "numResults": min(max(numResults, 1), 100)}
        if category:
            arguments["category"] = category

        # Call the tool via MCP
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
                timeout=aiohttp.ClientTimeout(total=60),
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

                # Also include the raw formatted text
                return {
                    "success": True,
                    "query": query,
                    "resultCount": len(search_results),
                    "results": search_results[:numResults],
                    "formatted_text": full_text[:5000] if full_text else "",
                }

    except aiohttp.ClientError as e:
        return {"success": False, "error": f"Network error: {str(e)}"}
    except asyncio.TimeoutError:
        return {"success": False, "error": "Request timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}


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
    "consortium_start": consortium_start_tool,
    "consortium_stop": consortium_stop_tool,
    "consortium_status": consortium_status_tool,
    "consortium_agree": consortium_agree_tool,
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
        "consortium_start": ["task"],
        "consortium_stop": ["task_id"],
        "consortium_status": [],
        "consortium_agree": [],
    }

    if func_name in required_params:
        missing = [p for p in required_params[func_name] if p not in func_args]
        if missing:
            return False, f"Missing required parameters: {missing}"

    return True, None
