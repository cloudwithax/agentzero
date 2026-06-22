"""General-purpose regression detection for tool execution.

After every tool execution round, this module checks if the results
indicate a behavioral failure — not a code error (exception/traceback),
but a tool that ran without producing its expected outcome.

Design follows the crusty HEARTBEAT self-check protocol:
- Detect regressions using general heuristics (not tool-specific patterns)
- Log MISS/FIX entries to a persistent file
- Wire to SelfHealManager for auto-fix generation
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Regression categories ──────────────────────────────────────────────

CATEGORY_EMPTY_OUTPUT = "empty_output"
CATEGORY_ERROR_IN_SUCCESS = "error_in_success"
CATEGORY_TRIVIAL_RESULT = "trivial_result"
CATEGORY_SILENT_FAILURE = "silent_failure"

CATEGORY_LABELS = {
    CATEGORY_EMPTY_OUTPUT: "tool returned empty/null content despite success",
    CATEGORY_ERROR_IN_SUCCESS: "tool returned error-like content inside a success result",
    CATEGORY_TRIVIAL_RESULT: "tool returned suspiciously small/output for its task",
    CATEGORY_SILENT_FAILURE: "tool returned minimal output with no actionable content",
}

# ── Heuristics configuration ──────────────────────────────────────────

_MIN_CONTENT_LENGTH = 8  # bytes: anything under this is "empty enough to flag"
_ERROR_KEYWORDS = (
    "error", "failed", "failure", "unable to", "could not",
    "not found", "permission denied", "timeout", "timed out",
    "crash", "exception", "invalid", "missing", "denied",
    "aborted", "refused", "unreachable",
)

# ── Log file ──────────────────────────────────────────────────────────

_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory")
_LOG_FILE = os.path.join(_LOG_DIR, "regressions.log")


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class RegressionReport:
    """A single regression detection result."""
    tool_name: str
    tool_args: dict[str, Any]
    tool_result_content: str
    category: str
    detected_miss: str
    suggested_fix: str = ""
    iteration: int = 0
    timestamp: float = field(default_factory=time.time)
    session_id: str = ""


# ── Detection logic ───────────────────────────────────────────────────

def _extract_content_string(raw_content: str) -> str:
    """Try to parse a tool result's content field into a readable string."""
    if not raw_content:
        return ""
    cleaned = raw_content.strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return json.dumps(parsed, indent=2)
        if isinstance(parsed, list):
            return json.dumps(parsed, indent=2)
        return str(parsed)
    except (json.JSONDecodeError, TypeError):
        return cleaned


def _extract_content_value(raw_content: str) -> str:
    """Extract just the meaningful payload from a tool result JSON.

    Tries ``content``, then ``stdout``, then ``matches`` (for grep), and
    finally falls back to the entire JSON string so heuristic checks
    operate on real data.
    """
    if not raw_content:
        return ""
    try:
        parsed = json.loads(raw_content.strip())
        if isinstance(parsed, dict):
            for key in ("content", "stdout", "matches"):
                val = parsed.get(key)
                if val is not None:
                    if isinstance(val, (dict, list)):
                        return json.dumps(val)
                    return str(val)
                if key in parsed:
                    return ""
            return raw_content.strip()
    except (json.JSONDecodeError, TypeError):
        pass
    return raw_content.strip()


def _is_result_success(raw_content: str) -> bool:
    """Check if the tool result signals explicit success."""
    if not raw_content:
        return True
    try:
        parsed = json.loads(raw_content.strip())
        if isinstance(parsed, dict):
            return parsed.get("success", True) is not False
    except (json.JSONDecodeError, TypeError):
        pass
    return True


def _is_trivially_empty(content_str: str) -> bool:
    """Check if content is effectively empty."""
    if not content_str:
        return True
    stripped = content_str.strip()
    if not stripped:
        return True
    if stripped in ("{}", "[]", '""', "''", "null", "None", "true", "false"):
        return True
    return len(stripped) < _MIN_CONTENT_LENGTH


# Tools whose output is external/fetched content that may legitimately
# contain error-like words (e.g. web pages mentioning "error-correction").
_CONTENT_RETRIEVAL_TOOLS = frozenset({
    "web_search", "web_fetch", "web_fetch_exa", "web_search_exa",
    "read", "grep", "glob",
})


def _contains_error_keywords(content_lower: str) -> bool:
    """Check if content contains error-like language despite success.

    Uses word-boundary matching for single-word keywords to avoid
    false positives like "error" matching "error-correction".
    Multi-word phrases (e.g. ``"unable to"``) use plain substring.
    """
    for kw in _ERROR_KEYWORDS:
        if " " in kw:
            if kw in content_lower:
                return True
        else:
            if re.search(rf"\b{re.escape(kw)}\b", content_lower):
                return True
    return False


def detect_regression(
    tool_name: str,
    tool_args: dict[str, Any],
    raw_content: str,
    iteration: int = 0,
    session_id: str = "",
) -> Optional[RegressionReport]:
    """General-purpose regression check for any tool result.

    Returns a ``RegressionReport`` if the result looks behaviorally wrong,
    or ``None`` if everything seems fine.

    Heuristics (all general — no tool-specific patterns):

    1. **Empty output** — tool said ``success: true`` but content is empty/null.
    2. **Error in success** — tool said ``success: true`` but content reads like an error.
    3. **Trivial result** — content is suspiciously small for what the tool should produce.
    4. **Silent failure** — content is present but has no actionable information.
    """
    # Skip regression detection when the tool explicitly reports failure —
    # that is a normal error path, not a behavioral regression where the
    # model should have gotten a real result but got nothing useful.
    if not _is_result_success(raw_content):
        return None

    # Use the actual "content" field value for empty/trivial heuristics,
    # and the full pretty-printed JSON for error-keyword scanning.
    content_value = _extract_content_value(raw_content)
    content_str = _extract_content_string(raw_content)
    content_lower = content_str.lower()

    # Heuristic 1: Empty output
    if _is_trivially_empty(content_value):
        return RegressionReport(
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result_content=raw_content[:500],
            category=CATEGORY_EMPTY_OUTPUT,
            detected_miss=(
                f"Tool '{tool_name}' returned success=true but content is "
                f"effectively empty (len={len(content_value)}). "
                f"Args: {json.dumps(tool_args, default=str)[:200]}"
            ),
            iteration=iteration,
            session_id=session_id,
        )

    # Heuristic 2: Error keywords in a success result
    # Skip for content-retrieval tools — their output is external text
    # that may legitimately contain words like "error" (e.g. "error-correction").
    has_content_retrieval = tool_name in _CONTENT_RETRIEVAL_TOOLS
    if not has_content_retrieval and _contains_error_keywords(content_lower):
        return RegressionReport(
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result_content=raw_content[:500],
            category=CATEGORY_ERROR_IN_SUCCESS,
            detected_miss=(
                f"Tool '{tool_name}' returned success=true but content "
                f"contains error-like language: {content_str[:200]}"
            ),
            iteration=iteration,
            session_id=session_id,
        )

    # Heuristic 3: Trivial result — very small output for tools that
    # should produce substantive results (e.g. bash, grep, web_search,
    # activate_skill, read, write).
    low_info_tools = {
        "bash", "grep", "glob", "read", "write", "edit",
        "web_search", "web_fetch",
        "activate_skill", "add_skill", "acp_send_message",
    }
    if tool_name in low_info_tools and len(content_value) < 20:
        return RegressionReport(
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result_content=raw_content[:500],
            category=CATEGORY_TRIVIAL_RESULT,
            detected_miss=(
                f"Tool '{tool_name}' returned success=true but the output "
                f"is suspiciously small ({len(content_value)} chars) for the "
                f"work it should have done. Args: "
                f"{json.dumps(tool_args, default=str)[:200]}"
            ),
            iteration=iteration,
            session_id=session_id,
        )

    return None


def detect_all_regressions(
    tool_calls: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
    iteration: int = 0,
    session_id: str = "",
) -> list[RegressionReport]:
    """Check a full round of tool execution results for regressions.

    Pairs each ``tool_calls[i]`` with ``tool_results[i]`` and runs
    ``detect_regression`` on each pair.
    """
    reports: list[RegressionReport] = []
    seen = set()

    for i, result in enumerate(tool_results):
        raw_content = result.get("content", "")

        # Parse the tool name from the matching tool call (same index).
        tool_name = ""
        tool_args: dict[str, Any] = {}
        if i < len(tool_calls):
            tc = tool_calls[i]
            tool_name = tc.get("function", {}).get("name", "")
            try:
                tool_args = json.loads(
                    tc.get("function", {}).get("arguments", "{}")
                )
            except (json.JSONDecodeError, TypeError):
                pass

        # Deduplicate by (tool_name, raw_content).
        dedup_key = f"{tool_name}:{raw_content[:100]}"
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        report = detect_regression(
            tool_name=tool_name,
            tool_args=tool_args,
            raw_content=raw_content,
            iteration=iteration,
            session_id=session_id,
        )
        if report:
            reports.append(report)

    return reports


# ── Logging (crusty-style MISS/FIX format) ──────────────────────────────

def _ensure_log_dir() -> None:
    os.makedirs(_LOG_DIR, exist_ok=True)


def log_regression(report: RegressionReport, fix_description: str = "") -> None:
    """Append a crusty-style MISS/FIX entry to ``memory/regressions.log``."""
    _ensure_log_dir()
    timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    fix_line = fix_description or "auto-fix via AgentSelfHealer"

    entry = (
        f"[ {timestamp_str} ]\n"
        f"TOOL: {report.tool_name}\n"
        f"CATEGORY: {report.category}\n"
        f"MISS: {report.detected_miss}\n"
        f"FIX: {fix_line}\n"
        f"---\n"
    )

    try:
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(entry + "\n")
        logger.debug("Logged regression to %s: %s", _LOG_FILE, report.category)
    except OSError as e:
        logger.warning("Failed to write regression log: %s", e)


def read_recent_regressions(hours: int = 24) -> list[dict[str, str]]:
    """Read regression entries from the last N hours for boot-time watchlist."""
    _ensure_log_dir()
    if not os.path.isfile(_LOG_FILE):
        return []

    cutoff = time.time() - hours * 3600
    entries: list[dict[str, str]] = []
    current: dict[str, str] = {}

    try:
        with open(_LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("[ ") and line.endswith(" UTC ]"):
                    if current and current.get("_ts", 0) >= cutoff:
                        entries.append(current)
                    current = {"_ts": time.time()}
                elif line.startswith("TOOL: "):
                    current["tool"] = line[6:]
                elif line.startswith("CATEGORY: "):
                    current["category"] = line[10:]
                elif line.startswith("MISS: "):
                    current["miss"] = line[6:]
                elif line.startswith("FIX: "):
                    current["fix"] = line[5:]
    except OSError:
        pass

    if current and current.get("_ts", 0) >= cutoff:
        entries.append(current)

    return entries


# ── Self-heal integration ──────────────────────────────────────────────

def format_regression_for_heal(report: RegressionReport) -> str:
    """Format a ``RegressionReport`` as an error string for ``SelfHealManager``.

    The resulting string looks like a structured error report so the
    self-healer can analyze it and produce code patches.
    """
    args_str = json.dumps(report.tool_args, default=str)[:500]
    result_str = report.tool_result_content[:500]

    return (
        f"Error: Behavioral regression detected in tool '{report.tool_name}'\n"
        f"Category: {report.category} ({CATEGORY_LABELS.get(report.category, '')})\n"
        f"Session: {report.session_id}\n"
        f"Iteration: {report.iteration}\n"
        f"\n"
        f"Details:\n"
        f"MISS: {report.detected_miss}\n"
        f"\n"
        f"Tool arguments: {args_str}\n"
        f"\n"
        f"Tool result content:\n"
        f"{result_str}\n"
        f"\n"
        f"This is NOT a code exception — the tool ran without errors but "
        f"produced an unexpected result. Analyze the tool implementation "
        f"and produce patches to fix the behavioral regression."
    )
