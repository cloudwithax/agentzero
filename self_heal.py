"""
Self-healing subsystem for AgentZero.

When a code/tool/config error occurs during agent execution, this module:
1. Classifies the error (code/tool vs transient)
2. Consults the agent's own NVIDIA API model for analysis and patches
3. Applies the fix in an isolated git worktree
4. Commits and merges back to the main worktree with a full audit trail
"""

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_SELF_HEAL_ENABLED = os.environ.get("SELF_HEAL_ENABLED", "1") == "1"
SELF_HEAL_MODEL = os.environ.get("SELF_HEAL_MODEL", "").strip() or None
_SELF_HEAL_COOLDOWN_SECONDS = int(os.environ.get("SELF_HEAL_COOLDOWN_SECONDS", "120"))
_SELF_HEAL_TIMEOUT_SECONDS = int(os.environ.get("SELF_HEAL_TIMEOUT_SECONDS", "180"))
_SELF_HEAL_AUTO_MERGE = os.environ.get("SELF_HEAL_AUTO_MERGE", "1") == "1"
_SELF_HEAL_MAX_ATTEMPTS_PER_SESSION = int(
    os.environ.get("SELF_HEAL_MAX_ATTEMPTS_PER_SESSION", "3")
)


class ErrorCategory(Enum):
    CODE = "code"
    TOOL = "tool"
    CONFIG = "config"
    TRANSIENT = "transient"
    UNKNOWN = "unknown"


_TRANSIENT_PATTERNS = [
    re.compile(r"(?i)rate\s*limit"),
    re.compile(r"(?i)timeout|timed?\s*out"),
    re.compile(r"(?i)connection\s*(reset|refused|error|aborted|closed)"),
    re.compile(r"(?i)network\s*(error|unreachable|failure)"),
    re.compile(r"(?i)temporary\s*failure"),
    re.compile(r"(?i)429"),
    re.compile(r"(?i)503\s*service\s*unavailable"),
    re.compile(r"(?i)socket\s*(error|closed|timeout)"),
    re.compile(r"(?i)dns\s*(error|resolution|failure)"),
    re.compile(r"(?i)retry(after|-)?\s*"),
]

_CODE_ERROR_TYPES = {
    "AttributeError",
    "ImportError",
    "ModuleNotFoundError",
    "NameError",
    "TypeError",
    "ValueError",
    "KeyError",
    "IndexError",
    "SyntaxError",
    "IndentationError",
    "UnboundLocalError",
    "NotImplementedError",
    "FileNotFoundError",
}

_TOOL_ERROR_PATTERNS = [
    re.compile(r"(?i)tool\s+execution\s+error"),
    re.compile(r"(?i)tool\s+call\s+failed"),
    re.compile(r"(?i)invalid\s+tool\s+args"),
    re.compile(r"(?i)tool\s+not\s+found"),
]


class ErrorClassifier:
    def classify(
        self,
        error_string: str,
        exception_type: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> ErrorCategory:
        error_lower = (error_string or "").lower()

        for pattern in _TRANSIENT_PATTERNS:
            if pattern.search(error_lower):
                return ErrorCategory.TRANSIENT

        if exception_type:
            if exception_type in _CODE_ERROR_TYPES:
                return ErrorCategory.CODE

        for pattern in _TOOL_ERROR_PATTERNS:
            if pattern.search(error_lower):
                return ErrorCategory.TOOL

        if exception_type and exception_type not in _CODE_ERROR_TYPES:
            return ErrorCategory.UNKNOWN

        code_signals = [
            "traceback",
            "attributeerror",
            "importerror",
            "modulenotfounderror",
            "nameerror",
            "typeerror",
            "valueerror",
            "keyerror",
            "indexerror",
            "syntaxerror",
            "filenotfounderror",
        ]
        for signal in code_signals:
            if signal in error_lower:
                return ErrorCategory.CODE

        return ErrorCategory.UNKNOWN


@dataclass
class SelfHealErrorReport:
    error_string: str
    category: ErrorCategory
    exception_type: Optional[str] = None
    traceback_snippet: Optional[str] = None
    session_id: Optional[str] = None
    iteration: int = 0
    tools_executed: list[str] = field(default_factory=list)
    user_query: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class FilePatch:
    filepath: str
    content: str
    description: str = ""


@dataclass
class HealResult:
    success: bool = False
    applied: bool = False
    summary: str = ""
    patches: list[FilePatch] = field(default_factory=list)
    worktree_path: Optional[str] = None
    commit_hash: Optional[str] = None
    merged: bool = False
    error: Optional[str] = None


class GitWorktreeManager:
    def __init__(self, repo_root: Optional[str] = None):
        self.repo_root = repo_root or os.getcwd()
        self._active_worktrees: dict[str, str] = {}

    def _git(
        self, *args: str, cwd: Optional[str] = None
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args],
            cwd=cwd or self.repo_root,
            capture_output=True,
            text=True,
            timeout=30,
        )

    @staticmethod
    def _sanitize_branch_id(raw: str) -> str:
        """Sanitize a session ID for safe use in git branch names.

        Replaces invalid characters with underscores and ensures the
        result is non-empty and doesn't start/end with a slash.
        """
        sanitized = re.sub(r"[^a-zA-Z0-9_.-]", "_", raw.strip())
        sanitized = sanitized.strip("._-")
        if not sanitized:
            sanitized = "unknown"
        if len(sanitized) > 100:
            sanitized = sanitized[:100]
        return sanitized

    def create_worktree(
        self, session_id: str, base_branch: str = "HEAD"
    ) -> Optional[str]:
        safe_id = self._sanitize_branch_id(session_id)
        worktree_dir = tempfile.mkdtemp(
            prefix=f"selfheal_{safe_id}_", dir=self.repo_root
        )
        shutil.rmtree(worktree_dir, ignore_errors=True)

        branch_name = f"selfheal/{safe_id}"

        r = self._git("worktree", "add", worktree_dir, "-b", branch_name, base_branch)
        if r.returncode != 0:
            logger.warning("Failed to create worktree: %s", r.stderr.strip())
            shutil.rmtree(worktree_dir, ignore_errors=True)
            return None

        self._active_worktrees[session_id] = worktree_dir
        logger.info("Created worktree at %s (branch %s)", worktree_dir, branch_name)
        return worktree_dir

    def commit_in_worktree(self, session_id: str, message: str) -> Optional[str]:
        worktree_path = self._active_worktrees.get(session_id)
        if not worktree_path or not os.path.isdir(worktree_path):
            return None

        self._git("add", "-A", cwd=worktree_path)
        r = self._git("commit", "-m", message, cwd=worktree_path)
        if r.returncode != 0:
            logger.warning("Commit failed in worktree: %s", r.stderr.strip())
            return None

        r = self._git("rev-parse", "HEAD", cwd=worktree_path)
        if r.returncode != 0:
            return None

        commit_hash = r.stdout.strip()
        logger.info("Committed fix in worktree: %s", commit_hash)
        return commit_hash

    def merge_to_main(self, session_id: str) -> bool:
        worktree_path = self._active_worktrees.get(session_id)
        if not worktree_path:
            return False

        branch_name = f"selfheal/{self._sanitize_branch_id(session_id)}"

        r = self._git("rev-parse", "--verify", branch_name)
        if r.returncode != 0:
            logger.warning("Branch %s not found for merge", branch_name)
            return False

        self._git("clean", "-fd")
        r = self._git("cherry-pick", branch_name)
        if r.returncode != 0:
            logger.warning("Cherry-pick failed, attempting merge: %s", r.stderr.strip())
            self._git("cherry-pick", "--abort")
            r = self._git("merge", branch_name, "--no-edit")
            if r.returncode != 0:
                logger.warning("Merge also failed: %s", r.stderr.strip())
                self._git("merge", "--abort")
                return False

        logger.info("Merged self-heal branch %s to main", branch_name)
        return True

    def remove_worktree(self, session_id: str) -> bool:
        worktree_path = self._active_worktrees.pop(session_id, None)
        if not worktree_path:
            return True

        branch_name = f"selfheal/{self._sanitize_branch_id(session_id)}"

        self._git("worktree", "remove", worktree_path, "--force")
        if os.path.isdir(worktree_path):
            shutil.rmtree(worktree_path, ignore_errors=True)

        self._git("branch", "-D", branch_name)
        logger.info("Removed worktree and branch for session %s", session_id)
        return True

    def cleanup_all(self) -> None:
        for session_id in list(self._active_worktrees.keys()):
            self.remove_worktree(session_id)


class AgentSelfHealer:
    """Heals errors by calling the agent's own NVIDIA API model."""

    def __init__(
        self,
        model: Optional[str] = None,
        timeout: int = _SELF_HEAL_TIMEOUT_SECONDS,
    ):
        self.model = model or SELF_HEAL_MODEL
        self.timeout = timeout

    async def consult(
        self,
        error_report: SelfHealErrorReport,
        worktree_path: str,
    ) -> Optional[dict[str, Any]]:
        prompt = self._build_prompt(error_report, worktree_path)
        try:
            result = await asyncio.wait_for(
                self._run_query(prompt),
                timeout=self.timeout,
            )
            return result
        except asyncio.TimeoutError:
            logger.error("Self-heal consultation timed out after %ds", self.timeout)
            return None
        except Exception:
            logger.error("Self-heal consultation failed", exc_info=True)
            return None

    async def _run_query(self, prompt: str) -> Optional[dict[str, Any]]:
        from api import api_call_with_retry, process_response

        try:
            from handler import BASE_URL, API_KEY, MODEL_ID, BASE_PAYLOAD
        except ImportError:
            logger.error("Cannot import handler config for self-heal")
            return None

        model = self.model or MODEL_ID
        headers = {"Authorization": f"Bearer {API_KEY}"}

        payload = {
            "model": model,
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 4096,
            "stream": False,
            "tools": [],
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a code-healing agent. Analyze errors and produce "
                        "precise fixes. You MUST respond with ONLY a JSON object — "
                        "no markdown, no explanation outside the JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                response_data = await api_call_with_retry(
                    session,
                    BASE_URL,
                    payload,
                    headers,
                    max_retries=3,
                    stream=False,
                )
        except Exception:
            logger.error("API call failed during self-heal", exc_info=True)
            return None

        if "error" in response_data:
            err_msg = response_data.get("error", {})
            if isinstance(err_msg, dict):
                err_msg = err_msg.get("message", str(err_msg))
            logger.error("Self-heal API error: %s", err_msg)
            return None

        result_text = ""
        choices = response_data.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            result_text = message.get("content", "") or ""

        if not result_text:
            logger.warning("Self-heal returned empty response")
            return None

        cleaned = result_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        json_match = re.search(r"\{[\s\S]*\}", cleaned)
        if json_match:
            json_str = json_match.group(0)
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                logger.warning("Failed to parse extracted JSON from self-heal response")
        else:
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                logger.warning("Failed to parse self-heal response as JSON")

        return None

    def _build_prompt(
        self, error_report: SelfHealErrorReport, worktree_path: str
    ) -> str:
        parts = [
            "You are a code-healing agent. Analyze the error below and produce a fix.",
            "",
            f"Error: {error_report.error_string}",
        ]

        if error_report.exception_type:
            parts.append(f"Exception type: {error_report.exception_type}")
        if error_report.traceback_snippet:
            parts.append(f"Traceback:\n{error_report.traceback_snippet}")
        if error_report.tools_executed:
            parts.append(f"Tools executed: {', '.join(error_report.tools_executed)}")
        if error_report.user_query:
            parts.append(f"User query context: {error_report.user_query}")

        # Include relevant source files inline since the healer API call
        # has ``tools=[]`` and cannot read files itself.
        tool_file = os.path.join(worktree_path, "tools.py")
        if os.path.isfile(tool_file):
            try:
                with open(tool_file, "r", encoding="utf-8") as f:
                    content = f.read()
                if len(content) > 18000:
                    content = content[:18000] + "\n# ... (truncated)"
                parts.append(f"\n--- tools.py ---\n{content}")
            except OSError:
                parts.append("\n--- tools.py ---\n(could not read)")

        parts.extend(
            [
                "",
                "Instructions:",
                "1. Analyze the source code above to find the root cause.",
                "2. Produce minimal, targeted patches that fix the error.",
                "3. Each patch must include the full file path (relative to the repo root) and the complete corrected file content.",
                "4. Do NOT modify files unrelated to the error.",
                "",
                "Return ONLY a JSON object with this exact structure (no other text):",
                "{",
                '  "patches": [',
                "    {",
                '      "filepath": "<relative path to file>",',
                '      "content": "<complete fixed file content>",',
                '      "description": "<brief description of the fix>"',
                "    }",
                "  ],",
                '  "explanation": "<what was wrong and how you fixed it>"',
                "}",
            ]
        )

        return "\n".join(parts)


def _apply_patches_to_worktree(
    worktree_path: str, patches: list[dict[str, Any]]
) -> list[FilePatch]:
    applied: list[FilePatch] = []
    for patch_data in patches:
        filepath = patch_data.get("filepath", "")
        content = patch_data.get("content", "")
        description = patch_data.get("description", "")

        if not filepath or not content:
            logger.warning("Skipping patch with missing filepath/content")
            continue

        full_path = os.path.join(worktree_path, filepath)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        try:
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            applied.append(
                FilePatch(filepath=filepath, content=content, description=description)
            )
            logger.info("Applied patch to %s", filepath)
        except OSError:
            logger.error("Failed to write patch to %s", full_path, exc_info=True)

    return applied


def _parse_error_enrichments(error_string: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    iteration_match = re.search(r"iteration=(\d+)", error_string)
    if iteration_match:
        result["iteration"] = int(iteration_match.group(1))

    tools_match = re.search(r"tools_executed=\[([^\]]*)\]", error_string)
    if tools_match:
        raw = tools_match.group(1).strip()
        result["tools_executed"] = [
            t.strip().strip("'\"") for t in raw.split(",") if t.strip()
        ]
    return result


def _extract_exception_from_error(error_string: str) -> Optional[str]:
    patterns = [
        re.compile(r"Exception type: (\w+)"),
        re.compile(r"(\w+Error):"),
        re.compile(r"(\w+Exception):"),
        re.compile(r"Traceback.*?(\w+Error|Exception):", re.DOTALL),
    ]
    for pattern in patterns:
        match = pattern.search(error_string)
        if match:
            return match.group(1)
    return None


def _extract_traceback_from_error(error_string: str) -> Optional[str]:
    match = re.search(
        r"Traceback \(most recent call last\):.*?(?=\n\n|\Z)", error_string, re.DOTALL
    )
    if match:
        return match.group(0)[-2000:]
    tb_match = re.search(
        r"traceback_snippet['\"]?\s*[:=]\s*['\"]?(.+?)(?:['\"]?\s*$)",
        error_string,
        re.MULTILINE,
    )
    if tb_match:
        return tb_match.group(1)[:2000]
    return None


class SelfHealManager:
    def __init__(
        self,
        acp_agent: Any = None,
        repo_root: Optional[str] = None,
    ):
        self._acp_agent = acp_agent
        self._classifier = ErrorClassifier()
        self._wt_manager = GitWorktreeManager(repo_root=repo_root)
        self._healer = AgentSelfHealer()
        self._heal_history: list[dict[str, Any]] = []
        self._last_heal_time: float = 0
        self._session_attempts: dict[str, int] = {}
        self._active_session_id: Optional[str] = None
        self._enabled = _SELF_HEAL_ENABLED
        self._auto_merge = _SELF_HEAL_AUTO_MERGE
        self._cooldown = _SELF_HEAL_COOLDOWN_SECONDS
        self._max_attempts = _SELF_HEAL_MAX_ATTEMPTS_PER_SESSION

    def get_status(self) -> dict[str, Any]:
        return {
            "total_heals": len(self._heal_history),
            "last_heal_time": self._last_heal_time,
            "active_worktrees": dict(self._wt_manager._active_worktrees),
            "session_attempts": dict(self._session_attempts),
        }

    def shutdown(self) -> None:
        try:
            self._wt_manager.cleanup_all()
        except Exception:
            logger.warning("Error during self-heal shutdown", exc_info=True)

    async def try_heal(
        self,
        error_string: str,
        context: Optional[dict[str, Any]] = None,
    ) -> HealResult:
        context = context or {}

        if not self._enabled:
            return HealResult(success=False, error="Self-heal is disabled")

        now = time.time()
        if now - self._last_heal_time < self._cooldown:
            remaining = int(self._cooldown - (now - self._last_heal_time))
            return HealResult(
                success=False, error=f"Cooldown active ({remaining}s remaining)"
            )

        exception_type = context.get("exception_type") or _extract_exception_from_error(
            error_string
        )
        traceback_snippet = context.get(
            "traceback_snippet"
        ) or _extract_traceback_from_error(error_string)

        category = self._classifier.classify(
            error_string,
            exception_type=exception_type,
            context=context,
        )

        if category == ErrorCategory.TRANSIENT:
            return HealResult(
                success=False,
                error=f"Transient error (category={category.value}), not healable",
            )

        session_id = context.get("session_id", "default")
        attempts = self._session_attempts.get(session_id, 0)
        if attempts >= self._max_attempts:
            return HealResult(
                success=False,
                error=f"Max heal attempts ({self._max_attempts}) reached for session {session_id}",
            )

        enrichments = _parse_error_enrichments(error_string)

        report = SelfHealErrorReport(
            error_string=error_string,
            category=category,
            exception_type=exception_type,
            traceback_snippet=traceback_snippet,
            session_id=session_id,
            iteration=context.get("iteration", enrichments.get("iteration", 0)),
            tools_executed=context.get(
                "tools_executed", enrichments.get("tools_executed", [])
            ),
            user_query=context.get("user_query"),
        )

        result = await self._execute_heal(report, session_id)

        self._session_attempts[session_id] = attempts + 1
        self._last_heal_time = time.time()
        self._heal_history.append(
            {
                "timestamp": self._last_heal_time,
                "session_id": session_id,
                "category": category.value,
                "success": result.success,
                "applied": result.applied,
                "summary": result.summary,
            }
        )

        return result

    async def _execute_heal(
        self, report: SelfHealErrorReport, session_id: str
    ) -> HealResult:
        worktree_path = self._wt_manager.create_worktree(session_id)
        if not worktree_path:
            return HealResult(success=False, error="Failed to create git worktree")

        try:
            heal_result = await self._healer.consult(report, worktree_path)

            if not heal_result:
                return HealResult(
                    success=False,
                    error="Self-heal consultation returned no result",
                    worktree_path=worktree_path,
                )

            patches_data = heal_result.get("patches", [])
            explanation = heal_result.get("explanation", "")

            if not patches_data:
                return HealResult(
                    success=False,
                    error="No patches returned from self-heal",
                    worktree_path=worktree_path,
                )

            applied = _apply_patches_to_worktree(worktree_path, patches_data)

            if not applied:
                return HealResult(
                    success=False,
                    error="Failed to apply any patches",
                    worktree_path=worktree_path,
                )

            commit_message = (
                f"self-heal: fix {report.category.value} error\n\n"
                f"Error: {report.error_string[:200]}\n"
                f"Exception: {report.exception_type or 'unknown'}\n"
                f"Session: {session_id}\n"
                f"Explanation: {explanation[:500]}\n"
                f"Files patched: {', '.join(p.filepath for p in applied)}"
            )

            commit_hash = self._wt_manager.commit_in_worktree(
                session_id, commit_message
            )

            merged = False
            if self._auto_merge and commit_hash:
                merged = self._wt_manager.merge_to_main(session_id)

            summary = (
                f"Applied {len(applied)} patch(es) "
                f"({', '.join(p.filepath for p in applied)})"
            )
            if explanation:
                summary += f" — {explanation[:200]}"
            if merged:
                summary += " [merged]"

            return HealResult(
                success=True,
                applied=True,
                summary=summary,
                patches=applied,
                worktree_path=worktree_path,
                commit_hash=commit_hash,
                merged=merged,
            )
        except Exception:
            logger.error("Heal execution failed", exc_info=True)
            return HealResult(
                success=False,
                error=f"Heal execution error: {traceback.format_exc()[:500]}",
                worktree_path=worktree_path,
            )
        finally:
            self._wt_manager.remove_worktree(session_id)
