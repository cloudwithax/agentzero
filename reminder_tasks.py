"""Cron-based reminder and recurring task scheduler."""

from __future__ import annotations

import asyncio
import datetime
import logging
import re
import uuid
from typing import Any, Awaitable, Callable, Optional

from memory import EnhancedMemoryStore

logger = logging.getLogger(__name__)


def _utc_now() -> datetime.datetime:
    """Return current UTC time."""
    return datetime.datetime.now(datetime.timezone.utc)


def _parse_iso_datetime(value: Any) -> Optional[datetime.datetime]:
    """Parse an ISO datetime string with timezone awareness."""
    if not value:
        return None

    try:
        parsed = datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=datetime.timezone.utc)

    return parsed.astimezone(datetime.timezone.utc)


def _to_cron_dow(dt_value: datetime.datetime) -> int:
    """Convert Python weekday to cron weekday (0=Sunday)."""
    return (dt_value.weekday() + 1) % 7


class CronExpression:
    """Minimal cron parser/evaluator for 5-field expressions."""

    MONTH_NAMES = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }
    DOW_NAMES = {
        "sun": 0,
        "mon": 1,
        "tue": 2,
        "wed": 3,
        "thu": 4,
        "fri": 5,
        "sat": 6,
    }

    def __init__(self, expression: str):
        self.expression = str(expression or "").strip()
        parts = self.expression.split()
        if len(parts) != 5:
            raise ValueError(
                "cron must include 5 fields: minute hour day month weekday"
            )

        minute, hour, day, month, weekday = parts
        self.minutes = self._parse_field(minute, 0, 59)
        self.hours = self._parse_field(hour, 0, 23)
        self.days = self._parse_field(day, 1, 31)
        self.months = self._parse_field(month, 1, 12, aliases=self.MONTH_NAMES)
        self.weekdays = self._parse_field(
            weekday,
            0,
            6,
            aliases=self.DOW_NAMES,
            map_seven_to_zero=True,
        )
        self.days_any = day.strip() == "*"
        self.weekdays_any = weekday.strip() == "*"

    def _parse_field(
        self,
        value: str,
        minimum: int,
        maximum: int,
        aliases: Optional[dict[str, int]] = None,
        map_seven_to_zero: bool = False,
    ) -> set[int]:
        field = str(value).strip().lower()
        if not field:
            raise ValueError("cron field cannot be empty")

        if field == "*":
            return set(range(minimum, maximum + 1))

        results: set[int] = set()
        for token in field.split(","):
            token = token.strip()
            if not token:
                raise ValueError("invalid empty cron token")

            if "/" in token:
                range_part, step_part = token.split("/", 1)
                if not step_part.isdigit():
                    raise ValueError(f"invalid step value: {token}")
                step = int(step_part)
                if step <= 0:
                    raise ValueError(f"step must be positive: {token}")
            else:
                range_part = token
                step = 1

            if range_part == "*":
                start = minimum
                end = maximum
            elif "-" in range_part:
                start_raw, end_raw = range_part.split("-", 1)
                start = self._parse_value(
                    start_raw,
                    minimum,
                    maximum,
                    aliases=aliases,
                    map_seven_to_zero=map_seven_to_zero,
                )
                end = self._parse_value(
                    end_raw,
                    minimum,
                    maximum,
                    aliases=aliases,
                    map_seven_to_zero=map_seven_to_zero,
                )
                if start > end:
                    raise ValueError(f"range start is greater than end: {token}")
            else:
                single = self._parse_value(
                    range_part,
                    minimum,
                    maximum,
                    aliases=aliases,
                    map_seven_to_zero=map_seven_to_zero,
                )
                start = single
                end = single

            for number in range(start, end + 1, step):
                if number < minimum or number > maximum:
                    raise ValueError(f"value out of bounds: {token}")
                results.add(number)

        if not results:
            raise ValueError(f"cron field has no values: {value}")

        return results

    def _parse_value(
        self,
        token: str,
        minimum: int,
        maximum: int,
        aliases: Optional[dict[str, int]] = None,
        map_seven_to_zero: bool = False,
    ) -> int:
        normalized = str(token).strip().lower()
        if not normalized:
            raise ValueError("empty cron token")

        if aliases and normalized in aliases:
            value = aliases[normalized]
        elif re.fullmatch(r"\d+", normalized):
            value = int(normalized)
        else:
            raise ValueError(f"invalid cron token: {token}")

        if map_seven_to_zero and value == 7:
            value = 0

        if value < minimum or value > maximum:
            raise ValueError(f"cron value out of bounds: {token}")

        return value

    def matches(self, dt_value: datetime.datetime) -> bool:
        """Return True if this cron expression matches a datetime."""
        if dt_value.minute not in self.minutes:
            return False
        if dt_value.hour not in self.hours:
            return False
        if dt_value.month not in self.months:
            return False

        day_match = dt_value.day in self.days
        weekday_match = _to_cron_dow(dt_value) in self.weekdays

        if self.days_any and self.weekdays_any:
            return True
        if self.days_any:
            return weekday_match
        if self.weekdays_any:
            return day_match

        # Cron semantics: if both DOM and DOW are restricted, either can match.
        return day_match or weekday_match

    def next_after(
        self,
        after_dt: datetime.datetime,
        max_search_minutes: int = 60 * 24 * 366 * 2,
    ) -> Optional[datetime.datetime]:
        """Find the next matching datetime strictly after the provided timestamp."""
        current = after_dt.astimezone(datetime.timezone.utc).replace(
            second=0,
            microsecond=0,
        )
        candidate = current + datetime.timedelta(minutes=1)

        for _ in range(max_search_minutes):
            if self.matches(candidate):
                return candidate
            candidate += datetime.timedelta(minutes=1)

        return None


class ReminderScheduler:
    """Persistent scheduler for one-off and recurring cron tasks."""

    STATE_KEY = "reminders.tasks.v1"

    def __init__(
        self,
        memory_store: EnhancedMemoryStore,
        ai_runner: Optional[Callable[[str, str], Awaitable[str]]] = None,
        poll_seconds: int = 20,
    ):
        self.memory_store = memory_store
        self.ai_runner = ai_runner
        self.poll_seconds = max(5, int(poll_seconds))
        self._tasks: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._runner_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._loaded = False

    async def start(self) -> None:
        """Start the background scheduler loop if not already running."""
        await self._load_if_needed()

        async with self._lock:
            if (
                isinstance(self._runner_task, asyncio.Task)
                and not self._runner_task.done()
            ):
                return
            self._stop_event.clear()
            self._runner_task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop the background scheduler loop."""
        async with self._lock:
            runner = self._runner_task
            self._runner_task = None
            self._stop_event.set()

        if isinstance(runner, asyncio.Task) and not runner.done():
            runner.cancel()
            try:
                await runner
            except asyncio.CancelledError:
                pass

    async def create_task(
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
        """Create and persist a reminder task."""
        await self.start()

        cron_expression = str(cron or "").strip()
        if not cron_expression:
            return {"success": False, "error": "cron is required"}

        try:
            parser = CronExpression(cron_expression)
        except ValueError as exc:
            return {"success": False, "error": f"invalid cron: {exc}"}

        normalized_name = str(name or "").strip()
        normalized_message = str(message or "").strip()
        normalized_prompt = str(ai_prompt or "").strip()

        if run_ai and not normalized_prompt:
            return {
                "success": False,
                "error": "ai_prompt is required when run_ai is true",
            }
        if not run_ai and not normalized_message:
            return {
                "success": False,
                "error": "message is required when run_ai is false",
            }

        normalized_task_id = self._normalize_task_id(task_id)
        if task_id and not normalized_task_id:
            return {
                "success": False,
                "error": "task_id must include letters, numbers, '-' or '_'",
            }
        if not normalized_task_id:
            normalized_task_id = self._generate_task_id()

        now = _utc_now()
        next_run = parser.next_after(now - datetime.timedelta(minutes=1))
        if next_run is None:
            return {"success": False, "error": "unable to compute next run time"}

        async with self._lock:
            if normalized_task_id in self._tasks:
                return {
                    "success": False,
                    "error": f"Task already exists: {normalized_task_id}",
                }

            timestamp = now.isoformat(timespec="seconds")
            task = {
                "task_id": normalized_task_id,
                "name": normalized_name,
                "cron": cron_expression,
                "message": normalized_message,
                "session_id": str(session_id or "").strip() or None,
                "one_off": bool(one_off),
                "run_ai": bool(run_ai),
                "ai_prompt": normalized_prompt,
                "status": "active",
                "enabled": True,
                "run_count": 0,
                "last_run_at": None,
                "next_run_at": next_run.isoformat(timespec="seconds"),
                "last_result": "",
                "last_error": "",
                "created_at": timestamp,
                "updated_at": timestamp,
                "max_runs": 1 if one_off else 0,
            }
            self._tasks[normalized_task_id] = task
            self._persist_locked()

        return {
            "success": True,
            "message": f"Created reminder task {normalized_task_id}",
            "task": self._snapshot(task),
        }

    async def list_tasks(self, include_disabled: bool = True) -> dict[str, Any]:
        """List reminder tasks in reverse chronological order."""
        await self._load_if_needed()

        async with self._lock:
            tasks = [
                self._snapshot(task)
                for task in self._tasks.values()
                if include_disabled or bool(task.get("enabled", False))
            ]

        tasks.sort(key=lambda item: item.get("created_at") or "", reverse=True)
        running_count = sum(1 for task in tasks if task.get("status") == "running")
        active_count = sum(1 for task in tasks if task.get("enabled") is True)

        return {
            "success": True,
            "count": len(tasks),
            "active_count": active_count,
            "running_count": running_count,
            "tasks": tasks,
        }

    async def get_task_status(self, task_id: str) -> dict[str, Any]:
        """Return status for one reminder task."""
        await self._load_if_needed()

        normalized_task_id = self._normalize_task_id(task_id)
        if not normalized_task_id:
            return {"success": False, "error": "task_id is required"}

        async with self._lock:
            task = self._tasks.get(normalized_task_id)
            if task is None:
                return {
                    "success": False,
                    "error": f"Task not found: {normalized_task_id}",
                }
            snapshot = self._snapshot(task)

        return {"success": True, "task": snapshot}

    async def cancel_task(self, task_id: str, reason: str = "") -> dict[str, Any]:
        """Cancel and disable a reminder task."""
        await self._load_if_needed()

        normalized_task_id = self._normalize_task_id(task_id)
        if not normalized_task_id:
            return {"success": False, "error": "task_id is required"}

        normalized_reason = str(reason or "").strip()

        async with self._lock:
            task = self._tasks.get(normalized_task_id)
            if task is None:
                return {
                    "success": False,
                    "error": f"Task not found: {normalized_task_id}",
                }

            task["enabled"] = False
            task["status"] = "cancelled"
            task["next_run_at"] = None
            if normalized_reason:
                task["last_error"] = normalized_reason
            task["updated_at"] = _utc_now().isoformat(timespec="seconds")
            self._persist_locked()
            snapshot = self._snapshot(task)

        return {
            "success": True,
            "message": f"Cancelled reminder task {normalized_task_id}",
            "task": snapshot,
        }

    async def run_task_now(self, task_id: str) -> dict[str, Any]:
        """Run a reminder task immediately."""
        await self._load_if_needed()

        normalized_task_id = self._normalize_task_id(task_id)
        if not normalized_task_id:
            return {"success": False, "error": "task_id is required"}

        async with self._lock:
            task = self._tasks.get(normalized_task_id)
            if task is None:
                return {
                    "success": False,
                    "error": f"Task not found: {normalized_task_id}",
                }
            if not bool(task.get("enabled", False)):
                return {
                    "success": False,
                    "error": f"Task is not active: {normalized_task_id}",
                }

        await self._execute_task(normalized_task_id, trigger="manual")
        return await self.get_task_status(normalized_task_id)

    async def _run_loop(self) -> None:
        """Background polling loop that executes due tasks."""
        try:
            while True:
                await self._run_due_tasks()
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=self.poll_seconds
                    )
                    return
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Reminder scheduler loop crashed")

    async def _run_due_tasks(self) -> None:
        """Run all due tasks as of now."""
        now = _utc_now()
        due_task_ids: list[str] = []

        async with self._lock:
            for task in self._tasks.values():
                if not bool(task.get("enabled", False)):
                    continue
                if str(task.get("status", "")) == "running":
                    continue

                next_run = _parse_iso_datetime(task.get("next_run_at"))
                if next_run and next_run <= now:
                    due_task_ids.append(task["task_id"])

        for task_id in due_task_ids:
            await self._execute_task(task_id, trigger="schedule")

    async def _execute_task(self, task_id: str, trigger: str) -> None:
        """Execute one reminder task and update schedule state."""
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None or not bool(task.get("enabled", False)):
                return

            task["status"] = "running"
            task["updated_at"] = _utc_now().isoformat(timespec="seconds")
            self._persist_locked()

            run_ai = bool(task.get("run_ai", False))
            ai_prompt = str(task.get("ai_prompt", "")).strip()
            message = str(task.get("message", "")).strip()
            session_id = task.get("session_id")
            cron = str(task.get("cron", "")).strip()
            max_runs = int(task.get("max_runs", 0) or 0)
            previous_run_count = int(task.get("run_count", 0) or 0)

        now = _utc_now()
        run_error = ""
        output = ""

        try:
            if run_ai:
                if not self.ai_runner:
                    raise RuntimeError("AI runner not configured")
                output = await self.ai_runner(ai_prompt, task_id)
            else:
                output = message
        except Exception as exc:
            run_error = str(exc)
            logger.exception("Reminder task execution failed for %s", task_id)

        if session_id and output:
            self.memory_store.add_conversation_message(
                role="assistant",
                content=f"[Scheduled Task {task_id}] {output}",
                session_id=str(session_id),
                metadata={
                    "type": "reminder_task_output",
                    "task_id": task_id,
                    "trigger": trigger,
                },
            )

        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return

            run_count = previous_run_count + 1
            task["run_count"] = run_count
            task["last_run_at"] = now.isoformat(timespec="seconds")
            task["last_result"] = output
            task["last_error"] = run_error

            should_complete = bool(task.get("one_off", False)) or (
                max_runs > 0 and run_count >= max_runs
            )

            next_run_value: Optional[str] = None
            if not should_complete:
                try:
                    parser = CronExpression(cron)
                    next_run_dt = parser.next_after(now)
                    if next_run_dt:
                        next_run_value = next_run_dt.isoformat(timespec="seconds")
                except Exception as exc:
                    run_error = run_error or f"failed to compute next run: {exc}"

            if should_complete:
                task["enabled"] = False
                task["status"] = "completed" if not run_error else "failed"
                task["next_run_at"] = None
            else:
                if next_run_value:
                    task["status"] = "active" if not run_error else "error"
                    task["next_run_at"] = next_run_value
                else:
                    task["enabled"] = False
                    task["status"] = "failed"
                    task["next_run_at"] = None
                    if not run_error:
                        task["last_error"] = "unable to compute next run"

            task["updated_at"] = _utc_now().isoformat(timespec="seconds")
            self._persist_locked()

    async def _load_if_needed(self) -> None:
        """Load persisted tasks from agent state exactly once."""
        if self._loaded:
            return

        async with self._lock:
            if self._loaded:
                return

            raw_state = self.memory_store.get_agent_state(self.STATE_KEY, default={})
            state = raw_state if isinstance(raw_state, dict) else {}
            raw_tasks = state.get("tasks", []) if isinstance(state, dict) else []
            loaded_tasks: dict[str, dict[str, Any]] = {}

            if isinstance(raw_tasks, list):
                for item in raw_tasks:
                    if not isinstance(item, dict):
                        continue
                    task_id = self._normalize_task_id(item.get("task_id"))
                    cron = str(item.get("cron", "")).strip()
                    if not task_id or not cron:
                        continue

                    try:
                        CronExpression(cron)
                    except Exception:
                        continue

                    loaded_tasks[task_id] = {
                        "task_id": task_id,
                        "name": str(item.get("name", "")).strip(),
                        "cron": cron,
                        "message": str(item.get("message", "")).strip(),
                        "session_id": str(item.get("session_id", "")).strip() or None,
                        "one_off": bool(item.get("one_off", False)),
                        "run_ai": bool(item.get("run_ai", False)),
                        "ai_prompt": str(item.get("ai_prompt", "")).strip(),
                        "status": str(item.get("status", "active")),
                        "enabled": bool(item.get("enabled", True)),
                        "run_count": int(item.get("run_count", 0) or 0),
                        "last_run_at": item.get("last_run_at"),
                        "next_run_at": item.get("next_run_at"),
                        "last_result": str(item.get("last_result", "")),
                        "last_error": str(item.get("last_error", "")),
                        "created_at": item.get("created_at")
                        or _utc_now().isoformat(timespec="seconds"),
                        "updated_at": item.get("updated_at")
                        or _utc_now().isoformat(timespec="seconds"),
                        "max_runs": int(item.get("max_runs", 0) or 0),
                    }

            self._tasks = loaded_tasks
            self._loaded = True
            self._persist_locked()

    def _persist_locked(self) -> None:
        """Persist in-memory tasks to agent state (lock must be held)."""
        serializable = []
        for task in self._tasks.values():
            serializable.append(
                {
                    "task_id": task.get("task_id"),
                    "name": task.get("name"),
                    "cron": task.get("cron"),
                    "message": task.get("message"),
                    "session_id": task.get("session_id"),
                    "one_off": bool(task.get("one_off", False)),
                    "run_ai": bool(task.get("run_ai", False)),
                    "ai_prompt": task.get("ai_prompt"),
                    "status": task.get("status"),
                    "enabled": bool(task.get("enabled", False)),
                    "run_count": int(task.get("run_count", 0) or 0),
                    "last_run_at": task.get("last_run_at"),
                    "next_run_at": task.get("next_run_at"),
                    "last_result": task.get("last_result", ""),
                    "last_error": task.get("last_error", ""),
                    "created_at": task.get("created_at"),
                    "updated_at": task.get("updated_at"),
                    "max_runs": int(task.get("max_runs", 0) or 0),
                }
            )

        self.memory_store.set_agent_state(
            self.STATE_KEY,
            {
                "version": 1,
                "tasks": sorted(
                    serializable, key=lambda item: item.get("created_at") or ""
                ),
            },
        )

    @staticmethod
    def _normalize_task_id(task_id: Any) -> str:
        """Normalize user-provided task ID to safe characters."""
        if task_id is None:
            return ""

        normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(task_id).strip())
        normalized = normalized.strip("_")
        return normalized[:80]

    @staticmethod
    def _generate_task_id() -> str:
        """Generate a new unique reminder task ID."""
        stamp = _utc_now().strftime("%Y%m%d_%H%M%S")
        return f"reminder_task_{stamp}_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _snapshot(task: dict[str, Any]) -> dict[str, Any]:
        """Build a public snapshot without internal objects."""
        return {
            "task_id": task.get("task_id", ""),
            "name": task.get("name", ""),
            "cron": task.get("cron", ""),
            "message": task.get("message", ""),
            "session_id": task.get("session_id"),
            "one_off": bool(task.get("one_off", False)),
            "run_ai": bool(task.get("run_ai", False)),
            "status": task.get("status", "unknown"),
            "enabled": bool(task.get("enabled", False)),
            "run_count": int(task.get("run_count", 0) or 0),
            "created_at": task.get("created_at"),
            "updated_at": task.get("updated_at"),
            "last_run_at": task.get("last_run_at"),
            "next_run_at": task.get("next_run_at"),
            "last_result": task.get("last_result", ""),
            "last_error": task.get("last_error", ""),
        }
