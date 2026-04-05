#!/usr/bin/env python3
"""Tests for cron-based reminder scheduler."""

import asyncio
import tempfile

from memory import EnhancedMemoryStore
from reminder_tasks import CronExpression, ReminderScheduler
from tools import (
    reminder_create_tool,
    reset_tool_runtime_session,
    set_reminder_controller,
    set_tool_runtime_session,
)


def _build_scheduler(ai_runner=None, delivery_callback=None) -> ReminderScheduler:
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    temp_db.close()
    memory_store = EnhancedMemoryStore(db_path=temp_db.name, api_key="test_key")
    return ReminderScheduler(
        memory_store=memory_store,
        ai_runner=ai_runner,
        delivery_callback=delivery_callback,
        poll_seconds=60,
    )


def test_cron_expression_parser() -> None:
    """Cron parser should support standard wildcards and steps."""
    expr = CronExpression("*/15 9-17 * * 1-5")
    assert 0 in expr.minutes and 15 in expr.minutes and 45 in expr.minutes
    assert 9 in expr.hours and 17 in expr.hours
    assert 1 in expr.weekdays and 5 in expr.weekdays


async def test_one_off_task_completes() -> None:
    """A one-off task should complete and disable after one run."""
    scheduler = _build_scheduler()
    created = await scheduler.create_task(
        cron="* * * * *",
        message="Pay rent",
        one_off=True,
        task_id="rent_reminder",
    )
    assert created["success"], created

    status = await scheduler.run_task_now("rent_reminder")
    assert status["success"], status
    task = status["task"]
    assert task["status"] == "completed", task
    assert task["enabled"] is False, task
    assert task["run_count"] == 1, task


async def test_recurring_ai_task_runs() -> None:
    """Recurring tasks with AI output should stay active after a run."""

    deliveries: list[tuple[str, str]] = []

    async def fake_ai_runner(prompt: str, task_id: str) -> str:
        return f"AI output for {task_id}: {prompt}"

    async def fake_delivery_callback(session_id: str, output: str) -> dict:
        deliveries.append((session_id, output))
        return {"success": True}

    scheduler = _build_scheduler(
        ai_runner=fake_ai_runner,
        delivery_callback=fake_delivery_callback,
    )
    created = await scheduler.create_task(
        cron="*/5 * * * *",
        run_ai=True,
        ai_prompt="Summarize top priorities",
        one_off=False,
        task_id="summary_task",
        session_id="tg_123",
    )
    assert created["success"], created

    status = await scheduler.run_task_now("summary_task")
    assert status["success"], status
    task = status["task"]
    assert task["enabled"] is True, task
    assert task["status"] in {"active", "running"}, task
    assert task["run_count"] == 1, task
    assert "AI output for summary_task" in task["last_result"], task
    assert task["next_run_at"], task
    assert deliveries == [
        ("tg_123", "AI output for summary_task: Summarize top priorities")
    ], deliveries


async def test_reminder_create_tool_uses_runtime_session() -> None:
    """Reminder creation should default to the active tool runtime session."""

    class FakeReminderController:
        def __init__(self):
            self.calls: list[dict] = []

        async def create_reminder_task(self, **kwargs):
            self.calls.append(kwargs)
            return {"success": True, "task": kwargs}

    controller = FakeReminderController()
    set_reminder_controller(controller)
    token = set_tool_runtime_session("tg_456")
    try:
        result = await reminder_create_tool(
            cron="* * * * *",
            message="hello",
        )
    finally:
        reset_tool_runtime_session(token)
        set_reminder_controller(None)

    assert result["success"], result
    assert controller.calls, controller.calls
    assert controller.calls[0]["session_id"] == "tg_456", controller.calls[0]


async def test_invalid_cron_rejected() -> None:
    """Invalid cron syntax should be rejected."""
    scheduler = _build_scheduler()
    created = await scheduler.create_task(
        cron="invalid cron",
        message="This should fail",
    )
    assert created["success"] is False
    assert "invalid cron" in created["error"].lower()


if __name__ == "__main__":
    test_cron_expression_parser()
    asyncio.run(test_one_off_task_completes())
    asyncio.run(test_recurring_ai_task_runs())
    asyncio.run(test_reminder_create_tool_uses_runtime_session())
    asyncio.run(test_invalid_cron_rejected())
    print("Reminder scheduler tests passed")
