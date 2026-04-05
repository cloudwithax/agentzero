#!/usr/bin/env python3
"""Tests for cron-based reminder scheduler."""

import asyncio
import tempfile

from memory import EnhancedMemoryStore
from reminder_tasks import CronExpression, ReminderScheduler


def _build_scheduler(ai_runner=None) -> ReminderScheduler:
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    temp_db.close()
    memory_store = EnhancedMemoryStore(db_path=temp_db.name, api_key="test_key")
    return ReminderScheduler(
        memory_store=memory_store, ai_runner=ai_runner, poll_seconds=60
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

    async def fake_ai_runner(prompt: str, task_id: str) -> str:
        return f"AI output for {task_id}: {prompt}"

    scheduler = _build_scheduler(ai_runner=fake_ai_runner)
    created = await scheduler.create_task(
        cron="*/5 * * * *",
        run_ai=True,
        ai_prompt="Summarize top priorities",
        one_off=False,
        task_id="summary_task",
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
    asyncio.run(test_invalid_cron_rejected())
    print("Reminder scheduler tests passed")
