#!/usr/bin/env python3
"""Tests for cron-based reminder scheduler."""

import asyncio
import datetime
import tempfile
import time

import reminder_tasks as reminder_module
from memory import EnhancedMemoryStore
from reminder_tasks import CronExpression, ReminderScheduler
from tools import (
    reminder_create_tool,
    reset_tool_runtime_session,
    set_reminder_controller,
    set_tool_runtime_session,
)


def _build_scheduler(
    ai_runner=None,
    delivery_callback=None,
    db_path: str | None = None,
) -> ReminderScheduler:
    if db_path is None:
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        temp_db.close()
        db_path = temp_db.name

    memory_store = EnhancedMemoryStore(db_path=db_path, api_key="test_key")
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

    async def fake_ai_runner(prompt: str, task_id: str, _run_ai_with_tools: bool = False) -> str:
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


async def test_one_off_same_day_prefers_today() -> None:
    """One-off same-day schedules should resolve to the remaining time today."""
    scheduler = _build_scheduler()
    original_utc_now = reminder_module._utc_now
    frozen_now = datetime.datetime(2026, 4, 4, 10, 15, tzinfo=datetime.timezone.utc)

    reminder_module._utc_now = lambda: frozen_now
    try:
        created = await scheduler.create_task(
            cron="30 10 4 4 6",
            message="same day reminder",
            one_off=True,
            task_id="same_day_task",
        )
    finally:
        reminder_module._utc_now = original_utc_now

    assert created["success"], created
    assert created["task"]["next_run_at"] == "2026-04-04T10:30:00+00:00", created


async def test_one_off_same_day_past_time_is_rejected() -> None:
    """One-off schedules pinned to today should not roll over after the time has passed."""
    scheduler = _build_scheduler()
    original_utc_now = reminder_module._utc_now
    frozen_now = datetime.datetime(2026, 4, 4, 10, 31, tzinfo=datetime.timezone.utc)

    reminder_module._utc_now = lambda: frozen_now
    try:
        created = await scheduler.create_task(
            cron="30 10 4 4 6",
            message="too late",
            one_off=True,
            task_id="past_same_day_task",
        )
    finally:
        reminder_module._utc_now = original_utc_now

    assert created["success"] is False, created
    assert "same-day reminder time has already passed" in created["error"], created


async def test_reminders_persist_to_db_and_reload_on_startup() -> None:
    """Reminder tasks should persist in SQLite and reload into memory on startup."""
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    temp_db.close()

    first_scheduler = _build_scheduler(db_path=temp_db.name)
    created = await first_scheduler.create_task(
        cron="0 9 * * 1-5",
        message="weekday reminder",
        task_id="persisted_task",
        session_id="tg_999",
    )
    assert created["success"], created

    stored_rows = first_scheduler.memory_store.get_reminder_tasks()
    assert any(row["task_id"] == "persisted_task" for row in stored_rows), stored_rows

    second_scheduler = _build_scheduler(db_path=temp_db.name)
    await second_scheduler.start()

    status = await second_scheduler.get_task_status("persisted_task")
    assert status["success"], status
    task = status["task"]
    assert task["task_id"] == "persisted_task", task
    assert task["message"] == "weekday reminder", task
    assert task["session_id"] == "tg_999", task
    assert task["enabled"] is True, task


async def test_unix_timestamp_run_at_creates_one_off() -> None:
    """A task created with run_at (Unix timestamp) should be one-off and fire at the right time."""
    scheduler = _build_scheduler()
    future_ts = int(time.time()) + 3600  # 1 hour from now
    created = await scheduler.create_task(
        run_at=future_ts,
        message="timestamp-based reminder",
        task_id="ts_reminder",
    )
    assert created["success"], created
    task = created["task"]
    assert task["one_off"] is True, task
    assert task["enabled"] is True, task
    # next_run_at should be roughly 1 hour from now
    next_run = datetime.datetime.fromisoformat(task["next_run_at"])
    expected = datetime.datetime.fromtimestamp(future_ts, tz=datetime.timezone.utc)
    diff = abs((next_run - expected).total_seconds())
    assert diff < 2, f"next_run_at off by {diff}s: {task['next_run_at']} vs {expected.isoformat()}"


async def test_run_at_rejects_past_timestamp() -> None:
    """A task with a past run_at timestamp should be rejected."""
    scheduler = _build_scheduler()
    past_ts = int(time.time()) - 60  # 1 minute ago
    created = await scheduler.create_task(
        run_at=past_ts,
        message="should fail",
        task_id="past_ts_task",
    )
    assert created["success"] is False, created
    assert "run_at must be in the future" in created["error"], created


async def test_run_ai_with_tools_flag_passthrough() -> None:
    """The run_ai_with_tools flag should be captured in task metadata and passed to the AI runner."""

    runner_calls: list[tuple] = []

    async def tool_aware_runner(prompt: str, task_id: str, run_ai_with_tools: bool) -> str:
        runner_calls.append((prompt, task_id, run_ai_with_tools))
        return f"tools={run_ai_with_tools}"

    scheduler = _build_scheduler(ai_runner=tool_aware_runner)
    created = await scheduler.create_task(
        cron="* * * * *",
        run_ai=True,
        ai_prompt="Test prompt",
        run_ai_with_tools=True,
        task_id="tool_task",
        one_off=True,
    )
    assert created["success"], created
    assert created["task"]["run_ai_with_tools"] is True, created["task"]

    await scheduler.run_task_now("tool_task")
    assert len(runner_calls) == 1, runner_calls
    assert runner_calls[0][2] is True  # run_ai_with_tools flag


async def test_reminder_create_tool_passes_run_at() -> None:
    """The tool layer should pass run_at and run_ai_with_tools to the controller."""

    class FakeReminderController:
        def __init__(self):
            self.calls: list[dict] = []

        async def create_reminder_task(self, **kwargs):
            self.calls.append(kwargs)
            return {"success": True, "task": kwargs}

    controller = FakeReminderController()
    set_reminder_controller(controller)
    token = set_tool_runtime_session("tg_789")
    try:
        future_ts = int(time.time()) + 120
        result = await reminder_create_tool(
            run_at=future_ts,
            message="timestamp scheduled",
            run_ai_with_tools=True,
        )
    finally:
        reset_tool_runtime_session(token)
        set_reminder_controller(None)

    assert result["success"], result
    assert controller.calls[0]["run_at"] == future_ts, controller.calls[0]
    assert controller.calls[0]["run_ai_with_tools"] is True, controller.calls[0]
    assert controller.calls[0]["session_id"] == "tg_789", controller.calls[0]


async def test_delay_seconds_creates_one_off_at_relative_time() -> None:
    """delay_seconds should schedule a one-off ~N seconds from now (no client clock math)."""
    scheduler = _build_scheduler()
    before = int(time.time())
    created = await scheduler.create_task(
        delay_seconds=60,
        message="take out the trash",
        task_id="delay_task",
    )
    assert created["success"], created
    task = created["task"]
    assert task["one_off"] is True, task
    next_run = datetime.datetime.fromisoformat(task["next_run_at"])
    delta = (next_run - datetime.datetime.fromtimestamp(
        before, tz=datetime.timezone.utc
    )).total_seconds()
    assert 58 <= delta <= 63, f"delay_seconds=60 scheduled at delta={delta}s"


async def test_delay_seconds_rejects_non_positive() -> None:
    """A zero/negative delay should be rejected with a clear error."""
    scheduler = _build_scheduler()
    created = await scheduler.create_task(
        delay_seconds=0,
        message="nope",
        task_id="bad_delay",
    )
    assert created["success"] is False, created
    assert "delay_seconds" in created["error"], created


async def test_run_at_takes_precedence_over_delay_seconds() -> None:
    """When both are given, an explicit run_at wins (delay is the convenience path)."""
    scheduler = _build_scheduler()
    future_ts = int(time.time()) + 3600
    created = await scheduler.create_task(
        run_at=future_ts,
        delay_seconds=60,
        message="absolute wins",
        task_id="both_task",
    )
    assert created["success"], created
    next_run = datetime.datetime.fromisoformat(created["task"]["next_run_at"])
    expected = datetime.datetime.fromtimestamp(future_ts, tz=datetime.timezone.utc)
    assert abs((next_run - expected).total_seconds()) < 2, created["task"]


async def test_reminder_create_tool_passes_delay_seconds() -> None:
    """The tool layer should forward delay_seconds to the controller."""

    class FakeReminderController:
        def __init__(self):
            self.calls: list[dict] = []

        async def create_reminder_task(self, **kwargs):
            self.calls.append(kwargs)
            return {"success": True, "task": kwargs}

    controller = FakeReminderController()
    set_reminder_controller(controller)
    token = set_tool_runtime_session("imessage_+15551230000")
    try:
        result = await reminder_create_tool(
            delay_seconds=60,
            message="remind me in a minute",
        )
    finally:
        reset_tool_runtime_session(token)
        set_reminder_controller(None)

    assert result["success"], result
    assert controller.calls[0]["delay_seconds"] == 60, controller.calls[0]
    assert controller.calls[0]["session_id"] == "imessage_+15551230000"


if __name__ == "__main__":
    test_cron_expression_parser()
    asyncio.run(test_one_off_task_completes())
    asyncio.run(test_recurring_ai_task_runs())
    asyncio.run(test_reminder_create_tool_uses_runtime_session())
    asyncio.run(test_invalid_cron_rejected())
    asyncio.run(test_one_off_same_day_prefers_today())
    asyncio.run(test_one_off_same_day_past_time_is_rejected())
    asyncio.run(test_reminders_persist_to_db_and_reload_on_startup())
    asyncio.run(test_unix_timestamp_run_at_creates_one_off())
    asyncio.run(test_run_at_rejects_past_timestamp())
    asyncio.run(test_run_ai_with_tools_flag_passthrough())
    asyncio.run(test_reminder_create_tool_passes_run_at())
    asyncio.run(test_delay_seconds_creates_one_off_at_relative_time())
    asyncio.run(test_delay_seconds_rejects_non_positive())
    asyncio.run(test_run_at_takes_precedence_over_delay_seconds())
    asyncio.run(test_reminder_create_tool_passes_delay_seconds())
    print("Reminder scheduler tests passed")
