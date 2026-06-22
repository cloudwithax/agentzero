"""Tests for the orchestrated 3-stage pipeline (plan → blind worker → finalize).

Deterministic — no live API. Run with:
    AGENTZERO_LIVE_TESTS=0 PYTHONPATH=. python3 tests/test_orchestrated_pipeline.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import handler
from handler import (
    ADVISOR_MODEL_ID,
    EXECUTOR_MODEL_ID,
    _WORKER_EXCLUDED_TOOLS,
    _is_trivial_query,
)
from tools import (
    reset_tool_runtime_session,
    send_message_tool,
    set_tool_runtime_session,
)
from tests._live_harness import _make_handler


class _StubTask:
    def __init__(self, type_):
        self.type = type_


# ─── Unit tests ────────────────────────────────────────────────────────────


def test_is_trivial_query() -> None:
    print("\nTest 1: _is_trivial_query gating")
    for q in [
        "hi", "hello", "hey there", "thanks", "thank you", "thx",
        "ok", "okay", "sure", "lol", "got it", "k", "  ", "",
    ]:
        assert _is_trivial_query(q, None, None) is True, f"should be trivial: {q!r}"

    # Classified non-trivial tasks must NOT be trivial.
    assert _is_trivial_query(
        "build me a landing page and publish it", _StubTask("coding")
    ) is False
    assert _is_trivial_query(
        "what is the capital of France and its population?", _StubTask("research")
    ) is False
    # Starts with "ok" but has real work + a coding classification → not trivial
    # (regex is anchored to the whole string; length clause needs generic/None).
    assert _is_trivial_query(
        "ok implement oauth across the app", _StubTask("coding")
    ) is False
    # Short + unclassified → trivial (safe single-model fallback).
    assert _is_trivial_query("whatever man", None) is True
    print("  PASS")


def test_worker_tool_filter_strips_delivery_tools() -> None:
    print("\nTest 2: worker tool filter")
    tools = list(handler.BASE_PAYLOAD.get("tools", []))
    filtered = [
        t
        for t in tools
        if t.get("function", {}).get("name") not in _WORKER_EXCLUDED_TOOLS
    ]
    names = {t["function"]["name"] for t in filtered}
    for excluded in _WORKER_EXCLUDED_TOOLS:
        assert excluded not in names, f"{excluded} should be stripped"
    for kept in ["bash", "read", "write", "web_search", "grep", "glob"]:
        assert kept in names, f"{kept} should remain available to the worker"
    print("  PASS")


def test_flag_default_on_and_parse() -> None:
    print("\nTest 3: pipeline flag default + parse contract")
    assert handler.ORCHESTRATED_PIPELINE_ENABLED is True
    # The module computes: env.get(..., "1").strip() != "0"
    assert ("1".strip() != "0") is True
    assert ("0".strip() != "0") is False
    assert ("  0  ".strip() != "0") is False
    print("  PASS")


async def test_sentinel_session_buffers_send_message() -> None:
    print("\nTest 4: sentinel worker session buffers send_message")
    token = set_tool_runtime_session("orchestrated_worker__tg_123456")
    try:
        result = await send_message_tool("hello from the worker")
    finally:
        reset_tool_runtime_session(token)
    assert result.get("success") is True, result
    assert result.get("channel") == "buffered", (
        f"sentinel session must buffer, not deliver: {result}"
    )
    print("  PASS")


# ─── _run_orchestrated_task flow tests (mocked api/process_response) ─────────


def _plan_choice(text):
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


async def _run_with_mocks(*, plan_resp, finalize_resp, worker_result="WORKER REPORT: built it at /tmp/x"):
    """Drive _run_orchestrated_task with mocked api_call_with_retry +
    process_response. Returns (result, captured)."""
    captured = {"api_payloads": []}

    async def fake_api(session, base_url, payload, headers, **kwargs):
        captured["api_payloads"].append(payload)
        sys_content = (
            payload["messages"][0]["content"] if payload.get("messages") else ""
        )
        if "planning half" in sys_content:
            return plan_resp
        if "replying to the user" in sys_content:
            return finalize_resp
        # worker's first API call
        return _plan_choice("worker thinking")

    async def fake_process_response(
        response_data, messages, session, base_url, api_key, base_payload, *a, **k
    ):
        captured["worker_payload"] = base_payload
        captured["worker_messages"] = messages
        return worker_result

    orig_api = handler.api_call_with_retry
    orig_proc = handler.process_response
    handler.api_call_with_retry = fake_api
    handler.process_response = fake_process_response
    try:
        h = _make_handler()
        result = await h._run_orchestrated_task(
            user_query="build me a small site and publish it",
            session_id="test_orch_sess",
            data={"messages": [{"role": "user", "content": "build me a small site"}]},
            memory_context="",
            skills_catalog_context="",
            active_skills_context="",
            session=None,
        )
    finally:
        handler.api_call_with_retry = orig_api
        handler.process_response = orig_proc
    return result, captured


async def test_three_stage_flow() -> None:
    print("\nTest 5: three-stage flow (plan → worker → finalize)")
    result, captured = await _run_with_mocks(
        plan_resp=_plan_choice("Build a one-page site. Publish it. Done when a URL exists."),
        finalize_resp=_plan_choice("all set — your site is live at example.here.now"),
    )
    assert result == "all set — your site is live at example.here.now", result

    # Stage 1 (plan) and stage 3 (finalize) use the advisor model, no tools.
    plan_payload = captured["api_payloads"][0]
    assert plan_payload["model"] == ADVISOR_MODEL_ID
    assert plan_payload["tools"] == []

    # Worker uses the executor model with delivery tools stripped.
    wp = captured["worker_payload"]
    assert wp["model"] == EXECUTOR_MODEL_ID, wp.get("model")
    worker_tool_names = {t["function"]["name"] for t in wp.get("tools", [])}
    assert "send_message" not in worker_tool_names
    assert "bash" in worker_tool_names

    # Worker's final user message is the orchestrator brief, and its system
    # prompt instructs reporting (blind to the orchestration).
    wm = captured["worker_messages"]
    assert wm[-1]["role"] == "user"
    assert "Publish it" in wm[-1]["content"]
    assert "report your results" in wm[0]["content"].lower()
    print("  PASS")


async def test_finalize_failure_degrades_to_worker_results() -> None:
    print("\nTest 6: finalize failure returns worker results")
    result, _ = await _run_with_mocks(
        plan_resp=_plan_choice("Do the thing."),
        finalize_resp={"error": {"message": "rate limit"}},
        worker_result="WORKER REPORT: finished the thing",
    )
    assert result == "WORKER REPORT: finished the thing", result
    print("  PASS")


async def test_plan_failure_raises_for_fallback() -> None:
    print("\nTest 7: plan failure raises (caller falls back)")
    raised = False
    try:
        await _run_with_mocks(
            plan_resp={"error": {"message": "rate limit"}},
            finalize_resp=_plan_choice("unused"),
        )
    except RuntimeError:
        raised = True
    assert raised, "stage-1 failure must raise so handle() can fall back"
    print("  PASS")


async def main() -> None:
    print("=" * 60)
    print("Orchestrated pipeline tests")
    print("=" * 60)
    test_is_trivial_query()
    test_worker_tool_filter_strips_delivery_tools()
    test_flag_default_on_and_parse()
    await test_sentinel_session_buffers_send_message()
    await test_three_stage_flow()
    await test_finalize_failure_degrades_to_worker_results()
    await test_plan_failure_raises_for_fallback()
    print("\n" + "=" * 60)
    print("All orchestrated pipeline tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
