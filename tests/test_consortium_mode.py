#!/usr/bin/env python3
"""Tests for consortium mode behavior in AgentHandler."""

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, patch

from dotenv import load_dotenv

from capabilities import Capability, CapabilityProfile, AdaptiveFormatter
from examples import ExampleBank, AdaptiveFewShotManager
from handler import AgentHandler
from memory import EnhancedMemoryStore
from planning import TaskAnalyzer, TaskPlanner
from tools import TOOLS


load_dotenv()


def create_test_handler(db_path: str) -> AgentHandler:
    """Create a handler with an isolated temporary database."""
    memory_store = EnhancedMemoryStore(
        db_path=db_path,
        api_key=os.environ.get("NVIDIA_API_KEY", "test-key"),
    )

    capability_profile = CapabilityProfile(
        capabilities={
            Capability.JSON_OUTPUT,
            Capability.TOOL_USE,
            Capability.CHAIN_OF_THOUGHT,
            Capability.REASONING,
            Capability.LONG_CONTEXT,
            Capability.FEW_SHOT,
            Capability.SELF_CORRECTION,
            Capability.STRUCTURED_OUTPUT,
        },
        model_name="test-model",
    )

    task_planner = TaskPlanner(capability_profile)
    task_analyzer = TaskAnalyzer()
    adaptive_formatter = AdaptiveFormatter(capability_profile)
    example_bank = AdaptiveFewShotManager(ExampleBank(exploration_rate=0.1))

    return AgentHandler(
        memory_store=memory_store,
        capability_profile=capability_profile,
        example_bank=example_bank,
        task_planner=task_planner,
        task_analyzer=task_analyzer,
        adaptive_formatter=adaptive_formatter,
    )


class DummyClientSession:
    """Minimal aiohttp.ClientSession replacement for consortium-route tests."""

    async def __aenter__(self):
        return AsyncMock()

    async def __aexit__(self, exc_type, exc, tb):
        return False


async def test_trigger_detection() -> None:
    """Ensure consortium routing uses intent detection with confidence gating."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        handler = create_test_handler(db_path)
        should_contact, is_confident = handler._detect_consortium_contact_intent(
            "Ask the consortium if this is a good idea"
        )
        assert should_contact and is_confident

        should_contact, is_confident = handler._detect_consortium_contact_intent(
            "Please use consortium mode for this decision"
        )
        assert should_contact and is_confident

        should_contact, is_confident = handler._detect_consortium_contact_intent(
            "What is a consortium?"
        )
        assert not should_contact and is_confident

        should_contact, is_confident = handler._detect_consortium_contact_intent(
            "Should I contact the consortium if this gets harder?"
        )
        assert not should_contact and not is_confident

        edge_cases = [
            "tell the consortium to fuck themselves",
            "did you know you can call the consortium",
            "yeah totally the consortium would LOVE to hear that",
        ]
        for phrase in edge_cases:
            should_contact, is_confident = handler._detect_consortium_contact_intent(
                phrase
            )
            assert not should_contact
            assert is_confident
            assert not handler._should_use_consortium_mode(phrase)

        assert not handler._should_use_consortium_mode(
            "Should I contact the consortium if this gets harder?"
        )
        print("✓ Trigger detection works")
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


async def test_handle_routes_to_consortium_mode() -> None:
    """Ensure handle() routes consortium requests through consortium mode path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        handler = create_test_handler(db_path)
        handler.memory_store.search_memories = AsyncMock(return_value=[])

        consortium_ack = "Acknowledged. I will consult the consortium now."
        consortium_response = (
            "the consortium has reached an agreement.\nConsensus summary here."
        )
        interim_callback = AsyncMock()

        with patch.object(
            handler,
            "_generate_consortium_acknowledgement",
            new=AsyncMock(return_value=consortium_ack),
        ) as ack_mock, patch.object(
            handler,
            "_run_consortium_mode",
            new=AsyncMock(return_value=consortium_response),
        ) as consortium_mock, patch(
            "handler.aiohttp.ClientSession", return_value=DummyClientSession()
        ):
            response = await handler.handle(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Ask the consortium whether Casper is the best mattress.",
                        }
                    ]
                },
                session_id="consortium_session",
                interim_response_callback=interim_callback,
            )

        ack_mock.assert_awaited_once()
        consortium_mock.assert_awaited_once()
        interim_callback.assert_awaited_once_with(consortium_ack)
        assert response == consortium_response
        assert "the consortium has reached an agreement" in response

        history = handler.memory_store.get_conversation_history(
            session_id="consortium_session", limit=10
        )
        assert any(
            msg["role"] == "assistant" and consortium_ack in msg["content"]
            for msg in history
        )
        assert any(
            msg["role"] == "assistant" and consortium_response in msg["content"]
            for msg in history
        )
        print("✓ Consortium routing works")
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


async def test_consortium_task_tools() -> None:
    """Ensure consortium task management tools can start/stop/status tasks."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        handler = create_test_handler(db_path)

        with patch.object(
            handler,
            "_run_consortium_mode",
            new=AsyncMock(return_value="the consortium has reached an agreement."),
        ):
            start_result = await TOOLS["consortium_start"](
                task="Ask the consortium about mattress options",
                task_id="mattress_consensus",
            )
            assert start_result["success"]

            task_id = start_result["task"]["task_id"]
            final_status = None
            for _ in range(50):
                status_result = await TOOLS["consortium_status"](task_id=task_id)
                assert status_result["success"]
                final_status = status_result["task"]
                if final_status["status"] == "completed":
                    break
                await asyncio.sleep(0.01)

            assert final_status is not None
            assert final_status["status"] == "completed"
            assert "consortium has reached an agreement" in final_status["result"]

            all_status = await TOOLS["consortium_status"]()
            assert all_status["success"]
            assert all_status["count"] >= 1

            stop_result = await TOOLS["consortium_stop"](
                task_id=task_id,
                reason="No longer needed",
            )
            assert stop_result["success"]

        async def slow_consortium_mode(*args, **kwargs):
            await asyncio.sleep(0.3)
            return "slow run complete"

        with patch.object(
            handler,
            "_run_consortium_mode",
            new=AsyncMock(side_effect=slow_consortium_mode),
        ):
            start_result = await TOOLS["consortium_start"](
                task="Run a long consortium analysis",
                task_id="long_task",
            )
            assert start_result["success"]
            long_task_id = start_result["task"]["task_id"]

            stop_result = await TOOLS["consortium_stop"](
                task_id=long_task_id,
                reason="cancel requested",
            )
            assert stop_result["success"]

            stopped_status = None
            for _ in range(50):
                status_result = await TOOLS["consortium_status"](task_id=long_task_id)
                assert status_result["success"]
                stopped_status = status_result["task"]
                if stopped_status["status"] == "stopped":
                    break
                await asyncio.sleep(0.01)

            assert stopped_status is not None
            assert stopped_status["status"] in {"stopping", "stopped"}
        print("✓ Consortium task tools work")
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


async def main() -> int:
    """Run consortium mode tests."""
    await test_trigger_detection()
    await test_handle_routes_to_consortium_mode()
    await test_consortium_task_tools()
    print("All consortium mode tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
