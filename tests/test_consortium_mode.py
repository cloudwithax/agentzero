#!/usr/bin/env python3
"""Live-API and deterministic tests for consortium mode."""

import asyncio
import os
import tempfile

from capabilities import Capability, CapabilityProfile, AdaptiveFormatter
from examples import ExampleBank, AdaptiveFewShotManager
from handler import AgentHandler
from memory import EnhancedMemoryStore
from planning import TaskAnalyzer, TaskPlanner
from tests._live_harness import (
    LIVE,
    live_agent_handle,
    skip_if_not_live,
    _make_handler,
    _make_store,
)


def create_test_handler(db_path: str) -> AgentHandler:
    memory_store = EnhancedMemoryStore(db_path=db_path, api_key=os.environ.get("NVIDIA_API_KEY", "test-key"))
    profile = CapabilityProfile(
        capabilities={
            Capability.JSON_OUTPUT, Capability.TOOL_USE, Capability.CHAIN_OF_THOUGHT,
            Capability.REASONING, Capability.LONG_CONTEXT, Capability.FEW_SHOT,
            Capability.SELF_CORRECTION, Capability.STRUCTURED_OUTPUT,
        },
        model_name="test-model",
    )
    return AgentHandler(
        memory_store=memory_store, capability_profile=profile,
        example_bank=AdaptiveFewShotManager(ExampleBank(exploration_rate=0.1)),
        task_planner=TaskPlanner(profile), task_analyzer=TaskAnalyzer(),
        adaptive_formatter=AdaptiveFormatter(profile),
    )


async def test_trigger_detection() -> None:
    """Consortium routing uses intent detection with confidence gating."""
    print("\nTest 1: Consortium trigger detection")
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    try:
        handler = create_test_handler(db_path)
        should_contact, is_confident = handler._detect_consortium_contact_intent(
            "Ask the consortium if this is a good idea"
        )
        assert should_contact and is_confident
        should_contact, is_confident = handler._detect_consortium_contact_intent(
            "hello how are you"
        )
        assert not (should_contact and is_confident)
    finally:
        os.unlink(db_path)
    print("  PASS")


async def test_live_agent_handles_normal_query() -> None:
    """Agent should handle a normal query without consortium routing."""
    if not LIVE:
        print("\nTest L1: Live agent query [SKIP]")
        return
    print("\nTest L1: Live agent handles normal query")
    skip_if_not_live()
    store = _make_store()
    handler = _make_handler(store)
    response = await live_agent_handle(
        handler,
        user_text="Just say 'hello consortium test' and nothing else. Stop after that.",
        session_id="test_consortium",
    )
    assert len(response) > 3, f"Empty response: {response[:200]}"
    print(f"  PASS — response: {response[:100]}")


async def main() -> None:
    print("=" * 60)
    print("Consortium mode tests")
    print("=" * 60)
    await test_trigger_detection()
    await test_live_agent_handles_normal_query()
    print("\n" + "=" * 60)
    print("Consortium tests complete")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
