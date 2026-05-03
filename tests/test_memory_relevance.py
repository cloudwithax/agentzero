"""Test memory relevance filtering in conversation context."""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from handler import AgentHandler
from memory import Memory, MemoryStore
from examples import AdaptiveFewShotManager
from planning import TaskPlanner, TaskAnalyzer
from capabilities import CapabilityProfile, AdaptiveFormatter
import json


async def test_memory_relevance_filtering():
    """Test that memories are filtered for relevance to the current query."""
    print("Testing memory relevance filtering...")

    memory_store = MemoryStore()
    capability_profile = CapabilityProfile()
    example_bank = AdaptiveFewShotManager()
    task_planner = TaskPlanner(profile=capability_profile)
    task_analyzer = TaskAnalyzer()
    adaptive_formatter = AdaptiveFormatter(profile=capability_profile)

    handler = AgentHandler(
        memory_store=memory_store,
        capability_profile=capability_profile,
        example_bank=example_bank,
        task_planner=task_planner,
        task_analyzer=task_analyzer,
        adaptive_formatter=adaptive_formatter,
    )

    test_cases = [
        {
            "name": "Relevant memory about user preferences",
            "memory_content": "User prefers coffee over tea in the morning",
            "query": "What does the user like to drink in the morning?",
            "expected_relevance": True,
        },
        {
            "name": "Irrelevant memory about unrelated topic",
            "memory_content": "User's favorite color is blue",
            "query": "What project are we working on together?",
            "expected_relevance": False,
        },
        {
            "name": "Partially relevant memory",
            "memory_content": "User is working on a Python project called 'agentzero'",
            "query": "What are the technical requirements for the project?",
            "expected_relevance": True,
        },
        {
            "name": "Memory with assistant identity",
            "memory_content": "Your name is Assistant",
            "query": "What is my name?",
            "expected_relevance": True,
        },
        {
            "name": "Assistant identity when not asked",
            "memory_content": "Your name is Assistant",
            "query": "What's the weather today?",
            "expected_relevance": False,
        },
    ]

    for case in test_cases:
        print(f"\nTest case: {case['name']}")
        memory = Memory(
            id=1,
            content=case["memory_content"],
            embedding=None,
            metadata={"type": "explicit_memory", "session_id": "test_session"},
        )

        score = handler._score_memory_relevance_to_query(memory, case["query"])

        if case["expected_relevance"]:
            assert score > 0.0, f"Expected relevance score > 0.0, got {score}"
            print(f"✓ Memory was correctly marked as relevant (score: {score:.3f})")
        else:
            assert score == 0.0, f"Expected relevance score == 0.0, got {score}"
            print(f"✓ Memory was correctly filtered out (score: {score:.3f})")

    print("\n✓ All relevance filtering tests passed!")


async def test_session_continuity_with_relevance():
    """Test that session continuity context respects relevance filtering."""
    print("\n\nTesting session continuity context with relevance filtering...")

    memory_store = MemoryStore()
    capability_profile = CapabilityProfile()
    example_bank = AdaptiveFewShotManager()
    task_planner = TaskPlanner(profile=capability_profile)
    task_analyzer = TaskAnalyzer()
    adaptive_formatter = AdaptiveFormatter(profile=capability_profile)

    handler = AgentHandler(
        memory_store=memory_store,
        capability_profile=capability_profile,
        example_bank=example_bank,
        task_planner=task_planner,
        task_analyzer=task_analyzer,
        adaptive_formatter=adaptive_formatter,
    )

    session_id = "test_session"

    # Add some test memories
    memories = [
        Memory(
            id=1,
            content="Your name is Assistant",
            embedding=None,
            metadata={"type": "explicit_memory", "session_id": session_id},
        ),
        Memory(
            id=2,
            content="User prefers coffee over tea in the morning",
            embedding=None,
            metadata={"type": "explicit_memory", "session_id": session_id},
        ),
        Memory(
            id=3,
            content="User's favorite color is blue",
            embedding=None,
            metadata={"type": "explicit_memory", "session_id": session_id},
        ),
        Memory(
            id=4,
            content="User is working on a Python project called 'agentzero'",
            embedding=None,
            metadata={"type": "explicit_memory", "session_id": session_id},
        ),
    ]

    for memory in memories:
        await handler.memory_store.add_memory(memory.content, metadata=memory.metadata)

    # Test with relevant query
    query = "What does the user like to drink in the morning?"
    context = handler._build_session_continuity_context(
        session_id=session_id,
        user_query=query,
    )

    print(f"\nQuery: {query}")
    print(f"Context:\n{context}")

    assert "coffee" in context.lower(), "Expected 'coffee' in context for relevant query"
    assert "blue" not in context.lower(), "Expected 'blue' to be filtered out for irrelevant query"
    assert "agentzero" not in context.lower(), "Expected 'agentzero' to be filtered out for irrelevant query"
    print("✓ Session continuity context correctly filtered for relevance")

    # Test with assistant identity query
    query = "What is my name?"
    context = handler._build_session_continuity_context(
        session_id=session_id,
        user_query=query,
    )

    print(f"\nQuery: {query}")
    print(f"Context:\n{context}")

    assert "assistant" in context.lower(), "Expected assistant identity in context"
    print("✓ Assistant identity correctly surfaced for relevant query")

    # Test with completely irrelevant query
    query = "What's the weather today?"
    context = handler._build_session_continuity_context(
        session_id=session_id,
        user_query=query,
    )

    print(f"\nQuery: {query}")
    print(f"Context:\n{context}")

    # With relevance filtering, only relevant memories are included
    assert "coffee" in context.lower(), "Expected coffee memory to be included for relevant query"
    assert "blue" not in context.lower(), "Expected blue memory to be filtered out for irrelevant query"
    assert "assistant" not in context.lower(), "Expected assistant identity to be filtered out for irrelevant query"
    print("✓ All memories correctly filtered for irrelevant query")

    print("\n✓ All session continuity tests passed!")


async def main():
    """Run all tests."""
    await test_memory_relevance_filtering()
    await test_session_continuity_with_relevance()
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())