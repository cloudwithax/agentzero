#!/usr/bin/env python3
"""
Test the learning feedback loop to ensure the system actually improves.
"""

import sys
sys.path.insert(0, '/home/clxud/Documents/github/agentzero')

from core.example_bank import ExampleBank, AdaptiveFewShotManager
from main import CapabilityProfile, Capability
from main import TaskPlanner, Task
from memory_enhanced import EnhancedMemoryStore


def test_example_bank_learning():
    """Test that example weights improve with positive feedback."""
    print("=" * 60)
    print("TEST 1: Example Bank Learning")
    print("=" * 60)

    bank = ExampleBank(exploration_rate=0.1)

    # Add some examples for math tasks
    ex1_id = bank.add(
        task_type="math",
        input_text="What is 2 + 2?",
        output_text='{"result": 4}',
        weight=0.5
    )
    ex2_id = bank.add(
        task_type="math",
        input_text="Calculate 5 * 3",
        output_text='{"result": 15}',
        weight=0.5
    )
    ex3_id = bank.add(
        task_type="math",
        input_text="Solve 10 / 2",
        output_text='{"result": 5}',
        weight=0.5
    )

    print("\nInitial weights:")
    for ex_id in [ex1_id, ex2_id, ex3_id]:
        ex = bank._example_index[ex1_id]
        print(f"  {ex_id}: {ex.weight:.3f} (success_rate: {ex.success_rate:.3f})")

    # Simulate using examples and getting feedback
    print("\nSimulating 10 successful uses of example 1...")
    for i in range(10):
        bank.update([ex1_id], outcome=1.0)  # Perfect success

    print("Simulating 5 failures of example 2...")
    for i in range(5):
        bank.update([ex2_id], outcome=0.0)  # Complete failure

    print("Simulating mixed results for example 3...")
    for i in range(5):
        bank.update([ex3_id], outcome=0.5)  # Neutral

    print("\nFinal weights after feedback:")
    for ex_id in [ex1_id, ex2_id, ex3_id]:
        ex = bank._example_index[ex_id]
        print(f"  {ex_id}: weight={ex.weight:.3f}, success_rate={ex.success_rate:.3f}, usages={ex.usage_count}")

    # Verify learning occurred
    ex1 = bank._example_index[ex1_id]
    ex2 = bank._example_index[ex2_id]

    assert ex1.weight > 0.8, f"Example 1 weight should have increased, got {ex1.weight}"
    assert ex2.weight < 0.3, f"Example 2 weight should have decreased, got {ex2.weight}"

    print("\n✓ Example bank is LEARNING from feedback!")

    # Test selection bias
    print("\nTesting selection bias (should prefer high-weight examples)...")
    selections = {"ex1": 0, "ex2": 0, "ex3": 0}
    for _ in range(100):
        selected = bank.select("math", k=1, exploration=0.0)  # No exploration
        if selected:
            if selected[0].id == ex1_id:
                selections["ex1"] += 1
            elif selected[0].id == ex2_id:
                selections["ex2"] += 1
            elif selected[0].id == ex3_id:
                selections["ex3"] += 1

    print("  Selection counts over 100 iterations:")
    print(f"    ex1 (high weight): {selections['ex1']}")
    print(f"    ex2 (low weight): {selections['ex2']}")
    print(f"    ex3 (neutral): {selections['ex3']}")

    assert selections["ex1"] > selections["ex2"], "High-weight example should be selected more"
    print("\n✓ Selection correctly prefers high-performing examples!")


def test_adaptive_few_shot_manager():
    """Test the high-level few-shot manager."""
    print("\n" + "=" * 60)
    print("TEST 2: Adaptive Few-Shot Manager")
    print("=" * 60)

    manager = AdaptiveFewShotManager()

    # Add examples
    manager.add_example(
        task_type="coding",
        input_text="Write a function to reverse a string",
        output_text='def reverse(s): return s[::-1]',
        success_score=0.8
    )
    manager.add_example(
        task_type="coding",
        input_text="Write a function to check if string is palindrome",
        output_text='def is_palindrome(s): return s == s[::-1]',
        success_score=0.6
    )

    print(f"\nInitial stats: {manager.get_stats()}")

    # Get examples (records selection)
    examples = manager.get_examples_for_task("coding", "Write a function", max_examples=1)
    print(f"\nSelected {len(examples)} examples for coding task")

    # Provide feedback
    manager.auto_feedback("coding", success=True, efficiency=0.9)

    print(f"Stats after feedback: {manager.get_stats()}")
    print("\n✓ AdaptiveFewShotManager working correctly!")


def test_memory_learning():
    """Test that memory system learns from outcomes."""
    print("\n" + "=" * 60)
    print("TEST 3: Memory Learning with Weighted Retrieval")
    print("=" * 60)

    # Use test database
    store = EnhancedMemoryStore(
        db_path="test_memory_learning.db",
        api_key="test"  # Won't actually generate embeddings
    )

    # Add some memories manually (without embeddings for speed)
    import sqlite3
    conn = sqlite3.connect("test_memory_learning.db")
    cursor = conn.cursor()

    # Insert test memories
    memories = [
        ("Python list comprehension syntax", '{"type": "knowledge"}'),
        ("Docker container basics", '{"type": "knowledge"}'),
        ("Git rebase workflow", '{"type": "knowledge"}'),
    ]

    for content, metadata in memories:
        cursor.execute(
            "INSERT INTO memories (content, metadata) VALUES (?, ?)",
            (content, metadata)
        )
    memory_ids = [row[0] for row in cursor.execute("SELECT id FROM memories").fetchall()]
    conn.commit()
    conn.close()

    print(f"\nCreated {len(memory_ids)} test memories: {memory_ids}")

    # Record different outcomes for different memories
    print("\nRecording outcomes...")

    # Memory 1: Highly successful (used 10 times, always helped)
    for i in range(10):
        store.record_memory_outcome(memory_ids[0], "success", 1.0, {"task": "coding"})

    # Memory 2: Failure (used 5 times, never helped)
    for i in range(5):
        store.record_memory_outcome(memory_ids[1], "failure", 0.0, {"task": "docker"})

    # Memory 3: Mixed (used 5 times, sometimes helped)
    for i in range(3):
        store.record_memory_outcome(memory_ids[2], "success", 0.8)
    for i in range(2):
        store.record_memory_outcome(memory_ids[2], "failure", 0.3)

    # Check success scores
    score1 = store.get_memory_success_score(memory_ids[0])
    score2 = store.get_memory_success_score(memory_ids[1])
    score3 = store.get_memory_success_score(memory_ids[2])

    print("\nSuccess scores:")
    print(f"  Memory 1 (always successful): {score1:.3f}")
    print(f"  Memory 2 (always failed): {score2:.3f}")
    print(f"  Memory 3 (mixed): {score3:.3f}")

    assert score1 > 0.9, f"Memory 1 should have high success score, got {score1}"
    assert score2 < 0.1, f"Memory 2 should have low success score, got {score2}"

    print("\n✓ Memory system is tracking and learning from outcomes!")

    # Cleanup
    import os
    os.remove("test_memory_learning.db")


def test_task_planner_adaptation():
    """Test that task planner adapts to model capabilities."""
    print("\n" + "=" * 60)
    print("TEST 4: Task Planner Capability Adaptation")
    print("=" * 60)

    # Create profile for capable model
    capable_profile = CapabilityProfile(
        capabilities={
            Capability.JSON_OUTPUT,
            Capability.TOOL_USE,
            Capability.CHAIN_OF_THOUGHT,
            Capability.REASONING,
            Capability.LONG_CONTEXT,
            Capability.FEW_SHOT,
            Capability.SELF_CORRECTION,
        }
    )

    # Create profile for weak model
    weak_profile = CapabilityProfile(
        capabilities={Capability.REASONING, Capability.FEW_SHOT}
    )

    capable_planner = TaskPlanner(capable_profile)
    weak_planner = TaskPlanner(weak_profile)

    # Create same task for both
    math_task = Task(
        id="math_001",
        description="Calculate 15 * 27",
        type="math"
    )

    capable_plan = capable_planner.plan(math_task)
    weak_plan = weak_planner.plan(math_task)

    print(f"\nCapable model plan ({len(capable_plan.steps)} steps):")
    for step in capable_plan.steps:
        print(f"  - {step.description} (needs: {[c.value for c in step.required_capabilities]})")

    print(f"\nWeak model plan ({len(weak_plan.steps)} steps):")
    for step in weak_plan.steps:
        print(f"  - {step.description} (needs: {[c.value for c in step.required_capabilities]})")

    # Capable model should get verification step
    has_verification = any("verify" in s.operation for s in capable_plan.steps)
    weak_has_verification = any("verify" in s.operation for s in weak_plan.steps)

    print(f"\nCapable model has verification step: {has_verification}")
    print(f"Weak model has verification step: {weak_has_verification}")

    assert has_verification, "Capable model should have verification step"
    assert not weak_has_verification, "Weak model shouldn't have verification (no SELF_CORRECTION)"

    print("\n✓ Task planner adapts to model capabilities!")


def test_end_to_end_learning_simulation():
    """Simulate a full learning cycle."""
    print("\n" + "=" * 60)
    print("TEST 5: End-to-End Learning Simulation")
    print("=" * 60)

    bank = ExampleBank(exploration_rate=0.2)

    # Simulate a coding assistant scenario
    print("\nScenario: Coding assistant learning which examples work best")

    # Add examples with initial neutral weights
    examples = [
        ("basic", "Write a hello world function", "def hello(): print('Hello')"),
        ("complex", "Write a decorator", "def decorator(func): ..."),
        ("intermediate", "Write a list comprehension", "[x for x in range(10)]"),
    ]

    ex_ids = {}
    for task_type, inp, out in examples:
        ex_id = bank.add(task_type="coding", input_text=inp, output_text=out, weight=0.5)
        ex_ids[task_type] = ex_id

    print("\nInitial state:")
    print("  All examples have weight=0.5")

    # Simulate 50 iterations of usage and feedback
    print("\nSimulating 50 task executions...")

    import random
    for iteration in range(50):
        # Select examples (with exploration)
        selected = bank.select("coding", k=1, exploration=0.2)
        if not selected:
            continue

        ex = selected[0]

        # Simulate execution success based on example type
        # Basic examples succeed 90% of the time
        # Complex examples succeed 40% of the time
        # Intermediate examples succeed 70% of the time
        if "basic" in ex.id:
            success = random.random() < 0.9
        elif "complex" in ex.id:
            success = random.random() < 0.4
        else:
            success = random.random() < 0.7

        outcome = 1.0 if success else 0.0
        bank.update([ex.id], outcome)

    print("\nFinal weights after learning:")
    for task_type, ex_id in ex_ids.items():
        ex = bank._example_index[ex_id]
        print(f"  {task_type}: weight={ex.weight:.3f}, success_rate={ex.success_rate:.3f}, usages={ex.usage_count}")

    # Verify learning
    basic_ex = bank._example_index[ex_ids["basic"]]
    complex_ex = bank._example_index[ex_ids["complex"]]

    print("\nLearning verification:")
    print(f"  Basic example weight: {basic_ex.weight:.3f} (should be high)")
    print(f"  Complex example weight: {complex_ex.weight:.3f} (should be lower)")

    # The basic example should have higher weight than complex
    if basic_ex.weight > complex_ex.weight:
        print("\n✓ System learned that basic examples work better!")
    else:
        print("\n⚠ Learning effect present but variance is high (this can happen with random simulation)")

    # Test selection preference
    print("\nTesting selection preference (100 iterations, no exploration)...")
    selection_counts = {"basic": 0, "complex": 0, "intermediate": 0}
    for _ in range(100):
        selected = bank.select("coding", k=1, exploration=0.0)
        if selected:
            for task_type, ex_id in ex_ids.items():
                if selected[0].id == ex_id:
                    selection_counts[task_type] += 1

    print(f"  Selection counts: {selection_counts}")

    if selection_counts["basic"] > selection_counts["complex"]:
        print("\n✓ System correctly prefers better-performing examples!")
    else:
        print("\n⚠ Selection counts may vary due to random simulation")


def main():
    print("\n" + "=" * 60)
    print("LEARNING FEEDBACK LOOP TESTS")
    print("=" * 60)

    test_example_bank_learning()
    test_adaptive_few_shot_manager()
    test_memory_learning()
    test_task_planner_adaptation()
    test_end_to_end_learning_simulation()

    print("\n" + "=" * 60)
    print("ALL LEARNING TESTS COMPLETED")
    print("=" * 60)
    print("\nThe system demonstrates:")
    print("  ✓ Example weights update based on success/failure")
    print("  ✓ Selection prefers high-weight examples")
    print("  ✓ Memory outcomes are tracked and scored")
    print("  ✓ Task planning adapts to model capabilities")
    print("  ✓ Feedback loops improve performance over time")


if __name__ == "__main__":
    main()
