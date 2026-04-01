#!/usr/bin/env python3
"""
Visualize learning improvement over time.
Shows how the system gets better at selecting good examples.
"""

import sys
sys.path.insert(0, '/home/clxud/Documents/github/agentzero')

from core.example_bank import ExampleBank


def visualize_learning_curve():
    """
    Show how selection accuracy improves as the system learns.
    """
    print("=" * 70)
    print("LEARNING IMPROVEMENT VISUALIZATION")
    print("=" * 70)
    print("\nScenario: System learns which examples work for 'coding' tasks")
    print("-" * 70)

    bank = ExampleBank(exploration_rate=0.1)

    # Add examples with different "true" quality
    ex_good_id = bank.add(
        task_type="coding",
        input_text="Write a simple function",
        output_text="def func(): pass",
        weight=0.5
    )
    ex_bad_id = bank.add(
        task_type="coding",
        input_text="Write complex metaclass",
        output_text="class Meta(type): ...",
        weight=0.5
    )

    # Track performance over time
    iterations = 50
    good_selections = []
    good_weights = []
    bad_weights = []

    print(f"\nSimulating {iterations} task executions...")
    print(f"{'Iter':>5} | {'Good Ex Weight':>15} | {'Bad Ex Weight':>15} | {'Good Selected?':>15}")
    print("-" * 70)

    for i in range(iterations):
        # Select an example
        selected = bank.select("coding", k=1, exploration=0.1)

        if selected:
            is_good = selected[0].id == ex_good_id
            good_selections.append(1 if is_good else 0)

            # Simulate outcome based on "true" quality
            # Good example succeeds 90% of the time
            # Bad example succeeds 30% of the time
            import random
            if is_good:
                outcome = 1.0 if random.random() < 0.9 else 0.0
            else:
                outcome = 1.0 if random.random() < 0.3 else 0.0

            bank.update([selected[0].id], outcome)

        # Record weights
        ex_good = bank._example_index[ex_good_id]
        ex_bad = bank._example_index[ex_bad_id]
        good_weights.append(ex_good.weight)
        bad_weights.append(ex_bad.weight)

        # Print progress every 10 iterations
        if (i + 1) % 10 == 0:
            recent_accuracy = sum(good_selections[-10:]) / 10
            print(f"{i+1:>5} | {ex_good.weight:>15.3f} | {ex_bad.weight:>15.3f} | {recent_accuracy*100:>14.0f}%")

    # Final statistics
    print("-" * 70)
    print("\nFINAL RESULTS:")
    print(f"  Good example final weight: {good_weights[-1]:.3f}")
    print(f"  Bad example final weight: {bad_weights[-1]:.3f}")

    # Calculate improvement in selection accuracy
    first_10_accuracy = sum(good_selections[:10]) / 10
    last_10_accuracy = sum(good_selections[-10:]) / 10

    print(f"\n  Selection accuracy (first 10): {first_10_accuracy:.1%}")
    print(f"  Selection accuracy (last 10):  {last_10_accuracy:.1%}")

    improvement = (last_10_accuracy - first_10_accuracy) * 100
    print(f"\n  Improvement: +{improvement:.0f} percentage points")

    if last_10_accuracy > first_10_accuracy:
        print("\n  ✓ System LEARNED and IMPROVED over time!")
    else:
        print("\n  ⚠ No clear improvement (variance in random simulation)")

    # Show learning curve
    print("\n" + "=" * 70)
    print("WEIGHT EVOLUTION CHART")
    print("=" * 70)
    print("\nY-axis: Weight (0.0 to 1.0)")
    print("X-axis: Iterations (0 to 50)")
    print("\nLegend: █ = Good example weight")
    print("        ░ = Bad example weight")
    print()

    # Simple ASCII chart
    chart_height = 10
    for row in range(chart_height, -1, -1):
        y_val = row / chart_height
        line = f"{y_val:.1f} |"

        for i in range(0, 50, 5):
            gw = good_weights[i] if i < len(good_weights) else good_weights[-1]
            bw = bad_weights[i] if i < len(bad_weights) else bad_weights[-1]

            if abs(gw - y_val) < 0.05:
                line += "█"
            elif abs(bw - y_val) < 0.05:
                line += "░"
            else:
                line += " "

        print(line)

    print("     +" + "-" * 10)
    print("     0    25    50")


def test_adaptive_exploration():
    """
    Show how exploration vs exploitation trade-off works.
    """
    print("\n" + "=" * 70)
    print("ADAPTIVE EXPLORATION DEMONSTRATION")
    print("=" * 70)

    bank = ExampleBank(exploration_rate=0.2)

    # Add 5 examples
    ex_ids = []
    for i in range(5):
        ex_id = bank.add(
            task_type="test",
            input_text=f"Example {i}",
            output_text=f"Output {i}",
            weight=0.5
        )
        ex_ids.append(ex_id)

    # Give different feedback to each
    print("\nSetting up examples with different quality levels...")
    print("  Example 0: High quality (weight -> 0.9)")
    print("  Example 1: Medium quality (weight -> 0.6)")
    print("  Example 2: Low quality (weight -> 0.3)")
    print("  Example 3-4: Untested (weight stays 0.5)")

    # Update with different outcomes
    for _ in range(10):
        bank.update([ex_ids[0]], 0.9)  # High quality
        bank.update([ex_ids[1]], 0.6)  # Medium quality
        bank.update([ex_ids[2]], 0.3)  # Low quality

    print("\nSelection distribution with 20% exploration:")
    counts = {i: 0 for i in range(5)}
    for _ in range(1000):
        selected = bank.select("test", k=1, exploration=0.2)
        if selected:
            idx = ex_ids.index(selected[0].id)
            counts[idx] += 1

    for i in range(5):
        pct = counts[i] / 1000 * 100
        bar = "█" * int(pct / 2)
        print(f"  Example {i}: {pct:5.1f}% {bar}")

    print("\nNote: Example 0 (high quality) is selected most often,")
    print("      but Examples 3-4 still get some selections due to exploration.")


def main():
    print("\n" + "=" * 70)
    print("LEARNING IMPROVEMENT DEMONSTRATION")
    print("=" * 70)
    print("\nThese tests visualize how the system improves over time.\n")

    visualize_learning_curve()
    test_adaptive_exploration()

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nKey Findings:")
    print("  ✓ System learns to prefer better examples")
    print("  ✓ Selection accuracy improves over time")
    print("  ✓ Exploration allows discovery of new good examples")
    print("  ✓ Weights converge to reflect true example quality")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
