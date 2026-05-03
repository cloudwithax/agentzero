#!/usr/bin/env python3
"""Run a larger live Sendblue-style sample and summarize recovery metrics."""

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from live_sendblue_smoke import run_smoke


DEFAULT_CASES: list[dict[str, Any]] = [
    {
        "name": "bash_read_baseline",
        "prompt": (
            "Live Sendblue batch case. You must use tools and actually execute them. "
            "First use the bash tool to run `printf sendblue-batch`. "
            "Then use the read tool to read the first line of AGENTS.md. "
            "Reply in exactly two lines:\n"
            "status: <bash output>\n"
            "first_line: <first line>"
        ),
        "expected_substrings": ["status:", "first_line:"],
        "required_tools": ["bash", "read"],
    },
    {
        "name": "grep_read_prompt_rules",
        "prompt": (
            "Tool execution is mandatory. Use grep on prompts/system_prompt.md to find "
            "pseudo-tool and workspace guidance, then use read on that file. "
            "Reply in exactly two lines:\n"
            "pseudo_tool_rule: yes|no\n"
            "workspace_rule: yes|no"
        ),
        "expected_substrings": ["pseudo_tool_rule:", "workspace_rule:"],
        "required_tools": ["grep", "read"],
    },
    {
        "name": "planning_heavy_repo_triage",
        "prompt": (
            "You must use tools and actually execute them. Figure out whether the repo "
            "already has logic in agentic_loop.py to detect invalid pseudo-tool syntax "
            "and hard-decision language. Do not read entire files. Use grep on "
            "agentic_loop.py first, and only use line-limited read() calls with limit <= 80 "
            "if grep results are ambiguous. Reply in exactly two lines:\n"
            "pseudo_tool_detection: yes|no\n"
            "hard_decision_detection: yes|no"
        ),
        "expected_substrings": [
            "pseudo_tool_detection:",
            "hard_decision_detection:",
        ],
        "required_tools": ["grep"],
    },
    {
        "name": "repo_workspace_paths",
        "prompt": (
            "Tool execution is mandatory. First use bash to print the current working "
            "directory. Then use grep to find the handler.py or tools.py references to "
            "MODEL_ID and PRIMARY_MODEL_ID. Use line-limited read() calls with limit <= 80 "
            "on the relevant sections. Reply in exactly three lines:\n"
            "cwd: <cwd>\n"
            "model_refs: <count>\n"
            "primary_model: <id>"
        ),
        "expected_substrings": ["cwd:", "model_refs:", "primary_model:"],
        "required_tools": ["bash", "grep", "read"],
    },
]


def _build_summary(results: list[dict[str, Any]], started_at: float) -> dict[str, Any]:
    total_cases = len(results)
    success_count = sum(1 for result in results if result.get("success"))
    recovery_opportunities = sum(
        1
        for result in results
        if result.get("agentic_metrics", {}).get("needed_recovery_before_tools")
    )
    recovered_cases = sum(
        1
        for result in results
        if result.get("agentic_metrics", {}).get("recovered_to_tool_execution")
    )
    pseudo_tool_opportunities = sum(
        1
        for result in results
        if result.get("agentic_metrics", {}).get(
            "pseudo_tool_rounds_before_first_tool", 0
        )
        > 0
    )
    pseudo_tool_recovered_cases = sum(
        1
        for result in results
        if result.get("agentic_metrics", {}).get(
            "pseudo_tool_rounds_before_first_tool", 0
        )
        > 0
        and result.get("agentic_metrics", {}).get("recovered_to_tool_execution")
    )

    def _rate(numerator: int, denominator: int) -> float | None:
        if denominator <= 0:
            return None
        return round((numerator / denominator) * 100.0, 1)

    return {
        "generated_at_epoch": round(time.time(), 3),
        "elapsed_seconds": round(time.time() - started_at, 2),
        "total_cases": total_cases,
        "success_count": success_count,
        "success_rate_percent": _rate(success_count, total_cases),
        "recovery_opportunities": recovery_opportunities,
        "recovered_cases": recovered_cases,
        "recovery_rate_percent": _rate(recovered_cases, recovery_opportunities),
        "pseudo_tool_opportunities": pseudo_tool_opportunities,
        "pseudo_tool_recovered_cases": pseudo_tool_recovered_cases,
        "pseudo_tool_recovery_rate_percent": _rate(
            pseudo_tool_recovered_cases, pseudo_tool_opportunities
        ),
    }


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a batch of live Sendblue-style smoke prompts."
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="case_names",
        help="Run only the named case(s). Defaults to all built-in cases.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write the JSON report.",
    )
    args = parser.parse_args()

    selected_cases = DEFAULT_CASES
    if args.case_names:
        allowed = set(args.case_names)
        selected_cases = [case for case in DEFAULT_CASES if case["name"] in allowed]
        missing = sorted(allowed - {case["name"] for case in selected_cases})
        if missing:
            raise SystemExit(f"Unknown case(s): {', '.join(missing)}")

    started_at = time.time()
    results: list[dict[str, Any]] = []

    for index, case in enumerate(selected_cases, start=1):
        phone_number = f"+1666{int(time.time() * 1000 + index) % 10000000:07d}"
        result = await run_smoke(
            prompt=case["prompt"],
            phone_number=phone_number,
            message_handle=f"batch-{case['name']}",
            expected_substrings=case["expected_substrings"],
            required_tools=case["required_tools"],
        )
        result["case_name"] = case["name"]
        results.append(result)

    report = {
        "summary": _build_summary(results, started_at),
        "cases": results,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    return 0 if all(result.get("success") for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
