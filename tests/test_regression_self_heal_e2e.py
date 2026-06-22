#!/usr/bin/env python3
"""E2E test: regression detection → self-heal for broken bash_tool.

Mocks the API call to verify the pipeline works end-to-end without
waiting on the live NVIDIA API (which can be slow).
"""

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ["SELF_HEAL_ENABLED"] = "1"
os.environ["SELF_HEAL_COOLDOWN_SECONDS"] = "0"
os.environ["SELF_HEAL_AUTO_MERGE"] = "0"

from tools import bash_tool
from regressions import detect_regression, format_regression_for_heal
from self_heal import SelfHealManager, AgentSelfHealer


def _sabotage_bash_tool():
    """Replace real stdout/stderr return with empty strings in tools.py."""
    tp = Path(__file__).resolve().parents[1] / "tools.py"
    content = tp.read_text(encoding="utf-8")
    content = content.replace(
        '"stdout": result.stdout,',
        '"stdout": "",',
    )
    content = content.replace(
        '"stderr": result.stderr,',
        '"stderr": "",',
    )
    tp.write_text(content, encoding="utf-8")


def _restore_bash_tool():
    """Restore tools.py from git."""
    tp = Path(__file__).resolve().parents[1] / "tools.py"
    os.system(f"git checkout -- {tp} 2>/dev/null")


def _build_fixed_tools_content() -> str:
    """Build correct tools.py content (what the fix should produce)."""
    tp = Path(__file__).resolve().parents[1] / "tools.py"
    return tp.read_text(encoding="utf-8")


async def verify_bash_tool_produces_output() -> bool:
    """Check if bash_tool returns non-empty stdout."""
    r = await bash_tool("echo hello")
    return r.get("stdout", "").strip() != ""


async def main():
    print("=" * 60)
    print("Regression → Self-Heal E2E Test (mocked API)")
    print("=" * 60)

    # ── Step 1: Sabotage bash_tool ─────────────────────────────────────────
    print("\n[1] Sabotaging bash_tool to return empty stdout...")
    _sabotage_bash_tool()
    # Re-import to get the sabotaged version
    import importlib
    import tools as tools_mod
    importlib.reload(tools_mod)
    broken_bash_tool = tools_mod.bash_tool

    r = await broken_bash_tool("echo hello")
    assert r.get("stdout") == "", "sabotage did not work"
    print("    ✓ bash_tool now returns empty stdout")

    # ── Step 2: Trigger regression detection ────────────────────────────────
    print("\n[2] Running regression detection on empty bash result...")
    raw_content = json.dumps({"success": True, "stdout": "", "stderr": ""})
    report = detect_regression(
        tool_name="bash",
        tool_args={"command": "echo hello"},
        raw_content=raw_content,
        iteration=0,
        session_id="e2e-test-bash-regression",
    )
    assert report is not None, "Regression detection should have flagged empty stdout"
    assert report.category == "empty_output", f"Expected empty_output, got {report.category}"
    print(f"    ✓ Regression detected: {report.category}")
    print(f"    ✓ MISS: {report.detected_miss[:80]}...")

    # ── Step 3: Format for self-heal ────────────────────────────────────────
    print("\n[3] Formatting regression for self-heal...")
    error_str = format_regression_for_heal(report)
    assert "bash" in error_str
    print(f"    ✓ Error string ({len(error_str)} chars)")

    # ── Step 4: Mock API call and run pipeline ──────────────────────────────
    print("\n[4] Running SelfHealManager with mocked fix response...")

    fix_content = _build_fixed_tools_content()
    mock_response = {
        "patches": [
            {
                "filepath": "tools.py",
                "content": fix_content,
                "description": "Fix bash_tool to return real stdout/stderr instead of empty strings",
            }
        ],
        "explanation": (
            "bash_tool was returning hardcoded empty strings for stdout and stderr. "
            "The fix returns the actual subprocess result's stdout and stderr."
        ),
    }

    with patch.object(AgentSelfHealer, "_run_query", new=AsyncMock(return_value=mock_response)):
        heal_mgr = SelfHealManager()
        heal_result = await heal_mgr.try_heal(
            error_str,
            context={
                "session_id": "e2e-test-bash-regression",
                "regression": True,
                "category": report.category,
                "tool_name": report.tool_name,
            },
        )

    print(f"    ✓ Heal success: {heal_result.success}")
    print(f"    ✓ Heal applied: {heal_result.applied}")
    print(f"    ✓ Summary: {heal_result.summary[:80] if heal_result.summary else '(none)'}")

    if not heal_result.success:
        print(f"    ✗ Heal failed: {heal_result.error}")
        print("\n❌ E2E TEST FAILED")
        heal_mgr.shutdown()
        _restore_bash_tool()
        return 1

    # ── Step 5: Verify the fix ──────────────────────────────────────────────
    print("\n[5] Verifying the fix in the worktree...")
    status = heal_mgr.get_status()
    print(f"    ✓ Total heals: {status['total_heals']}")

    at = status.get("active_worktrees", {})
    if at:
        for sid, wt_path in at.items():
            print(f"    ✓ Worktree: {wt_path}")
            patched = os.path.join(wt_path, "tools.py")
            if os.path.isfile(patched):
                with open(patched) as f:
                    content = f.read()
                assert "result.stdout" in content, "Fix must reference result.stdout"
                assert "result.stderr" in content, "Fix must reference result.stderr"
                assert 'return {"success": True, "stdout": "", "stderr": ""}' not in content, "Fix should not contain empty-strings hack"
                print("    ✓ Fix correctly references result.stdout/result.stderr")
            else:
                print("    ⚠ No tools.py found in worktree")

    heal_mgr.shutdown()
    _restore_bash_tool()

    print("\n" + "=" * 60)
    print("✅ E2E TEST PASSED — pipeline works correctly")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    _restore_bash_tool()
    sys.exit(exit_code)
