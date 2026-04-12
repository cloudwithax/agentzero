#!/usr/bin/env python3
"""Tests for self_heal module."""

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestErrorClassifier(unittest.TestCase):
    def setUp(self):
        from self_heal import ErrorClassifier

        self.classifier = ErrorClassifier()

    def test_classify_rate_limit_as_transient(self):
        result = self.classifier.classify("Error: rate limit exceeded")
        from self_heal import ErrorCategory

        self.assertEqual(result, ErrorCategory.TRANSIENT)

    def test_classify_timeout_as_transient(self):
        result = self.classifier.classify("Connection timed out")
        from self_heal import ErrorCategory

        self.assertEqual(result, ErrorCategory.TRANSIENT)

    def test_classify_network_error_as_transient(self):
        result = self.classifier.classify("network unreachable error")
        from self_heal import ErrorCategory

        self.assertEqual(result, ErrorCategory.TRANSIENT)

    def test_classify_429_as_transient(self):
        result = self.classifier.classify("HTTP 429 too many requests")
        from self_heal import ErrorCategory

        self.assertEqual(result, ErrorCategory.TRANSIENT)

    def test_classify_attributeerror_as_code(self):
        result = self.classifier.classify(
            "object has no attribute",
            exception_type="AttributeError",
        )
        from self_heal import ErrorCategory

        self.assertEqual(result, ErrorCategory.CODE)

    def test_classify_importerror_as_code(self):
        result = self.classifier.classify(
            "No module named 'foo'",
            exception_type="ImportError",
        )
        from self_heal import ErrorCategory

        self.assertEqual(result, ErrorCategory.CODE)

    def test_classify_keyerror_as_code(self):
        result = self.classifier.classify(
            "KeyError: 'missing_key'",
            exception_type="KeyError",
        )
        from self_heal import ErrorCategory

        self.assertEqual(result, ErrorCategory.CODE)

    def test_classify_tool_execution_error_as_tool(self):
        result = self.classifier.classify("Tool execution error for bash")
        from self_heal import ErrorCategory

        self.assertEqual(result, ErrorCategory.TOOL)

    def test_classify_unknown_without_exception_type(self):
        result = self.classifier.classify("Something went wrong")
        from self_heal import ErrorCategory

        self.assertEqual(result, ErrorCategory.UNKNOWN)

    def test_classify_traceback_signal_as_code(self):
        result = self.classifier.classify(
            "Traceback (most recent call last):\n  File ...",
        )
        from self_heal import ErrorCategory

        self.assertEqual(result, ErrorCategory.CODE)


class TestParseErrorEnrichments(unittest.TestCase):
    def test_extract_iteration(self):
        from self_heal import _parse_error_enrichments

        result = _parse_error_enrichments("Error: API failed [iteration=3]")
        self.assertEqual(result.get("iteration"), 3)

    def test_extract_tools_executed(self):
        from self_heal import _parse_error_enrichments

        result = _parse_error_enrichments("Error: failed [tools_executed=[bash, grep]]")
        self.assertEqual(result.get("tools_executed"), ["bash", "grep"])

    def test_empty_string(self):
        from self_heal import _parse_error_enrichments

        result = _parse_error_enrichments("")
        self.assertEqual(result, {})


class TestExtractExceptionFromError(unittest.TestCase):
    def test_extract_exception_type(self):
        from self_heal import _extract_exception_from_error

        result = _extract_exception_from_error("Exception type: AttributeError")
        self.assertEqual(result, "AttributeError")

    def test_extract_from_colon_format(self):
        from self_heal import _extract_exception_from_error

        result = _extract_exception_from_error("ValueError: invalid value")
        self.assertEqual(result, "ValueError")

    def test_no_exception_found(self):
        from self_heal import _extract_exception_from_error

        result = _extract_exception_from_error("generic error message")
        self.assertIsNone(result)


class TestGitWorktreeManager(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_dir = os.getcwd()
        cls.test_repo = tempfile.mkdtemp(prefix="selfheal_test_repo_")
        os.chdir(cls.test_repo)
        subprocess.run(["git", "init"], check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"], check=True, capture_output=True
        )
        with open("test_file.py", "w") as f:
            f.write("# test\n")
        subprocess.run(["git", "add", "."], check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "initial"], check=True, capture_output=True
        )

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.original_dir)
        shutil.rmtree(cls.test_repo, ignore_errors=True)

    def test_create_worktree(self):
        from self_heal import GitWorktreeManager

        manager = GitWorktreeManager(repo_root=self.test_repo)
        worktree = manager.create_worktree("test_session_1")
        self.assertIsNotNone(worktree)
        self.assertTrue(os.path.isdir(worktree))
        manager.remove_worktree("test_session_1")

    def test_commit_in_worktree(self):
        from self_heal import GitWorktreeManager

        manager = GitWorktreeManager(repo_root=self.test_repo)
        worktree = manager.create_worktree("test_session_2")
        self.assertIsNotNone(worktree)

        test_file = os.path.join(worktree, "test_file.py")
        with open(test_file, "a") as f:
            f.write("# added\n")

        commit_hash = manager.commit_in_worktree("test_session_2", "test commit")
        self.assertIsNotNone(commit_hash)
        self.assertEqual(len(commit_hash), 40)

        manager.remove_worktree("test_session_2")

    def test_merge_to_main(self):
        from self_heal import GitWorktreeManager

        manager = GitWorktreeManager(repo_root=self.test_repo)

        worktree = manager.create_worktree("test_session_3")
        self.assertIsNotNone(worktree)

        test_file = os.path.join(worktree, "merged_file.py")
        with open(test_file, "w") as f:
            f.write("# merged content\n")

        manager.commit_in_worktree("test_session_3", "merge test commit")

        merged = manager.merge_to_main("test_session_3")
        self.assertTrue(merged)

        merged_file = os.path.join(self.test_repo, "merged_file.py")
        self.assertTrue(os.path.exists(merged_file))

        manager.remove_worktree("test_session_3")


class TestSelfHealManagerUnit(unittest.TestCase):
    def test_disabled_returns_failure(self):
        from self_heal import SelfHealManager, HealResult

        manager = SelfHealManager()
        manager._enabled = False
        result = asyncio.run(manager.try_heal("test error"))
        self.assertFalse(result.success)
        self.assertIn("disabled", result.error.lower())

    def test_cooldown_prevents_heal(self):
        from self_heal import SelfHealManager

        manager = SelfHealManager()
        manager._cooldown = 10
        manager._last_heal_time = time.time()
        result = asyncio.run(manager.try_heal("test error"))
        self.assertFalse(result.success)
        self.assertIn("cooldown", result.error.lower())

    def test_max_attempts_reached(self):
        from self_heal import SelfHealManager

        manager = SelfHealManager()
        manager._session_attempts["test_session"] = 10
        manager._max_attempts = 3
        result = asyncio.run(
            manager.try_heal("test error", context={"session_id": "test_session"})
        )
        self.assertFalse(result.success)
        self.assertIn("max", result.error.lower())

    def test_transient_error_not_healed(self):
        from self_heal import SelfHealManager

        manager = SelfHealManager()
        result = asyncio.run(manager.try_heal("rate limit exceeded"))
        self.assertFalse(result.success)
        self.assertIn("transient", result.error.lower())


class TestSelfHealManagerIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_dir = os.getcwd()
        cls.test_repo = tempfile.mkdtemp(prefix="selfheal_integration_repo_")
        os.chdir(cls.test_repo)
        subprocess.run(["git", "init"], check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"], check=True, capture_output=True
        )
        with open("main.py", "w") as f:
            f.write("print('hello')\n")
        subprocess.run(["git", "add", "."], check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "initial"], check=True, capture_output=True
        )

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.original_dir)
        shutil.rmtree(cls.test_repo, ignore_errors=True)

    def test_try_heal_skips_transient(self):
        from self_heal import SelfHealManager

        manager = SelfHealManager(repo_root=self.test_repo)
        manager._wt_manager = None  # Disable worktree creation

        result = asyncio.run(
            manager.try_heal("Error: rate limit exceeded [iteration=1]")
        )
        self.assertFalse(result.success)
        self.assertIn("transient", result.error.lower())


class TestClaudeSDKHealer(unittest.TestCase):
    def test_build_prompt_includes_error(self):
        from self_heal import ClaudeAgentSDKHealer, SelfHealErrorReport, ErrorCategory

        healer = ClaudeAgentSDKHealer()
        report = SelfHealErrorReport(
            error_string="AttributeError: foo",
            category=ErrorCategory.CODE,
            exception_type="AttributeError",
        )
        prompt = healer._build_prompt(report, "/tmp/worktree")
        self.assertIn("AttributeError: foo", prompt)
        self.assertIn("Exception type: AttributeError", prompt)

    def test_build_prompt_includes_tools_executed(self):
        from self_heal import ClaudeAgentSDKHealer, SelfHealErrorReport, ErrorCategory

        healer = ClaudeAgentSDKHealer()
        report = SelfHealErrorReport(
            error_string="Tool error",
            category=ErrorCategory.TOOL,
            tools_executed=["bash", "grep"],
        )
        prompt = healer._build_prompt(report, "/tmp/worktree")
        self.assertIn("Tools executed: bash, grep", prompt)


class TestApplyPatchesToWorktree(unittest.TestCase):
    def test_apply_single_patch(self):
        from self_heal import _apply_patches_to_worktree

        with tempfile.TemporaryDirectory() as worktree:
            patches = [
                {
                    "filepath": "test.py",
                    "content": "# fixed content\n",
                    "description": "Fixed the bug",
                }
            ]
            applied = _apply_patches_to_worktree(worktree, patches)
            self.assertEqual(len(applied), 1)
            self.assertEqual(applied[0].filepath, "test.py")

            with open(os.path.join(worktree, "test.py")) as f:
                content = f.read()
            self.assertEqual(content, "# fixed content\n")

    def test_apply_multiple_patches(self):
        from self_heal import _apply_patches_to_worktree

        with tempfile.TemporaryDirectory() as worktree:
            patches = [
                {"filepath": "a.py", "content": "a"},
                {"filepath": "b.py", "content": "b"},
            ]
            applied = _apply_patches_to_worktree(worktree, patches)
            self.assertEqual(len(applied), 2)

    def test_skip_patch_with_missing_filepath(self):
        from self_heal import _apply_patches_to_worktree

        with tempfile.TemporaryDirectory() as worktree:
            patches = [
                {"content": "no filepath"},
                {"filepath": "valid.py", "content": "valid"},
            ]
            applied = _apply_patches_to_worktree(worktree, patches)
            self.assertEqual(len(applied), 1)
            self.assertEqual(applied[0].filepath, "valid.py")

    def test_create_nested_directories(self):
        from self_heal import _apply_patches_to_worktree

        with tempfile.TemporaryDirectory() as worktree:
            patches = [
                {"filepath": "sub/dir/file.py", "content": "nested"},
            ]
            applied = _apply_patches_to_worktree(worktree, patches)
            self.assertEqual(len(applied), 1)
            self.assertTrue(os.path.exists(os.path.join(worktree, "sub/dir/file.py")))


class TestHealResult(unittest.TestCase):
    def test_default_values(self):
        from self_heal import HealResult

        result = HealResult()
        self.assertFalse(result.success)
        self.assertFalse(result.applied)
        self.assertEqual(result.summary, "")
        self.assertEqual(result.patches, [])

    def test_with_values(self):
        from self_heal import HealResult, FilePatch

        patch = FilePatch(filepath="test.py", content="fixed")
        result = HealResult(
            success=True,
            applied=True,
            summary="Fixed test.py",
            patches=[patch],
            commit_hash="abc123",
            merged=True,
        )
        self.assertTrue(result.success)
        self.assertEqual(result.commit_hash, "abc123")
        self.assertTrue(result.merged)


class TestSelfHealErrorReport(unittest.TestCase):
    def test_default_values(self):
        from self_heal import SelfHealErrorReport, ErrorCategory

        report = SelfHealErrorReport(
            error_string="test error",
            category=ErrorCategory.CODE,
        )
        self.assertEqual(report.error_string, "test error")
        self.assertEqual(report.iteration, 0)
        self.assertEqual(report.tools_executed, [])
        self.assertIsNone(report.exception_type)

    def test_timestamp_auto_set(self):
        from self_heal import SelfHealErrorReport, ErrorCategory

        before = time.time()
        report = SelfHealErrorReport(
            error_string="test",
            category=ErrorCategory.TOOL,
        )
        after = time.time()
        self.assertGreaterEqual(report.timestamp, before)
        self.assertLessEqual(report.timestamp, after)


class TestEndToEndWithRealClaudeSDK(unittest.TestCase):
    """Real E2E tests that call Claude Code via the SDK."""

    @classmethod
    def setUpClass(cls):
        cls.original_dir = os.getcwd()
        cls.test_repo = tempfile.mkdtemp(prefix="selfheal_e2e_repo_")
        os.chdir(cls.test_repo)
        subprocess.run(["git", "init"], check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"], check=True, capture_output=True
        )

        broken_code = """
def calculate_total(prices):
    total = sum(price_dict['amount'] for price_dict in prices)
    return total

def format_report(data):
    return f"Report: {data['totl']}"

class DataProcessor:
    def process(self, items):
        return items.convert_to_upper()
"""
        with open("_test_broken_for_self_heal.py", "w") as f:
            f.write(broken_code)

        subprocess.run(["git", "add", "."], check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "initial broken code"],
            check=True,
            capture_output=True,
        )

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.original_dir)
        shutil.rmtree(cls.test_repo, ignore_errors=True)

    def test_real_claude_sdk_heals_broken_code(self):
        """Real E2E: Claude analyzes broken code and produces patches."""
        from self_heal import (
            SelfHealManager,
            SelfHealErrorReport,
            ErrorCategory,
        )

        manager = SelfHealManager(repo_root=self.test_repo)
        manager._cooldown = 0

        error_report = SelfHealErrorReport(
            error_string=(
                "Error: Tool execution error for execute_user_code:\n"
                "AttributeError: 'list' object has no attribute 'convert_to_upper'\n\n"
                "Traceback (most recent call last):\n"
                '  File "_test_broken_for_self_heal.py", line 14, in process\n'
                "    return items.convert_to_upper()\n"
                "AttributeError: 'list' object has no attribute 'convert_to_upper'"
            ),
            category=ErrorCategory.CODE,
            exception_type="AttributeError",
            traceback_snippet="AttributeError: 'list' object has no attribute 'convert_to_upper'",
            session_id="e2e_test_session",
        )

        result = asyncio.run(manager._execute_heal(error_report, "e2e_test_session"))

        print("\n=== Real Claude Agent SDK Consultation ===")
        print(f"Error: {error_report.error_string[:100]}...")
        print(f"File: {self.test_repo}/_test_broken_for_self_heal.py:14")
        print("=" * 40)

        if result.patches:
            print(f"\nClaude response received:")
            print(f"  Patches returned: {len(result.patches)}")
            print(f"  Explanation: {result.summary[:150]}...")
            for p in result.patches:
                print(
                    f"    -> {p.filepath}: {p.description[:80] if p.description else 'no description'}"
                )
            print(
                f"Applied {len(result.patches)} / {len(result.patches)} patches to worktree"
            )
            print(f"Committed fix: {result.commit_hash}")

        self.assertIsNotNone(result, "Heal result should not be None")
        self.assertTrue(result.success, f"Heal should succeed but got: {result.error}")
        self.assertTrue(result.applied, "At least one patch should be applied")
        self.assertGreater(len(result.patches), 0, "Should have at least one patch")

        fixed_file = os.path.join(self.test_repo, "_test_broken_for_self_heal.py")
        with open(fixed_file) as f:
            fixed_content = f.read()

        self.assertNotIn(
            "convert_to_upper",
            fixed_content,
            "Bug should be fixed - convert_to_upper method call removed",
        )


class TestSelfHealStatusTool(unittest.TestCase):
    def test_tool_registered(self):
        from tools import TOOLS

        self.assertIn("self_heal_status", TOOLS)

    def test_tool_callable(self):
        from tools import TOOLS

        result = asyncio.run(TOOLS["self_heal_status"]())
        self.assertIn("success", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
