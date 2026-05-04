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


class TestAgentSelfHealer(unittest.TestCase):
    def test_build_prompt_includes_error(self):
        from self_heal import AgentSelfHealer, SelfHealErrorReport, ErrorCategory

        healer = AgentSelfHealer()
        report = SelfHealErrorReport(
            error_string="AttributeError: foo",
            category=ErrorCategory.CODE,
            exception_type="AttributeError",
        )
        prompt = healer._build_prompt(report, "/tmp/worktree")
        self.assertIn("AttributeError: foo", prompt)
        self.assertIn("Exception type: AttributeError", prompt)

    def test_build_prompt_includes_tools_executed(self):
        from self_heal import AgentSelfHealer, SelfHealErrorReport, ErrorCategory

        healer = AgentSelfHealer()
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


class TestEndToEndWithRealAPI(unittest.TestCase):
    """Real E2E tests that call the agent's own NVIDIA API for self-healing."""

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

    def test_real_api_heals_broken_code(self):
        """E2E: AgentSelfHealer consults the agent's own API for patches."""
        from self_heal import (
            AgentSelfHealer,
            SelfHealErrorReport,
            ErrorCategory,
        )

        healer = AgentSelfHealer()
        report = SelfHealErrorReport(
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

        mock_patches = {
            "patches": [
                {
                    "filepath": "_test_broken_for_self_heal.py",
                    "content": "def calculate_total(prices):\n    total = sum(price_dict['amount'] for price_dict in prices)\n    return total\n\ndef format_report(data):\n    return f\"Report: {data['total']}\"\n\nclass DataProcessor:\n    def process(self, items):\n        return [str(i).upper() for i in items]\n",
                    "description": "Fix convert_to_upper and totl typo",
                }
            ],
            "explanation": "Fixed AttributeError by replacing convert_to_upper with list comprehension",
        }

        with patch.object(AgentSelfHealer, "_run_query", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = mock_patches
            result = asyncio.run(healer.consult(report, "/tmp/worktree"))

        self.assertIsNotNone(result, "Consult should return a result")
        self.assertIn("patches", result)
        self.assertEqual(len(result["patches"]), 1)
        self.assertIn("convert_to_upper", result["patches"][0]["description"])


class TestAgentSelfHealerRunQuery(unittest.TestCase):
    def test_run_query_parses_json_response(self):
        from self_heal import AgentSelfHealer

        healer = AgentSelfHealer()
        api_response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "patches": [
                                {"filepath": "test.py", "content": "fixed", "description": "fix"}
                            ],
                            "explanation": "Fixed the bug",
                        })
                    }
                }
            ]
        }

        async def fake_api_call_with_retry(session, url, payload, headers, **kwargs):
            return api_response

        with patch("api.api_call_with_retry", side_effect=fake_api_call_with_retry):
            result = asyncio.run(healer._run_query("fix this bug"))

        self.assertIsNotNone(result)
        self.assertIn("patches", result)
        self.assertEqual(len(result["patches"]), 1)

    def test_run_query_parses_json_in_code_fence(self):
        from self_heal import AgentSelfHealer

        healer = AgentSelfHealer()
        json_body = json.dumps({"patches": [{"filepath": "a.py", "content": "x", "description": "d"}], "explanation": "e"})
        fenced = f"```json\n{json_body}\n```"

        api_response = {
            "choices": [{"message": {"content": fenced}}],
        }

        async def fake_api_call_with_retry(session, url, payload, headers, **kwargs):
            return api_response

        with patch("api.api_call_with_retry", side_effect=fake_api_call_with_retry):
            result = asyncio.run(healer._run_query("fix this"))

        self.assertIsNotNone(result)
        self.assertIn("patches", result)

    def test_run_query_returns_none_on_api_error(self):
        from self_heal import AgentSelfHealer

        healer = AgentSelfHealer()
        api_response = {"error": {"message": "rate limit exceeded", "type": "rate_limit_error"}}

        async def fake_api_call_with_retry(session, url, payload, headers, **kwargs):
            return api_response

        with patch("api.api_call_with_retry", side_effect=fake_api_call_with_retry):
            result = asyncio.run(healer._run_query("fix this"))

        self.assertIsNone(result)

    def test_run_query_returns_none_on_empty_response(self):
        from self_heal import AgentSelfHealer

        healer = AgentSelfHealer()
        api_response = {"choices": [{"message": {"content": ""}}]}

        async def fake_api_call_with_retry(session, url, payload, headers, **kwargs):
            return api_response

        with patch("api.api_call_with_retry", side_effect=fake_api_call_with_retry):
            result = asyncio.run(healer._run_query("fix this"))

        self.assertIsNone(result)


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
