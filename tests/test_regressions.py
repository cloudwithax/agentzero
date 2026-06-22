#!/usr/bin/env python3
"""Tests for the general-purpose regression detection system."""

import json
import os
import shutil
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, patch

import regressions as reg_mod
from regressions import (
    CATEGORY_EMPTY_OUTPUT,
    CATEGORY_ERROR_IN_SUCCESS,
    CATEGORY_TRIVIAL_RESULT,
    RegressionReport,
    _contains_error_keywords,
    _extract_content_string,
    _is_trivially_empty,
    detect_all_regressions,
    detect_regression,
    format_regression_for_heal,
    log_regression,
    read_recent_regressions,
)


class TestExtractContentString(unittest.TestCase):
    def test_empty_string(self):
        self.assertEqual(_extract_content_string(""), "")

    def test_json_object(self):
        result = _extract_content_string('{"success": true, "content": "hello"}')
        self.assertIn("success", result)
        self.assertIn("hello", result)

    def test_json_array(self):
        result = _extract_content_string('["a", "b"]')
        self.assertIn("a", result)

    def test_plain_text(self):
        self.assertEqual(_extract_content_string("hello world"), "hello world")


class TestIsTriviallyEmpty(unittest.TestCase):
    def test_none(self):
        self.assertTrue(_is_trivially_empty(""))

    def test_empty_dict(self):
        self.assertTrue(_is_trivially_empty("{}"))

    def test_empty_list(self):
        self.assertTrue(_is_trivially_empty("[]"))

    def test_null_string(self):
        self.assertTrue(_is_trivially_empty("null"))

    def test_none_string(self):
        self.assertTrue(_is_trivially_empty("None"))

    def test_short_string(self):
        self.assertTrue(_is_trivially_empty("ok"))

    def test_normal_string(self):
        self.assertFalse(_is_trivially_empty("reasonable output content"))


class TestContainsErrorKeywords(unittest.TestCase):
    def test_error_detected(self):
        self.assertTrue(_contains_error_keywords("an error occurred"))

    def test_failed_detected(self):
        self.assertTrue(_contains_error_keywords("the command failed"))

    def test_permission_denied(self):
        self.assertTrue(_contains_error_keywords("permission denied"))

    def test_clean_text(self):
        self.assertFalse(_contains_error_keywords("all systems operational"))

    def test_timeout(self):
        self.assertTrue(_contains_error_keywords("connection timed out"))


class TestDetectRegression(unittest.TestCase):
    def test_empty_output_triggers_regression(self):
        report = detect_regression(
            "bash",
            {"command": "ls"},
            json.dumps({"success": True, "content": ""}),
        )
        self.assertIsNotNone(report)
        self.assertEqual(report.category, CATEGORY_EMPTY_OUTPUT)
        self.assertEqual(report.tool_name, "bash")

    def test_null_content_triggers_regression(self):
        report = detect_regression(
            "read",
            {"filepath": "/tmp/x"},
            json.dumps({"success": True, "content": None}),
        )
        self.assertIsNotNone(report)

    def test_error_in_success_triggers_regression(self):
        report = detect_regression(
            "bash",
            {"command": "deploy"},
            json.dumps({"success": True, "content": "build failed: syntax error"}),
        )
        self.assertIsNotNone(report)
        self.assertEqual(report.category, CATEGORY_ERROR_IN_SUCCESS)

    def test_trivial_output_flagged_for_low_info_tools(self):
        report = detect_regression(
            "grep",
            {"pattern": "foo"},
            json.dumps({"success": True, "content": "x"}),
        )
        self.assertIsNotNone(report)
        # Content "x" is fewer than _MIN_CONTENT_LENGTH=8, so it is
        # flagged as empty_output before the trivial-result check runs.
        self.assertEqual(report.category, CATEGORY_EMPTY_OUTPUT)

    def test_normal_output_no_regression(self):
        report = detect_regression(
            "bash",
            {"command": "ls"},
            json.dumps({"success": True, "content": "file1.txt\nfile2.txt\n"}),
        )
        self.assertIsNone(report)

    def test_success_false_not_flagged_as_regression(self):
        report = detect_regression(
            "bash",
            {"command": "ls"},
            json.dumps({"success": False, "error": "command not found"}),
        )
        self.assertIsNone(report)

    def test_large_output_not_flagged(self):
        report = detect_regression(
            "read",
            {"filepath": "/tmp/x"},
            json.dumps({"success": True, "content": "a" * 100}),
        )
        self.assertIsNone(report)


class TestDetectAllRegressions(unittest.TestCase):
    def test_no_regressions(self):
        calls = [
            {"function": {"name": "bash", "arguments": '{"command": "ls"}'}},
        ]
        results = [
            {
                "tool_call_id": "call_1",
                "role": "tool",
                "content": json.dumps({"success": True, "content": "file1.txt\nfile2.txt\nfile3.txt"}),
            },
        ]
        reports = detect_all_regressions(calls, results)
        self.assertEqual(len(reports), 0)

    def test_detects_empty_output_regression(self):
        calls = [
            {"function": {"name": "bash", "arguments": '{"command": "ls"}'}},
        ]
        results = [
            {
                "tool_call_id": "call_1",
                "role": "tool",
                "content": json.dumps({"success": True, "content": ""}),
            },
        ]
        reports = detect_all_regressions(calls, results)
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0].category, CATEGORY_EMPTY_OUTPUT)

    def test_detects_multiple_regressions(self):
        calls = [
            {"function": {"name": "bash", "arguments": '{"command": "x"}'}},
            {"function": {"name": "grep", "arguments": '{"pattern": "foo"}'}},
        ]
        results = [
            {
                "tool_call_id": "call_1",
                "role": "tool",
                "content": json.dumps({"success": True, "content": ""}),
            },
            {
                "tool_call_id": "call_2",
                "role": "tool",
                "content": json.dumps({"success": True, "content": "x"}),
            },
        ]
        reports = detect_all_regressions(calls, results)
        self.assertEqual(len(reports), 2)


class TestFormatRegressionForHeal(unittest.TestCase):
    def test_format_includes_tool_name_and_miss(self):
        report = RegressionReport(
            tool_name="bash",
            tool_args={"command": "ls"},
            tool_result_content='{"success": true, "content": ""}',
            category=CATEGORY_EMPTY_OUTPUT,
            detected_miss="Tool returned empty content",
            session_id="test_session",
        )
        formatted = format_regression_for_heal(report)
        self.assertIn("bash", formatted)
        self.assertIn("empty content", formatted)
        self.assertIn("Behavioral regression", formatted)


class TestLogRegression(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="regression_test_")
        self.tmp_log = os.path.join(self.tmpdir, "regressions.log")
        self._orig_log_file = reg_mod._LOG_FILE
        reg_mod._LOG_FILE = self.tmp_log

    def tearDown(self):
        reg_mod._LOG_FILE = self._orig_log_file
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_log_writes_entry(self):
        report = RegressionReport(
            tool_name="bash",
            tool_args={"command": "ls"},
            tool_result_content="empty",
            category=CATEGORY_EMPTY_OUTPUT,
            detected_miss="Tool returned empty content",
            session_id="test_session",
        )
        log_regression(report)

        self.assertTrue(os.path.isfile(self.tmp_log))
        with open(self.tmp_log) as f:
            content = f.read()
        self.assertIn("TOOL: bash", content)
        self.assertIn("MISS: Tool returned empty content", content)
        self.assertIn("CATEGORY: empty_output", content)

    def test_read_recent_regressions(self):
        report = RegressionReport(
            tool_name="grep",
            tool_args={"pattern": "foo"},
            tool_result_content="",
            category=CATEGORY_EMPTY_OUTPUT,
            detected_miss="grep returned nothing",
            session_id="test_session",
        )
        log_regression(report)

        entries = read_recent_regressions(hours=48)
        self.assertGreaterEqual(len(entries), 1)
        self.assertEqual(entries[0].get("tool"), "grep")
        self.assertEqual(entries[0].get("category"), CATEGORY_EMPTY_OUTPUT)
        self.assertIn("grep", entries[0].get("miss", ""))


if __name__ == "__main__":
    unittest.main(verbosity=2)
