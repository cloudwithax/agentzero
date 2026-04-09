"""Tests for the encrypted credential store."""

import asyncio
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory import MemoryStore


class TestCredentialStore(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.store = MemoryStore(db_path=self._tmp.name)

    def tearDown(self):
        os.unlink(self._tmp.name)

    def test_store_and_get_credential(self):
        result = self.store.store_credential(
            "github_token", "ghp_abc123", {"description": "GitHub PAT"}
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["key"], "github_token")

        retrieved = self.store.get_credential("github_token")
        self.assertTrue(retrieved["success"])
        self.assertEqual(retrieved["key"], "github_token")
        self.assertEqual(retrieved["value"], "ghp_abc123")
        self.assertEqual(retrieved["metadata"]["description"], "GitHub PAT")

    def test_store_overwrites_existing_key(self):
        self.store.store_credential("api_key", "old_value")
        self.store.store_credential("api_key", "new_value", {"description": "updated"})

        retrieved = self.store.get_credential("api_key")
        self.assertTrue(retrieved["success"])
        self.assertEqual(retrieved["value"], "new_value")
        self.assertEqual(retrieved["metadata"]["description"], "updated")

    def test_get_nonexistent_credential(self):
        result = self.store.get_credential("nonexistent")
        self.assertFalse(result["success"])
        self.assertIn("not found", result["error"])

    def test_delete_credential(self):
        self.store.store_credential("temp_key", "temp_value")
        result = self.store.delete_credential("temp_key")
        self.assertTrue(result["success"])

        retrieved = self.store.get_credential("temp_key")
        self.assertFalse(retrieved["success"])

    def test_delete_nonexistent_credential(self):
        result = self.store.delete_credential("nonexistent")
        self.assertFalse(result["success"])

    def test_list_credentials(self):
        self.store.store_credential("key_a", "val_a", {"description": "A"})
        self.store.store_credential("key_b", "val_b", {"description": "B"})

        result = self.store.list_credentials()
        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 2)

        keys = [entry["key"] for entry in result["credentials"]]
        self.assertIn("key_a", keys)
        self.assertIn("key_b", keys)

        for entry in result["credentials"]:
            self.assertNotIn("value", entry)
            self.assertIn("metadata", entry)
            self.assertIn("created_at", entry)

    def test_value_is_encrypted_at_rest(self):
        import sqlite3

        secret = "super_secret_password_12345"
        self.store.store_credential("test_key", secret)

        conn = sqlite3.connect(self._tmp.name)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT encrypted_value FROM credentials WHERE key = ?", ("test_key",)
        )
        row = cursor.fetchone()
        conn.close()

        self.assertIsNotNone(row)
        stored_value = row[0]
        self.assertNotEqual(stored_value, secret)
        self.assertNotIn(secret, stored_value)

    def test_empty_key_rejected(self):
        result = self.store.store_credential("", "value")
        self.assertFalse(result["success"])

    def test_empty_value_rejected(self):
        result = self.store.store_credential("key", "")
        self.assertFalse(result["success"])

    def test_store_credential_without_metadata(self):
        result = self.store.store_credential("bare_key", "bare_value")
        self.assertTrue(result["success"])

        retrieved = self.store.get_credential("bare_key")
        self.assertTrue(retrieved["success"])
        self.assertEqual(retrieved["value"], "bare_value")
        self.assertEqual(retrieved["metadata"], {})


class TestCredentialTools(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.store = MemoryStore(db_path=self._tmp.name)
        import tools

        tools.set_memory_store(self.store)
        self.tools = tools

    def tearDown(self):
        os.unlink(self._tmp.name)

    def test_store_credential_tool(self):
        result = asyncio.run(
            self.tools.store_credential_tool(
                key="openai_key", value="sk-1234", description="OpenAI API key"
            )
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["key"], "openai_key")

    def test_get_credential_tool(self):
        asyncio.run(self.tools.store_credential_tool(key="test_key", value="test_val"))
        result = asyncio.run(self.tools.get_credential_tool(key="test_key"))
        self.assertTrue(result["success"])
        self.assertEqual(result["value"], "test_val")

    def test_delete_credential_tool(self):
        asyncio.run(self.tools.store_credential_tool(key="del_me", value="val"))
        result = asyncio.run(self.tools.delete_credential_tool(key="del_me"))
        self.assertTrue(result["success"])

    def test_list_credentials_tool(self):
        asyncio.run(
            self.tools.store_credential_tool(key="k1", value="v1", description="first")
        )
        asyncio.run(
            self.tools.store_credential_tool(key="k2", value="v2", description="second")
        )
        result = asyncio.run(self.tools.list_credentials_tool())
        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 2)

    def test_tool_validation(self):
        is_valid, error = self.tools.validate_tool_args(
            "store_credential", {"key": "k"}
        )
        self.assertFalse(is_valid)
        is_valid, error = self.tools.validate_tool_args(
            "store_credential", {"key": "k", "value": "v"}
        )
        self.assertTrue(is_valid)
        is_valid, error = self.tools.validate_tool_args("get_credential", {"key": "k"})
        self.assertTrue(is_valid)
        is_valid, error = self.tools.validate_tool_args(
            "delete_credential", {"key": "k"}
        )
        self.assertTrue(is_valid)
        is_valid, error = self.tools.validate_tool_args("list_credentials", {})
        self.assertTrue(is_valid)

    def test_tools_registered(self):
        self.assertIn("store_credential", self.tools.TOOLS)
        self.assertIn("get_credential", self.tools.TOOLS)
        self.assertIn("delete_credential", self.tools.TOOLS)
        self.assertIn("list_credentials", self.tools.TOOLS)


if __name__ == "__main__":
    unittest.main()
