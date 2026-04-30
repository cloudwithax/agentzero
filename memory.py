"""Memory system for persistent storage and retrieval."""

import base64
import json
import math
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

import aiohttp
import numpy as np

try:
    from cryptography.fernet import Fernet, InvalidToken
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    _CRYPTO_AVAILABLE = True
except ImportError:
    _CRYPTO_AVAILABLE = False


@dataclass
class Memory:
    """A single memory entry."""

    id: Optional[int]
    content: str
    embedding: Optional[list[float]]
    metadata: dict[str, Any]
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class MemoryStore:
    """SQLite-based persistent memory store with embedding support."""

    EMBEDDING_API_URL = "https://integrate.api.nvidia.com/v1/embeddings"
    EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-1b-v2"

    def __init__(
        self,
        db_path: str = "agent_memory.db",
        api_key: Optional[str] = None,
        embedding_dim: int = 1024,
    ):
        self.db_path = db_path
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        self.embedding_dim = embedding_dim
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Main memories table with vector storage
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                embedding BLOB,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Index for faster lookups
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memories_created 
            ON memories(created_at)
        """
        )

        # Conversations table for session history
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Topics/tags table for categorization
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Memory-topic associations
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_topics (
                memory_id INTEGER,
                topic_id INTEGER,
                PRIMARY KEY (memory_id, topic_id),
                FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
                FOREIGN KEY (topic_id) REFERENCES topics(id) ON DELETE CASCADE
            )
        """
        )

        # Agent-wide state for long-running behavior (e.g., dream schedule)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_state (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Reminder tasks table for durable scheduled task persistence.
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS reminders (
                task_id TEXT PRIMARY KEY,
                name TEXT,
                cron TEXT NOT NULL,
                message TEXT,
                session_id TEXT,
                one_off INTEGER DEFAULT 0,
                run_ai INTEGER DEFAULT 0,
                ai_prompt TEXT,
                status TEXT,
                enabled INTEGER DEFAULT 1,
                run_count INTEGER DEFAULT 0,
                last_run_at TEXT,
                next_run_at TEXT,
                last_result TEXT,
                last_error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                max_runs INTEGER DEFAULT 0
            )
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_reminders_next_run
            ON reminders(enabled, next_run_at)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_reminders_created_at
            ON reminders(created_at)
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS credentials (
                key TEXT PRIMARY KEY,
                encrypted_value BLOB NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_credentials_updated
            ON credentials(updated_at)
        """
        )

        conn.commit()
        conn.close()

    @staticmethod
    def _derive_fernet_key(password: str, salt: bytes) -> bytes:
        if not _CRYPTO_AVAILABLE:
            raise RuntimeError(
                "cryptography package is required for credential encryption"
            )
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=600_000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode("utf-8")))

    def _get_or_create_encryption_key(self) -> tuple[Fernet, bytes]:
        if not _CRYPTO_AVAILABLE:
            raise RuntimeError(
                "cryptography package is required for credential encryption"
            )
        key_b64 = os.environ.get("CREDENTIALS_ENCRYPTION_KEY", "").strip()
        if key_b64:
            try:
                return Fernet(key_b64.encode("utf-8")), b""
            except Exception:
                pass
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT value FROM agent_state WHERE key = 'credentials.encryption_key'"
        )
        row = cursor.fetchone()
        if row:
            stored = row[0]
            if isinstance(stored, str):
                stored = stored.strip('"')
            try:
                fernet = Fernet(stored.encode("utf-8"))
                conn.close()
                return fernet, b""
            except Exception:
                pass
        new_key = Fernet.generate_key()
        cursor.execute(
            """
            INSERT INTO agent_state (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key)
            DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP
        """,
            ("credentials.encryption_key", new_key.decode("utf-8")),
        )
        conn.commit()
        conn.close()
        return Fernet(new_key), b""

    def _encrypt_value(self, plaintext: str) -> str:
        fernet, _ = self._get_or_create_encryption_key()
        return fernet.encrypt(plaintext.encode("utf-8")).decode("utf-8")

    def _decrypt_value(self, ciphertext: str) -> str:
        fernet, _ = self._get_or_create_encryption_key()
        return fernet.decrypt(ciphertext.encode("utf-8")).decode("utf-8")

    def store_credential(
        self,
        key: str,
        value: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        key = str(key or "").strip()
        if not key:
            return {"success": False, "error": "key must be a non-empty string"}
        value = str(value or "")
        if not value:
            return {"success": False, "error": "value must be a non-empty string"}
        if not _CRYPTO_AVAILABLE:
            return {
                "success": False,
                "error": "cryptography package not installed; cannot encrypt credentials",
            }
        try:
            encrypted = self._encrypt_value(value)
        except Exception as exc:
            return {"success": False, "error": f"encryption failed: {exc}"}
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO credentials (key, encrypted_value, metadata, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key)
            DO UPDATE SET encrypted_value = excluded.encrypted_value,
                          metadata = excluded.metadata,
                          updated_at = CURRENT_TIMESTAMP
        """,
            (key, encrypted, json.dumps(metadata or {})),
        )
        conn.commit()
        conn.close()
        return {"success": True, "key": key}

    def get_credential(self, key: str) -> dict[str, Any]:
        key = str(key or "").strip()
        if not key:
            return {"success": False, "error": "key must be a non-empty string"}
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT encrypted_value, metadata, created_at, updated_at FROM credentials WHERE key = ?",
            (key,),
        )
        row = cursor.fetchone()
        conn.close()
        if not row:
            return {"success": False, "error": f"credential '{key}' not found"}
        encrypted_value, metadata_blob, created_at, updated_at = row
        if not _CRYPTO_AVAILABLE:
            return {
                "success": False,
                "error": "cryptography package not installed; cannot decrypt credentials",
            }
        try:
            decrypted = self._decrypt_value(encrypted_value)
        except InvalidToken:
            return {
                "success": False,
                "error": "decryption failed: invalid encryption key or corrupted data",
            }
        except Exception as exc:
            return {"success": False, "error": f"decryption failed: {exc}"}
        return {
            "success": True,
            "key": key,
            "value": decrypted,
            "metadata": self._parse_metadata_blob(metadata_blob),
            "created_at": created_at,
            "updated_at": updated_at,
        }

    def delete_credential(self, key: str) -> dict[str, Any]:
        key = str(key or "").strip()
        if not key:
            return {"success": False, "error": "key must be a non-empty string"}
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM credentials WHERE key = ?", (key,))
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        if deleted:
            return {"success": True, "key": key}
        return {"success": False, "error": f"credential '{key}' not found"}

    def list_credentials(self) -> dict[str, Any]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT key, metadata, created_at, updated_at FROM credentials ORDER BY key"
        )
        rows = cursor.fetchall()
        conn.close()
        entries = []
        for key, metadata_blob, created_at, updated_at in rows:
            entries.append(
                {
                    "key": key,
                    "metadata": self._parse_metadata_blob(metadata_blob),
                    "created_at": created_at,
                    "updated_at": updated_at,
                }
            )
        return {"success": True, "count": len(entries), "credentials": entries}

    @staticmethod
    def _parse_metadata_blob(metadata_blob: Optional[str]) -> dict[str, Any]:
        """Safely parse JSON metadata stored in SQLite."""
        if not metadata_blob:
            return {}
        try:
            parsed = json.loads(metadata_blob)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
        """Parse SQLite timestamp strings into datetime."""
        if not value:
            return None

        candidate = str(value).strip()
        if not candidate:
            return None

        try:
            return datetime.fromisoformat(candidate.replace(" ", "T"))
        except ValueError:
            return None

    async def generate_embedding(
        self, text: str, input_type: str = "passage"
    ) -> list[float]:
        """Generate embedding vector using NVIDIA API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "input": text,
            "model": self.EMBEDDING_MODEL,
            "encoding_format": "float",
            "input_type": input_type,
            "truncate": "END",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.EMBEDDING_API_URL, json=payload, headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Embedding API error: {error_text}")

                data = await response.json()
                return data["data"][0]["embedding"]

    def _embedding_to_bytes(self, embedding: list[float]) -> bytes:
        """Convert embedding list to bytes for storage."""
        return np.array(embedding, dtype=np.float32).tobytes()

    def _bytes_to_embedding(self, data: bytes) -> list[float]:
        """Convert bytes back to embedding list."""
        return np.frombuffer(data, dtype=np.float32).tolist()

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a_arr = np.array(a)
        b_arr = np.array(b)
        return float(
            np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
        )

    async def add_memory(
        self,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        generate_embedding: bool = False,
        topics: Optional[list[str]] = None,
    ) -> Optional[int]:
        """Add a new memory to the store."""
        embedding = None
        if generate_embedding and self.api_key:
            try:
                embedding = await self.generate_embedding(content, input_type="passage")
            except Exception as e:
                print(f"Warning: Failed to generate embedding: {e}")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO memories (content, embedding, metadata, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """,
            (
                content,
                self._embedding_to_bytes(embedding) if embedding else None,
                json.dumps(metadata or {}),
            ),
        )

        memory_id = cursor.lastrowid

        # Associate with topics if provided
        if topics:
            for topic_name in topics:
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO topics (name) VALUES (?)
                """,
                    (topic_name,),
                )
                cursor.execute(
                    """
                    SELECT id FROM topics WHERE name = ?
                """,
                    (topic_name,),
                )
                topic_id = cursor.fetchone()[0]
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO memory_topics (memory_id, topic_id)
                    VALUES (?, ?)
                """,
                    (memory_id, topic_id),
                )

        conn.commit()
        conn.close()

        return memory_id

    async def search_memories(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.15,
        topic: Optional[str] = None,
    ) -> list[tuple[Memory, float]]:
        """Search memories by keyword matching (fast, no API calls)."""
        if not query or not query.strip():
            return []

        # Extract meaningful keywords from the query (skip short/common words)
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "in", "on", "at", "to", "for", "of", "with", "from", "by",
            "and", "or", "but", "not", "so", "if", "it", "its", "i", "you",
            "he", "she", "we", "they", "my", "your", "his", "her", "our",
            "me", "us", "them", "do", "does", "did", "can", "will", "would",
            "what", "where", "when", "why", "how", "who", "which", "that",
            "this", "these", "those", "there", "here", "just", "now", "then",
            "very", "really", "about", "also", "please", "tell", "know",
            "remember", "remind", "name", "live", "like", "want", "need",
        }
        keywords = [
            w.lower() for w in re.findall(r"[a-zA-Z]{3,}", query)
            if w.lower() not in stopwords
        ]

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Build scored results from text matching
        scored: dict[int, tuple[Memory, float]] = {}

        for kw in keywords[:8]:  # max 8 keywords
            like_pattern = f"%{kw}%"
            if topic:
                cursor.execute(
                    """
                    SELECT m.id, m.content, m.embedding, m.metadata, m.created_at, m.updated_at
                    FROM memories m
                    JOIN memory_topics mt ON m.id = mt.memory_id
                    JOIN topics t ON mt.topic_id = t.id
                    WHERE t.name = ? AND LOWER(m.content) LIKE ?
                """,
                    (topic, like_pattern),
                )
            else:
                cursor.execute(
                    """
                    SELECT id, content, embedding, metadata, created_at, updated_at
                    FROM memories
                    WHERE LOWER(content) LIKE ?
                """,
                    (like_pattern,),
                )

            for row in cursor.fetchall():
                memory_id, content, embedding_bytes, metadata_str, created_at, updated_at = row
                if memory_id in scored:
                    scored[memory_id] = (scored[memory_id][0], scored[memory_id][1] + 1.0)
                else:
                    memory = Memory(
                        id=memory_id,
                        content=content,
                        embedding=None,
                        metadata=json.loads(metadata_str or "{}"),
                        created_at=created_at,
                        updated_at=updated_at,
                    )
                    scored[memory_id] = (memory, 1.0)

        conn.close()

        results = sorted(scored.values(), key=lambda x: x[1], reverse=True)
        return [(m, min(s / len(keywords or [1]), 1.0)) for m, s in results[:top_k]]

    def get_memories_by_topic(self, topic: str, limit: int = 10) -> list[Memory]:
        """Get all memories associated with a topic."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT m.id, m.content, m.embedding, m.metadata, m.created_at, m.updated_at
            FROM memories m
            JOIN memory_topics mt ON m.id = mt.memory_id
            JOIN topics t ON mt.topic_id = t.id
            WHERE t.name = ?
            ORDER BY m.created_at DESC
            LIMIT ?
        """,
            (topic, limit),
        )

        rows = cursor.fetchall()
        conn.close()

        memories = []
        for row in rows:
            memory_id, content, embedding_bytes, metadata, created_at, updated_at = row
            embedding = (
                self._bytes_to_embedding(embedding_bytes) if embedding_bytes else None
            )
            memories.append(
                Memory(
                    id=memory_id,
                    content=content,
                    embedding=embedding,
                    metadata=json.loads(metadata or "{}"),
                    created_at=created_at,
                    updated_at=updated_at,
                )
            )

        return memories

    def get_recent_memories(self, limit: int = 10) -> list[Memory]:
        """Get most recent memories."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, content, embedding, metadata, created_at, updated_at
            FROM memories
            ORDER BY created_at DESC
            LIMIT ?
        """,
            (limit,),
        )

        rows = cursor.fetchall()
        conn.close()

        memories = []
        for row in rows:
            memory_id, content, embedding_bytes, metadata, created_at, updated_at = row
            embedding = (
                self._bytes_to_embedding(embedding_bytes) if embedding_bytes else None
            )
            memories.append(
                Memory(
                    id=memory_id,
                    content=content,
                    embedding=embedding,
                    metadata=json.loads(metadata or "{}"),
                    created_at=created_at,
                    updated_at=updated_at,
                )
            )

        return memories

    def get_speech_pattern_memories(
        self, session_id: Optional[str] = None, limit: int = 5
    ) -> list[Memory]:
        """Get speech pattern memories for a session or globally."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if session_id:
            cursor.execute(
                """
                SELECT id, content, embedding, metadata, created_at, updated_at
                FROM memories
                WHERE metadata LIKE '%"type": "speech_pattern"%'
                AND (metadata LIKE ? OR metadata LIKE '%"session_id": null%')
                ORDER BY created_at DESC
                LIMIT ?
            """,
                (f'%"session_id": "{session_id}"%', limit),
            )
        else:
            cursor.execute(
                """
                SELECT id, content, embedding, metadata, created_at, updated_at
                FROM memories
                WHERE metadata LIKE '%"type": "speech_pattern"%'
                ORDER BY created_at DESC
                LIMIT ?
            """,
                (limit,),
            )

        rows = cursor.fetchall()
        conn.close()

        memories = []
        for row in rows:
            memory_id, content, embedding_bytes, metadata, created_at, updated_at = row
            embedding = (
                self._bytes_to_embedding(embedding_bytes) if embedding_bytes else None
            )
            memories.append(
                Memory(
                    id=memory_id,
                    content=content,
                    embedding=embedding,
                    metadata=json.loads(metadata or "{}"),
                    created_at=created_at,
                    updated_at=updated_at,
                )
            )

        return memories

    def add_conversation_message(
        self,
        role: str,
        content: str,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> int:
        """Add a conversation message to history."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO conversations (session_id, role, content, metadata)
            VALUES (?, ?, ?, ?)
        """,
            (session_id, role, content, json.dumps(metadata or {})),
        )

        msg_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return msg_id

    def update_conversation_message_content(
        self,
        message_id: int,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Update stored content (and optional metadata) for a conversation row."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        normalized_content = "" if content is None else str(content)
        if metadata is None:
            cursor.execute(
                "UPDATE conversations SET content = ? WHERE id = ?",
                (normalized_content, int(message_id)),
            )
        else:
            cursor.execute(
                "UPDATE conversations SET content = ?, metadata = ? WHERE id = ?",
                (normalized_content, json.dumps(metadata), int(message_id)),
            )

        updated = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return updated

    def get_conversation_messages_with_untranscribed_voice_memos(
        self,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return user rows that still contain unresolved voice memo URL markers."""
        limit = max(1, int(limit))

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, session_id, role, content, metadata, created_at
            FROM conversations
            WHERE role = 'user'
              AND LOWER(content) LIKE '%[voice memo attachments not transcribed]%'
            ORDER BY created_at DESC
            LIMIT ?
        """,
            (limit,),
        )

        rows = cursor.fetchall()
        conn.close()

        messages: list[dict[str, Any]] = []
        for row in rows:
            messages.append(
                {
                    "id": row[0],
                    "session_id": row[1],
                    "role": row[2],
                    "content": row[3],
                    "metadata": json.loads(row[4] or "{}"),
                    "created_at": row[5],
                }
            )

        return messages

    def get_conversation_history(
        self, session_id: Optional[str] = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Get conversation history."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if session_id:
            cursor.execute(
                """
                SELECT id, session_id, role, content, metadata, created_at
                FROM conversations
                WHERE session_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
            """,
                (session_id, limit),
            )
        else:
            cursor.execute(
                """
                SELECT id, session_id, role, content, metadata, created_at
                FROM conversations
                ORDER BY created_at DESC, id DESC
                LIMIT ?
            """,
                (limit,),
            )

        rows = cursor.fetchall()
        conn.close()

        messages = []
        for row in rows:
            messages.append(
                {
                    "id": row[0],
                    "session_id": row[1],
                    "role": row[2],
                    "content": row[3],
                    "metadata": json.loads(row[4] or "{}"),
                    "created_at": row[5],
                }
            )

        return messages

    def get_recent_session_ids_by_prefix(
        self,
        session_prefix: str,
        limit: int = 10,
    ) -> list[str]:
        """Return recent session IDs whose IDs start with the provided prefix."""
        prefix = str(session_prefix or "").strip()
        if not prefix:
            return []

        limit = max(1, int(limit))
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT session_id, MAX(created_at) AS last_seen
            FROM conversations
            WHERE session_id LIKE ?
            GROUP BY session_id
            ORDER BY last_seen DESC
            LIMIT ?
        """,
            (f"{prefix}%", limit),
        )
        rows = cursor.fetchall()
        conn.close()

        session_ids: list[str] = []
        for row in rows:
            session_id = row[0]
            if isinstance(session_id, str) and session_id.strip():
                session_ids.append(session_id.strip())

        return session_ids

    def get_recent_conversation_messages_for_prefix(
        self,
        session_prefix: str,
        limit: int = 20,
        session_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Return recent conversation messages for a session prefix or explicit session."""
        prefix = str(session_prefix or "").strip()
        if not prefix:
            return []

        limit = max(1, int(limit))
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if session_id:
            cursor.execute(
                """
                SELECT id, session_id, role, content, metadata, created_at
                FROM conversations
                WHERE session_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
            """,
                (session_id, limit),
            )
        else:
            cursor.execute(
                """
                SELECT id, session_id, role, content, metadata, created_at
                FROM conversations
                WHERE session_id LIKE ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
            """,
                (f"{prefix}%", limit),
            )

        rows = cursor.fetchall()
        conn.close()

        messages: list[dict[str, Any]] = []
        for row in rows:
            messages.append(
                {
                    "id": row[0],
                    "session_id": row[1],
                    "role": row[2],
                    "content": row[3],
                    "metadata": json.loads(row[4] or "{}"),
                    "created_at": row[5],
                }
            )

        return messages

    def delete_memory(self, memory_id: int) -> bool:
        """Delete a memory by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        deleted = cursor.rowcount > 0

        conn.commit()
        conn.close()

        return deleted

    def clear_all_memories(self) -> int:
        """Clear all memories."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM memories")
        count = cursor.fetchone()[0]

        cursor.execute("DELETE FROM memories")

        conn.commit()
        conn.close()

        return count

    def get_memory_stats(self) -> dict[str, Any]:
        """Get statistics about the memory store."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM memories")
        total_memories = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL")
        embedded_memories = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM conversations")
        total_conversations = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM topics")
        total_topics = cursor.fetchone()[0]

        conn.close()

        return {
            "total_memories": total_memories,
            "embedded_memories": embedded_memories,
            "total_conversations": total_conversations,
            "total_topics": total_topics,
            "db_path": self.db_path,
        }

    def get_conversation_message_count(self, session_id: Optional[str] = None) -> int:
        """Return total conversation messages, optionally scoped to a session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if session_id:
            cursor.execute(
                "SELECT COUNT(*) FROM conversations WHERE session_id = ?", (session_id,)
            )
        else:
            cursor.execute("SELECT COUNT(*) FROM conversations")

        count = cursor.fetchone()[0]
        conn.close()
        return int(count)

    def get_session_memory_cadence(
        self,
        session_id: str,
        memory_types: Optional[set[str]] = None,
    ) -> dict[str, Any]:
        """Return memory cadence stats for a session based on memory metadata."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT metadata FROM memories")
        rows = cursor.fetchall()
        conn.close()

        count = 0
        latest_message_index: Optional[int] = None

        for row in rows:
            metadata = self._parse_metadata_blob(row[0])
            if metadata.get("session_id") != session_id:
                continue

            memory_type = str(metadata.get("type", "")).strip()
            if memory_types and memory_type not in memory_types:
                continue

            count += 1
            message_index = metadata.get("message_index")
            if isinstance(message_index, str) and message_index.isdigit():
                message_index = int(message_index)

            if isinstance(message_index, int):
                if latest_message_index is None or message_index > latest_message_index:
                    latest_message_index = message_index

        return {
            "memory_count": count,
            "latest_message_index": latest_message_index,
        }

    def get_conversation_activity_by_hour(
        self,
        lookback_days: int = 21,
    ) -> dict[str, Any]:
        """Return hourly conversation activity across recent history."""
        lookback_days = max(1, int(lookback_days))
        cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT created_at FROM conversations WHERE created_at >= ?", (cutoff,)
        )
        rows = cursor.fetchall()
        conn.close()

        hour_counts = {hour: 0 for hour in range(24)}
        distinct_days: set[str] = set()

        for (created_at_raw,) in rows:
            parsed = self._parse_timestamp(created_at_raw)
            if not parsed:
                continue
            hour_counts[parsed.hour] += 1
            distinct_days.add(parsed.date().isoformat())

        return {
            "lookback_days": lookback_days,
            "message_count": len(rows),
            "distinct_days": len(distinct_days),
            "hour_counts": hour_counts,
        }

    def infer_offpeak_hours(
        self,
        lookback_days: int = 21,
        min_days: int = 14,
        window_hours: int = 6,
    ) -> dict[str, Any]:
        """Infer a contiguous low-traffic hour window for dream-mode maintenance."""
        stats = self.get_conversation_activity_by_hour(lookback_days=lookback_days)
        min_days = max(1, int(min_days))
        window_hours = max(2, min(12, int(window_hours)))

        if stats["distinct_days"] < min_days:
            return {
                "hours": [],
                "reason": "insufficient_history",
                "stats": stats,
            }

        hour_counts = stats["hour_counts"]
        best_start = 0
        best_score: Optional[int] = None

        for start in range(24):
            window = [(start + offset) % 24 for offset in range(window_hours)]
            score = sum(hour_counts[hour] for hour in window)
            if best_score is None or score < best_score:
                best_score = score
                best_start = start

        selected_hours = sorted(
            [(best_start + offset) % 24 for offset in range(window_hours)]
        )

        return {
            "hours": selected_hours,
            "reason": "ok",
            "window_score": best_score if best_score is not None else 0,
            "stats": stats,
        }

    def set_agent_state(self, key: str, value: Any):
        """Persist a small JSON-serializable state value."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO agent_state (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key)
            DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP
        """,
            (key, json.dumps(value)),
        )
        conn.commit()
        conn.close()

    def get_agent_state(self, key: str, default: Any = None) -> Any:
        """Retrieve a previously persisted JSON state value."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM agent_state WHERE key = ?", (key,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return default

        try:
            return json.loads(row[0])
        except Exception:
            return default

    def replace_reminder_tasks(self, tasks: list[dict[str, Any]]) -> None:
        """Replace all persisted reminder tasks with the provided snapshot."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM reminders")

        for task in tasks:
            if not isinstance(task, dict):
                continue

            cursor.execute(
                """
                INSERT INTO reminders (
                    task_id,
                    name,
                    cron,
                    message,
                    session_id,
                    one_off,
                    run_ai,
                    ai_prompt,
                    status,
                    enabled,
                    run_count,
                    last_run_at,
                    next_run_at,
                    last_result,
                    last_error,
                    created_at,
                    updated_at,
                    max_runs
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    str(task.get("task_id", "")).strip(),
                    str(task.get("name", "")),
                    str(task.get("cron", "")).strip(),
                    str(task.get("message", "")),
                    str(task.get("session_id", "")).strip() or None,
                    1 if bool(task.get("one_off", False)) else 0,
                    1 if bool(task.get("run_ai", False)) else 0,
                    str(task.get("ai_prompt", "")),
                    str(task.get("status", "active")),
                    1 if bool(task.get("enabled", False)) else 0,
                    int(task.get("run_count", 0) or 0),
                    task.get("last_run_at"),
                    task.get("next_run_at"),
                    str(task.get("last_result", "")),
                    str(task.get("last_error", "")),
                    str(task.get("created_at", "")),
                    str(task.get("updated_at", "")),
                    int(task.get("max_runs", 0) or 0),
                ),
            )

        conn.commit()
        conn.close()

    def get_reminder_tasks(self) -> list[dict[str, Any]]:
        """Return all persisted reminder tasks from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                task_id,
                name,
                cron,
                message,
                session_id,
                one_off,
                run_ai,
                ai_prompt,
                status,
                enabled,
                run_count,
                last_run_at,
                next_run_at,
                last_result,
                last_error,
                created_at,
                updated_at,
                max_runs
            FROM reminders
            ORDER BY created_at ASC
        """
        )
        rows = cursor.fetchall()
        conn.close()

        reminders: list[dict[str, Any]] = []
        for row in rows:
            reminders.append(
                {
                    "task_id": row[0],
                    "name": row[1] or "",
                    "cron": row[2] or "",
                    "message": row[3] or "",
                    "session_id": row[4],
                    "one_off": bool(row[5]),
                    "run_ai": bool(row[6]),
                    "ai_prompt": row[7] or "",
                    "status": row[8] or "active",
                    "enabled": bool(row[9]),
                    "run_count": int(row[10] or 0),
                    "last_run_at": row[11],
                    "next_run_at": row[12],
                    "last_result": row[13] or "",
                    "last_error": row[14] or "",
                    "created_at": row[15],
                    "updated_at": row[16],
                    "max_runs": int(row[17] or 0),
                }
            )

        return reminders

    def get_memories_for_consolidation(
        self,
        limit: int = 24,
        min_age_hours: int = 24,
    ) -> list[Memory]:
        """Return memory candidates eligible for dream consolidation."""
        limit = max(1, int(limit))
        min_age_hours = max(1, int(min_age_hours))
        cutoff = (datetime.now() - timedelta(hours=min_age_hours)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, content, embedding, metadata, created_at, updated_at
            FROM memories
            WHERE created_at <= ?
            ORDER BY created_at DESC
            LIMIT ?
        """,
            (cutoff, limit * 3),
        )
        rows = cursor.fetchall()
        conn.close()

        disallowed_types = {"system_prompt", "long_term_memory"}
        candidates: list[Memory] = []

        for row in rows:
            (
                memory_id,
                content,
                embedding_bytes,
                metadata_blob,
                created_at,
                updated_at,
            ) = row
            metadata = self._parse_metadata_blob(metadata_blob)
            memory_type = str(metadata.get("type", "")).strip()

            if memory_type in disallowed_types:
                continue
            if metadata.get("dream_consolidated"):
                continue

            embedding = (
                self._bytes_to_embedding(embedding_bytes) if embedding_bytes else None
            )

            candidates.append(
                Memory(
                    id=memory_id,
                    content=content,
                    embedding=embedding,
                    metadata=metadata,
                    created_at=created_at,
                    updated_at=updated_at,
                )
            )

            if len(candidates) >= limit:
                break

        return candidates

    def mark_memories_dream_consolidated(
        self,
        memory_ids: list[int],
        significance: Optional[float] = None,
    ) -> int:
        """Mark source memories as consolidated by the dream cycle."""
        unique_ids = sorted({int(m_id) for m_id in memory_ids if int(m_id) > 0})
        if not unique_ids:
            return 0

        consolidated_at = datetime.now().isoformat(timespec="seconds")
        clamped_significance = None
        if significance is not None:
            clamped_significance = max(0.0, min(1.0, float(significance)))

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        updated = 0

        for memory_id in unique_ids:
            cursor.execute("SELECT metadata FROM memories WHERE id = ?", (memory_id,))
            row = cursor.fetchone()
            if not row:
                continue

            metadata = self._parse_metadata_blob(row[0])
            metadata["dream_consolidated"] = True
            metadata["dream_consolidated_at"] = consolidated_at
            if clamped_significance is not None:
                metadata["dream_significance"] = clamped_significance

            cursor.execute(
                """
                UPDATE memories
                SET metadata = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """,
                (json.dumps(metadata), memory_id),
            )
            if cursor.rowcount:
                updated += 1

        conn.commit()
        conn.close()
        return updated

    def get_system_prompt(self) -> Optional[str]:
        """Get the custom system prompt."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT content FROM memories
            WHERE metadata LIKE '%"type": "system_prompt"%'
            ORDER BY created_at DESC
            LIMIT 1
        """
        )

        row = cursor.fetchone()
        conn.close()

        return row[0] if row else None

    def set_system_prompt(self, prompt: str) -> int:
        """Store a custom system prompt."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            DELETE FROM memories
            WHERE metadata LIKE '%"type": "system_prompt"%'
        """
        )

        cursor.execute(
            """
            INSERT INTO memories (content, embedding, metadata, updated_at)
            VALUES (?, NULL, ?, CURRENT_TIMESTAMP)
        """,
            (prompt, json.dumps({"type": "system_prompt"})),
        )

        memory_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return memory_id

    def clear_conversation_history(self, session_id: Optional[str] = None) -> int:
        """Clear conversation history for a session (or all if no session specified).

        Args:
            session_id: Session ID to clear, or None to clear all conversations

        Returns:
            Number of conversations deleted
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if session_id:
            cursor.execute(
                "SELECT COUNT(*) FROM conversations WHERE session_id = ?", (session_id,)
            )
            count = cursor.fetchone()[0]
            cursor.execute(
                "DELETE FROM conversations WHERE session_id = ?", (session_id,)
            )
        else:
            cursor.execute("SELECT COUNT(*) FROM conversations")
            count = cursor.fetchone()[0]
            cursor.execute("DELETE FROM conversations")

        conn.commit()
        conn.close()

        return count

    def clear_system_prompt(self) -> bool:
        """Clear the custom system prompt."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            DELETE FROM memories
            WHERE metadata LIKE '%"type": "system_prompt"%'
        """
        )

        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()

        return deleted


@dataclass
class WeightedMemory:
    """A memory with its calculated relevance score."""

    memory: Memory
    similarity: float
    success_score: float
    recency_score: float
    final_score: float


class EnhancedMemoryStore(MemoryStore):
    """Memory store with learned retrieval weights."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_enhanced_tables()

    def _init_enhanced_tables(self):
        """Initialize additional tables for tracking memory outcomes."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER,
                outcome_type TEXT,
                outcome_score REAL,
                context TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_access_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER,
                access_type TEXT,
                query TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
            )
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memory_outcomes_memory
            ON memory_outcomes(memory_id)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memory_access_memory
            ON memory_access_log(memory_id)
        """
        )

        conn.commit()
        conn.close()

    def record_memory_outcome(
        self,
        memory_id: int,
        outcome_type: str,
        outcome_score: Optional[float] = None,
        context: Optional[dict] = None,
    ):
        """Record the outcome of using a memory."""
        if outcome_score is None:
            outcome_score = 1.0 if outcome_type == "success" else 0.0

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO memory_outcomes (memory_id, outcome_type, outcome_score, context) VALUES (?, ?, ?, ?)",
            (memory_id, outcome_type, outcome_score, json.dumps(context or {})),
        )

        conn.commit()
        conn.close()

    def get_memory_success_score(self, memory_id: int) -> float:
        """Calculate the success score for a memory."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT outcome_score, COUNT(*) FROM memory_outcomes WHERE memory_id = ? GROUP BY outcome_score",
            (memory_id,),
        )

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return 0.5

        total_weight = 0
        weighted_sum = 0
        for score, count in rows:
            weighted_sum += score * count
            total_weight += count

        return weighted_sum / total_weight if total_weight > 0 else 0.5

    def log_memory_access(
        self,
        memory_id: int,
        access_type: str = "retrieval",
        query: Optional[str] = None,
    ):
        """Log that a memory was accessed."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO memory_access_log (memory_id, access_type, query) VALUES (?, ?, ?)",
            (memory_id, access_type, query),
        )
        conn.commit()
        conn.close()

    def get_memory_recency_score(self, memory_id: int) -> float:
        """Calculate recency score for a memory."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT MAX(created_at) FROM memory_access_log WHERE memory_id = ?",
            (memory_id,),
        )
        row = cursor.fetchone()
        conn.close()

        if not row or not row[0]:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT created_at FROM memories WHERE id = ?", (memory_id,))
            row = cursor.fetchone()
            conn.close()
            if not row or not row[0]:
                return 0.5

        last_access = (
            datetime.fromisoformat(row[0].replace(" ", "T"))
            if " " in row[0]
            else datetime.fromisoformat(row[0])
        )
        age_hours = (datetime.now() - last_access).total_seconds() / 3600
        return math.exp(-age_hours / 24)

    async def search_memories_weighted(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.1,
        topic: Optional[str] = None,
        similarity_weight: float = 0.5,
        success_weight: float = 0.3,
        recency_weight: float = 0.2,
    ) -> list[WeightedMemory]:
        """Search memories using learned weighted scoring."""
        base_results = await self.search_memories(
            query=query, top_k=top_k * 3, threshold=threshold, topic=topic
        )
        weighted_results = []

        for memory, similarity in base_results:
            success_score = self.get_memory_success_score(memory.id)
            recency_score = self.get_memory_recency_score(memory.id)
            final_score = (
                similarity_weight * similarity
                + success_weight * success_score
                + recency_weight * recency_score
            )

            weighted_results.append(
                WeightedMemory(
                    memory=memory,
                    similarity=similarity,
                    success_score=success_score,
                    recency_score=recency_score,
                    final_score=final_score,
                )
            )
            self.log_memory_access(memory.id, "retrieval", query)

        weighted_results.sort(key=lambda x: x.final_score, reverse=True)
        return weighted_results[:top_k]
