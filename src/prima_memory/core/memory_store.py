"""
memory_store.py

Central database interface for PRIMA persistent memory.
All database operations must go through this module.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = PROJECT_ROOT / "data" / "memory.db"


class MemoryStore:
    """Interface for interacting with PRIMA memory database."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or DB_PATH

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    # ---------------------------------------------------------
    # Memory Notes
    # ---------------------------------------------------------

    def insert_memory(
        self,
        memory_id: str,
        content: str,
        created_at: str,
        context: Optional[str],
        keywords: List[str],
        tags: List[str],
        embedding: bytes,
    ) -> None:
        """Insert a new memory note."""

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_notes
                (id, content, created_at, last_accessed, retrieval_count,
                 context, keywords, tags, embedding)
                VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    content,
                    created_at,
                    created_at,
                    context,
                    json.dumps(keywords),
                    json.dumps(tags),
                    embedding,
                ),
            )

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a memory note."""

        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM memory_notes WHERE id = ?",
                (memory_id,),
            ).fetchone()

        if row is None:
            return None

        return dict(row)

    def update_memory(
        self,
        memory_id: str,
        context: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Update semantic fields of a memory."""

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE memory_notes
                SET context = COALESCE(?, context),
                    keywords = COALESCE(?, keywords),
                    tags = COALESCE(?, tags)
                WHERE id = ?
                """,
                (
                    context,
                    json.dumps(keywords) if keywords else None,
                    json.dumps(tags) if tags else None,
                    memory_id,
                ),
            )

    # ---------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------

    def get_all_embeddings(self) -> List[Tuple[str, bytes]]:
        """Return all memory embeddings."""

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, embedding FROM memory_notes"
            ).fetchall()

        return [(row["id"], row["embedding"]) for row in rows]

    def update_access_stats(self, memory_id: str, timestamp: str) -> None:
        """Update retrieval statistics."""

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE memory_notes
                SET last_accessed = ?, retrieval_count = retrieval_count + 1
                WHERE id = ?
                """,
                (timestamp, memory_id),
            )

    # ---------------------------------------------------------
    # Links
    # ---------------------------------------------------------

    def insert_link(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        strength: float,
    ) -> None:
        """Insert semantic relationship between memories."""

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_links
                (source_id, target_id, relation_type, strength)
                VALUES (?, ?, ?, ?)
                """,
                (source_id, target_id, relation_type, strength),
            )

    def get_links(self, memory_id: str) -> List[Dict[str, Any]]:
        """Retrieve all links connected to a memory."""

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM memory_links
                WHERE source_id = ? OR target_id = ?
                """,
                (memory_id, memory_id),
            ).fetchall()

        return [dict(row) for row in rows]

    # ---------------------------------------------------------
    # Evolution
    # ---------------------------------------------------------

    def log_evolution(
        self,
        memory_id: str,
        timestamp: str,
        action: str,
        details: Dict[str, Any],
    ) -> None:
        """Record memory evolution event."""

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_evolution
                (memory_id, timestamp, action, details)
                VALUES (?, ?, ?, ?)
                """,
                (memory_id, timestamp, action, json.dumps(details)),
            )