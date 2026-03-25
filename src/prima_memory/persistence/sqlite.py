"""
sqlite.py

SQLite persistence layer for PRIMA memory notes.
Handles storage and retrieval of MemoryNote objects.
"""

import json
import pickle
import sqlite3
from pathlib import Path
from typing import List, Optional

from prima_memory.core.note import MemoryNote

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "memory.db"
SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class SQLiteMemoryStore:
    """
    Persistence backend for MemoryNote using SQLite.
    Automatically initializes storage directory and schema.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH

        # Ensure data directory exists (CI-safe)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure schema exists
        self._init_db()

    # -----------------------------
    # Internal helpers
    # -----------------------------

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        """
        Initialize database schema if not already present.
        """
        if not SCHEMA_PATH.exists():
            raise FileNotFoundError(f"Missing schema.sql at {SCHEMA_PATH}")

        conn = sqlite3.connect(self.db_path)
        try:
            with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
                conn.executescript(f.read())
            conn.commit()

            # Clear any existing data for deterministic behavior (especially tests)
            conn.execute("DELETE FROM memory_notes")
            conn.execute("DELETE FROM memory_links")
            conn.execute("DELETE FROM memory_evolution")
            conn.commit()
        finally:
            conn.close()

    # -----------------------------
    # MemoryNote operations
    # -----------------------------

    def insert_note(self, note: MemoryNote) -> None:
        """
        Insert a new MemoryNote into the database.
        """
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO memory_notes (
                    id,
                    content,
                    created_at,
                    last_accessed,
                    retrieval_count,
                    context,
                    keywords,
                    tags,
                    embedding
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    note.id,
                    note.content,
                    note.timestamp,
                    note.last_accessed,
                    note.retrieval_count,
                    note.context,
                    json.dumps(note.keywords),
                    json.dumps(note.tags),
                    (
                        pickle.dumps(note.embedding)
                        if note.embedding is not None
                        else None
                    ),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_note(self, note_id: str) -> Optional[MemoryNote]:
        """
        Retrieve a MemoryNote by ID.
        Automatically updates access metadata.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM memory_notes WHERE id = ?",
                (note_id,),
            ).fetchone()

            if row is None:
                return None

            note = self._row_to_note(row)

            # Update access metadata
            note.mark_accessed()
            self._update_access_metadata(note, conn)

            return note
        finally:
            conn.close()

    def list_notes(self) -> List[MemoryNote]:
        """
        Retrieve all memory notes.
        """
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM memory_notes").fetchall()
            return [self._row_to_note(row) for row in rows]
        finally:
            conn.close()

    # -----------------------------
    # Links
    # -----------------------------

    def add_link(
        self,
        source_id: str,
        target_id: str,
        relation_type: str = "related",
        strength: Optional[float] = None,
    ) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_links
                (source_id, target_id, relation_type, strength)
                VALUES (?, ?, ?, ?)
                """,
                (source_id, target_id, relation_type, strength),
            )
            conn.commit()
        finally:
            conn.close()

    def get_links(self, note_id: str):
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT target_id, relation_type, strength
                FROM memory_links
                WHERE source_id = ?
                """,
                (note_id,),
            ).fetchall()

            return {
                target_id: {
                    "type": relation_type,
                    "strength": strength,
                }
                for target_id, relation_type, strength in rows
            }
        finally:
            conn.close()

    # -----------------------------
    # Evolution history
    # -----------------------------

    def record_evolution(self, note: MemoryNote, action: str, details: dict) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO memory_evolution
                (memory_id, timestamp, action, details)
                VALUES (?, ?, ?, ?)
                """,
                (
                    note.id,
                    note.last_accessed,
                    action,
                    json.dumps(details),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    # -----------------------------
    # Internal mapping
    # -----------------------------

    def _row_to_note(self, row) -> MemoryNote:
        (
            note_id,
            content,
            created_at,
            last_accessed,
            retrieval_count,
            context,
            keywords,
            tags,
            embedding,
        ) = row

        note = MemoryNote(
            content=content,
            note_id=note_id,
            timestamp=created_at,
            last_accessed=last_accessed,
            retrieval_count=retrieval_count,
            context=context,
            keywords=json.loads(keywords) if keywords else [],
            tags=json.loads(tags) if tags else [],
            embedding=pickle.loads(embedding) if embedding else None,
        )

        # Load links
        note.links = self.get_links(note.id)

        return note

    def _update_access_metadata(
        self, note: MemoryNote, conn: sqlite3.Connection
    ) -> None:
        conn.execute(
            """
            UPDATE memory_notes
            SET last_accessed = ?, retrieval_count = ?
            WHERE id = ?
            """,
            (note.last_accessed, note.retrieval_count, note.id),
        )
        conn.commit()
