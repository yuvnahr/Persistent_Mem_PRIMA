"""
evolution.py

Memory evolution module for PRIMA.

Responsible for refining existing MemoryNotes based on
new information and repeated access patterns.

Implements the A-MEM evolution stage (Ps3).
"""

from typing import List

from prima_memory.core.note import MemoryNote
from prima_memory.persistence.sqlite import SQLiteMemoryStore


class MemoryEvolver:
    """
    Evolves memory notes by refining semantic metadata.
    """

    def __init__(
        self,
        store: SQLiteMemoryStore,
        min_retrievals: int = 2,
    ):
        """
        Args:
            store (SQLiteMemoryStore):
                Persistence backend.
            min_retrievals (int):
                Minimum retrieval count before a memory is eligible
                for evolution.
        """
        self.store = store
        self.min_retrievals = min_retrievals

    # -----------------------------
    # Public API
    # -----------------------------

    def evolve(
        self,
        source: MemoryNote,
        related: List[MemoryNote],
    ) -> None:
        """
        Evolve related memories using a new source memory.

        Args:
            source (MemoryNote):
                Newly added or focused memory.
            related (List[MemoryNote]):
                Semantically related existing memories.
        """

        for target in related:
            if target.id == source.id:
                continue

            if not self._should_evolve(target):
                continue

            self._evolve_single(target, source)

    # -----------------------------
    # Decision logic
    # -----------------------------

    def _should_evolve(self, note: MemoryNote) -> bool:
        """
        Decide whether a memory is eligible for evolution.
        """

        return note.retrieval_count >= self.min_retrievals

    # -----------------------------
    # Evolution actions
    # -----------------------------

    def _evolve_single(
        self,
        target: MemoryNote,
        source: MemoryNote,
    ) -> None:
        """
        Apply semantic evolution to a single memory.
        """

        changes = {}

        # 1. Merge tags
        new_tags = sorted(set(target.tags) | set(source.tags))
        if new_tags != target.tags:
            changes["tags"] = {
                "old": target.tags,
                "new": new_tags,
            }
            target.tags = new_tags

        # 2. Refine context (simple concatenation heuristic)
        if source.context and source.context not in (target.context or ""):
            old_context = target.context
            target.context = (
                f"{target.context}; {source.context}"
                if target.context
                else source.context
            )
            changes["context"] = {
                "old": old_context,
                "new": target.context,
            }

        # 3. Persist changes
        if changes:
            self.store.record_evolution(
                note=target,
                action="semantic_refinement",
                details=changes,
            )

            # Persist updated fields
            self._persist_updates(target)

    # -----------------------------
    # Persistence helpers
    # -----------------------------

    def _persist_updates(self, note: MemoryNote) -> None:
        """
        Write updated semantic fields back to SQLite.
        """
        import json
        import sqlite3

        conn = sqlite3.connect(self.store.db_path)
        try:
            conn.execute(
                """
                UPDATE memory_notes
                SET context = ?, tags = ?
                WHERE id = ?
                """,
                (
                    note.context,
                    json.dumps(note.tags),
                    note.id,
                ),
            )
            conn.commit()
        finally:
            conn.close()
