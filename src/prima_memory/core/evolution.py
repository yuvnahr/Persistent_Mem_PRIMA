"""
evolution.py

Memory evolution module for PRIMA.

Responsible for refining existing MemoryNotes based on
new information and repeated access patterns.

Implements the A-MEM evolution stage (Ps3).
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, List

from prima_memory.core.memory_store import MemoryStore
from prima_memory.core.note import MemoryNote


class MemoryEvolver:
    """
    Evolves memory notes by refining semantic metadata.
    """

    def __init__(
        self,
        store: MemoryStore,
        min_retrievals: int = 2,
    ) -> None:
        self.store = store
        self.min_retrievals = min_retrievals

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def evolve(
        self,
        source: MemoryNote,
        related: List[MemoryNote],
    ) -> None:

        for target in related:
            if target is None:
                continue

            if target.id == source.id:
                continue

            if not self._should_evolve(target):
                continue

            self._evolve_single(target, source)

    # --------------------------------------------------
    # Eligibility
    # --------------------------------------------------

    def _should_evolve(self, note: MemoryNote) -> bool:
        return note.retrieval_count >= self.min_retrievals

    # --------------------------------------------------
    # Evolution logic
    # --------------------------------------------------

    def _evolve_single(
        self,
        target: MemoryNote,
        source: MemoryNote,
    ) -> None:

        changes: Dict[str, Dict[str, Any]] = {}

        # -------------------------
        # Merge tags
        # -------------------------

        new_tags = sorted(set(target.tags) | set(source.tags))

        if new_tags != target.tags:
            changes["tags"] = {"old": target.tags, "new": new_tags}
            target.tags = new_tags

        # -------------------------
        # Merge keywords
        # -------------------------

        new_keywords = sorted(set(target.keywords) | set(source.keywords))

        if new_keywords != target.keywords:
            changes["keywords"] = {"old": target.keywords, "new": new_keywords}
            target.keywords = new_keywords

        # -------------------------
        # Context refinement
        # -------------------------

        if source.context and source.context not in (target.context or ""):
            old_context = target.context

            target.context = (
                f"{target.context}; {source.context}"
                if target.context
                else source.context
            )

            changes["context"] = {"old": old_context, "new": target.context}

        if not changes:
            return

        # --------------------------------------------------
        # Persist changes (MemoryStore API)
        # --------------------------------------------------

        update_memory = getattr(self.store, "update_memory", None)

        if callable(update_memory):
            update_memory(
                memory_id=target.id,
                context=target.context,
                keywords=target.keywords,
                tags=target.tags,
            )

        # --------------------------------------------------
        # SQLiteMemoryStore fallback
        # --------------------------------------------------

        else:

            update_note = getattr(self.store, "update_note", None)

            if callable(update_note):
                update_note(target)

            # Force persistence of semantic fields
            conn = sqlite3.connect(self.store.db_path)

            try:
                conn.execute(
                    """
                    UPDATE memory_notes
                    SET context = ?, tags = ?, keywords = ?
                    WHERE id = ?
                    """,
                    (
                        target.context,
                        json.dumps(target.tags),
                        json.dumps(target.keywords),
                        target.id,
                    ),
                )
                conn.commit()

            finally:
                conn.close()

        # --------------------------------------------------
        # Log evolution event
        # --------------------------------------------------

        timestamp = getattr(source, "created_at", None)

        log_evolution = getattr(self.store, "log_evolution", None)
        record_evolution = getattr(self.store, "record_evolution", None)

        if callable(log_evolution):

            log_evolution(
                memory_id=target.id,
                timestamp=timestamp,
                action="semantic_refinement",
                details=changes,
            )

        elif callable(record_evolution):

            record_evolution(
                note=target,
                action="semantic_refinement",
                details=changes,
            )
