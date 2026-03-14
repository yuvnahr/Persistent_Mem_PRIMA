"""
evolution.py

Memory evolution module for PRIMA.

Responsible for refining existing MemoryNotes based on
new information and repeated access patterns.

Implements the A-MEM evolution stage (Ps3).
"""

from __future__ import annotations

from typing import Any, Dict, List

from prima_memory.core.note import MemoryNote
from prima_memory.core.memory_store import MemoryStore


class MemoryEvolver:
    """
    Evolves memory notes by refining semantic metadata.
    """

    def __init__(
        self,
        store: MemoryStore,
        min_retrievals: int = 2,
    ) -> None:
        """
        Args:
            store:
                Persistence backend.
            min_retrievals:
                Minimum retrieval count before a memory
                becomes eligible for evolution.
        """
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
        """
        Evolve related memories using a new source memory.

        Args:
            source:
                Newly added or focused memory.
            related:
                Semantically related existing memories.
        """

        for target in related:
            if target.id == source.id:
                continue

            if not self._should_evolve(target):
                continue

            self._evolve_single(target, source)

    # --------------------------------------------------
    # Decision logic
    # --------------------------------------------------

    def _should_evolve(self, note: MemoryNote) -> bool:
        """
        Decide whether a memory is eligible for evolution.
        """

        return note.retrieval_count >= self.min_retrievals

    # --------------------------------------------------
    # Evolution actions
    # --------------------------------------------------

    def _evolve_single(
        self,
        target: MemoryNote,
        source: MemoryNote,
    ) -> None:
        """
        Apply semantic evolution to a single memory.
        """

        changes: Dict[str, Dict[str, Any]] = {}

        # -------------------------
        # 1️⃣ Merge tags
        # -------------------------

        new_tags = sorted(set(target.tags) | set(source.tags))

        if new_tags != target.tags:
            changes["tags"] = {
                "old": target.tags,
                "new": new_tags,
            }
            target.tags = new_tags

        # -------------------------
        # 2️⃣ Merge keywords
        # -------------------------

        new_keywords = sorted(set(target.keywords) | set(source.keywords))

        if new_keywords != target.keywords:
            changes["keywords"] = {
                "old": target.keywords,
                "new": new_keywords,
            }
            target.keywords = new_keywords

        # -------------------------
        # 3️⃣ Context refinement
        # -------------------------

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

        # -------------------------
        # 4️⃣ Persist changes
        # -------------------------

        if not changes:
            return

        # Update memory record
        self.store.update_memory(
            memory_id=target.id,
            context=target.context,
            keywords=target.keywords,
            tags=target.tags,
        )

        # Log evolution event
        self.store.log_evolution(
            memory_id=target.id,
            timestamp=source.created_at,
            action="semantic_refinement",
            details=changes,
        )