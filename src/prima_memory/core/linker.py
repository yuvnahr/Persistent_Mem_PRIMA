"""
linker.py

Memory linking module for PRIMA.

Responsible for creating semantic links between MemoryNotes
based on similarity and metadata overlap.

Implements the A-MEM linking stage (Ps2).
"""

from __future__ import annotations

from typing import List, Tuple

from prima_memory.core.note import MemoryNote
from prima_memory.core.memory_store import MemoryStore


class MemoryLinker:
    """
    Creates semantic links between memory notes.
    """

    def __init__(
        self,
        store: MemoryStore,
        similarity_threshold: float = 0.6,
    ) -> None:
        """
        Args:
            store:
                Persistence backend.
            similarity_threshold:
                Minimum similarity required to create a link.
        """
        self.store = store
        self.similarity_threshold = similarity_threshold

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def link(
        self,
        source: MemoryNote,
        retrieved: List[Tuple[MemoryNote, float]],
    ) -> None:
        """
        Create links between a source memory and retrieved memories.

        Args:
            source:
                Newly created or active memory.
            retrieved:
                Candidate memories with similarity scores.
        """

        for target, score in retrieved:
            if target.id == source.id:
                continue

            if not self._should_link(source, target, score):
                continue

            relation_type = self._infer_relation_type(source, target)

            # Normalize score for safety
            strength = max(0.0, min(score, 1.0))

            self.store.insert_link(
                source_id=source.id,
                target_id=target.id,
                relation_type=relation_type,
                strength=strength,
            )

    # --------------------------------------------------
    # Decision logic
    # --------------------------------------------------

    def _should_link(
        self,
        source: MemoryNote,
        target: MemoryNote,
        similarity_score: float,
    ) -> bool:
        """
        Determine if two memories should be linked.
        """

        # Similarity gate
        if similarity_score < self.similarity_threshold:
            return False

        # Tag overlap
        if set(source.tags) & set(target.tags):
            return True

        # Keyword overlap
        if set(source.keywords) & set(target.keywords):
            return True

        # Context heuristic
        if source.context and target.context:
            s_ctx = source.context.lower()
            t_ctx = target.context.lower()

            if s_ctx in t_ctx or t_ctx in s_ctx:
                return True

        return False

    def _infer_relation_type(
        self,
        source: MemoryNote,
        target: MemoryNote,
    ) -> str:
        """
        Infer relationship type between memories.

        Phase-2 currently uses heuristic inference.
        Later this will be replaced with LLM reasoning.
        """

        if set(source.tags) & set(target.tags):
            return "related"

        if set(source.keywords) & set(target.keywords):
            return "similar"

        if source.context and target.context:
            return "contextual"

        return "associated"