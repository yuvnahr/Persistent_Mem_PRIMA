"""
linker.py

Memory linking module for PRIMA.

Responsible for creating semantic links between MemoryNotes
based on similarity and metadata overlap.

Implements the A-MEM linking stage (Ps2).
"""

from typing import List

from prima_memory.core.note import MemoryNote
from prima_memory.persistence.sqlite import SQLiteMemoryStore


class MemoryLinker:
    """
    Creates links between memory notes.
    """

    def __init__(
        self,
        store: SQLiteMemoryStore,
        similarity_threshold: float = 0.6,
    ):
        """
        Args:
            store (SQLiteMemoryStore):
                Persistence backend for writing links.
            similarity_threshold (float):
                Minimum similarity score to consider linking.
        """
        self.store = store
        self.similarity_threshold = similarity_threshold

    # -----------------------------
    # Public API
    # -----------------------------

    def link(
        self,
        source: MemoryNote,
        retrieved: List[tuple[MemoryNote, float]],
    ) -> None:
        """
        Create links between a source memory and retrieved memories.

        Args:
            source (MemoryNote):
                Newly created or focused memory.
            retrieved (List[(MemoryNote, similarity_score)]):
                Candidate memories with similarity scores.
        """

        for target, score in retrieved:
            if target.id == source.id:
                continue

            if not self._should_link(source, target, score):
                continue

            relation_type = self._infer_relation_type(source, target)

            self.store.add_link(
                source_id=source.id,
                target_id=target.id,
                relation_type=relation_type,
                strength=score,
            )

    # -----------------------------
    # Decision logic
    # -----------------------------

    def _should_link(
        self,
        source: MemoryNote,
        target: MemoryNote,
        similarity_score: float,
    ) -> bool:
        """
        Decide whether two memories should be linked.
        """

        # 1. Similarity gate
        if similarity_score < self.similarity_threshold:
            return False

        # 2. Tag overlap
        if set(source.tags) & set(target.tags):
            return True

        # 3. Context overlap (cheap heuristic)
        if source.context and target.context:
            if source.context.lower() in target.context.lower():
                return True
            if target.context.lower() in source.context.lower():
                return True

        return False

    def _infer_relation_type(
        self,
        source: MemoryNote,
        target: MemoryNote,
    ) -> str:
        """
        Infer relationship type between two memories.

        This is a placeholder for LLM-based reasoning later.
        """

        if set(source.tags) & set(target.tags):
            return "related"

        return "associated"
