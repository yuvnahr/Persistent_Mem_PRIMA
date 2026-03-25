"""
linker.py

Memory linking module for PRIMA.

Responsible for creating semantic links between MemoryNotes
based on LLM analysis of relationships.

Implements the A-MEM linking stage (Ps2).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from prima_memory.core.memory_store import MemoryStore
from prima_memory.core.note import MemoryNote
from prima_memory.llm.llm_service import LLMService


class MemoryLinker:
    """
    Creates semantic links between memory notes using LLM analysis.
    """

    def __init__(
        self,
        store: MemoryStore,
        llm_service: Optional[LLMService] = None,
        similarity_threshold: float = 0.5,
    ) -> None:
        self.store = store
        self.llm_service = llm_service
        self.similarity_threshold = similarity_threshold

    def link(
        self,
        source: MemoryNote,
        retrieved: List[Tuple[MemoryNote, float]],
    ) -> None:
        """
        Create links between a source memory and retrieved memories using LLM.

        Implements A-MEM Ps2: Use LLM to analyze relationships and decide links.
        """
        if not retrieved:
            return

        # Fallback for no LLM: link based on similarity threshold
        if self.llm_service is None:
            for mem, score in retrieved:
                if mem is None or mem.id == source.id:
                    continue
                if score >= self.similarity_threshold:
                    self._create_link(source, mem, score)
            return

        # Extract the memory notes from retrieved tuples
        nearest_memories = [mem for mem, _ in retrieved]

        # Use LLM to decide which memories to link
        memories_to_link = self.llm_service.decide_links(source, nearest_memories)

        # Create links for decided memories
        for memory_id in memories_to_link:
            # Find the corresponding memory and similarity score
            target_memory = None
            similarity_score = 0.0

            for mem, score in retrieved:
                if mem.id == memory_id:
                    target_memory = mem
                    similarity_score = score
                    break

            if target_memory:
                self._create_link(source, target_memory, similarity_score)

    def _create_link(
        self,
        source: MemoryNote,
        target: MemoryNote,
        strength: float,
    ) -> None:
        """
        Create a bidirectional link between two memories.
        """
        relation_type = "related"  # Could be refined based on LLM analysis

        # Create link in both directions
        insert_link = getattr(self.store, "insert_link", None)

        if callable(insert_link):
            # Source -> Target
            insert_link(
                source_id=source.id,
                target_id=target.id,
                relation_type=relation_type,
                strength=strength,
            )
            # Target -> Source (bidirectional)
            insert_link(
                source_id=target.id,
                target_id=source.id,
                relation_type=relation_type,
                strength=strength,
            )
