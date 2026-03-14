"""
retriever.py

Memory retrieval module for PRIMA.

Bridges semantic similarity search (FAISS embeddings) with persistent
memory storage (SQLite).

Phase-2 upgrade:
- supports linked memory expansion
- enables memory graph retrieval
"""

from __future__ import annotations

from typing import List, Set

from prima_memory.core.embedding import EmbeddingIndex
from prima_memory.core.note import MemoryNote
from prima_memory.persistence.sqlite import SQLiteMemoryStore


class MemoryRetriever:
    """
    Retrieves relevant MemoryNote objects given a query.
    """

    def __init__(
        self,
        store: SQLiteMemoryStore,
        embedder: EmbeddingIndex,
    ) -> None:
        self.store = store
        self.embedder = embedder

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        expand_links: bool = False,
    ) -> List[MemoryNote]:
        """
        Retrieve the most relevant memories.

        Args:
            query:
                Natural language query.
            top_k:
                Number of semantic matches.
            expand_links:
                If True, also retrieve linked memories.

        Returns:
            Ordered list of MemoryNote objects.
        """

        # 1️⃣ Embed query
        query_embedding = self.embedder.embed_text(query)

        # 2️⃣ Semantic search
        search_results = self.embedder.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        if not search_results:
            return []

        memories: List[MemoryNote] = []
        seen: Set[str] = set()

        # 3️⃣ Load top-k semantic matches
        for memory_id, _score in search_results:
            note = self.store.get_note(memory_id)

            if note is None:
                continue

            memories.append(note)
            seen.add(note.id)

            # 4️⃣ Phase-2: expand semantic links
            if expand_links:
                linked = self._retrieve_linked(note.id)

                for linked_note in linked:
                    if linked_note.id not in seen:
                        memories.append(linked_note)
                        seen.add(linked_note.id)

        return memories

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------

    def _retrieve_linked(self, memory_id: str) -> List[MemoryNote]:
        """
        Retrieve memories linked to a given memory.

        Used for graph expansion during Phase-2 retrieval.
        """

        links = self.store.get_links(memory_id)

        linked_notes: List[MemoryNote] = []

        for link in links:
            other_id = (
                link["target_id"]
                if link["source_id"] == memory_id
                else link["source_id"]
            )

            note = self.store.get_note(other_id)

            if note is not None:
                linked_notes.append(note)

        return linked_notes
