"""
retriever.py

Memory retrieval module for PRIMA.

Bridges semantic similarity search (FAISS embeddings) with persistent
memory storage (SQLite). Responsible only for recalling relevant memories.

This module does NOT:
- create memories
- link memories
- evolve memories
"""

from typing import List

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
    ):
        """
        Args:
            store (SQLiteMemoryStore):
                Persistent memory backend.
            embedder (EmbeddingIndex):
                FAISS-based embedding index.
        """
        self.store = store
        self.embedder = embedder

    # -----------------------------
    # Public API
    # -----------------------------

    def retrieve(self, query: str, top_k: int = 5) -> List[MemoryNote]:
        """
        Retrieve the top-k most relevant memories for a query.

        Args:
            query (str):
                Natural language query.
            top_k (int):
                Number of memories to retrieve.

        Returns:
            List[MemoryNote]:
                Ordered list of retrieved memory notes.
        """

        # 1. Embed query
        query_embedding = self.embedder.embed_text(query)

        # 2. Search vector index
        search_results = self.embedder.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        if not search_results:
            return []

        # 3. Load MemoryNotes from SQLite (preserve order)
        memories: List[MemoryNote] = []

        for memory_id, _score in search_results:
            note = self.store.get_note(memory_id)
            if note is not None:
                memories.append(note)

        return memories
