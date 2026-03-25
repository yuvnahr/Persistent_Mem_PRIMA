"""
memory_orchestrator.py

Orchestrator for the complete A-MEM pipeline in PRIMA.

Coordinates the four stages:
1. Note Construction (with LLM metadata generation)
2. Link Generation (with LLM relationship analysis)
3. Memory Evolution (with LLM evolution decisions)
4. Retrieval (with linked memory expansion)

This replaces the scattered logic in run_agent.py with a clean, orchestrated approach.
"""

from __future__ import annotations

from typing import List, Optional

from prima_memory.core.embedding import EmbeddingIndex
from prima_memory.core.evolution import MemoryEvolver
from prima_memory.core.linker import MemoryLinker
from prima_memory.core.memory_store import MemoryStore
from prima_memory.core.note import MemoryNote
from prima_memory.core.retriever import MemoryRetriever
from prima_memory.llm.llm_service import LLMService


class MemoryOrchestrator:
    """
    Orchestrates the complete A-MEM memory pipeline.
    """

    def __init__(
        self,
        store: MemoryStore,
        embedder: EmbeddingIndex,
        llm_service: LLMService,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.llm_service = llm_service

        # Initialize components
        self.retriever = MemoryRetriever(store=store, embedder=embedder)
        self.linker = MemoryLinker(store=store, llm_service=llm_service)
        self.evolver = MemoryEvolver(store=store, llm_service=llm_service)

    # --------------------------------------------------
    # Main A-MEM Pipeline
    # --------------------------------------------------

    def add_memory(self, content: str, timestamp: Optional[str] = None) -> str:
        """
        Add a new memory following the complete A-MEM pipeline.

        Args:
            content: Raw memory content
            timestamp: Optional timestamp, defaults to current time

        Returns:
            Memory ID of the created note
        """
        # 1️⃣ Note Construction (A-MEM Ps1)
        note = self._construct_note(content, timestamp)

        # 2️⃣ Get related memories for linking and evolution
        embedding = self.embedder.embed_memory_note(note)
        note.embedding = embedding

        # Find nearest neighbors
        retrieved = self._get_nearest_memories(note, top_k=5)

        # 3️⃣ Link Generation (A-MEM Ps2)
        if retrieved:
            self.linker.link(source=note, retrieved=retrieved)

        # 4️⃣ Memory Evolution (A-MEM Ps3)
        if retrieved:
            related_notes = [mem for mem, _ in retrieved]
            self.evolver.evolve(source=note, related=related_notes)

        # 5️⃣ Persist the new memory
        self._persist_memory(note)

        # 6️⃣ Update embedding index
        self.embedder.add(note.id, note.embedding)

        return note.id

    def retrieve_memories(
        self, query: str, top_k: int = 5, expand_links: bool = True
    ) -> List[MemoryNote]:
        """
        Retrieve relevant memories with optional link expansion.

        Implements A-MEM retrieval with linked memory access.
        """
        return self.retriever.retrieve(
            query=query,
            top_k=top_k,
            expand_links=expand_links,
        )

    # --------------------------------------------------
    # Internal pipeline steps
    # --------------------------------------------------

    def _construct_note(
        self, content: str, timestamp: Optional[str] = None
    ) -> MemoryNote:
        """
        Construct a memory note with LLM-generated metadata.

        Implements A-MEM Ps1.
        """
        # Create basic note
        note = MemoryNote(content=content, timestamp=timestamp)

        # Generate metadata using LLM
        metadata = self.llm_service.generate_metadata(content, note.timestamp)

        note.context = metadata["context"]
        note.keywords = metadata["keywords"]
        note.tags = metadata["tags"]

        return note

    def _get_nearest_memories(
        self, note: MemoryNote, top_k: int
    ) -> List[tuple[MemoryNote, float]]:
        """
        Find the nearest existing memories for linking and evolution.
        """
        if not note.embedding:
            return []

        # Search for similar memories
        search_results = self.embedder.search(
            query_embedding=note.embedding,
            top_k=top_k,
        )

        retrieved = []
        for memory_id, score in search_results:
            # Get the full memory note
            memory_note = self._get_memory_note(memory_id)
            if memory_note:
                retrieved.append((memory_note, score))

        return retrieved

    def _get_memory_note(self, memory_id: str) -> Optional[MemoryNote]:
        """
        Retrieve a memory note from storage.
        """
        # Try different store methods
        get_note = getattr(self.store, "get_note", None)
        if callable(get_note):
            return get_note(memory_id)

        get_memory = getattr(self.store, "get_memory", None)
        if callable(get_memory):
            data = get_memory(memory_id)
            if data:
                return MemoryNote.from_dict(data)

        return None

    def _persist_memory(self, note: MemoryNote) -> None:
        """
        Persist a memory note to storage.
        """
        # Convert embedding to bytes for storage
        embedding_bytes = None
        if note.embedding:
            import pickle

            embedding_bytes = pickle.dumps(note.embedding)

        # Try different store methods
        insert_memory = getattr(self.store, "insert_memory", None)
        if callable(insert_memory):
            insert_memory(
                memory_id=note.id,
                content=note.content,
                created_at=note.timestamp,
                context=note.context,
                keywords=note.keywords,
                tags=note.tags,
                embedding=embedding_bytes,
            )
            return

        # Fallback: manual insertion
        import json
        import sqlite3

        conn = sqlite3.connect(self.store.db_path)

        try:
            conn.execute(
                """
                INSERT INTO memory_notes
                (id, content, created_at, last_accessed, retrieval_count,
                 context, keywords, tags, embedding)
                VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?)
                """,
                (
                    note.id,
                    note.content,
                    note.timestamp,
                    note.last_accessed,
                    note.context,
                    json.dumps(note.keywords),
                    json.dumps(note.tags),
                    embedding_bytes,
                ),
            )
            conn.commit()

        finally:
            conn.close()
