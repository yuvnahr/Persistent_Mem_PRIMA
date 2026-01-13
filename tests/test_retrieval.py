"""
test_retrieval.py

End-to-end sanity test for PRIMA memory retrieval.

Validates:
1. SQLite persistence of MemoryNotes
2. FAISS-based semantic similarity search
3. Retrieval of full MemoryNote objects
4. Access metadata updates (retrieval_count)
"""

from prima_memory.core.embedding import EmbeddingIndex
from prima_memory.core.note import MemoryNote
from prima_memory.core.retriever import MemoryRetriever
from prima_memory.persistence.sqlite import SQLiteMemoryStore


def test_retrieval_sanity():
    store = SQLiteMemoryStore()
    embedder = EmbeddingIndex()
    retriever = MemoryRetriever(store=store, embedder=embedder)

    notes = [
        MemoryNote(
            content="Agentic memory systems and long-term reasoning",
            context="Discussion about agent memory architectures",
            tags=["memory", "agentic"],
        ),
        MemoryNote(
            content="Memory evolution and semantic consolidation in agents",
            context="How memories change over time",
            tags=["memory", "evolution"],
        ),
        MemoryNote(
            content="How to cook pasta with olive oil and garlic",
            context="Cooking recipe",
            tags=["cooking"],
        ),
    ]

    # Insert notes into SQLite and FAISS
    for note in notes:
        note.embedding = embedder.embed_text(note.content)
        store.insert_note(note)
        embedder.add(note.id, note.embedding)

    # Query memory retriever
    results = retriever.retrieve("agent memory systems", top_k=3)

    # --- Assertions ---

    assert len(results) > 0
    assert "memory" in results[0].tags

    # Retrieval should update access metadata
    for note in results:
        assert note.retrieval_count >= 1
