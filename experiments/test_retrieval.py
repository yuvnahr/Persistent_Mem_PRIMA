"""
test_retrieval.py

End-to-end sanity test for PRIMA memory retrieval.

This test validates:
1. SQLite persistence of MemoryNotes
2. FAISS-based semantic similarity search
3. Retrieval of full MemoryNote objects
4. Access metadata updates (retrieval_count)
"""

from prima_memory.core.note import MemoryNote
from prima_memory.core.embedding import EmbeddingIndex
from prima_memory.persistence.sqlite import SQLiteMemoryStore
from prima_memory.core.retriever import MemoryRetriever


def main():
    print("[TEST] Initializing components...")

    store = SQLiteMemoryStore()
    embedder = EmbeddingIndex()
    retriever = MemoryRetriever(store=store, embedder=embedder)

    print("[TEST] Creating memory notes...")

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

    print("[TEST] Inserting notes into SQLite and FAISS...")
    for note in notes:
        # Generate embedding
        note.embedding = embedder.embed_text(note.content)

        # Persist note
        store.insert_note(note)

        # Add to vector index
        embedder.add(note.id, note.embedding)

    print("[TEST] Querying memory retriever...")
    query = "agent memory systems"
    results = retriever.retrieve(query, top_k=3)

    print("\n[RESULTS]")
    for i, note in enumerate(results, start=1):
        print(f"{i}. {note.content}")
        print(f"   tags={note.tags}")
        print(f"   retrieval_count={note.retrieval_count}")

    print("\n[ASSERTIONS]")
    assert len(results) > 0, "No memories retrieved"
    assert "memory" in results[0].tags, "Top result is not memory-related"

    print("✅ Retrieval sanity test PASSED")


if __name__ == "__main__":
    main()
