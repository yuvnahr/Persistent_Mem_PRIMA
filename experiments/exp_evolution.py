"""
test_evolution.py

Sanity test for PRIMA memory evolution.

Validates:
1. Eligibility-based memory evolution
2. Semantic refinement (tags, context)
3. Proper logging in memory_evolution table
4. No mutation of raw content
"""
"""
import sqlite3
import json

from prima_memory.core.note import MemoryNote
from prima_memory.core.embedding import EmbeddingIndex
from prima_memory.core.retriever import MemoryRetriever
from prima_memory.core.evolution import MemoryEvolver
from prima_memory.persistence.sqlite import SQLiteMemoryStore


def main():
    print("[TEST] Initializing components...")

    store = SQLiteMemoryStore()
    embedder = EmbeddingIndex()
    retriever = MemoryRetriever(store=store, embedder=embedder)
    evolver = MemoryEvolver(store=store, min_retrievals=2)

    print("[TEST] Creating memory notes...")

    base_note = MemoryNote(
        content="Agentic memory systems for long-term reasoning",
        context="Agent memory architectures",
        tags=["memory"],
    )

    reinforcing_note = MemoryNote(
        content="Memory evolution improves agent reasoning over time",
        context="Memory refinement and evolution",
        tags=["evolution"],
    )

    notes = [base_note, reinforcing_note]

    print("[TEST] Persisting notes and embeddings...")
    for note in notes:
        note.embedding = embedder.embed_text(note.content)
        store.insert_note(note)
        embedder.add(note.id, note.embedding)

    print("[TEST] Simulating repeated retrievals...")
    # Retrieve base_note twice to trigger evolution eligibility
    retriever.retrieve("agent memory systems", top_k=1)
    retriever.retrieve("agent memory systems", top_k=1)

    print("[TEST] Retrieving related memories...")
    retrieved_ids_scores = embedder.search(
        embedder.embed_text(base_note.content),
        top_k=2,
    )

    related_notes = []
    for mem_id, _score in retrieved_ids_scores:
        related_notes.append(store.get_note(mem_id))

    print("[TEST] Running memory evolution...")
    evolver.evolve(
        source=reinforcing_note,
        related=related_notes,
    )

    print("[TEST] Reloading evolved memory...")
    evolved_note = store.get_note(base_note.id)

    print("\n[EVOLVED NOTE]")
    print(f"Content: {evolved_note.content}")
    print(f"Context: {evolved_note.context}")
    print(f"Tags: {evolved_note.tags}")

    print("\n[ASSERTIONS]")

    # Raw content must never change
    assert evolved_note.content == base_note.content, "Raw content was mutated"

    # Tags should be merged
    assert "memory" in evolved_note.tags
    assert "evolution" in evolved_note.tags

    # Context should be refined
    assert "Memory refinement" in evolved_note.context or "evolution" in evolved_note.context.lower()

    print("[TEST] Checking evolution history table...")
    conn = sqlite3.connect(store.db_path)
    try:
        rows = conn.execute(
            "SELECT action, details FROM memory_evolution WHERE memory_id = ?",
            (base_note.id,),
        ).fetchall()
    finally:
        conn.close()

    assert len(rows) > 0, "No evolution history recorded"

    action, details_json = rows[0]
    details = json.loads(details_json)

    assert action == "semantic_refinement"
    assert "tags" in details or "context" in details

    print("✅ Evolution sanity test PASSED")


if __name__ == "__main__":
    main()
"""