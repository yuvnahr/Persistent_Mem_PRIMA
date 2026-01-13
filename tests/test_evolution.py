"""
test_evolution.py

Sanity test for PRIMA memory evolution.

Validates:
1. Eligibility-based memory evolution
2. Semantic refinement (tags, context)
3. Proper logging in memory_evolution table
4. No mutation of raw content
"""

import json
import sqlite3

from prima_memory.core.embedding import EmbeddingIndex
from prima_memory.core.evolution import MemoryEvolver
from prima_memory.core.note import MemoryNote
from prima_memory.core.retriever import MemoryRetriever
from prima_memory.persistence.sqlite import SQLiteMemoryStore


def test_evolution_sanity():
    store = SQLiteMemoryStore()
    embedder = EmbeddingIndex()
    retriever = MemoryRetriever(store=store, embedder=embedder)
    evolver = MemoryEvolver(store=store, min_retrievals=2)

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

    # Persist notes and embeddings
    for note in (base_note, reinforcing_note):
        note.embedding = embedder.embed_text(note.content)
        store.insert_note(note)
        embedder.add(note.id, note.embedding)

    # Simulate repeated retrievals (eligibility trigger)
    retriever.retrieve("agent memory systems", top_k=1)
    retriever.retrieve("agent memory systems", top_k=1)

    # Retrieve related memories
    retrieved_ids_scores = embedder.search(
        embedder.embed_text(base_note.content),
        top_k=2,
    )

    related_notes = [store.get_note(mem_id) for mem_id, _ in retrieved_ids_scores]

    # Run evolution
    evolver.evolve(
        source=reinforcing_note,
        related=related_notes,
    )

    # Reload evolved memory
    evolved_note = store.get_note(base_note.id)

    # --- Assertions ---

    # Raw content must never change
    assert evolved_note.content == base_note.content

    # Tags should be merged
    assert "memory" in evolved_note.tags
    assert "evolution" in evolved_note.tags

    # Context should be refined
    assert (
        "evolution" in evolved_note.context.lower()
        or "refinement" in evolved_note.context.lower()
    )

    # Evolution history must be recorded
    conn = sqlite3.connect(store.db_path)
    try:
        rows = conn.execute(
            "SELECT action, details FROM memory_evolution WHERE memory_id = ?",
            (base_note.id,),
        ).fetchall()
    finally:
        conn.close()

    assert len(rows) > 0

    action, details_json = rows[0]
    details = json.loads(details_json)

    assert action == "semantic_refinement"
    assert "tags" in details or "context" in details
