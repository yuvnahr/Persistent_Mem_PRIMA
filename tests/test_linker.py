"""
test_linker.py

Sanity test for PRIMA memory linking.

Validates:
1. Semantic similarity + metadata-based link creation
2. Correct persistence of links in SQLite
3. No self-links
"""

from prima_memory.core.note import MemoryNote
from prima_memory.core.embedding import EmbeddingIndex
from prima_memory.core.retriever import MemoryRetriever
from prima_memory.core.linker import MemoryLinker
from prima_memory.persistence.sqlite import SQLiteMemoryStore


def test_linker_sanity():
    store = SQLiteMemoryStore()
    embedder = EmbeddingIndex()
    retriever = MemoryRetriever(store=store, embedder=embedder)
    linker = MemoryLinker(store=store, similarity_threshold=0.5)

    note_main = MemoryNote(
        content="Agentic memory systems for long-term reasoning",
        context="Agent memory architectures",
        tags=["memory", "agentic"],
    )

    note_related = MemoryNote(
        content="Memory evolution and refinement in intelligent agents",
        context="Agent memory architectures",
        tags=["memory", "evolution"],
    )

    note_unrelated = MemoryNote(
        content="How to bake a chocolate cake",
        context="Cooking recipes",
        tags=["cooking"],
    )

    # Persist notes and embeddings
    for note in (note_main, note_related, note_unrelated):
        note.embedding = embedder.embed_text(note.content)
        store.insert_note(note)
        embedder.add(note.id, note.embedding)

    # Retrieve candidate memories
    retrieved_ids_scores = embedder.search(
        embedder.embed_text(note_main.content),
        top_k=3,
    )

    retrieved_notes = [
        (store.get_note(mem_id), score) for mem_id, score in retrieved_ids_scores
    ]

    # Run linker
    linker.link(
        source=note_main,
        retrieved=retrieved_notes,
    )

    # Fetch stored links
    links = store.get_links(note_main.id)

    # --- Assertions ---

    assert note_related.id in links
    assert note_unrelated.id not in links
    assert note_main.id not in links
