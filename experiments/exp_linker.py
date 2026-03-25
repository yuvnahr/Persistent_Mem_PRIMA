"""
test_linker.py

Sanity test for PRIMA memory linking.

Validates:
1. Semantic similarity + metadata-based link creation
2. Correct persistence of links in SQLite
3. No self-links
"""

"""
from prima_memory.core.note import MemoryNote
from prima_memory.core.embedding import EmbeddingIndex
from prima_memory.core.retriever import MemoryRetriever
from prima_memory.core.linker import MemoryLinker
from prima_memory.persistence.sqlite import SQLiteMemoryStore


def main():
    print("[TEST] Initializing components...")

    store = SQLiteMemoryStore()
    embedder = EmbeddingIndex()
    retriever = MemoryRetriever(store=store, embedder=embedder)
    linker = MemoryLinker(store=store, similarity_threshold=0.5)

    print("[TEST] Creating memory notes...")

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

    notes = [note_main, note_related, note_unrelated]

    print("[TEST] Persisting notes and embeddings...")
    for note in notes:
        note.embedding = embedder.embed_text(note.content)
        store.insert_note(note)
        embedder.add(note.id, note.embedding)

    print("[TEST] Retrieving candidate memories...")
    retrieved_ids_scores = embedder.search(
        embedder.embed_text(note_main.content),
        top_k=3,
    )

    retrieved_notes = []
    for mem_id, score in retrieved_ids_scores:
        retrieved_notes.append((store.get_note(mem_id), score))

    print("[TEST] Running linker...")
    linker.link(
        source=note_main,
        retrieved=retrieved_notes,
    )

    print("[TEST] Fetching stored links...")
    links = store.get_links(note_main.id)

    print("\n[LINKS FOUND]")
    for target_id, meta in links.items():
        print(f"→ {target_id} | type={meta['type']} | strength={meta['strength']:.3f}")

    print("\n[ASSERTIONS]")
    assert note_related.id in links, "Expected related memory to be linked"
    assert note_unrelated.id not in links, "Unrelated memory should not be linked"
    assert note_main.id not in links, "Self-link should not exist"

    print("✅ Linker sanity test PASSED")


if __name__ == "__main__":
    main()
"""
