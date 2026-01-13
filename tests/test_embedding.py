from prima_memory.core.embedding import EmbeddingIndex


def test_embedding_sanity():
    idx = EmbeddingIndex()

    e1 = idx.embed_text("memory systems and long-term learning")
    e2 = idx.embed_text("agentic memory and evolution")
    e3 = idx.embed_text("cooking pasta recipe")

    idx.add("m1", e1)
    idx.add("m2", e2)
    idx.add("m3", e3)

    query = idx.embed_text("agent memory systems")
    results = idx.search(query, top_k=3)

    assert results[0][0] in {"m1", "m2"}
    assert results[-1][0] == "m3"
