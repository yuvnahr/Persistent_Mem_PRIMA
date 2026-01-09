from prima_memory.core.embedding import EmbeddingIndex

def main():
    print("[TEST] Initializing EmbeddingIndex...")
    idx = EmbeddingIndex()

    print("[TEST] Generating embeddings...")
    e1 = idx.embed_text("memory systems and long-term learning")
    e2 = idx.embed_text("agentic memory and evolution")
    e3 = idx.embed_text("cooking pasta recipe")

    print("[TEST] Adding embeddings to index...")
    idx.add("m1", e1)
    idx.add("m2", e2)
    idx.add("m3", e3)

    print("[TEST] Querying index...")
    query = idx.embed_text("agent memory systems")
    results = idx.search(query, top_k=3)

    print("\n[RESULTS]")
    for mem_id, score in results:
        print(f"  {mem_id} -> similarity: {score:.4f}")

if __name__ == "__main__":
    main()

