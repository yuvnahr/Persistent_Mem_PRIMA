# scripts/run_agent.py

from prima_memory.core.note import MemoryNote
from prima_memory.core.embedding import EmbeddingIndex
from prima_memory.core.retriever import MemoryRetriever
from prima_memory.core.linker import MemoryLinker
from prima_memory.core.evolution import MemoryEvolver
from prima_memory.persistence.sqlite import SQLiteMemoryStore
from prima_memory.llm.hf_model import HFModel
from prima_memory.llm.prompts import build_agent_prompt


def main():
    # Initialize components
    store = SQLiteMemoryStore()
    embedder = EmbeddingIndex()
    retriever = MemoryRetriever(store=store, embedder=embedder)
    linker = MemoryLinker(store=store)
    evolver = MemoryEvolver(store=store)
    llm = HFModel()

    print("PRIMA Agent ready. Type 'exit' to quit.\n")

    while True:
        user_input = input("User: ").strip()
        if user_input.lower() == "exit":
            break

        # Recall
        memories = retriever.retrieve(user_input, top_k=5)

        # Reason
        prompt = build_agent_prompt(
            query=user_input,
            memories=memories
        )
        response = llm.generate(prompt)

        print(f"\nAgent: {response}\n")

        # Store new memory
        note = MemoryNote(
            content=user_input,
            context="User interaction",
            tags=["interaction"],
        )
        note.embedding = embedder.embed_text(note.content)

        store.insert_note(note)
        embedder.add(note.id, note.embedding)

        # Link & evolve
        linker.link(
            source=note,
            retrieved=[(m, 1.0) for m in memories],
        )
        evolver.evolve(
            source=note,
            related=memories,
        )


if __name__ == "__main__":
    main()

