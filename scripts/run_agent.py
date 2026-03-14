"""
run_agent.py

Interactive PRIMA agent demonstrating the full A-MEM pipeline:

1. Retrieve relevant memories
2. Generate response using LLM
3. Store interaction as new memory
4. Link related memories
5. Evolve existing memories
"""

from __future__ import annotations

import datetime
from typing import List

from prima_memory.core.note import MemoryNote
from prima_memory.core.embedding import EmbeddingIndex
from prima_memory.core.retriever import MemoryRetriever
from prima_memory.core.linker import MemoryLinker
from prima_memory.core.evolution import MemoryEvolver
from prima_memory.core.memory_store import MemoryStore
from prima_memory.llm.hf_model import HFModel
from prima_memory.llm.prompts import build_agent_prompt


def build_interaction_note(user_input: str, embedding: List[float]) -> MemoryNote:
    """
    Create a MemoryNote from user interaction.
    """

    now = datetime.datetime.utcnow().isoformat()

    note = MemoryNote(
        content=user_input,
        context="User interaction",
        tags=["interaction"],
        keywords=["user_input"],
        created_at=now,
    )

    note.embedding = embedding

    return note


def main() -> None:
    """
    Launch interactive PRIMA agent.
    """

    # -------------------------
    # Initialize components
    # -------------------------

    store = MemoryStore()
    embedder = EmbeddingIndex()

    retriever = MemoryRetriever(store=store, embedder=embedder)
    linker = MemoryLinker(store=store)
    evolver = MemoryEvolver(store=store)

    llm = HFModel()

    print("PRIMA Agent ready. Type 'exit' to quit.\n")

    # -------------------------
    # Interactive loop
    # -------------------------

    while True:
        user_input = input("User: ").strip()

        if user_input.lower() == "exit":
            break

        # ---------------------------------
        # 1️⃣ Retrieve memories
        # ---------------------------------

        memories = retriever.retrieve(
            user_input,
            top_k=5,
            expand_links=True,
        )

        # ---------------------------------
        # 2️⃣ Generate response
        # ---------------------------------

        prompt = build_agent_prompt(
            query=user_input,
            memories=memories,
        )

        response = llm.generate(prompt)

        print(f"\nAgent: {response}\n")

        # ---------------------------------
        # 3️⃣ Create new memory
        # ---------------------------------

        embedding = embedder.embed_text(user_input)

        note = build_interaction_note(
            user_input=user_input,
            embedding=embedding,
        )

        # Store memory
        store.insert_memory(
            memory_id=note.id,
            content=note.content,
            created_at=note.created_at,
            context=note.context,
            keywords=note.keywords,
            tags=note.tags,
            embedding=note.embedding,
        )

        # Add to embedding index
        embedder.add(note.id, note.embedding)

        # ---------------------------------
        # 4️⃣ Link memories
        # ---------------------------------

        linker.link(
            source=note,
            retrieved=[(m, 1.0) for m in memories],
        )

        # ---------------------------------
        # 5️⃣ Evolve memories
        # ---------------------------------

        evolver.evolve(
            source=note,
            related=memories,
        )


if __name__ == "__main__":
    main()