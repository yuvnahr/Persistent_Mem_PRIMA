"""
run_agent.py

Interactive PRIMA agent demonstrating the full A-MEM pipeline:

1. Note Construction (LLM metadata generation)
2. Link Generation (LLM relationship analysis)
3. Memory Evolution (LLM evolution decisions)
4. Retrieval (with linked memory expansion)
"""

from __future__ import annotations

import datetime

import torch

from prima_memory.core.agentic_memory_system import AgenticMemorySystem
from prima_memory.llm.prompts import build_agent_prompt


def main() -> None:
    """
    Launch interactive PRIMA agent with full A-MEM pipeline.
    """

    # -------------------------
    # Initialize A-MEM System
    # -------------------------

    memory_system = AgenticMemorySystem(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        llm_model_name="microsoft/DialoGPT-medium",  # Can be changed to larger models
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print("PRIMA Agent (A-MEM) ready. Type 'exit' to quit.\n")

    # -------------------------
    # Interactive loop
    # -------------------------

    while True:
        user_input = input("User: ").strip()

        if user_input.lower() == "exit":
            break

        # ---------------------------------
        # 1️⃣ Retrieve relevant memories
        # ---------------------------------

        memories = memory_system.search(user_input, k=5)

        # Convert to MemoryNote list for prompt building
        memory_notes = []
        for mem_dict in memories:
            from prima_memory.core.note import MemoryNote

            note = MemoryNote(
                content=mem_dict["content"],
                note_id=mem_dict["id"],
                context=mem_dict["context"],
                keywords=mem_dict["keywords"],
                tags=mem_dict["tags"],
            )
            memory_notes.append(note)

        # ---------------------------------
        # 2️⃣ Generate response using LLM
        # ---------------------------------

        prompt = build_agent_prompt(
            query=user_input,
            memories=memory_notes,
        )

        response = memory_system.llm.generate(prompt)

        print(f"\nAgent: {response}\n")

        # ---------------------------------
        # 3️⃣ Store interaction as memory
        # ---------------------------------

        # Add the user input as a new memory (full A-MEM pipeline)
        memory_id = memory_system.add_note(
            content=user_input,
            time=datetime.datetime.utcnow().strftime("%Y%m%d%H%M"),
        )

        print(f"Memory stored with ID: {memory_id}")


if __name__ == "__main__":
    main()
