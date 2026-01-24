"""
prompts.py

Prompt construction utilities for PRIMA agents.
Memory-aware but memory-safe prompting.
"""

from typing import List

from prima_memory.core.note import MemoryNote


def build_agent_prompt(
    query: str,
    memories: List[MemoryNote],
    max_memories: int = 5,
) -> str:
    """
    Build a memory-aware prompt for the agent.

    Args:
        query (str): User input / observation.
        memories (List[MemoryNote]): Retrieved relevant memories.
        max_memories (int): Maximum number of memories to include.

    Returns:
        str: Formatted prompt string.
    """

    prompt_parts: list[str] = []

    # User query
    prompt_parts.append("User query:")
    prompt_parts.append(query.strip())
    prompt_parts.append("")

    # Memory section
    if memories:
        prompt_parts.append("Relevant past memories:")

        for i, mem in enumerate(memories[:max_memories], start=1):
            summary = mem.content.strip()

            context = mem.context or "General"
            tags = ", ".join(mem.tags) if mem.tags else "none"

            prompt_parts.append(f"{i}. [{context}] {summary} (tags: {tags})")

        prompt_parts.append("")
    else:
        prompt_parts.append("No relevant past memories found.\n")

    # Instruction to the model
    prompt_parts.append(
        "Instruction:\n"
        "Use the past memories if they are relevant to the query. "
        "Do not repeat them verbatim unless necessary. "
        "If they are not helpful, answer based on your general knowledge."
    )

    return "\n".join(prompt_parts)
