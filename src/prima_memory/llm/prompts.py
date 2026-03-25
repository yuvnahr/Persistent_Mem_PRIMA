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


# A-MEM specific prompts (Ps1, Ps2, Ps3 from the paper)


def build_metadata_prompt(content: str, timestamp: str) -> str:
    """
    Ps1: Generate metadata for new memory note.

    Args:
        content: Raw interaction content
        timestamp: Creation timestamp

    Returns:
        Prompt for LLM to generate keywords, context, tags
    """
    return f"""Generate a structured analysis of the following content by:
1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
2. Extracting core themes and contextual elements
3. Creating relevant categorical tags

Format the response as a JSON object:
{{
    "keywords": [
        // several specific, distinct keywords that capture key concepts and terminology
        // Order from most to least important
        // Don't include keywords that are the name of the speaker or time
    ],
    "context": 
        // one sentence summarizing:
        // - Main topic/domain
        // - Key arguments/points
        // - Intended audience/purpose
    ,
    "tags": [
        // several broad categories/themes for classification
        // Include domain, format, and type tags
        // At least three tags, but don't be too redundant.
    ]
}}

Content for analysis:
{content}

Timestamp: {timestamp}"""


def build_linking_prompt(
    new_memory: MemoryNote, nearest_memories: List[MemoryNote]
) -> str:
    """
    Ps2: Decide which memories to link the new memory to.

    Args:
        new_memory: The newly created memory
        nearest_memories: Top-k similar existing memories

    Returns:
        Prompt for LLM to decide links
    """
    memories_text = ""
    for i, mem in enumerate(nearest_memories, 1):
        memories_text += f"{i}. ID: {mem.id}\n"
        memories_text += f"   Content: {mem.content}\n"
        memories_text += f"   Context: {mem.context or 'None'}\n"
        memories_text += (
            f"   Keywords: {', '.join(mem.keywords) if mem.keywords else 'None'}\n"
        )
        memories_text += f"   Tags: {', '.join(mem.tags) if mem.tags else 'None'}\n\n"

    return f"""Analyze the new memory and determine which of the nearest memories should be linked to it.

New memory:
Content: {new_memory.content}
Context: {new_memory.context or 'None'}
Keywords: {', '.join(new_memory.keywords) if new_memory.keywords else 'None'}
Tags: {', '.join(new_memory.tags) if new_memory.tags else 'None'}

Nearest memories:
{memories_text}

Based on semantic similarity, shared concepts, and contextual relationships, decide which memories should be linked to the new memory.

Return a JSON object with:
{{
    "links": [
        {{"memory_id": "id_of_memory_to_link", "reason": "brief explanation"}},
        // ... include all memories that should be linked
    ]
}}"""


def build_evolution_prompt(
    new_memory: MemoryNote, target_memory: MemoryNote, neighborhood: List[MemoryNote]
) -> str:
    """
    Ps3: Evolve an existing memory based on new information.

    Args:
        new_memory: The newly added memory
        target_memory: The memory to potentially evolve
        neighborhood: Other related memories (excluding target)

    Returns:
        Prompt for LLM to evolve the target memory
    """
    neighborhood_text = ""
    for i, mem in enumerate(neighborhood, 1):
        if mem.id != target_memory.id:
            neighborhood_text += f"{i}. Content: {mem.content}\n"
            neighborhood_text += f"   Context: {mem.context or 'None'}\n"
            neighborhood_text += (
                f"   Keywords: {', '.join(mem.keywords) if mem.keywords else 'None'}\n"
            )
            neighborhood_text += (
                f"   Tags: {', '.join(mem.tags) if mem.tags else 'None'}\n\n"
            )

    return f"""Analyze how the target memory should evolve based on the new memory and related context.

New memory:
Content: {new_memory.content}
Context: {new_memory.context or 'None'}
Keywords: {', '.join(new_memory.keywords) if new_memory.keywords else 'None'}
Tags: {', '.join(new_memory.tags) if new_memory.tags else 'None'}

Target memory to evolve:
Content: {target_memory.content}
Current Context: {target_memory.context or 'None'}
Current Keywords: {', '.join(target_memory.keywords) if target_memory.keywords else 'None'}
Current Tags: {', '.join(target_memory.tags) if target_memory.tags else 'None'}

Related neighborhood memories:
{neighborhood_text}

Based on the new information and relationships, suggest how to update the target memory's context, keywords, and tags. The updates should integrate new insights while preserving valuable existing information.

Return a JSON object with:
{{
    "should_evolve": true/false,
    "updated_context": "refined context description (keep existing if no improvement needed)",
    "updated_keywords": ["refined", "keyword", "list"],
    "updated_tags": ["refined", "tag", "list"],
    "evolution_reason": "brief explanation of changes"
}}"""
