"""
seed_dummy_memory.py

Resets the database and seeds 50 deterministic dummy memories
for testing PRIMA memory retrieval, linking, and evolution.
"""

from __future__ import annotations

import datetime
import hashlib
from typing import List

from prima_memory.core.memory_store import MemoryStore


def fake_embedding(text: str) -> bytes:
    """
    Create deterministic embedding bytes from text.
    This avoids needing a real embedding model during tests.
    """
    return hashlib.sha256(text.encode("utf-8")).digest()


def generate_memories() -> List[dict]:
    """Generate 50 structured memory notes."""

    groups = {
        "agentic_memory": {
            "tag": "concept",
            "context": "Concept explanation for agentic memory systems.",
            "items": [
                "Agentic memory allows AI agents to reason over past actions.",
                "Agentic memory structures knowledge beyond simple storage.",
                "Agentic memory enables agents to recall task history.",
                "Agentic memory helps autonomous agents maintain context.",
                "Agentic memory supports adaptive reasoning in AI.",
                "Agentic memory enables knowledge refinement over time.",
                "Agentic memory links related experiences together.",
                "Agentic memory allows agents to evolve internal knowledge.",
                "Agentic memory organizes information dynamically.",
                "Agentic memory improves decision-making in long tasks.",
            ],
        },
        "vector_db": {
            "tag": "comparison",
            "context": "Contrast between vector databases and agentic memory.",
            "items": [
                "Vector databases store embeddings for similarity search.",
                "Vector databases enable semantic retrieval.",
                "Vector databases cannot evolve knowledge structures.",
                "Vector search finds semantically similar text.",
                "Vector storage does not track reasoning history.",
                "Vector DB retrieval depends on embedding similarity.",
                "Vector databases scale similarity search efficiently.",
                "Vector DBs store embeddings but not conceptual links.",
                "Vector DB retrieval is static compared to evolving memory.",
                "Vector databases support large-scale embedding storage.",
            ],
        },
        "llm_agents": {
            "tag": "architecture",
            "context": "Architecture of LLM-based autonomous agents.",
            "items": [
                "LLM agents combine reasoning, tools, and memory.",
                "LLM agents require persistent memory across sessions.",
                "LLM agents orchestrate planning and execution.",
                "LLM agents rely on context for coherent reasoning.",
                "LLM agents benefit from structured memory storage.",
                "LLM agents coordinate tasks through an orchestrator.",
                "LLM agents use prompts to guide reasoning behavior.",
                "LLM agents require retrieval for long conversations.",
                "LLM agents combine knowledge with dynamic reasoning.",
                "LLM agents integrate memory with task execution.",
            ],
        },
        "retrieval": {
            "tag": "retrieval",
            "context": "Memory retrieval mechanisms for AI systems.",
            "items": [
                "Similarity search retrieves semantically related memories.",
                "Retrieval systems rely on embedding distance metrics.",
                "Top-k retrieval selects most relevant memories.",
                "Memory retrieval improves contextual reasoning.",
                "Retrieval pipelines combine search and filtering.",
                "Efficient retrieval enables scalable memory systems.",
                "Embedding similarity enables semantic recall.",
                "Memory retrieval supports knowledge reuse.",
                "Retrieval allows agents to recall past experiences.",
                "Memory recall improves task continuity.",
            ],
        },
        "memory_systems": {
            "tag": "architecture",
            "context": "Design principles of persistent memory systems.",
            "items": [
                "Persistent memory enables long-term AI learning.",
                "Memory graphs organize knowledge relationships.",
                "Memory linking connects related concepts.",
                "Memory evolution updates knowledge structures.",
                "Persistent storage allows knowledge reuse.",
                "Memory architectures support knowledge accumulation.",
                "Semantic memory graphs enable reasoning chains.",
                "Memory linking forms conceptual clusters.",
                "Memory systems improve agent adaptability.",
                "Structured memory improves knowledge organization.",
            ],
        },
    }

    memories = []

    idx = 1

    for group, data in groups.items():
        for content in data["items"]:
            memories.append(
                {
                    "id": f"m{idx}",
                    "content": content,
                    "context": data["context"],
                    "keywords": [group],
                    "tags": [data["tag"]],
                }
            )
            idx += 1

    return memories


def seed_memories() -> None:
    store = MemoryStore()

    now = datetime.datetime.utcnow().isoformat()

    memories = generate_memories()

    for mem in memories:
        store.insert_memory(
            memory_id=mem["id"],
            content=mem["content"],
            created_at=now,
            context=mem["context"],
            keywords=mem["keywords"],
            tags=mem["tags"],
            embedding=fake_embedding(mem["content"]),
        )

    print(f"[PRIMA] Seeded {len(memories)} memories.")


if __name__ == "__main__":
    seed_memories()
