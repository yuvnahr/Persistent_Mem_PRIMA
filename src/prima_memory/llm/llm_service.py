"""
llm_service.py

LLM service for A-MEM operations in PRIMA.

Handles metadata generation, linking decisions, and memory evolution
using Hugging Face models on CUDA.
"""

import json
from typing import Any, Dict, List, Optional

from prima_memory.core.note import MemoryNote
from prima_memory.llm.hf_model import HFModel
from prima_memory.llm.prompts import (
    build_evolution_prompt,
    build_linking_prompt,
    build_metadata_prompt,
)


class LLMService:
    """
    LLM service for A-MEM memory operations.
    """

    def __init__(self, model: HFModel):
        self.model = model

    def generate_metadata(self, content: str, timestamp: str) -> Dict[str, Any]:
        """
        Generate keywords, context, and tags for a memory note.

        Implements A-MEM Ps1.

        Args:
            content: Raw memory content
            timestamp: Creation timestamp

        Returns:
            Dict with 'keywords', 'context', 'tags'
        """
        prompt = build_metadata_prompt(content, timestamp)

        try:
            response = self.model.generate(prompt)

            # Try to parse JSON response
            result = json.loads(response)

            # Validate structure
            keywords = result.get("keywords", [])
            context = result.get("context", "General")
            tags = result.get("tags", [])

            # Ensure types
            if not isinstance(keywords, list):
                keywords = []
            if not isinstance(context, str):
                context = "General"
            if not isinstance(tags, list):
                tags = []

            return {"keywords": keywords, "context": context, "tags": tags}

        except (json.JSONDecodeError, Exception) as e:
            print(f"LLM metadata generation failed: {e}")
            # Fallback: extract basic keywords from content
            words = content.lower().split()[:5]  # Simple fallback
            return {
                "keywords": words,
                "context": "General discussion",
                "tags": ["general"],
            }

    def decide_links(
        self, new_memory: MemoryNote, nearest_memories: List[MemoryNote]
    ) -> List[str]:
        """
        Decide which memories to link to the new memory.

        Implements A-MEM Ps2.

        Args:
            new_memory: Newly created memory
            nearest_memories: Top-k similar existing memories

        Returns:
            List of memory IDs to link to
        """
        if not nearest_memories:
            return []

        prompt = build_linking_prompt(new_memory, nearest_memories)

        try:
            response = self.model.generate(prompt)
            result = json.loads(response)

            links = result.get("links", [])
            if isinstance(links, list):
                # Extract memory IDs
                memory_ids = []
                for link in links:
                    if isinstance(link, dict) and "memory_id" in link:
                        memory_ids.append(link["memory_id"])
                return memory_ids
            else:
                return []

        except (json.JSONDecodeError, Exception) as e:
            print(f"LLM linking decision failed: {e}")
            # Fallback: link to all nearest memories
            return [mem.id for mem in nearest_memories]

    def evolve_memory(
        self,
        new_memory: MemoryNote,
        target_memory: MemoryNote,
        neighborhood: List[MemoryNote],
    ) -> Optional[Dict[str, Any]]:
        """
        Decide how to evolve a target memory based on new information.

        Implements A-MEM Ps3.

        Args:
            new_memory: Newly added memory
            target_memory: Memory to potentially evolve
            neighborhood: Other related memories

        Returns:
            Dict with evolution decisions, or None if no evolution needed
        """
        # Filter out the target from neighborhood
        filtered_neighborhood = [
            mem for mem in neighborhood if mem.id != target_memory.id
        ]

        prompt = build_evolution_prompt(
            new_memory, target_memory, filtered_neighborhood
        )

        try:
            response = self.model.generate(prompt)
            result = json.loads(response)

            should_evolve = result.get("should_evolve", False)

            if not should_evolve:
                return None

            return {
                "updated_context": result.get("updated_context", target_memory.context),
                "updated_keywords": result.get(
                    "updated_keywords", target_memory.keywords
                ),
                "updated_tags": result.get("updated_tags", target_memory.tags),
                "evolution_reason": result.get(
                    "evolution_reason", "LLM suggested evolution"
                ),
            }

        except (json.JSONDecodeError, Exception) as e:
            print(f"LLM evolution decision failed: {e}")
            return None
