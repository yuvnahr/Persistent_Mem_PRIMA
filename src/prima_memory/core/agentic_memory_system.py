import json
import logging
from typing import Dict, List, Optional, Tuple

import chromadb
import torch
from sentence_transformers import SentenceTransformer

from prima_memory.core.note import MemoryNote
from prima_memory.llm.hf_model import HFModel

logger = logging.getLogger(__name__)


class AgenticMemorySystem:
    """
    Agentic Memory System replicating A-mem-sys logic.
    Uses ChromaDB for storage/retrieval, HF models on CUDA for LLM.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model_name: str = "microsoft/DialoGPT-medium",
        evo_threshold: int = 100,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model_name = model_name
        self.llm_model_name = llm_model_name
        self.evo_threshold = evo_threshold
        self.device = device

        # In-memory storage (as in original)
        self.memories: Dict[str, MemoryNote] = {}

        # ChromaDB setup (replicating original)
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(name="memories")

        # Embedding model (SentenceTransformer, as in original)
        self.embedder = SentenceTransformer(model_name)
        if device == "cuda":
            self.embedder.to(device)

        # HF LLM model (our addition for CUDA)
        self.llm = HFModel(model_name=llm_model_name, device=device)

        # Evolution counter
        self.evo_cnt = 0

        # Evolution prompt (from original)
        self._evolution_system_prompt = """
                                You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
                                Analyze the the new memory note according to keywords and context, also with their several nearest neighbors memory.
                                Make decisions about its evolution.

                                The new memory context:
                                {context}
                                content: {content}
                                keywords: {keywords}

                                The nearest neighbors memories (each line starts with memory_id):
                                {nearest_neighbors_memories}

                                Based on this information, determine:
                                1. Should this memory be evolved? Consider its relationships with other memories.
                                2. What specific actions should be taken (strengthen, update_neighbor)?
                                   2.1 If choose to strengthen the connection, which memory should it be connected to? Use the memory_id from the neighbors above. Can you give the updated tags of this memory?
                                2.2 If choose to update_neighbor, you can update the context and tags of these memories based on the understanding of these memories. If the context and the tags are not updated, the new context and tags should be the same as the original ones. Generate the new context and tags in the sequential order of the input neighbors.
                                Tags should be determined by the content of these characteristic of these memories, which can be used to retrieve them later and categorize them.
                                Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.
                                The number of neighbors is {neighbor_number}.
                                Return your decision in JSON format with the following structure:
                                {{
                                    "should_evolve": True or False,
                                    "actions": ["strengthen", "update_neighbor"],
                                    "suggested_connections": ["memory_id_1", "memory_id_2", ...],
                                    "tags_to_update": ["tag_1",..."tag_n"],
                                    "new_context_neighborhood": ["new context",...,"new context"],
                                    "new_tags_neighborhood": [["tag_1",...,"tag_n"],...["tag_1",...,"tag_n"]],
                                }}
                                """

    def analyze_content(self, content: str) -> Dict:
        """Analyze content using HF LLM (adapted from original)."""
        prompt = """Generate a structured analysis of the following content by:
            1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
            2. Extracting core themes and contextual elements
            3. Creating relevant categorical tags

            Format the response as a JSON object:
            {
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
            }

            Content for analysis:
            """ + content
        response = self.llm.generate(prompt)
        try:
            return json.loads(response)
        except (json.JSONDecodeError, TypeError, ValueError):
            return {"keywords": [], "context": "General", "tags": []}

    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        """Add note following original A-mem-sys logic."""
        # Create note
        if time:
            kwargs["timestamp"] = time
        note = MemoryNote(content=content, **kwargs)

        # Analyze if needed (original logic)
        needs_analysis = not note.keywords or note.context == "General" or not note.tags
        if needs_analysis:
            analysis = self.analyze_content(content)
            note.keywords = analysis.get("keywords", [])
            note.context = analysis.get("context", "General")
            note.tags = analysis.get("tags", [])

        # Embed (using all text components as in original)
        text_components = [
            note.content,
            " ".join(note.keywords),
            " ".join(note.tags),
            note.context,
        ]
        combined_text = " ".join(text_components).strip()
        embedding = self.embedder.encode(
            combined_text, normalize_embeddings=True
        ).tolist()
        note.embedding = embedding

        # Store in memory
        self.memories[note.id] = note

        # Add to ChromaDB (original logic)
        metadata = {
            "id": note.id,
            "content": note.content,
            "keywords": json.dumps(note.keywords),
            "links": json.dumps(note.links),
            "retrieval_count": note.retrieval_count,
            "timestamp": note.timestamp,
            "last_accessed": note.last_accessed,
            "context": note.context,
            "evolution_history": json.dumps(note.evolution_history),
            "category": getattr(note, "category", None),
            "tags": json.dumps(note.tags),
        }
        self.collection.add(
            documents=[note.content],
            metadatas=[metadata],
            ids=[note.id],
            embeddings=[embedding],
        )

        # Process memory (evolution, as in original)
        evo_label, note = self.process_memory(note)
        self.memories[note.id] = note

        # Consolidate if threshold reached
        self.evo_cnt += 1
        if self.evo_cnt % self.evo_threshold == 0:
            self.consolidate_memories()

        return note.id

    def process_memory(self, note: MemoryNote) -> Tuple[bool, MemoryNote]:
        """Process memory for evolution (exact original logic)."""
        if not self.memories:
            return False, note

        try:
            # Get nearest neighbors
            neighbors_text, memory_ids = self.find_related_memories(note.content, k=5)
            if not neighbors_text or not memory_ids:
                return False, note

            # Query LLM for evolution decision
            prompt = self._evolution_system_prompt.format(
                content=note.content,
                context=note.context,
                keywords=note.keywords,
                nearest_neighbors_memories=neighbors_text,
                neighbor_number=len(memory_ids),
            )

            response = self.llm.generate(prompt)
            response_json = json.loads(response)
            should_evolve = response_json["should_evolve"]

            if should_evolve:
                actions = response_json["actions"]
                for action in actions:
                    if action == "strengthen":
                        suggest_connections = response_json["suggested_connections"]
                        new_tags = response_json["tags_to_update"]
                        if isinstance(suggest_connections, list):
                            for conn_id in suggest_connections:
                                note.links[conn_id] = {"strength": 1.0}
                        note.tags = new_tags
                    elif action == "update_neighbor":
                        new_context_neighborhood = response_json[
                            "new_context_neighborhood"
                        ]
                        new_tags_neighborhood = response_json["new_tags_neighborhood"]
                        # Update each neighbor memory using its actual ID
                        for i in range(
                            min(len(memory_ids), len(new_tags_neighborhood))
                        ):
                            memory_id = memory_ids[i]

                            # Skip if memory doesn't exist
                            if memory_id not in self.memories:
                                continue

                            # Get the memory to update
                            neighbor_memory = self.memories[memory_id]

                            # Update tags
                            if i < len(new_tags_neighborhood):
                                neighbor_memory.tags = new_tags_neighborhood[i]

                            # Update context
                            if i < len(new_context_neighborhood):
                                neighbor_memory.context = new_context_neighborhood[i]

                            # Save the updated memory back
                            self.memories[memory_id] = neighbor_memory

            return should_evolve, note

        except (json.JSONDecodeError, KeyError, Exception) as e:
            logger.error(f"Error in memory evolution: {str(e)}")
            return False, note

    def find_related_memories(self, query: str, k: int = 5) -> Tuple[str, List[str]]:
        """Find related memories using ChromaDB retrieval

        Returns:
            Tuple[str, List[str]]: (formatted_memory_string, list_of_memory_ids)
        """
        if not self.memories:
            return "", []

        try:
            # Get results from ChromaDB
            results = self.collection.query(query_texts=[query], n_results=k)

            # Convert to list of memories
            memory_str = ""
            memory_ids = []

            if (
                "ids" in results
                and results["ids"]
                and len(results["ids"]) > 0
                and len(results["ids"][0]) > 0
            ):
                for i, doc_id in enumerate(results["ids"][0]):
                    # Get metadata from ChromaDB results
                    if i < len(results["metadatas"][0]):
                        metadata = results["metadatas"][0][i]
                        # Format memory string with actual memory ID
                        memory_str += f"memory_id:{doc_id}\ttalk start time:{metadata.get('timestamp', '')}\tmemory content: {metadata.get('content', '')}\tmemory context: {metadata.get('context', '')}\tmemory keywords: {metadata.get('keywords', '')}\tmemory tags: {metadata.get('tags', '')}\n"
                        memory_ids.append(doc_id)

            return memory_str, memory_ids
        except Exception as e:
            logger.error(f"Error in find_related_memories: {str(e)}")
            return "", []

    def search(
        self,
        query: str,
        k: int = 5,
        top_k: Optional[int] = None,
    ) -> List[MemoryNote]:
        """Search for memories using a hybrid retrieval approach."""
        if top_k is not None:
            k = top_k

        search_results = self.collection.query(query_texts=[query], n_results=k)
        memories: List[MemoryNote] = []

        if (
            "ids" in search_results
            and search_results["ids"]
            and len(search_results["ids"]) > 0
        ):
            for i, doc_id in enumerate(search_results["ids"][0]):
                memory = self.memories.get(doc_id)
                if memory:
                    memory.mark_accessed()
                    memory.retrieval_count += 1
                    memories.append(memory)

        return memories[:k]

    def consolidate_memories(self):
        """Consolidate memories: update retriever with new documents"""
        # Reset ChromaDB collection
        self.client.delete_collection("memories")
        self.collection = self.client.create_collection(name="memories")

        # Re-add all memory documents with their complete metadata
        for memory in self.memories.values():
            metadata = {
                "id": memory.id,
                "content": memory.content,
                "keywords": json.dumps(memory.keywords),
                "links": json.dumps(memory.links),
                "retrieval_count": memory.retrieval_count,
                "timestamp": memory.timestamp,
                "last_accessed": memory.last_accessed,
                "context": memory.context,
                "evolution_history": json.dumps(memory.evolution_history),
                "category": getattr(memory, "category", None),
                "tags": json.dumps(memory.tags),
            }
            # Re-embed
            text_components = [
                memory.content,
                " ".join(memory.keywords),
                " ".join(memory.tags),
                memory.context,
            ]
            combined_text = " ".join(text_components).strip()
            embedding = self.embedder.encode(
                combined_text, normalize_embeddings=True
            ).tolist()
            self.collection.add(
                documents=[memory.content],
                metadatas=[metadata],
                ids=[memory.id],
                embeddings=[embedding],
            )

    def read(self, memory_id: str) -> Optional[MemoryNote]:
        """Retrieve a memory note by its ID."""
        return self.memories.get(memory_id)

    def update(self, memory_id: str, **kwargs) -> bool:
        """Update a memory note."""
        if memory_id not in self.memories:
            return False

        note = self.memories[memory_id]

        # Update fields
        for key, value in kwargs.items():
            if hasattr(note, key):
                setattr(note, key, value)

        # Update in ChromaDB
        self._update_chromadb(memory_id)

        return True

    def delete(self, memory_id: str) -> bool:
        """Delete a memory note by its ID."""
        if memory_id in self.memories:
            # Delete from ChromaDB
            self.collection.delete(ids=[memory_id])
            # Delete from local storage
            del self.memories[memory_id]
            return True
        return False

    def _update_chromadb(self, memory_id: str):
        """Update ChromaDB entry."""
        if memory_id not in self.memories:
            return
        memory = self.memories[memory_id]
        metadata = {
            "id": memory.id,
            "content": memory.content,
            "keywords": json.dumps(memory.keywords),
            "links": json.dumps(memory.links),
            "retrieval_count": memory.retrieval_count,
            "timestamp": memory.timestamp,
            "last_accessed": memory.last_accessed,
            "context": memory.context,
            "evolution_history": json.dumps(memory.evolution_history),
            "category": memory.category,
            "tags": json.dumps(memory.tags),
        }
        # Re-embed
        text_components = [
            memory.content,
            " ".join(memory.keywords),
            " ".join(memory.tags),
            memory.context,
        ]
        combined_text = " ".join(text_components).strip()
        embedding = self.embedder.encode(
            combined_text, normalize_embeddings=True
        ).tolist()
        self.collection.update(
            ids=[memory_id],
            documents=[memory.content],
            metadatas=[metadata],  # type: ignore
            embeddings=[embedding],
        )
