"""
embedding.py

Semantic embedding and vector index management for PRIMA.

This module is responsible for:
- Encoding text into dense vectors
- Maintaining a similarity index (ChromaDB)
- Mapping memory IDs to vector positions

This is a derived layer: embeddings can be recomputed at any time.
"""

from typing import List, Tuple

import chromadb
from sentence_transformers import SentenceTransformer

from prima_memory.core.note import MemoryNote


class EmbeddingIndex:
    """
    Manages vector embeddings and similarity search for MemoryNotes.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_name: str = "memory_embeddings",
    ):
        """
        Initialize embedding model and ChromaDB collection.

        Args:
            model_name (str):
                HuggingFace / SentenceTransformer model name.
            collection_name (str):
                Name of the ChromaDB collection.
        """
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=collection_name)

        # ID mapping (ChromaDB handles this internally)
        self.id_to_pos: dict[str, int] = {}
        self.pos_to_id: dict[int, str] = {}

        self._next_pos = 0

    # -----------------------------
    # Embedding
    # -----------------------------

    def embed_text(self, text: str) -> List[float]:
        """
        Generate a normalized embedding for a given text.
        """
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_memory_note(self, note: MemoryNote) -> List[float]:
        """
        Generate embedding for a memory note using all textual components.

        Follows A-MEM approach: concat(content, keywords, tags, context)
        """
        # Concatenate all textual components as in A-MEM
        text_components = [
            note.content,
            " ".join(note.keywords) if note.keywords else "",
            " ".join(note.tags) if note.tags else "",
            note.context if note.context else "",
        ]

        combined_text = " ".join(text_components).strip()

        return self.embed_text(combined_text)

    # -----------------------------
    # Index management
    # -----------------------------

    def add(
        self, memory_id: str, embedding: List[float], metadata: dict = None
    ) -> None:
        """
        Add a memory embedding to the index.

        Args:
            memory_id (str): MemoryNote ID
            embedding (List[float]): Dense vector
            metadata (dict): Optional metadata
        """
        if metadata is None:
            metadata = {"type": "memory"}

        self.collection.add(
            embeddings=[embedding],  # type: ignore[arg-type]
            ids=[memory_id],
            metadatas=[metadata],
        )

    # -----------------------------
    # Similarity search
    # -----------------------------

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Search for the top-k most similar memories.

        Args:
            query_embedding (List[float]):
                Query vector
            top_k (int):
                Number of neighbors to retrieve

        Returns:
            List of (memory_id, similarity_score)
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],  # type: ignore[arg-type]
            n_results=top_k,
        )

        if not results["ids"][0]:
            return []

        memory_results = []
        for memory_id, distance in zip(results["ids"][0], results["distances"][0]):
            # ChromaDB returns cosine distance, convert to similarity
            similarity = 1.0 - distance
            memory_results.append((memory_id, similarity))

        return memory_results

    # -----------------------------
    # Rebuilding (important later)
    # -----------------------------

    def rebuild(self, embeddings: List[Tuple[str, List[float]]]) -> None:
        """
        Rebuild the entire index from scratch.

        Used after:
        - bulk memory load
        - major evolution changes
        """
        # Delete existing collection and recreate
        try:
            self.client.delete_collection(self.collection.name)
        except Exception as e:
            # tolerate missing collection or deletion issues
            print(f"Chroma collection delete error: {e}")
        self.collection = self.client.create_collection(name=self.collection.name)

        for memory_id, embedding in embeddings:
            self.add(memory_id, embedding)
