"""
embedding.py

Semantic embedding and vector index management for PRIMA.

This module is responsible for:
- Encoding text into dense vectors
- Maintaining a similarity index (FAISS)
- Mapping memory IDs to vector positions

This is a derived layer: embeddings can be recomputed at any time.
"""

from typing import List, Tuple
import numpy as np

import faiss
from sentence_transformers import SentenceTransformer


class EmbeddingIndex:
    """
    Manages vector embeddings and similarity search for MemoryNotes.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 384,
    ):
        """
        Initialize embedding model and FAISS index.

        Args:
            model_name (str):
                HuggingFace / SentenceTransformer model name.
            embedding_dim (int):
                Dimensionality of the embedding vectors.
        """
        self.model = SentenceTransformer(model_name)

        # FAISS index using cosine similarity
        self.index = faiss.IndexFlatIP(embedding_dim)

        # ID <-> index mapping
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

    # -----------------------------
    # Index management
    # -----------------------------

    def add(self, memory_id: str, embedding: List[float]) -> None:
        """
        Add a memory embedding to the index.

        Args:
            memory_id (str): MemoryNote ID
            embedding (List[float]): Dense vector
        """
        if memory_id in self.id_to_pos:
            # Avoid duplicate inserts
            return

        vec = np.array([embedding], dtype="float32")
        self.index.add(vec)

        self.id_to_pos[memory_id] = self._next_pos
        self.pos_to_id[self._next_pos] = memory_id
        self._next_pos += 1

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
        if self.index.ntotal == 0:
            return []

        query = np.array([query_embedding], dtype="float32")
        scores, indices = self.index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            memory_id = self.pos_to_id.get(idx)
            if memory_id:
                results.append((memory_id, float(score)))

        return results

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
        self.index.reset()
        self.id_to_pos.clear()
        self.pos_to_id.clear()
        self._next_pos = 0

        for memory_id, embedding in embeddings:
            self.add(memory_id, embedding)
