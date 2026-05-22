"""
Vector store management using FAISS for efficient similarity search.

Handles:
- FAISS index creation and management
- Metadata storage and retrieval
- Similarity search with metadata filtering
- Index persistence (save/load)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from faiss import read_index, write_index

try:
    import faiss
except ImportError:
    faiss = None

from utils.config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class VectorStore:
    """FAISS-based vector store with metadata management."""

    def __init__(self, embedding_dim: int = 384, index_path: Optional[str] = None):
        """
        Initialize vector store.

        Args:
            embedding_dim: Dimension of embeddings (default: 384)
            index_path: Path to save FAISS index (default: from settings)

        Raises:
            ImportError: If FAISS is not installed
            ValueError: If embedding_dim is invalid
        """
        if faiss is None:
            raise ImportError("FAISS is not installed. Install with: pip install faiss-cpu")

        if embedding_dim <= 0:
            raise ValueError("Embedding dimension must be positive")

        self.embedding_dim = embedding_dim
        self.index_path = Path(index_path or settings.faiss_index_path)
        self.metadata_path = Path(settings.metadata_index_path)

        # Create directories if they don't exist
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(embedding_dim)  # L2 distance
        self.metadata: List[Dict] = []

        logger.info(f"Initialized vector store with dimension {embedding_dim}")

    def add_embeddings(
        self, embeddings: np.ndarray, metadata: List[Dict], ids: Optional[List[str]] = None
    ) -> List[int]:
        """
        Add embeddings and metadata to the vector store.

        Args:
            embeddings: Numpy array of embeddings (n x embedding_dim)
            metadata: List of metadata dictionaries for each embedding
            ids: Optional list of IDs for tracking

        Returns:
            List of vector IDs assigned

        Raises:
            ValueError: If shapes don't match
            TypeError: If inputs have wrong types
        """
        if not isinstance(embeddings, np.ndarray):
            raise TypeError("Embeddings must be numpy array")

        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: {embeddings.shape[1]} vs {self.embedding_dim}"
            )

        if len(embeddings) != len(metadata):
            raise ValueError(f"Number of embeddings ({len(embeddings)}) must match metadata ({len(metadata)})")

        # Normalize embeddings for L2 distance (equivalent to cosine similarity)
        embeddings_normalized = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)

        # Add to FAISS index
        self.index.add(embeddings_normalized.astype(np.float32))

        # Store metadata
        start_idx = len(self.metadata)
        assigned_ids = []

        for i, meta in enumerate(metadata):
            vector_id = start_idx + i
            meta_with_id = {
                **meta,
                "vector_id": vector_id,
                "id": ids[i] if ids else meta.get("id", f"chunk_{vector_id}"),
            }
            self.metadata.append(meta_with_id)
            assigned_ids.append(vector_id)

        logger.info(f"Added {len(embeddings)} embeddings to vector store")
        return assigned_ids

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        similarity_threshold: Optional[float] = None,
        metadata_filters: Optional[Dict] = None,
    ) -> Tuple[List[Dict], List[float]]:
        """
        Search for similar embeddings.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return (default: 5)
            similarity_threshold: Optional minimum similarity threshold (0-1)
            metadata_filters: Optional metadata filter dictionary
                Example: {"crops": "wheat", "language": "en"}

        Returns:
            Tuple of (list of metadata dicts, list of similarity scores)

        Raises:
            ValueError: If query_embedding has wrong dimension
        """
        if query_embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Query embedding dimension mismatch: {query_embedding.shape[0]} vs {self.embedding_dim}"
            )

        # Normalize query embedding
        query_normalized = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        query_normalized = query_normalized.astype(np.float32).reshape(1, -1)

        # Search FAISS with a generous over-fetch to survive deduplication and
        # filtering.  We ask for min(k*6, total) candidates so that even when
        # many duplicates or filtered entries are present we still fill k slots.
        fetch_k = min(k * 6, len(self.metadata))
        distances, indices = self.index.search(query_normalized, fetch_k)

        # Convert L2 distances to similarity scores (cosine)
        # L2 distance to cosine similarity: sim = 1 - (dist^2 / 2)
        similarities = 1.0 - (distances[0] ** 2) / 2.0
        similarities = np.clip(similarities, -1.0, 1.0)  # Clamp to valid range

        # Apply filtering, deduplication, and thresholding
        results = []
        scores = []
        seen_content: set = set()  # track content hashes to skip duplicates

        for idx, similarity in zip(indices[0], similarities):
            if idx == -1 or idx >= len(self.metadata):
                continue

            # Check similarity threshold
            if similarity_threshold and similarity < similarity_threshold:
                continue

            # Check metadata filters — only enforce keys that actually exist in
            # the metadata so missing fields don't silently drop all results.
            metadata = self.metadata[idx]
            if metadata_filters and not self._match_filters(metadata, metadata_filters):
                continue

            # Deduplicate on content (strip + first 200 chars as fingerprint)
            content_fp = metadata.get("content", "")[:200].strip()
            if content_fp and content_fp in seen_content:
                continue
            seen_content.add(content_fp)

            results.append(metadata)
            scores.append(float(similarity))

            if len(results) == k:
                break

        logger.info(f"Search returned {len(results)} results (fetch_k={fetch_k})")
        return results, scores

    def search_batch(
        self,
        query_embeddings: np.ndarray,
        k: int = 5,
        similarity_threshold: Optional[float] = None,
        metadata_filters: Optional[Dict] = None,
    ) -> List[Tuple[List[Dict], List[float]]]:
        """
        Search for multiple query embeddings.

        Args:
            query_embeddings: Array of query embeddings (n x embedding_dim)
            k: Number of results per query
            similarity_threshold: Optional minimum similarity
            metadata_filters: Optional metadata filters

        Returns:
            List of (results, scores) tuples

        Raises:
            ValueError: If shape is invalid
        """
        if query_embeddings.shape[1] != self.embedding_dim:
            raise ValueError("Query embedding dimension mismatch")

        results_list = []
        for i in range(query_embeddings.shape[0]):
            results, scores = self.search(
                query_embeddings[i],
                k=k,
                similarity_threshold=similarity_threshold,
                metadata_filters=metadata_filters,
            )
            results_list.append((results, scores))

        return results_list

    def _match_filters(self, metadata: Dict, filters: Dict) -> bool:
        """
        Check if metadata matches all filters.

        Args:
            metadata: Metadata dictionary to check
            filters: Filter dictionary

        Returns:
            True if all filters match
        """
        for key, value in filters.items():
            # If the metadata record doesn't have this field at all, skip the
            # filter rather than excluding the result.  Documents indexed before
            # the filter fields were added would otherwise be invisible.
            if key not in metadata:
                continue

            meta_value = metadata[key]

            # Handle list values (e.g., crops list contains "wheat")
            if isinstance(meta_value, list):
                if isinstance(value, list):
                    if not any(v in meta_value for v in value):
                        return False
                else:
                    if value not in meta_value:
                        return False
            else:
                if meta_value != value:
                    return False

        return True

    def save(self) -> bool:
        """
        Save FAISS index and metadata to disk.

        Returns:
            True if successful

        Raises:
            Exception: If save fails
        """
        try:
            # Save FAISS index
            write_index(self.index, str(self.index_path))
            logger.info(f"Saved FAISS index to {self.index_path}")

            # Save metadata
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved metadata to {self.metadata_path}")

            return True

        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise

    def load(self) -> bool:
        """
        Load FAISS index and metadata from disk.

        Returns:
            True if successful

        Raises:
            FileNotFoundError: If files don't exist
            Exception: If load fails
        """
        try:
            if not self.index_path.exists():
                raise FileNotFoundError(f"FAISS index not found: {self.index_path}")

            if not self.metadata_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

            # Load FAISS index
            self.index = read_index(str(self.index_path))
            logger.info(f"Loaded FAISS index from {self.index_path}")

            # Load metadata
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded {len(self.metadata)} metadata entries")

            return True

        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise

    def get_stats(self) -> Dict:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with stats
        """
        return {
            "total_vectors": self.index.ntotal,
            "total_metadata": len(self.metadata),
            "embedding_dimension": self.embedding_dim,
            "index_path": str(self.index_path),
            "metadata_path": str(self.metadata_path),
        }

    def reset(self):
        """Reset the vector store."""
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.metadata = []
        logger.info("Vector store reset")


def create_vector_store(embedding_dim: int = 384, index_path: Optional[str] = None) -> VectorStore:
    """
    Create a vector store instance.

    Args:
        embedding_dim: Embedding dimension
        index_path: Optional path for index

    Returns:
        VectorStore instance

    Raises:
        ImportError: If FAISS not installed
    """
    return VectorStore(embedding_dim, index_path)


if __name__ == "__main__":
    # Example usage
    print("Creating vector store...")
    vector_store = create_vector_store(embedding_dim=384)

    # Create sample embeddings
    print("Creating sample embeddings...")
    embeddings = np.random.randn(10, 384).astype(np.float32)
    
    metadata = [
        {
            "content": f"Sample text {i}",
            "crop": "wheat" if i % 2 == 0 else "paddy",
            "language": "en",
        }
        for i in range(10)
    ]

    # Add to vector store
    print("Adding embeddings...")
    ids = vector_store.add_embeddings(embeddings, metadata)
    print(f"Added {len(ids)} embeddings")

    # Search
    print("\nSearching...")
    query = embeddings[0]
    results, scores = vector_store.search(query, k=3)
    
    print(f"Found {len(results)} results:")
    for i, (result, score) in enumerate(zip(results, scores)):
        print(f"  {i+1}. Score: {score:.4f}, Content: {result['content']}")

    # Stats
    print("\nVector store stats:")
    print(vector_store.get_stats())
