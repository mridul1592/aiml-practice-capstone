"""
Retriever for semantic search with metadata-aware retrieval.

Handles:
- Query embedding and similarity search
- Metadata-aware retrieval
- Query normalization and preprocessing
"""

import re
from typing import Dict, List, Optional

from rag.embedder import Embedder
from rag.vector_store import VectorStore
from utils.logger import setup_logger

logger = setup_logger(__name__)


class QueryPreprocessor:
    """Preprocess queries for better retrieval."""

    def preprocess(self, query: str) -> Dict[str, str]:
        """
        Preprocess query.

        Args:
            query: Raw query text

        Returns:
            Dictionary with preprocessed query
        """
        # Normalize whitespace
        normalized = " ".join(query.split())

        # Remove special characters
        cleaned = re.sub(r"[^\w\s]", " ", normalized)
        cleaned = " ".join(cleaned.split())

        return {
            "original": query,
            "normalized": cleaned,
        }


class Retriever:
    """Semantic retriever for English text."""

    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        """
        Initialize retriever.

        Args:
            embedder: Embedder instance
            vector_store: VectorStore instance

        Raises:
            ValueError: If inputs are None
        """
        if embedder is None or vector_store is None:
            raise ValueError("Embedder and VectorStore cannot be None")

        self.embedder = embedder
        self.vector_store = vector_store
        self.preprocessor = QueryPreprocessor()

        logger.info("Retriever initialized")

    def retrieve(
        self,
        query: str,
        k: int = 5,
        similarity_threshold: Optional[float] = None,
        filters: Optional[Dict] = None,
        return_scores: bool = True,
    ) -> List[Dict]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Query text
            k: Number of results
            similarity_threshold: Minimum similarity score
            filters: Metadata filters (e.g., crop, region)
            return_scores: Whether to include similarity scores

        Returns:
            List of relevant chunks with scores

        Raises:
            ValueError: If query is empty
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        logger.info(f"Retrieving for query: {query}")

        # Preprocess query
        processed = self.preprocessor.preprocess(query)
        logger.info(f"Processed query: {processed['normalized']}")

        # Generate query embedding
        query_embedding = self.embedder.embed_text(processed["normalized"])

        # Search in vector store
        results, scores = self.vector_store.search(
            query_embedding, k=k, similarity_threshold=similarity_threshold, metadata_filters=filters
        )

        # Format results
        formatted_results = []
        for result, score in zip(results, scores):
            formatted_result = {
                **result,
                "similarity_score": float(score),
            }

            if not return_scores:
                formatted_result.pop("similarity_score", None)

            formatted_results.append(formatted_result)

        logger.info(f"Retrieved {len(formatted_results)} results")
        return formatted_results

    def retrieve_by_crop(
        self,
        query: str,
        crop: str,
        k: int = 5,
        similarity_threshold: Optional[float] = None,
    ) -> List[Dict]:
        """
        Retrieve documents for a specific crop.

        Args:
            query: Query text
            crop: Crop name (wheat, paddy, etc.)
            k: Number of results
            similarity_threshold: Minimum similarity

        Returns:
            List of relevant chunks for the crop
        """
        filters = {"crop": crop}
        return self.retrieve(query, k, similarity_threshold, filters)

    def retrieve_by_metadata(
        self,
        query: str,
        k: int = 5,
        similarity_threshold: Optional[float] = None,
        crops: Optional[List[str]] = None,
        diseases: Optional[List[str]] = None,
        regions: Optional[List[str]] = None,
        seasons: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Retrieve with multiple metadata filters.

        Args:
            query: Query text
            k: Number of results
            similarity_threshold: Minimum similarity
            crops: List of crops to filter by
            diseases: List of diseases to filter by
            regions: List of regions to filter by
            seasons: List of seasons to filter by

        Returns:
            List of filtered results
        """
        filters = {}

        if crops:
            filters["crops"] = crops
        if diseases:
            filters["diseases"] = diseases
        if regions:
            filters["regions"] = regions
        if seasons:
            filters["seasons"] = seasons

        return self.retrieve(query, k, similarity_threshold, filters)


def create_retriever(embedder: Embedder, vector_store: VectorStore) -> Retriever:
    """
    Create a retriever instance.

    Args:
        embedder: Embedder instance
        vector_store: VectorStore instance

    Returns:
        Retriever instance

    Raises:
        ValueError: If inputs are None
    """
    return Retriever(embedder, vector_store)
