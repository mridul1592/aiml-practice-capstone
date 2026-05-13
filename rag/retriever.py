"""
Retriever for semantic search with language detection and filtering.

Handles:
- Multi-language query processing
- Language detection
- Metadata-aware retrieval
- Query normalization and preprocessing
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from rag.embedder import Embedder
from rag.vector_store import VectorStore
from utils.config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class LanguageDetector:
    """Detect language of text."""

    def __init__(self):
        """Initialize language detector."""
        self.supported_languages = settings.supported_languages

    def detect_language(self, text: str, threshold: float = 0.5) -> str:
        """
        Detect language of text using multiple heuristics.

        Args:
            text: Text to analyze
            threshold: Confidence threshold (0-1)

        Returns:
            Language code (en, hi, pa) or 'en' as default
        """
        if not text:
            return "en"

        text_lower = text.lower()

        # Simple pattern-based detection
        hindi_score = self._score_hindi(text)
        punjabi_score = self._score_punjabi(text)
        english_score = self._score_english(text)

        scores = {
            "hi": hindi_score,
            "pa": punjabi_score,
            "en": english_score,
        }

        # Get language with highest score
        detected_lang = max(scores, key=scores.get)

        # If confidence below threshold, default to English
        if scores[detected_lang] < threshold:
            detected_lang = "en"

        logger.info(f"Detected language: {detected_lang} (scores: {scores})")
        return detected_lang

    def _score_hindi(self, text: str) -> float:
        """Score text for Hindi language (Devanagari script)."""
        import re

        devanagari_pattern = r"[ा-ॿ]"
        matches = len(re.findall(devanagari_pattern, text))
        return matches / max(len(text), 1)

    def _score_punjabi(self, text: str) -> float:
        """Score text for Punjabi language (Gurmukhi script)."""
        import re

        gurmukhi_pattern = r"[ਅ-ੱ]"
        matches = len(re.findall(gurmukhi_pattern, text))
        return matches / max(len(text), 1)

    def _score_english(self, text: str) -> float:
        """Score text for English language."""
        import re

        english_pattern = r"[a-zA-Z]"
        matches = len(re.findall(english_pattern, text))
        return matches / max(len(text), 1)


class QueryPreprocessor:
    """Preprocess queries for better retrieval."""

    def __init__(self):
        """Initialize query preprocessor."""
        self.lang_detector = LanguageDetector()

    def preprocess(self, query: str) -> Dict[str, str]:
        """
        Preprocess query.

        Args:
            query: Raw query text

        Returns:
            Dictionary with preprocessed query and metadata
        """
        # Detect language
        language = self.lang_detector.detect_language(query)

        # Normalize whitespace
        normalized = " ".join(query.split())

        # Remove special characters (but keep script characters)
        import re

        cleaned = re.sub(r"[^\w\s\u0900-\u097F\u0A00-\u0A7F]", " ", normalized)
        cleaned = " ".join(cleaned.split())

        return {
            "original": query,
            "normalized": cleaned,
            "language": language,
        }


class Retriever:
    """Semantic retriever with language support."""

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
            query: Query text (any language)
            k: Number of results
            similarity_threshold: Minimum similarity score
            filters: Metadata filters (e.g., crop, language, region)
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
        logger.info(f"Processed query: {processed['normalized']} (language: {processed['language']})")

        # Add language to filters if not specified
        if filters is None:
            filters = {}

        if "language" not in filters:
            filters["language"] = processed["language"]

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

    def retrieve_by_language(
        self,
        query: str,
        language: str,
        k: int = 5,
        similarity_threshold: Optional[float] = None,
    ) -> List[Dict]:
        """
        Retrieve documents in a specific language.

        Args:
            query: Query text
            language: Language code (en, hi, pa)
            k: Number of results
            similarity_threshold: Minimum similarity

        Returns:
            List of relevant chunks in target language
        """
        filters = {"language": language}
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

        Example:
            results = retriever.retrieve_by_metadata(
                "Pest control methods",
                crops=["wheat"],
                regions=["punjab"],
                seasons=["kharif"]
            )
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

    def get_stats(self) -> Dict:
        """
        Get retriever statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "embedding_model": self.embedder.get_model_info(),
            "vector_store": self.vector_store.get_stats(),
        }


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


if __name__ == "__main__":
    # Example usage
    from rag.embedder import Embedder
    from rag.vector_store import VectorStore

    print("Creating retriever...")
    embedder = Embedder()
    vector_store = VectorStore(embedding_dim=384)

    # Add some sample data
    embeddings = np.random.randn(5, 384).astype(np.float32)
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)

    metadata = [
        {
            "content": "Wheat pests control methods",
            "crop": "wheat",
            "language": "en",
            "diseases": ["shoot_fly"],
        },
        {
            "content": "Paddy irrigation schedule",
            "crop": "paddy",
            "language": "en",
            "seasons": ["kharif"],
        },
    ]

    vector_store.add_embeddings(embeddings[:2], metadata)

    # Create retriever
    retriever = create_retriever(embedder, vector_store)

    # Test retrieval
    print("\nTesting retrieval...")
    results = retriever.retrieve("How to control pests in wheat?", k=2)

    print(f"Found {len(results)} results:")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result.get('content', 'N/A')}")
        print(f"     Similarity: {result.get('similarity_score', 'N/A'):.4f}")
