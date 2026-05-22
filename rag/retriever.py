"""
Retriever for semantic search with language detection and filtering.

Handles:
- Multi-language query processing
- Language detection
- Metadata-aware retrieval
- Query normalization and preprocessing
- Hybrid search: BM25 keyword + FAISS semantic, merged with Reciprocal Rank Fusion
"""

import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from rag.embedder import Embedder
from rag.vector_store import VectorStore
from utils.config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class LanguageDetector:
    """Detect language of text supporting 10 Indian languages."""

    def __init__(self):
        """Initialize language detector."""
        self.supported_languages = settings.supported_languages

    def detect_language(self, text: str, threshold: float = 0.05) -> str:
        """
        Detect language of text using script-based detection.

        Args:
            text: Text to analyze
            threshold: Confidence threshold (0-1)

        Returns:
            Language code (en, hi, pa, ta, te, or, kn, mr, ml, bn) or 'en' as default
        """
        if not text:
            return "en"

        # Score all supported languages
        scores = {
            "en": self._score_english(text),
            "hi": self._score_hindi(text),  # Devanagari (Hindi, Marathi)
            "pa": self._score_punjabi(text),  # Gurmukhi
            "bn": self._score_bengali(text),  # Bengali
            "or": self._score_odia(text),  # Odia
            "ta": self._score_tamil(text),  # Tamil
            "te": self._score_telugu(text),  # Telugu
            "kn": self._score_kannada(text),  # Kannada
            "ml": self._score_malayalam(text),  # Malayalam
            "mr": self._score_marathi(text),  # Marathi (uses Devanagari like Hindi)
        }

        # Get language with highest score
        detected_lang = max(scores, key=scores.get)

        # If confidence below threshold, default to English
        if scores[detected_lang] < threshold:
            detected_lang = "en"

        logger.info(f"Detected language: {detected_lang} (scores: {scores})")
        return detected_lang

    def _score_english(self, text: str) -> float:
        """Score text for English language."""
        import re
        pattern = r"[a-zA-Z]"
        matches = len(re.findall(pattern, text))
        return matches / len(text) if text else 0

    def _score_hindi(self, text: str) -> float:
        """Score text for Hindi (Devanagari script)."""
        import re
        pattern = r"[ऀ-ॿ]"
        matches = len(re.findall(pattern, text))
        return matches / len(text) if text else 0

    def _score_marathi(self, text: str) -> float:
        """Score text for Marathi (Devanagari script)."""
        import re
        # Marathi uses Devanagari, same as Hindi
        pattern = r"[ऀ-ॿ]"
        matches = len(re.findall(pattern, text))
        return matches / len(text) if text else 0

    def _score_punjabi(self, text: str) -> float:
        """Score text for Punjabi (Gurmukhi script)."""
        import re
        pattern = r"[਀-੿]"
        matches = len(re.findall(pattern, text))
        return matches / len(text) if text else 0

    def _score_bengali(self, text: str) -> float:
        """Score text for Bengali (Bengali script)."""
        import re
        pattern = r"[অ-৿]"  # Bengali Unicode range U+0980–U+09FF
        matches = len(re.findall(pattern, text))
        return matches / len(text) if text else 0

    def _score_odia(self, text: str) -> float:
        """Score text for Odia (Odia script)."""
        import re
        pattern = r"[ଅ-ୟ]"  # Odia Unicode range U+0B00–U+0B7F
        matches = len(re.findall(pattern, text))
        return matches / len(text) if text else 0

    def _score_tamil(self, text: str) -> float:
        """Score text for Tamil (Tamil script)."""
        import re
        pattern = r"[அ-ி]"  # Tamil Unicode range U+0B80–U+0BFF
        matches = len(re.findall(pattern, text))
        return matches / len(text) if text else 0

    def _score_telugu(self, text: str) -> float:
        """Score text for Telugu (Telugu script)."""
        import re
        pattern = r"[అ-౿]"  # Telugu Unicode range U+0C00–U+0C7F
        matches = len(re.findall(pattern, text))
        return matches / len(text) if text else 0

    def _score_kannada(self, text: str) -> float:
        """Score text for Kannada (Kannada script)."""
        import re
        pattern = r"[ಅ-ಿ]"  # Kannada Unicode range U+0C80–U+0CFF
        matches = len(re.findall(pattern, text))
        return matches / len(text) if text else 0

    def _score_malayalam(self, text: str) -> float:
        """Score text for Malayalam (Malayalam script)."""
        import re
        pattern = r"[അ-ൿ]"  # Malayalam Unicode range U+0D00–U+0D7F
        matches = len(re.findall(pattern, text))
        return matches / len(text) if text else 0


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
        # Detect language FIRST (before any cleaning)
        language = self.lang_detector.detect_language(query)

        # Normalize whitespace
        normalized = " ".join(query.split())

        # Remove special characters (but keep script characters)
        # Keep: alphanumeric, spaces, Hindi (Devanagari), Punjabi (Gurmukhi)
        import re
        cleaned = re.sub(
            r"[^\w\s\u0900-\u097F\u0A00-\u0A7F]",
            " ",
            normalized,
            flags=re.UNICODE
        )
        cleaned = " ".join(cleaned.split())

        return {
            "original": query,
            "normalized": cleaned,
            "language": language,
        }


class BM25Index:
    """
    Lightweight BM25 keyword index built on top of the existing vector-store
    metadata.  Lazily initialised on first use so cold-start cost is paid only
    when hybrid search is actually requested.

    Tokenisation strategy:
    - Lower-case the text
    - Split on whitespace and non-alphanumeric boundaries
    - Remove common English stop-words (they add noise to BM25 scores)
    - Keep tokens ≥ 2 characters
    This works for English and for the ASCII-portion of Indian-language docs.
    """

    _TOKEN_RE = re.compile(r"[^\w]+", re.UNICODE)

    # Common English stop-words that carry no domain signal
    _STOP_WORDS = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can", "not", "no",
        "how", "what", "why", "when", "where", "which", "who", "whom",
        "this", "that", "these", "those", "it", "its", "as", "if", "so",
        "up", "out", "about", "into", "then", "than", "also", "just",
    }

    def __init__(self):
        self._bm25 = None          # BM25Okapi instance
        self._index_to_meta: List[Dict] = []   # parallel list to bm25 corpus

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, metadata_list: List[Dict]) -> None:
        """
        Build the BM25 corpus from a list of metadata dicts (same list that
        lives in VectorStore.metadata).

        Args:
            metadata_list: list of dicts, each with at least a 'content' key
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank-bm25 is required for hybrid search. "
                "Install with: pip install rank-bm25"
            )

        corpus: List[List[str]] = []
        self._index_to_meta = []

        for doc in metadata_list:
            content = doc.get("content", "")
            tokens = self._tokenize(content)
            corpus.append(tokens)
            self._index_to_meta.append(doc)

        self._bm25 = BM25Okapi(corpus)
        logger.info(f"BM25 index built: {len(corpus)} documents")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 20) -> List[Tuple[Dict, float]]:
        """
        Run a BM25 keyword search.

        Args:
            query: Raw query text
            k:     Number of top results to return

        Returns:
            List of (metadata_dict, bm25_score) tuples, sorted descending.
        """
        if self._bm25 is None:
            raise RuntimeError("BM25 index has not been built yet. Call build() first.")

        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)          # ndarray, one score per doc

        # Pair with metadata and sort
        paired = sorted(
            zip(self._index_to_meta, scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return paired[:k]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @classmethod
    def _tokenize(cls, text: str) -> List[str]:
        """Lower-case, split on non-word chars, remove stop-words, keep tokens ≥ 2 chars."""
        tokens = cls._TOKEN_RE.split(text.lower())
        return [t for t in tokens if len(t) >= 2 and t not in cls._STOP_WORDS]

    @property
    def is_built(self) -> bool:
        return self._bm25 is not None


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion helper
# ---------------------------------------------------------------------------

def _reciprocal_rank_fusion(
    ranked_lists: List[List[Dict]],
    k_rrf: int = 60,
) -> List[Tuple[Dict, float]]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion.

    RRF score for document d = sum_r [ 1 / (k_rrf + rank_r(d)) ]
    where rank_r is the 1-based position of d in ranked list r.

    Args:
        ranked_lists: Each element is an ordered list of metadata dicts.
                      A dict is identified by its 'id' or 'vector_id' field.
        k_rrf:        Smoothing constant (default 60, standard in literature).

    Returns:
        List of (metadata_dict, rrf_score) sorted by rrf_score descending.
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            doc_id = str(doc.get("id") or doc.get("vector_id") or id(doc))
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank)
            doc_map[doc_id] = doc

    merged = sorted(doc_map.keys(), key=lambda d: scores[d], reverse=True)
    return [(doc_map[d], scores[d]) for d in merged]


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
        self._bm25_index = BM25Index()   # built lazily on first hybrid call

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

        # Generate query embedding (uses instruction prefix for instruct models)
        query_embedding = self.embedder.embed_query(processed["normalized"])

        # Search in vector store (don't filter by language if documents don't have it)
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

    # ------------------------------------------------------------------
    # Hybrid retrieval (BM25 + semantic → RRF)
    # ------------------------------------------------------------------

    def retrieve_hybrid(
        self,
        query: str,
        k: int = 10,
        similarity_threshold: Optional[float] = None,
        filters: Optional[Dict] = None,
        return_scores: bool = True,
        bm25_candidates: int = 40,
        semantic_candidates: int = 40,
        rrf_k: int = 60,
    ) -> List[Dict]:
        """
        Hybrid retrieval: run BM25 keyword search and FAISS semantic search
        independently, then merge the candidate lists using Reciprocal Rank
        Fusion (RRF) and return the top-k deduplicated results.

        Why this helps:
        - Semantic search excels at paraphrased / conceptual matches.
        - BM25 excels at exact keyword matches (crop names, pesticide names,
          technical terms) that may not rank high in embedding space.
        - RRF rewards documents that rank well in *both* lists, combining
          the strengths of each approach.

        Args:
            query:               Query text (any language)
            k:                   Final number of results to return
            similarity_threshold: Applied to semantic leg only
            filters:             Metadata filters applied to *both* legs
            return_scores:       Whether to attach scores to results
            bm25_candidates:     Candidates retrieved from BM25 before RRF
            semantic_candidates: Candidates retrieved from FAISS before RRF
            rrf_k:               RRF smoothing constant (default 60)

        Returns:
            List of chunk dicts with 'similarity_score' (RRF score) field
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        logger.info(f"Hybrid retrieval for: {query[:80]}")

        # ── 1. Preprocess query ──────────────────────────────────────────────
        processed = self.preprocessor.preprocess(query)
        normalized_query = processed["normalized"]

        # ── 2. Semantic leg ──────────────────────────────────────────────────
        # embed_query adds instruction prefix for instruct-tuned models
        query_embedding = self.embedder.embed_query(normalized_query)
        sem_results, _ = self.vector_store.search(
            query_embedding,
            k=semantic_candidates,
            similarity_threshold=similarity_threshold,
            metadata_filters=filters,
        )
        logger.info(f"Semantic leg: {len(sem_results)} candidates")

        # ── 3. BM25 leg ──────────────────────────────────────────────────────
        if not self._bm25_index.is_built:
            logger.info("Building BM25 index from vector store metadata…")
            self._bm25_index.build(self.vector_store.metadata)

        bm25_raw = self._bm25_index.search(normalized_query, k=bm25_candidates)

        # Apply metadata filters to BM25 results (same semantics as FAISS leg)
        if filters:
            bm25_raw = [
                (doc, score) for doc, score in bm25_raw
                if self.vector_store._match_filters(doc, filters)
            ]

        bm25_results = [doc for doc, _ in bm25_raw]
        logger.info(f"BM25 leg: {len(bm25_results)} candidates")

        # ── 4. Reciprocal Rank Fusion ────────────────────────────────────────
        merged = _reciprocal_rank_fusion([sem_results, bm25_results], k_rrf=rrf_k)

        # ── 5. Content-based deduplication (same as VectorStore.search) ──────
        seen: set = set()
        final: List[Dict] = []
        for doc, rrf_score in merged:
            fp = doc.get("content", "")[:200].strip()
            if fp and fp in seen:
                continue
            seen.add(fp)
            entry = {**doc, "similarity_score": round(rrf_score, 6)}
            if not return_scores:
                entry.pop("similarity_score", None)
            final.append(entry)
            if len(final) == k:
                break

        logger.info(
            f"Hybrid retrieval complete: {len(final)} results "
            f"(from {len(sem_results)} semantic + {len(bm25_results)} BM25 candidates)"
        )
        return final

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
