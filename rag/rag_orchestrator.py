"""
RAG orchestration layer.

Coordinates:
- Embedder loading
- Vector store loading
- Retriever execution
- LLM response generation
- Source attribution (filename enrichment from processed-JSON range map)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rag.embedder import Embedder
from rag.generator import ResponseGenerator, create_generator
from rag.reranker import BGEReranker, create_reranker
from rag.retriever import Retriever
from rag.vector_store import VectorStore
from utils.config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Source-map helpers — map vector_id → source PDF filename at runtime
# without requiring a re-index.
# ---------------------------------------------------------------------------

def _build_source_map(processed_dir: str = "./data/processed") -> List[Tuple[int, int, str]]:
    """
    Build a list of (start_vector_id, end_vector_id, filename) tuples by
    reading the processed JSON files in sorted order (same order the
    EmbeddingPipeline used when building the FAISS index).

    Returns an empty list if the processed directory doesn't exist or is empty.
    """
    p = Path(processed_dir)
    if not p.exists():
        return []

    source_map: List[Tuple[int, int, str]] = []
    running_id = 0

    for json_file in sorted(p.glob("*_processed.json")):
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            source_file = data.get("source_file", json_file.stem.replace("_processed", ""))
            n = data.get("total_chunks", len(data.get("chunks", [])))
            if n > 0:
                source_map.append((running_id, running_id + n - 1, source_file))
                running_id += n
        except Exception as exc:
            logger.warning(f"Could not read {json_file}: {exc}")

    logger.info(f"Source map built: {len(source_map)} documents, {running_id} total chunks")
    return source_map


def _lookup_filename(vector_id: int, source_map: List[Tuple[int, int, str]]) -> str:
    """Return the source PDF filename for a given vector_id, or 'Unknown'."""
    for start, end, filename in source_map:
        if start <= vector_id <= end:
            return filename
    return "Unknown"


def _enrich_chunks_with_filename(
    chunks: List[Dict], source_map: List[Tuple[int, int, str]]
) -> List[Dict]:
    """
    Add or replace the 'filename' key on every chunk dict that is missing or
    set to the 'Unknown' sentinel, using the source_map range lookup.

    Note: 'Unknown' is a truthy string so `not chunk.get('filename')` would
    incorrectly skip it — we check both conditions explicitly.
    """
    if not source_map:
        return chunks
    enriched = []
    for chunk in chunks:
        fname = chunk.get("filename")
        if not fname or fname == "Unknown":
            vid = chunk.get("vector_id", -1)
            chunk = {**chunk, "filename": _lookup_filename(vid, source_map)}
        enriched.append(chunk)
    return enriched


class RAGPipeline:
    """End-to-end RAG pipeline for query processing."""

    def __init__(
        self,
        embedding_dim: Optional[int] = None,
        index_path: Optional[str] = None,
        llm_provider: Optional[str] = None,
        llm_model_name: Optional[str] = None,
    ):
        """
        Initialize RAG pipeline.

        Args:
            embedding_dim: Embedding dimension (default: from settings)
            index_path: Optional path to FAISS index
            llm_provider: LLM provider (ollama, openai) (default: from settings)
            llm_model_name: LLM model name override

        Raises:
            Exception: If embedder, vector store, or generator fails to load
        """
        self.embedding_dim = embedding_dim or settings.embedding_dimension
        self.index_path = index_path
        self.llm_provider = llm_provider or settings.llm_provider
        self.llm_model_name = llm_model_name

        logger.info("Initializing RAG pipeline...")

        self.embedder = self.load_embedder()
        self.vector_store = self.load_vector_store()
        self.retriever = self.load_retriever()
        self.generator = self.load_generator()
        self.reranker = self.load_reranker()

        # Build source map for filename attribution (no re-indexing needed)
        self._source_map = _build_source_map()

        logger.info("RAG pipeline initialized successfully")

    def load_embedder(self, model_name: Optional[str] = None) -> Embedder:
        """
        Load or create embedder.

        Args:
            model_name: Optional model name override

        Returns:
            Embedder instance
        """
        logger.info("Loading embedder...")
        embedder = Embedder(model_name=model_name or settings.embedding_model)
        logger.info(f"Embedder loaded: {embedder.get_model_info()['model_name']}")
        return embedder

    def load_vector_store(self) -> VectorStore:
        """
        Load vector store from disk.

        Returns:
            VectorStore instance with loaded index

        Raises:
            Exception: If index or metadata not found
        """
        logger.info("Loading vector store from disk...")
        store = VectorStore(embedding_dim=self.embedding_dim, index_path=self.index_path)

        try:
            store.load()
            stats = store.get_stats()
            logger.info(f"Vector store loaded: {stats['total_vectors']} vectors")
        except Exception as e:
            logger.error(f"Failed to load vector store: {str(e)}")
            raise

        return store

    def load_retriever(self) -> Retriever:
        """
        Load retriever with embedder and vector store.

        Returns:
            Retriever instance
        """
        logger.info("Initializing retriever...")
        retriever = Retriever(self.embedder, self.vector_store)
        logger.info("Retriever initialized")
        return retriever

    def load_reranker(self) -> Optional[BGEReranker]:
        """
        Load the BGE cross-encoder reranker.

        Returns:
            BGEReranker instance, or None if reranking is disabled in settings.
        """
        if not settings.use_reranker:
            logger.info("Reranker disabled (USE_RERANKER=false)")
            return None

        logger.info(f"Loading reranker: {settings.reranker_model}")
        try:
            reranker = create_reranker()
            logger.info("Reranker loaded successfully")
            return reranker
        except Exception as e:
            logger.warning(
                f"Failed to load reranker ({e}). Continuing without reranking."
            )
            return None

    def load_generator(self) -> ResponseGenerator:
        """
        Load response generator with LLM client.

        Returns:
            ResponseGenerator instance

        Raises:
            Exception: If LLM client creation fails
        """
        logger.info(f"Initializing generator with provider: {self.llm_provider}")
        try:
            if self.llm_model_name:
                model_name = self.llm_model_name
            elif self.llm_provider == "ollama":
                model_name = settings.ollama_model
            elif self.llm_provider == "openai":
                model_name = "gpt-3.5-turbo"
            else:
                model_name = settings.ollama_model

            generator = create_generator(
                provider=self.llm_provider,
                model_name=model_name,
            )
            logger.info("Generator initialized")
            return generator
        except Exception as e:
            logger.error(f"Failed to initialize generator: {str(e)}")
            raise

    def query(
        self,
        query_text: str,
        k: int = 10,
        similarity_threshold: Optional[float] = None,
        language: Optional[str] = None,
        crop: Optional[str] = None,
        region: Optional[str] = None,
        season: Optional[str] = None,
        disease: Optional[str] = None,
        temperature: float = 0.7,
        use_hybrid: bool = True,
        use_reranker: Optional[bool] = None,
    ) -> Dict:
        """
        Execute a RAG query.

        Args:
            query_text: Query text
            k: Number of context chunks to retrieve
            similarity_threshold: Minimum similarity score (0-1)
            language: Response language code (en, hi, pa); auto-detected if not provided
            crop: Filter by crop (wheat, paddy, etc.)
            region: Filter by region
            season: Filter by season
            disease: Filter by disease
            temperature: LLM temperature for generation
            use_hybrid: If True, use BM25 + semantic hybrid search (recommended).
                        If False, use semantic-only search.
            use_reranker: Override reranking on/off for this call.
                          Defaults to settings.use_reranker.

        Returns:
            Dictionary with query, response, language, sources, and confidence

        Raises:
            ValueError: If query_text is empty
            Exception: If retrieval or generation fails
        """
        if not query_text or not query_text.strip():
            raise ValueError("Query text cannot be empty")

        # Resolve reranker flag: per-call override → settings default
        _use_reranker = use_reranker if use_reranker is not None else settings.use_reranker
        _use_reranker = _use_reranker and (self.reranker is not None)

        logger.info(f"Processing query: {query_text}")

        # Detect language if not provided
        if not language:
            from rag.retriever import LanguageDetector
            language = LanguageDetector().detect_language(query_text)
        logger.info(f"Response language: {language}")

        # Build metadata filters
        filters = {}
        if crop:
            filters["crop"] = crop
        if region:
            filters["region"] = region
        if season:
            filters["season"] = season
        if disease:
            filters["disease"] = disease

        # Retrieve exactly k chunks; the reranker reorders them, not filters.
        try:
            if use_hybrid:
                logger.info(f"Hybrid retrieval (BM25 + semantic), k={k}...")
                context_chunks = self.retriever.retrieve_hybrid(
                    query_text,
                    k=k,
                    similarity_threshold=similarity_threshold,
                    filters=filters if filters else None,
                    return_scores=True,
                )
            else:
                logger.info(f"Semantic-only retrieval, k={k}...")
                context_chunks = self.retriever.retrieve(
                    query_text,
                    k=k,
                    similarity_threshold=similarity_threshold,
                    filters=filters if filters else None,
                    return_scores=True,
                )
            logger.info(f"Retrieved {len(context_chunks)} chunks")

            # Enrich with source filename (runtime lookup, no re-index needed)
            context_chunks = _enrich_chunks_with_filename(context_chunks, self._source_map)

            # ── Reranking ──────────────────────────────────────────────────────
            if _use_reranker:
                logger.info(f"Reranking {len(context_chunks)} chunks by cross-encoder score...")
                context_chunks = self.reranker.rerank(
                    query=query_text,
                    chunks=context_chunks,
                    top_k=None,   # keep all k — just reorder by relevance
                )
                logger.info(f"Reranking complete")
        except Exception as e:
            logger.error(f"Retrieval/reranking failed: {str(e)}")
            raise

        # Generate response
        try:
            logger.info("Generating response...")
            response = self.generator.generate(
                query=query_text,
                context_chunks=context_chunks,
                temperature=temperature,
                language=language,
            )
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise

        result = {
            "query": query_text,
            "response": response["response"],
            "language": language,
            "confidence": response["confidence"],
            "sources": response["sources"],
            "num_context_chunks": len(context_chunks),
            "retrieved_chunks": context_chunks,
            "retrieval_mode": "hybrid" if use_hybrid else "semantic",
            "reranker_used": _use_reranker,
        }

        logger.info("Query processing completed")
        return result

    def get_stats(self) -> Dict:
        """
        Get pipeline statistics.

        Returns:
            Dictionary with component stats
        """
        return {
            "embedder": self.embedder.get_model_info(),
            "vector_store": self.vector_store.get_stats(),
            "retriever_support": {
                "supported_languages": settings.supported_languages,
            },
        }
