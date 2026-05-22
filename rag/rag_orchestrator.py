"""
RAG orchestration layer.

Coordinates:
- Embedder loading
- Vector store loading
- Retriever execution
- LLM response generation
"""

from typing import Dict, List, Optional

from rag.embedder import Embedder
from rag.generator import ResponseGenerator, create_generator
from rag.retriever import Retriever
from rag.vector_store import VectorStore
from utils.config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


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
        k: int = 5,
        similarity_threshold: Optional[float] = None,
        language: Optional[str] = None,
        crop: Optional[str] = None,
        region: Optional[str] = None,
        season: Optional[str] = None,
        disease: Optional[str] = None,
        temperature: float = 0.7,
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

        Returns:
            Dictionary with query, response, language, sources, and confidence

        Raises:
            ValueError: If query_text is empty
            Exception: If retrieval or generation fails
        """
        if not query_text or not query_text.strip():
            raise ValueError("Query text cannot be empty")

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

        # Retrieve context chunks
        try:
            logger.info(f"Retrieving {k} context chunks...")
            context_chunks = self.retriever.retrieve(
                query_text,
                k=k,
                similarity_threshold=similarity_threshold,
                filters=filters if filters else None,
                return_scores=True,
            )
            logger.info(f"Retrieved {len(context_chunks)} chunks")
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
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
