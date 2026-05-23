"""
Embedding Pipeline Orchestrator.

Coordinates:
- Chunk loading
- Embedding generation
- Vector store creation and population
- Index persistence
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ingestion.orchestrator import IngestionPipeline
from rag.embedder import Embedder
from rag.vector_store import VectorStore
from utils.config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class EmbeddingPipeline:
    """Orchestrate the embedding and vector store creation pipeline."""

    def __init__(self):
        """Initialize embedding pipeline."""
        self.embedder: Optional[Embedder] = None
        self.vector_store: Optional[VectorStore] = None
        self.ingestion_pipeline = IngestionPipeline()

    def load_or_create_embedder(self, model_name: Optional[str] = None) -> Embedder:
        """
        Load or create embedder.

        Args:
            model_name: Optional model name override

        Returns:
            Embedder instance
        """
        if self.embedder is None:
            logger.info("Creating embedder...")
            self.embedder = Embedder(model_name)

        return self.embedder

    def load_or_create_vector_store(self, embedding_dim: int = 384) -> VectorStore:
        """
        Load or create vector store.

        Args:
            embedding_dim: Embedding dimension

        Returns:
            VectorStore instance
        """
        if self.vector_store is None:
            logger.info("Creating vector store...")
            self.vector_store = VectorStore(embedding_dim)

        return self.vector_store

    def generate_embeddings_from_chunks(
        self,
        chunks: List[Dict],
        model_name: Optional[str] = None,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> List[Dict]:
        """
        Generate embeddings for chunks.

        Args:
            chunks: List of chunk dictionaries
            model_name: Optional model name override
            batch_size: Batch size for embedding generation
            show_progress: Whether to show progress

        Returns:
            Chunks with embeddings added

        Raises:
            ValueError: If chunks list is empty
        """
        if not chunks:
            raise ValueError("Chunks list cannot be empty")

        logger.info(f"Generating embeddings for {len(chunks)} chunks")

        embedder = self.load_or_create_embedder(model_name)
        enriched_chunks = embedder.embed_chunks(
            chunks, batch_size=batch_size, show_progress=show_progress
        )

        return enriched_chunks

    def build_vector_store(
        self,
        chunks: List[Dict],
        model_name: Optional[str] = None,
        batch_size: int = 32,
        save_index: bool = True,
    ) -> VectorStore:
        """
        Build vector store from chunks.

        Args:
            chunks: List of chunk dictionaries
            model_name: Optional model name override
            batch_size: Batch size for embedding
            save_index: Whether to save index to disk

        Returns:
            Populated VectorStore

        Raises:
            ValueError: If chunks empty or missing content field
        """
        if not chunks:
            raise ValueError("Chunks list cannot be empty")

        logger.info(f"Building vector store from {len(chunks)} chunks")

        # Generate embeddings
        embedder = self.load_or_create_embedder(model_name)
        enriched_chunks = embedder.embed_chunks(chunks, batch_size=batch_size)

        # Extract embeddings and metadata
        embeddings = np.array([chunk["embedding"] for chunk in enriched_chunks])
        metadata = [
            {k: v for k, v in chunk.items() if k != "embedding"} for chunk in enriched_chunks
        ]

        # Create and populate vector store
        vector_store = self.load_or_create_vector_store(embeddings.shape[1])
        vector_store.add_embeddings(embeddings, metadata)

        # Save if requested
        if save_index:
            vector_store.save()

        logger.info(f"Vector store built with {vector_store.get_stats()['total_vectors']} vectors")

        return vector_store

    def ingest_and_embed(
        self,
        pdf_directory: str,
        model_name: Optional[str] = None,
        batch_size: int = 32,
        save_index: bool = True,
    ) -> Dict:
        """
        Complete pipeline: Ingest PDFs → Generate chunks → Create embeddings → Build vector store.

        Args:
            pdf_directory: Directory containing PDFs
            model_name: Optional model name override
            batch_size: Batch size for embedding
            save_index: Whether to save index

        Returns:
            Dictionary with results and statistics

        Raises:
            ValueError: If PDF directory invalid
        """
        if not Path(pdf_directory).exists():
            raise ValueError(f"PDF directory not found: {pdf_directory}")

        logger.info(f"Starting complete ingestion + embedding pipeline from {pdf_directory}")

        # Step 1: Ingest PDFs
        logger.info("Step 1: Ingesting PDFs...")
        ingestion_results = self.ingestion_pipeline.ingest_batch(pdf_directory)

        if ingestion_results["processed_files"] == 0:
            logger.error("No PDFs were processed")
            return ingestion_results

        # Step 2: Load all chunks (inject source filename into each chunk)
        logger.info("Step 2: Loading chunks...")
        raw_chunks = self.ingestion_pipeline.get_all_chunks()

        if not raw_chunks:
            logger.error("No chunks found after ingestion")
            return ingestion_results

        # get_all_chunks() now injects 'filename' from the top-level source_file
        # field in each processed JSON, so raw_chunks already carry the correct
        # filename.  This guard is kept as a safety net only.
        chunks = []
        for chunk in raw_chunks:
            if not chunk.get("filename") or chunk.get("filename") == "Unknown":
                chunk = {**chunk, "filename": "Unknown"}
            chunks.append(chunk)

        # Step 3: Build vector store
        logger.info("Step 3: Building vector store...")
        vector_store = self.build_vector_store(chunks, model_name, batch_size, save_index)

        # Combine results
        results = {
            **ingestion_results,
            "embedding_stats": vector_store.get_stats(),
            "embedding_model": self.embedder.get_model_info() if self.embedder else None,
        }

        logger.info("Complete pipeline finished successfully")
        return results

    def search(
        self,
        query_text: str,
        k: int = 5,
        similarity_threshold: Optional[float] = None,
        metadata_filters: Optional[Dict] = None,
        load_index: bool = False,
    ) -> Dict:
        """
        Search the vector store with a query.

        Args:
            query_text: Query text
            k: Number of results
            similarity_threshold: Optional minimum similarity
            metadata_filters: Optional metadata filters
            load_index: Whether to load index from disk first

        Returns:
            Dictionary with results

        Raises:
            ValueError: If vector store not initialized
        """
        if load_index:
            logger.info("Loading vector store from disk...")
            vector_store = self.load_or_create_vector_store()
            vector_store.load()
            self.vector_store = vector_store
        elif self.vector_store is None:
            raise ValueError("Vector store not initialized. Build or load an index first.")

        # Generate query embedding
        embedder = self.load_or_create_embedder()
        query_embedding = embedder.embed_text(query_text)

        # Search
        logger.info(f"Searching for: {query_text}")
        results, scores = self.vector_store.search(
            query_embedding, k=k, similarity_threshold=similarity_threshold, metadata_filters=metadata_filters
        )

        return {
            "query": query_text,
            "num_results": len(results),
            "results": [
                {
                    **result,
                    "similarity": float(score),
                }
                for result, score in zip(results, scores)
            ],
        }

    def get_stats(self) -> Dict:
        """
        Get pipeline statistics.

        Returns:
            Statistics dictionary
        """
        stats = {}

        if self.embedder:
            stats["embedder"] = self.embedder.get_model_info()

        if self.vector_store:
            stats["vector_store"] = self.vector_store.get_stats()

        return stats


def run_embedding_pipeline(
    pdf_directory: str,
    model_name: Optional[str] = None,
    batch_size: int = 32,
    save_summary: Optional[str] = None,
) -> Dict:
    """
    Convenience function to run complete embedding pipeline.

    Args:
        pdf_directory: Directory containing PDFs
        model_name: Optional model name override
        batch_size: Batch size for embedding
        save_summary: Optional path to save results JSON

    Returns:
        Results dictionary
    """
    pipeline = EmbeddingPipeline()
    results = pipeline.ingest_and_embed(pdf_directory, model_name, batch_size)

    if save_summary:
        summary_path = Path(save_summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        with open(summary_path, "w", encoding="utf-8") as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = json.dumps(results, indent=2, default=str)
            f.write(json_results)

        logger.info(f"Results saved to {summary_path}")

    return results


if __name__ == "__main__":
    # Example usage
    pipeline = EmbeddingPipeline()

    # Option 1: Full pipeline
    print("Running complete embedding pipeline...")
    results = pipeline.ingest_and_embed("Agri_docs", batch_size=16)

    print("\n=== PIPELINE RESULTS ===")
    print(f"Files processed: {results['processed_files']}")
    print(f"Total chunks: {results['total_chunks']}")
    print(f"Vectors created: {results['embedding_stats']['total_vectors']}")

    # Option 2: Search
    print("\n=== SEARCH EXAMPLE ===")
    search_results = pipeline.search("How to control wheat pests?", k=3)
    print(f"Query: {search_results['query']}")
    print(f"Results found: {search_results['num_results']}")
    for i, result in enumerate(search_results["results"]):
        print(f"  {i+1}. Score: {result['similarity']:.4f}")
        print(f"     Content: {result['content'][:100]}...")
