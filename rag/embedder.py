"""
Embedding generator for multilingual agricultural documents.

Uses SentenceTransformers for generating embeddings that work across
English, Hindi, and Punjabi text.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from utils.config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class Embedder:
    """Generate multilingual embeddings for document chunks."""

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize embedder with SentenceTransformer model.

        Args:
            model_name: HuggingFace model name (default: from settings)

        Raises:
            Exception: If model loading fails
        """
        self.model_name = model_name or settings.embedding_model
        self.expected_dim = settings.embedding_dimension

        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

            if self.embedding_dim != self.expected_dim:
                logger.warning(
                    f"Model embedding dimension ({self.embedding_dim}) "
                    f"differs from config ({self.expected_dim})"
                )

            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")

        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise

    def embed_text(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed
            normalize: Whether to normalize embeddings (default: True)

        Returns:
            Embedding vector (1D numpy array)

        Raises:
            ValueError: If text is empty
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        try:
            embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=normalize)
            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    def embed_batch(
        self, texts: List[str], normalize: bool = True, batch_size: int = 32, show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed
            normalize: Whether to normalize embeddings (default: True)
            batch_size: Batch size for processing (default: 32)
            show_progress: Whether to show progress bar (default: True)

        Returns:
            Embedding matrix (n_texts x embedding_dim)

        Raises:
            ValueError: If texts list is empty
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        # Filter out empty texts
        non_empty_texts = [t for t in texts if t and t.strip()]
        if not non_empty_texts:
            raise ValueError("All texts are empty")

        logger.info(f"Generating embeddings for {len(non_empty_texts)} texts")

        try:
            embeddings = self.model.encode(
                non_empty_texts,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                batch_size=batch_size,
                show_progress_bar=show_progress,
            )

            logger.info(f"Generated embeddings shape: {embeddings.shape}")
            return embeddings

        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise

    def embed_chunks(
        self,
        chunks: List[Dict],
        content_field: str = "content",
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> List[Dict]:
        """
        Generate embeddings for a list of chunks.

        Args:
            chunks: List of chunk dictionaries
            content_field: Field name containing text to embed (default: "content")
            batch_size: Batch size for processing (default: 32)
            show_progress: Whether to show progress (default: True)

        Returns:
            List of chunks with added "embedding" field

        Raises:
            ValueError: If chunks list is empty
            KeyError: If content_field not in chunk
        """
        if not chunks:
            raise ValueError("Chunks list cannot be empty")

        # Validate all chunks have content field
        for i, chunk in enumerate(chunks):
            if content_field not in chunk:
                raise KeyError(f"Chunk {i} missing '{content_field}' field")

        # Extract content texts
        texts = [chunk[content_field] for chunk in chunks]

        # Generate embeddings
        embeddings = self.embed_batch(texts, batch_size=batch_size, show_progress=show_progress)

        # Add embeddings to chunks
        enriched_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_with_embedding = {
                **chunk,
                "embedding": embedding,
                "embedding_model": self.model_name,
                "embedding_dim": int(self.embedding_dim),
            }
            enriched_chunks.append(chunk_with_embedding)

        logger.info(f"Added embeddings to {len(enriched_chunks)} chunks")
        return enriched_chunks

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score between 0 and 1

        Raises:
            ValueError: If embeddings have wrong dimension
        """
        if len(embedding1) != len(embedding2):
            raise ValueError(
                f"Embedding dimensions must match: {len(embedding1)} vs {len(embedding2)}"
            )

        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )

        return float(similarity)

    def compute_similarities(self, embedding: np.ndarray, embeddings_matrix: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarities between one embedding and a matrix of embeddings.

        Args:
            embedding: Query embedding vector
            embeddings_matrix: Matrix of embeddings (n x d)

        Returns:
            Array of similarity scores

        Raises:
            ValueError: If dimensions don't match
        """
        if embedding.shape[0] != embeddings_matrix.shape[1]:
            raise ValueError(
                f"Embedding dimensions must match: {embedding.shape[0]} vs {embeddings_matrix.shape[1]}"
            )

        # Compute cosine similarities
        similarities = np.dot(embeddings_matrix, embedding) / (
            np.linalg.norm(embeddings_matrix, axis=1) * np.linalg.norm(embedding)
        )

        return similarities

    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": int(self.embedding_dim),
            "framework": "SentenceTransformers",
        }


def create_embedder(model_name: Optional[str] = None) -> Embedder:
    """
    Create an embedder instance.

    Args:
        model_name: Optional model name override

    Returns:
        Embedder instance

    Raises:
        Exception: If model loading fails
    """
    return Embedder(model_name)


if __name__ == "__main__":
    # Example usage
    embedder = Embedder()

    # Embed single text
    print("Embedding single text...")
    embedding = embedder.embed_text("How to control wheat pests in Punjab?")
    print(f"Embedding shape: {embedding.shape}")
    print(f"First 5 values: {embedding[:5]}")

    # Embed batch
    print("\nEmbedding batch of texts...")
    texts = [
        "Wheat cultivation best practices",
        "गेहूँ की खेती में कीटनाशक का उपयोग",
        "ਗੇਹੂੰ ਦੀ ਫਸਲ ਦੀ ਸੁਰੱਖਿਆ",
    ]
    embeddings = embedder.embed_batch(texts, show_progress=False)
    print(f"Embeddings shape: {embeddings.shape}")

    # Compute similarity
    print("\nComputing similarity...")
    sim = embedder.compute_similarity(embeddings[0], embeddings[1])
    print(f"Similarity between text 1 and 2: {sim:.4f}")

    # Model info
    print("\nModel info:")
    print(embedder.get_model_info())
