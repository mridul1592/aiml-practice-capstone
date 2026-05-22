"""
Embedding generator for multilingual agricultural documents.

Uses SentenceTransformers or Native HuggingFace Transformers for generating 
embeddings that work across English, Hindi, and Punjabi text.
"""

from typing import Any, Dict, List, Optional, Tuple
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

from utils.config import settings
from utils.logger import setup_logger


logger = setup_logger(__name__)


class Embedder:
    """Generate multilingual embeddings for document chunks."""

    # Task instruction used as prefix for query embeddings when using
    # E5-instruct family models (and any other instruct-tuned bi-encoders).
    # Documents are embedded without a prefix; only query vectors get this.
    _QUERY_INSTRUCTION: str = (
        "Given an agricultural query, retrieve relevant passages that answer the query"
    )

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize embedder with SentenceTransformer or AutoModel.

        Args:
            model_name: HuggingFace model name (default: from settings)

        Raises:
            Exception: If model loading fails
        """
        self.model_name = model_name or settings.embedding_model
        self.expected_dim = settings.embedding_dimension
        self.device = settings.device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Check routing flag
        self.use_transformers = self.model_name.startswith("BAAI/")

        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            
            if self.use_transformers:
                logger.info("Using native HuggingFace transformers pipeline.")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, use_fast=True
                )
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if "cuda" in str(self.device) else torch.float32,
                ).to(self.device)
                self.embedding_dim = self.model.config.hidden_size
            else:
                logger.info("Using SentenceTransformers pipeline.")
                self.model = SentenceTransformer(self.model_name, device=self.device)
                self.embedding_dim = self.model.get_embedding_dimension()

            # Cross-validate embedding dimensions with configuration file
            if self.embedding_dim != self.expected_dim:
                logger.warning(
                    f"Model embedding dimension ({self.embedding_dim}) "
                    f"differs from config ({self.expected_dim})"
                )

            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")

        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise
    
    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def _is_instruct_model(self) -> bool:
        """True for instruction-tuned embedding models (E5-instruct, etc.)."""
        return "instruct" in self.model_name.lower()

    # ------------------------------------------------------------------
    # Public embedding API
    # ------------------------------------------------------------------

    def embed_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Generate an embedding for a *query* string.

        For instruction-tuned models (e.g. intfloat/multilingual-e5-large-instruct)
        this automatically prepends the task instruction so the query vector is
        in the same embedding space as the document vectors produced by embed_text().

        For non-instruct models this is identical to embed_text().

        Args:
            query:     Raw query text
            normalize: Whether to L2-normalise the output (default: True)

        Returns:
            1-D numpy array of shape (embedding_dim,)
        """
        if not query or not query.strip():
            raise ValueError("Query text cannot be empty")

        if self._is_instruct_model and not self.use_transformers:
            # E5-instruct style: "Instruct: <task>\nQuery: <text>"
            text = f"Instruct: {self._QUERY_INSTRUCTION}\nQuery: {query}"
            logger.debug("Prepended instruction prefix for instruct model query")
        else:
            text = query

        return self.embed_text(text, normalize=normalize)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Retrieve metadata about the currently loaded embedding model.

        Returns:
            A dictionary containing model details.
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "expected_dimension": self.expected_dim,
            "device": str(self.device),
            "pipeline_type": "Native Transformers (AutoModel)" if self.use_transformers else "SentenceTransformers"
        }


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
            if self.use_transformers:
                # Returns matrix shape (1, dim), convert to 1D array
                return self._embed_with_transformers([text], normalize=normalize)[0]
            
            # SentenceTransformer fallback execution
            embedding = self.model.encode(
                text, 
                convert_to_numpy=True, 
                normalize_embeddings=normalize,
                device=self.device,
            )
            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def _embed_with_transformers(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """Helper to run inference via HuggingFace Transformers."""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=1024,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Apply basic pooling logic
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            emb = outputs.pooler_output
        else:
            emb = outputs.last_hidden_state[:, 0, :]

        if normalize:
            emb = F.normalize(emb, p=2, dim=1)

        return emb.cpu().numpy()

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

        # Adjust batch sizes based on runtime platform context
        if "cuda" in str(self.device):
            optimized_batch_size = min(batch_size * 2, 128)
        else:
            optimized_batch_size = batch_size

        logger.info(f"Generating embeddings for {len(non_empty_texts)} texts on {str(self.device).upper()}")
        logger.info(f"Batch size: {optimized_batch_size}")

        try:
            start_time = time.time()

            if self.use_transformers:
                # Chunk data into structural batches manually for native Transformers
                all_embeddings = []
                for i in range(0, len(non_empty_texts), optimized_batch_size):
                    batch_texts = non_empty_texts[i:i + optimized_batch_size]
                    batch_emb = self._embed_with_transformers(batch_texts, normalize=normalize)
                    all_embeddings.append(batch_emb)
                embeddings = np.vstack(all_embeddings)
            else:
                # Use SentenceTransformer batch processing engine
                embeddings = self.model.encode(
                    non_empty_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=normalize,
                    batch_size=optimized_batch_size,
                    show_progress_bar=show_progress,
                    device=self.device,
                )

            elapsed_time = time.time() - start_time
            texts_per_sec = len(non_empty_texts) / elapsed_time
            
            logger.info(f"Generated embeddings shape: {embeddings.shape}")
            logger.info(f"Time elapsed: {elapsed_time:.2f}s ({texts_per_sec:.1f} texts/sec)")
            
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
        Generate embeddings for a list of chunks and append them in place.

        Args:
            chunks: List of chunk dictionaries
            content_field: Field name containing text to embed (default: "content")
            batch_size: Batch size for processing (default: 32)
            show_progress: Whether to show progress (default: True)

        Returns:
            List of chunks with added "embedding" field
        """
        if not chunks:
            raise ValueError("Chunks list cannot be empty")

        # Validate structured dictionary layout
        for i, chunk in enumerate(chunks):
            if content_field not in chunk:
                raise KeyError(f"Chunk {i} missing '{content_field}' field")

        # Process embeddings uniformly 
        texts = [chunk[content_field] for chunk in chunks]
        embeddings = self.embed_batch(
            texts, normalize=True, batch_size=batch_size, show_progress=show_progress
        )

        # Assign output embedding structures back safely
        for chunk, emb in zip(chunks, embeddings):
            chunk["embedding"] = emb.tolist()

        return chunks
