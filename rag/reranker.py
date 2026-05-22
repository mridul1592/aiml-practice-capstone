"""
Cross-encoder reranker using BAAI/bge-reranker-base.

Reranks a candidate set of retrieved chunks by scoring each (query, passage)
pair with a dedicated cross-encoder model, which is significantly more accurate
than the bi-encoder similarity used during first-stage retrieval.

Typical usage in the RAG pipeline
----------------------------------
1. Retrieve k * N candidates with hybrid BM25 + FAISS (fast but approximate).
2. Pass all candidates through the reranker → get precise relevance scores.
3. Keep only the top-k most relevant chunks for the LLM context window.

This two-stage design keeps latency manageable: the fast bi-encoder handles
the bulk of the corpus, and the cross-encoder only sees a small candidate set.
"""

from typing import Dict, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils.config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class BGEReranker:
    """
    Cross-encoder reranker backed by BAAI/bge-reranker-base (or any compatible
    AutoModelForSequenceClassification checkpoint).

    The model receives tokenised (query, passage) pairs and outputs a scalar
    relevance logit — higher is more relevant.  Results are sorted by this
    score so the most relevant chunks bubble to the top.
    """

    # Maximum combined token length fed to the cross-encoder per pair.
    # bge-reranker-base has a 512-token limit.
    MAX_LENGTH: int = 512

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Load the reranker model and tokeniser.

        Args:
            model_name: HuggingFace model id (default: settings.reranker_model)
            device:     Torch device string (default: settings.device or auto-detect)

        Raises:
            Exception: If model loading fails
        """
        self.model_name = model_name or settings.reranker_model
        self.device = (
            device
            or getattr(settings, "device", None)
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        logger.info(f"Loading reranker: {self.model_name} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if "cuda" in str(self.device) else torch.float32,
        ).to(self.device)
        self.model.eval()

        logger.info("Reranker loaded successfully")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        chunks: List[Dict],
        top_k: Optional[int] = None,
        batch_size: int = 32,
    ) -> List[Dict]:
        """
        Score each (query, chunk) pair and return chunks sorted by relevance.

        Args:
            query:      User query string
            chunks:     Retrieved chunk dicts (each must have a 'content' key)
            top_k:      If set, return only the top_k highest-scoring chunks.
                        If None, return all chunks re-sorted by reranker score.
            batch_size: Number of pairs to score in one forward pass.

        Returns:
            List of chunk dicts, each with an added 'rerank_score' field,
            sorted descending by that score.
        """
        if not chunks:
            return chunks

        logger.info(f"Reranking {len(chunks)} chunks (top_k={top_k})")

        # Score all pairs in batches
        all_scores: List[float] = []
        pairs = [
            [query, chunk.get("content", "")[: self.MAX_LENGTH * 4]]
            for chunk in chunks
        ]

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            all_scores.extend(self._score_batch(batch))

        # Attach scores and sort descending
        scored = sorted(
            zip(all_scores, chunks),
            key=lambda x: x[0],
            reverse=True,
        )

        result = [
            {**chunk, "rerank_score": round(float(score), 6)}
            for score, chunk in (scored[:top_k] if top_k is not None else scored)
        ]

        if result:
            logger.info(
                f"Reranking complete — top score: {result[0]['rerank_score']:.4f}, "
                f"bottom score: {result[-1]['rerank_score']:.4f}"
            )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_batch(self, pairs: List[List[str]]) -> List[float]:
        """
        Run a single forward pass through the cross-encoder for a batch of
        (query, passage) pairs.

        Args:
            pairs: List of [query, passage] lists

        Returns:
            List of raw logit scores (one per pair)
        """
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self.MAX_LENGTH,
                return_tensors="pt",
            ).to(self.device)

            logits = self.model(**inputs).logits  # shape: (batch, 1) or (batch, 2)

            # bge-reranker outputs a single logit per pair
            if logits.shape[-1] == 1:
                scores = logits.squeeze(-1)
            else:
                # Binary classification head — use the positive-class logit
                scores = logits[:, 1]

        return scores.float().cpu().tolist()


def create_reranker(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> BGEReranker:
    """
    Convenience factory for creating a BGEReranker.

    Args:
        model_name: Optional model override
        device:     Optional device override

    Returns:
        BGEReranker instance
    """
    return BGEReranker(model_name=model_name, device=device)
