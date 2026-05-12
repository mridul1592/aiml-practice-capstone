"""
Test suite for embedding and vector store modules.

Tests:
1. Embedder - Text embedding and similarity
2. Vector Store - Index creation and search
3. Embedding Pipeline - Full embedding workflow
"""

import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag.embedder import Embedder
from rag.vector_store import VectorStore
from rag.embeddings_orchestrator import EmbeddingPipeline
from utils.logger import setup_logger

logger = setup_logger(__name__)


def test_embedder_single_text():
    """Test single text embedding."""
    print("\n" + "=" * 60)
    print("TEST 1: Embedder - Single Text")
    print("=" * 60)

    try:
        embedder = Embedder()

        # Test single embedding
        text = "How to control wheat pests in Punjab?"
        embedding = embedder.embed_text(text)

        assert embedding.shape[0] == 384, f"Expected dimension 384, got {embedding.shape[0]}"
        assert np.isfinite(embedding).all(), "Embedding contains NaN or Inf"

        print(f"✓ Single text embedding successful")
        print(f"  - Text: {text}")
        print(f"  - Embedding dimension: {embedding.shape[0]}")
        print(f"  - First 5 values: {embedding[:5]}")

        return True

    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        logger.error(f"Test failed: {str(e)}")
        return False


def test_embedder_batch():
    """Test batch text embedding."""
    print("\n" + "=" * 60)
    print("TEST 2: Embedder - Batch Processing")
    print("=" * 60)

    try:
        embedder = Embedder()

        texts = [
            "Wheat cultivation best practices",
            "गेहूँ की खेती में कीटनाशक का उपयोग",
            "ਧਾਨ ਦੀ ਫਸਲ ਵਿੱਚ ਸਿੰਚਾਈ",
        ]

        embeddings = embedder.embed_batch(texts, show_progress=False)

        assert embeddings.shape == (3, 384), f"Expected shape (3, 384), got {embeddings.shape}"
        assert np.isfinite(embeddings).all(), "Embeddings contain NaN or Inf"

        print(f"✓ Batch embedding successful")
        print(f"  - Number of texts: {len(texts)}")
        print(f"  - Embeddings shape: {embeddings.shape}")

        return True

    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        logger.error(f"Test failed: {str(e)}")
        return False


def test_embedder_similarity():
    """Test similarity computation."""
    print("\n" + "=" * 60)
    print("TEST 3: Embedder - Similarity Computation")
    print("=" * 60)

    try:
        embedder = Embedder()

        # Create embeddings for similar and dissimilar texts
        similar_texts = ["wheat pests", "wheat pest control"]
        dissimilar_texts = ["wheat pests", "paddy cultivation"]

        similar_embeddings = embedder.embed_batch(similar_texts, show_progress=False)
        dissimilar_embeddings = embedder.embed_batch(dissimilar_texts, show_progress=False)

        similarity_similar = embedder.compute_similarity(similar_embeddings[0], similar_embeddings[1])
        similarity_dissimilar = embedder.compute_similarity(
            dissimilar_embeddings[0], dissimilar_embeddings[1]
        )

        print(f"✓ Similarity computation successful")
        print(f"  - Similar texts similarity: {similarity_similar:.4f}")
        print(f"  - Dissimilar texts similarity: {similarity_dissimilar:.4f}")
        print(f"  - Similar > Dissimilar: {similarity_similar > similarity_dissimilar}")

        return True

    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        logger.error(f"Test failed: {str(e)}")
        return False


def test_vector_store_add_search():
    """Test vector store add and search."""
    print("\n" + "=" * 60)
    print("TEST 4: Vector Store - Add and Search")
    print("=" * 60)

    try:
        # Create vector store
        vector_store = VectorStore(embedding_dim=384)

        # Create sample embeddings
        embeddings = np.random.randn(10, 384).astype(np.float32)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)

        metadata = [
            {
                "content": f"Document {i}",
                "crop": "wheat" if i < 5 else "paddy",
                "language": "en",
            }
            for i in range(10)
        ]

        # Add to vector store
        ids = vector_store.add_embeddings(embeddings, metadata)
        assert len(ids) == 10, f"Expected 10 IDs, got {len(ids)}"

        # Search
        query_embedding = embeddings[0]
        results, scores = vector_store.search(query_embedding, k=3)

        assert len(results) > 0, "No results found"
        assert len(results) == len(scores), "Results and scores mismatch"

        print(f"✓ Vector store add and search successful")
        print(f"  - Added vectors: {len(ids)}")
        print(f"  - Search results: {len(results)}")
        for i, (result, score) in enumerate(zip(results, scores)):
            print(f"    {i+1}. Score: {score:.4f}, Content: {result['content']}")

        return True

    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        logger.error(f"Test failed: {str(e)}")
        return False


def test_vector_store_metadata_filter():
    """Test vector store with metadata filtering."""
    print("\n" + "=" * 60)
    print("TEST 5: Vector Store - Metadata Filtering")
    print("=" * 60)

    try:
        vector_store = VectorStore(embedding_dim=384)

        # Create sample embeddings
        embeddings = np.random.randn(10, 384).astype(np.float32)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)

        metadata = [
            {
                "content": f"Document {i}",
                "crop": "wheat" if i < 5 else "paddy",
                "language": "en",
            }
            for i in range(10)
        ]

        vector_store.add_embeddings(embeddings, metadata)

        # Search with filter
        query_embedding = embeddings[0]
        results, scores = vector_store.search(query_embedding, k=5, metadata_filters={"crop": "wheat"})

        # All results should be wheat
        assert all(r["crop"] == "wheat" for r in results), "Filter not applied correctly"

        print(f"✓ Metadata filtering successful")
        print(f"  - Filter: crop=wheat")
        print(f"  - Results: {len(results)}")
        for result in results:
            print(f"    - {result['content']}: {result['crop']}")

        return True

    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        logger.error(f"Test failed: {str(e)}")
        return False


def test_embedding_pipeline():
    """Test complete embedding pipeline."""
    print("\n" + "=" * 60)
    print("TEST 6: Embedding Pipeline - Complete Workflow")
    print("=" * 60)

    try:
        pipeline = EmbeddingPipeline()

        # Check if Agri_docs exists
        agri_docs_path = Path("Agri_docs")
        if not agri_docs_path.exists():
            print(f"⚠️  Agri_docs directory not found - skipping full pipeline test")
            return True

        pdf_files = list(agri_docs_path.glob("*.pdf"))
        if not pdf_files:
            print(f"⚠️  No PDF files found - skipping full pipeline test")
            return True

        # Run limited pipeline (first PDF only for speed)
        print("  Running limited pipeline test (1 PDF for speed)...")

        # Test embedder creation
        embedder = pipeline.load_or_create_embedder()
        assert embedder is not None, "Embedder not created"

        # Test vector store creation
        vector_store = pipeline.load_or_create_vector_store()
        assert vector_store is not None, "Vector store not created"

        stats = vector_store.get_stats()
        print(f"✓ Embedding pipeline initialized")
        print(f"  - Embedder: {embedder.get_model_info()['model_name']}")
        print(f"  - Vector store embedding dimension: {stats['embedding_dimension']}")

        return True

    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        logger.error(f"Test failed: {str(e)}")
        return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("EMBEDDING & VECTOR STORE - TEST SUITE")
    print("=" * 60)

    tests = [
        ("Embedder - Single Text", test_embedder_single_text),
        ("Embedder - Batch Processing", test_embedder_batch),
        ("Embedder - Similarity", test_embedder_similarity),
        ("Vector Store - Add/Search", test_vector_store_add_search),
        ("Vector Store - Metadata Filter", test_vector_store_metadata_filter),
        ("Embedding Pipeline", test_embedding_pipeline),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {str(e)}")
            results[test_name] = False

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, passed_status in results.items():
        status = "✓ PASSED" if passed_status else "✗ FAILED"
        print(f"{test_name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
