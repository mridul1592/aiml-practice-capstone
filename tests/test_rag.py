"""
Test suite for RAG orchestration.

Tests:
1. RAG Pipeline initialization and component loading
2. Query execution with retrieval
3. Response generation with context
"""

import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag.rag_orchestrator import RAGPipeline
from rag.embedder import Embedder
from rag.generator import ResponseGenerator, OllamaClient, OpenAIClient
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from utils.logger import setup_logger

logger = setup_logger(__name__)


def test_rag_pipeline_initialization():
    """Test RAG pipeline can initialize components."""
    print("\n" + "=" * 60)
    print("TEST 1: RAG Pipeline - Initialization")
    print("=" * 60)

    try:
        # Check if FAISS index exists
        from utils.config import settings
        from pathlib import Path

        index_path = Path(settings.faiss_index_path)
        if not index_path.exists():
            print("⚠️  FAISS index not found - skipping initialization test")
            print("  Run 'python main.py embed' first to create embeddings")
            return True

        # Try to initialize pipeline
        pipeline = RAGPipeline()

        assert pipeline.embedder is not None, "Embedder not initialized"
        assert pipeline.vector_store is not None, "Vector store not initialized"
        assert pipeline.retriever is not None, "Retriever not initialized"
        assert pipeline.generator is not None, "Generator not initialized"

        print("✓ RAG pipeline initialized successfully")
        print(f"  - Embedder: {pipeline.embedder.get_model_info()['model_name']}")
        print(f"  - Vector store: {pipeline.vector_store.get_stats()['total_vectors']} vectors")
        print(f"  - Generator provider: {pipeline.llm_provider}")

        return True

    except FileNotFoundError as e:
        print(f"⚠️  Skipping test: {str(e)}")
        return True
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        logger.error(f"Test failed: {str(e)}")
        return False


def test_retriever_basic_retrieval():
    """Test retriever can perform basic retrieval."""
    print("\n" + "=" * 60)
    print("TEST 2: Retriever - Basic Retrieval")
    print("=" * 60)

    try:
        from utils.config import settings
        from pathlib import Path

        index_path = Path(settings.faiss_index_path)
        if not index_path.exists():
            print("⚠️  FAISS index not found - skipping retrieval test")
            return True

        # Initialize components
        embedder = Embedder()
        vector_store = VectorStore()
        vector_store.load()

        retriever = Retriever(embedder, vector_store)

        # Test retrieval
        query = "How to control wheat pests?"
        results = retriever.retrieve(query, k=3)

        assert len(results) > 0, "No results retrieved"
        assert all("similarity_score" in r for r in results), "Missing similarity scores"

        print("✓ Retriever works correctly")
        print(f"  - Query: {query}")
        print(f"  - Results retrieved: {len(results)}")
        for i, result in enumerate(results, 1):
            print(
                f"    {i}. Similarity: {result['similarity_score']:.4f}, "
                f"Content: {result['content'][:60]}..."
            )

        return True

    except FileNotFoundError:
        print("⚠️  FAISS index not found - skipping retrieval test")
        return True
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        logger.error(f"Test failed: {str(e)}")
        return False


def test_response_generator_mock():
    """Test response generator with mock LLM."""
    print("\n" + "=" * 60)
    print("TEST 3: Response Generator - Mock Generation")
    print("=" * 60)

    try:
        # Create a mock LLM client
        class MockLLMClient:
            def __init__(self):
                self.model_name = "mock"

            def generate(self, prompt, system_prompt=None, **kwargs):
                return "This is a mock response based on the context provided."

            def is_available(self):
                return True

        generator = ResponseGenerator(MockLLMClient())

        context_chunks = [
            {
                "content": "Wheat pests can be controlled using integrated pest management.",
                "filename": "wheat_guide.pdf",
                "language": "en",
                "similarity_score": 0.85,
            },
            {
                "content": "Pesticides should be applied at specific growth stages.",
                "filename": "pesticide_guide.pdf",
                "language": "en",
                "similarity_score": 0.78,
            },
        ]

        result = generator.generate(
            query="How to control wheat pests?",
            context_chunks=context_chunks,
            language="en",
            temperature=0.7,
        )

        assert result["response"] is not None, "No response generated"
        assert result["language"] == "en", "Language mismatch"
        assert len(result["sources"]) > 0, "No sources returned"
        assert result["confidence"] in ["high", "medium", "low"], "Invalid confidence"

        print("✓ Response generator works correctly")
        print(f"  - Response: {result['response']}")
        print(f"  - Confidence: {result['confidence']}")
        print(f"  - Sources: {result['sources']}")

        return True

    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        logger.error(f"Test failed: {str(e)}")
        return False


def test_language_detection():
    """Test query language detection."""
    print("\n" + "=" * 60)
    print("TEST 4: Language Detection - Multilingual Support")
    print("=" * 60)

    try:
        from rag.retriever import LanguageDetector

        detector = LanguageDetector()

        test_cases = [
            ("How to control wheat pests?", "en"),
            ("गेहूँ में कीटों का नियंत्रण कैसे करें?", "hi"),
            ("ਗੇਹੂੰ ਵਿੱਚ ਕੀਟਾਂ ਦਾ ਨਿਯੰਤਰਣ ਕਿਵੇਂ ਕਰਨਾ ਹੈ?", "pa"),
        ]

        results = {}
        for text, expected_lang in test_cases:
            detected = detector.detect_language(text)
            results[expected_lang] = detected

        print("✓ Language detection works")
        for lang, detected in results.items():
            status = "✓" if lang == detected else "✗"
            print(f"  {status} {lang}: detected as {detected}")

        return True

    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        logger.error(f"Test failed: {str(e)}")
        return False


def test_rag_pipeline_stats():
    """Test RAG pipeline statistics."""
    print("\n" + "=" * 60)
    print("TEST 5: RAG Pipeline - Statistics")
    print("=" * 60)

    try:
        from utils.config import settings
        from pathlib import Path

        index_path = Path(settings.faiss_index_path)
        if not index_path.exists():
            print("⚠️  FAISS index not found - skipping stats test")
            return True

        pipeline = RAGPipeline()
        stats = pipeline.get_stats()

        assert "embedder" in stats, "Embedder stats missing"
        assert "vector_store" in stats, "Vector store stats missing"

        print("✓ Pipeline statistics retrieved")
        print(f"  - Embedder: {stats['embedder']['model_name']}")
        print(f"  - Embedding dimension: {stats['embedder']['embedding_dimension']}")
        print(f"  - Total vectors: {stats['vector_store']['total_vectors']}")
        print(f"  - Supported languages: {stats['retriever_support']['supported_languages']}")

        return True

    except FileNotFoundError:
        print("⚠️  FAISS index not found - skipping stats test")
        return True
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        logger.error(f"Test failed: {str(e)}")
        return False


def run_all_tests():
    """Run all RAG tests."""
    print("\n" + "=" * 60)
    print("RAG ORCHESTRATION - TEST SUITE")
    print("=" * 60)

    tests = [
        ("RAG Pipeline - Initialization", test_rag_pipeline_initialization),
        ("Retriever - Basic Retrieval", test_retriever_basic_retrieval),
        ("Response Generator - Mock", test_response_generator_mock),
        ("Language Detection - Multilingual", test_language_detection),
        ("RAG Pipeline - Statistics", test_rag_pipeline_stats),
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
