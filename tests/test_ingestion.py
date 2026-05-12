"""
Test script for PDF ingestion pipeline.

Tests individual components and the complete pipeline.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ingestion.chunking import DocumentChunker
from ingestion.metadata_extractor import MetadataExtractor
from ingestion.pdf_parser import PDFParser
from ingestion.orchestrator import IngestionPipeline
from utils.logger import setup_logger

logger = setup_logger(__name__)


def test_pdf_parser():
    """Test PDF parsing functionality."""
    print("\n" + "=" * 60)
    print("TEST 1: PDF Parser")
    print("=" * 60)

    parser = PDFParser()

    # Test with first PDF from Agri_docs
    pdf_path = "Agri_docs/Doubling-of-Farmers-Income-updated-092020-min.pdf"

    if not Path(pdf_path).exists():
        print(f"⚠️  PDF not found: {pdf_path}")
        return False

    result = parser.parse_pdf(pdf_path)

    if result:
        print(f"✓ Successfully parsed: {result['filename']}")
        print(f"  - Text length: {len(result['text'])} characters")
        print(f"  - Preview: {result['text'][:150]}...")
        return True
    else:
        print("✗ Failed to parse PDF")
        return False


def test_metadata_extractor():
    """Test metadata extraction functionality."""
    print("\n" + "=" * 60)
    print("TEST 2: Metadata Extractor")
    print("=" * 60)

    extractor = MetadataExtractor()

    sample_text = """
    Kharif Season Paddy Cultivation in Punjab and Haryana
    
    This manual provides comprehensive guidance on paddy (rice) cultivation
    during the kharif season. Key topics include disease management such as
    leaf spot and blast control, optimal fertilizer usage, and irrigation
    scheduling for maximum yield.
    """

    metadata = extractor.extract_metadata(sample_text, "test_paddy_kharif_2025.pdf")

    print(f"✓ Extracted metadata:")
    print(f"  - Language: {metadata.get('language')}")
    print(f"  - Crops: {metadata.get('crops')}")
    print(f"  - Diseases: {metadata.get('diseases')}")
    print(f"  - Seasons: {metadata.get('seasons')}")
    print(f"  - Regions: {metadata.get('regions')}")

    return True


def test_chunking():
    """Test document chunking functionality."""
    print("\n" + "=" * 60)
    print("TEST 3: Document Chunking")
    print("=" * 60)

    sample_text = """
    # Wheat Cultivation Complete Guide
    
    ## Chapter 1: Soil Preparation and Management
    
    Proper soil preparation is critical for successful wheat cultivation.
    The soil should be well-draining, rich in organic matter, and free from
    weeds and crop residues. Farmers should conduct soil testing to determine
    nutrient levels and pH. Lime should be added if soil pH is below 6.0.
    
    The field should be ploughed 3-4 times, with the first ploughing done 4-6
    weeks before sowing. This allows for proper decomposition of organic matter
    and elimination of weeds. The final ploughing should be done 1-2 weeks
    before sowing to create a good seedbed.
    
    ## Chapter 2: Seed Selection and Treatment
    
    Use certified, high-quality seeds from recognized seed producers. The seed
    should be healthy, free from diseases, and have good germination capacity
    (above 85%). Always treat seeds with appropriate fungicides before sowing
    to protect against seed-borne diseases.
    
    Recommended seed rate: 100-125 kg/hectare for timely sowing and 125-150
    kg/hectare for late sowing. Seeds should be soaked in water for 24 hours
    before treatment.
    
    ## Chapter 3: Disease Management
    
    Common wheat diseases include Karnal bunt, loose smut, and various leaf spots.
    Implementation of crop rotation, use of resistant varieties, and proper
    spacing can significantly reduce disease incidence. For severe infestations,
    farmers should consult local agricultural extension services for appropriate
    pesticide recommendations.
    """

    chunker = DocumentChunker(chunk_size=300, chunk_overlap=50, language="en")
    chunks = chunker.chunk_document(sample_text)

    print(f"✓ Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i + 1}:")
        print(f"    - Section: {chunk.get('section', 'N/A')}")
        print(f"    - Tokens: {chunk.get('tokens')}")
        print(f"    - Preview: {chunk['content'][:80]}...")

    return True


def test_full_pipeline():
    """Test the complete ingestion pipeline."""
    print("\n" + "=" * 60)
    print("TEST 4: Complete Ingestion Pipeline")
    print("=" * 60)

    pipeline = IngestionPipeline()

    # Check if Agri_docs exists
    agri_docs_path = Path("Agri_docs")
    if not agri_docs_path.exists():
        print(f"⚠️  Agri_docs directory not found")
        return False

    pdf_files = list(agri_docs_path.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files in Agri_docs/")

    if not pdf_files:
        print("⚠️  No PDF files found")
        return False

    # Process first PDF only for testing
    print(f"\nProcessing first PDF: {pdf_files[0].name}")
    chunks = pipeline.ingest_pdf(str(pdf_files[0]))

    if chunks:
        print(f"✓ Successfully created {len(chunks)} chunks")
        
        # Show statistics
        total_tokens = sum(c.get("tokens", 0) for c in chunks)
        avg_tokens = total_tokens // len(chunks) if chunks else 0
        
        print(f"  - Total tokens: {total_tokens}")
        print(f"  - Average tokens per chunk: {avg_tokens}")

        # Show sample chunk
        if chunks:
            sample_chunk = chunks[0]
            print(f"\n  Sample chunk:")
            print(f"    - Filename: {sample_chunk.get('filename')}")
            print(f"    - Crops: {sample_chunk.get('crops')}")
            print(f"    - Language: {sample_chunk.get('language')}")
            print(f"    - Content preview: {sample_chunk.get('content')[:100]}...")

        return True
    else:
        print("✗ Failed to create chunks")
        return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("PDF INGESTION PIPELINE - TEST SUITE")
    print("=" * 60)

    tests = [
        ("PDF Parser", test_pdf_parser),
        ("Metadata Extractor", test_metadata_extractor),
        ("Document Chunking", test_chunking),
        ("Full Pipeline", test_full_pipeline),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test '{test_name}' failed with error: {str(e)}")
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
