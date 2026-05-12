"""
Main entry point for Agricultural RAG System.

Supports commands for:
- Ingesting PDFs
- Running tests
- Starting API/UI
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from ingestion.orchestrator import IngestionPipeline
from rag.embeddings_orchestrator import EmbeddingPipeline
from utils.logger import setup_logger

logger = setup_logger(__name__)


def ingest_command(args) -> int:
    """
    Execute PDF ingestion command.

    Args:
        args: Parsed arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    pdf_directory = args.pdf_dir
    output_summary = args.summary

    if not Path(pdf_directory).exists():
        logger.error(f"PDF directory not found: {pdf_directory}")
        return 1

    logger.info(f"Starting PDF ingestion from: {pdf_directory}")

    pipeline = IngestionPipeline()
    results = pipeline.ingest_batch(pdf_directory)

    # Print results
    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"Total files: {results['total_files']}")
    print(f"Successfully processed: {results['processed_files']}")
    print(f"Failed: {results['failed_files']}")
    print(f"Total chunks created: {results['total_chunks']}")
    print("=" * 60)

    # Print detailed summary
    print("\nDETAILED RESULTS:")
    for doc in results["documents"]:
        filename = doc["filename"]
        status = doc["status"]

        if status == "success":
            chunks = doc.get("chunks", 0)
            print(f"  ✓ {filename}: {chunks} chunks")
        else:
            print(f"  ✗ {filename}: {status}")
            if "error" in doc:
                print(f"    Error: {doc['error']}")

    # Save summary to file if requested
    if output_summary:
        summary_file = Path(output_summary)
        summary_file.parent.mkdir(parents=True, exist_ok=True)

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Summary saved to: {summary_file}")
        print(f"\nSummary saved to: {summary_file}")

    return 0 if results["failed_files"] == 0 else 1


def test_command(args) -> int:
    """
    Execute test command.

    Args:
        args: Parsed arguments

    Returns:
        Exit code
    """
    test_module = args.test_module if args.test_module else "tests.test_ingestion"

    print(f"Running tests from: {test_module}")

    try:
        # Import and run the test module
        if test_module == "tests.test_ingestion":
            from tests.test_ingestion import run_all_tests

            success = run_all_tests()
            return 0 if success else 1
        elif test_module == "tests.test_embeddings":
            from tests.test_embeddings import run_all_tests

            success = run_all_tests()
            return 0 if success else 1
        else:
            logger.error(f"Unknown test module: {test_module}")
            return 1

    except Exception as e:
        logger.error(f"Error running tests: {str(e)}")
        return 1


def embed_command(args) -> int:
    """
    Execute embedding command.

    Args:
        args: Parsed arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    pdf_directory = args.pdf_dir
    output_summary = args.summary
    model_name = args.model

    if not Path(pdf_directory).exists():
        logger.error(f"PDF directory not found: {pdf_directory}")
        return 1

    logger.info(f"Starting embedding pipeline from: {pdf_directory}")

    try:
        pipeline = EmbeddingPipeline()
        results = pipeline.ingest_and_embed(
            pdf_directory, model_name=model_name, batch_size=32, save_index=True
        )

        # Print results
        print("\n" + "=" * 60)
        print("EMBEDDING PIPELINE SUMMARY")
        print("=" * 60)
        print(f"Files processed: {results['processed_files']}")
        print(f"Failed files: {results['failed_files']}")
        print(f"Total chunks created: {results['total_chunks']}")
        print(f"Total vectors created: {results['embedding_stats']['total_vectors']}")
        print(f"Embedding dimension: {results['embedding_stats']['embedding_dimension']}")

        if results["embedding_model"]:
            print(f"Embedding model: {results['embedding_model']['model_name']}")

        print(f"Index saved to: {results['embedding_stats']['index_path']}")
        print(f"Metadata saved to: {results['embedding_stats']['metadata_path']}")
        print("=" * 60)

        # Save summary if requested
        if output_summary:
            summary_file = Path(output_summary)
            summary_file.parent.mkdir(parents=True, exist_ok=True)

            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"Summary saved to: {summary_file}")
            print(f"\nSummary saved to: {summary_file}")

        return 0 if results["failed_files"] == 0 else 1

    except Exception as e:
        logger.error(f"Error running embedding pipeline: {str(e)}")
        return 1


def api_command(args) -> int:
    """
    Start FastAPI server.

    Args:
        args: Parsed arguments

    Returns:
        Exit code
    """
    from utils.config import settings

    print(f"Starting FastAPI server...")
    print(f"Host: {settings.api_host}:{settings.api_port}")
    print(f"Debug mode: {settings.api_debug}")

    try:
        import uvicorn

        uvicorn.run(
            "api.fastapi_app:app",
            host=settings.api_host,
            port=settings.api_port,
            reload=settings.api_debug,
        )
        return 0
    except Exception as e:
        logger.error(f"Failed to start API: {str(e)}")
        return 1


def ui_command(args) -> int:
    """
    Start Streamlit UI.

    Args:
        args: Parsed arguments

    Returns:
        Exit code
    """
    import subprocess

    print("Starting Streamlit UI...")

    try:
        subprocess.run(["streamlit", "run", "ui/streamlit_app.py"], check=True)
        return 0
    except subprocess.CalledProcessError:
        logger.error("Failed to start Streamlit UI")
        return 1
    except FileNotFoundError:
        logger.error("Streamlit not found. Install with: pip install streamlit")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Agricultural RAG System - Main Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest PDFs from Agri_docs directory
  python main.py ingest --pdf-dir Agri_docs

  # Ingest and save summary
  python main.py ingest --pdf-dir Agri_docs --summary results.json

  # Run ingestion tests
  python main.py test --test-module tests.test_ingestion

  # Run embedding tests
  python main.py test --test-module tests.test_embeddings

  # Generate embeddings and build vector store
  python main.py embed --pdf-dir Agri_docs

  # Generate embeddings and save summary
  python main.py embed --pdf-dir Agri_docs --summary embedding_results.json

  # Use custom embedding model
  python main.py embed --model sentence-transformers/multilingual-MiniLM-L12-v2

  # Start API server
  python main.py api

  # Start Streamlit UI
  python main.py ui
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest PDFs from a directory")
    ingest_parser.add_argument(
        "--pdf-dir",
        default="Agri_docs",
        help="Directory containing PDF files (default: Agri_docs)",
    )
    ingest_parser.add_argument(
        "--summary",
        help="Save ingestion summary to JSON file",
    )
    ingest_parser.set_defaults(func=ingest_command)

    # Test command
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument(
        "--test-module",
        choices=["tests.test_ingestion", "tests.test_embeddings"],
        help="Specific test module to run (default: tests.test_ingestion)",
    )
    test_parser.set_defaults(func=test_command)

    # Embed command
    embed_parser = subparsers.add_parser("embed", help="Generate embeddings and build vector store")
    embed_parser.add_argument(
        "--pdf-dir",
        default="Agri_docs",
        help="Directory containing PDF files (default: Agri_docs)",
    )
    embed_parser.add_argument(
        "--model",
        help="Optional embedding model name override",
    )
    embed_parser.add_argument(
        "--summary",
        help="Save embedding results to JSON file",
    )
    embed_parser.set_defaults(func=embed_command)

    # API command
    api_parser = subparsers.add_parser("api", help="Start FastAPI server")
    api_parser.set_defaults(func=api_command)

    # UI command
    ui_parser = subparsers.add_parser("ui", help="Start Streamlit UI")
    ui_parser.set_defaults(func=ui_command)

    # Parse arguments
    args = parser.parse_args()

    # If no command specified, print help
    if not args.command:
        parser.print_help()
        return 1

    # Execute command
    try:
        return args.func(args)
    except Exception as e:
        logger.error(f"Error executing command: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
