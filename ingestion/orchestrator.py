"""
Ingestion Orchestrator for the Agricultural RAG system.

Coordinates:
- PDF parsing
- Metadata extraction
- Document chunking
- JSON output generation
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from ingestion.chunking import DocumentChunker
from ingestion.metadata_extractor import MetadataExtractor
from ingestion.pdf_parser import PDFParser
from utils.config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class IngestionPipeline:
    """Orchestrate the complete ingestion pipeline."""

    def __init__(self):
        """Initialize pipeline components."""
        self.pdf_parser = PDFParser()
        self.metadata_extractor = MetadataExtractor()
        self.output_path = Path(settings.data_processed_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def ingest_pdf(self, pdf_path: str) -> Optional[List[Dict]]:
        """
        Process a single PDF through the complete pipeline.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of processed chunks or None if failed
        """
        logger.info(f"Starting ingestion for {pdf_path}")

        # Step 1: Parse PDF
        parsed_doc = self.pdf_parser.parse_pdf(pdf_path)
        if not parsed_doc:
            logger.error(f"Failed to parse PDF: {pdf_path}")
            return None

        text = parsed_doc["text"]
        filename = parsed_doc["filename"]

        # Step 2: Extract metadata
        metadata = self.metadata_extractor.extract_metadata(text, filename)
        logger.info(f"Extracted metadata: {metadata}")

        # Step 3: Chunk document
        language = metadata.get("language", "en")
        chunker = DocumentChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            language=language,
        )

        chunks = chunker.chunk_document(text)
        logger.info(f"Created {len(chunks)} chunks")

        # Step 4: Enrich chunks with metadata
        enriched_chunks = chunker.add_metadata_to_chunks(chunks, metadata)

        # Step 5: Save processed document
        self._save_processed_document(filename, enriched_chunks)

        logger.info(f"Successfully ingested {filename}")
        return enriched_chunks

    def ingest_batch(self, pdf_directory: str) -> Dict[str, List]:
        """
        Ingest all PDFs from a directory.

        Args:
            pdf_directory: Directory containing PDF files

        Returns:
            Dictionary with statistics and results
        """
        pdf_path = Path(pdf_directory)

        if not pdf_path.exists():
            logger.error(f"Directory not found: {pdf_directory}")
            return {}

        pdf_files = list(pdf_path.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files to ingest")

        results = {
            "total_files": len(pdf_files),
            "processed_files": 0,
            "failed_files": 0,
            "total_chunks": 0,
            "documents": [],
        }

        for pdf_file in pdf_files:
            try:
                chunks = self.ingest_pdf(str(pdf_file))
                if chunks:
                    results["processed_files"] += 1
                    results["total_chunks"] += len(chunks)
                    results["documents"].append(
                        {
                            "filename": pdf_file.name,
                            "chunks": len(chunks),
                            "status": "success",
                        }
                    )
                else:
                    results["failed_files"] += 1
                    results["documents"].append(
                        {
                            "filename": pdf_file.name,
                            "status": "failed",
                        }
                    )
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {str(e)}")
                results["failed_files"] += 1
                results["documents"].append(
                    {
                        "filename": pdf_file.name,
                        "status": "error",
                        "error": str(e),
                    }
                )

        logger.info(
            f"Batch ingestion complete: {results['processed_files']}/{results['total_files']} "
            f"files processed, {results['total_chunks']} total chunks created"
        )

        return results

    def _save_processed_document(self, filename: str, chunks: List[Dict]) -> bool:
        """
        Save processed document chunks to JSON.

        Args:
            filename: Original PDF filename
            chunks: List of processed chunks

        Returns:
            True if saved successfully
        """
        try:
            # Create output filename
            base_name = Path(filename).stem
            output_file = self.output_path / f"{base_name}_processed.json"

            # Prepare output structure
            document_output = {
                "source_file": filename,
                "total_chunks": len(chunks),
                "chunks": chunks,
            }

            # Save to JSON
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(document_output, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved processed document to {output_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save processed document: {str(e)}")
            return False

    def get_all_chunks(self, processed_dir: Optional[str] = None) -> List[Dict]:
        """
        Load all processed chunks from JSON files.

        Args:
            processed_dir: Directory containing processed JSON files (default: settings.data_processed_path)

        Returns:
            List of all chunks
        """
        if processed_dir is None:
            processed_dir = settings.data_processed_path

        all_chunks = []
        processed_path = Path(processed_dir)

        if not processed_path.exists():
            logger.warning(f"Processed directory not found: {processed_dir}")
            return []

        json_files = list(processed_path.glob("*_processed.json"))
        logger.info(f"Found {len(json_files)} processed document files")

        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    chunks = data.get("chunks", [])
                    all_chunks.extend(chunks)
                    logger.info(f"Loaded {len(chunks)} chunks from {json_file.name}")
            except Exception as e:
                logger.error(f"Error loading {json_file.name}: {str(e)}")

        logger.info(f"Total chunks loaded: {len(all_chunks)}")
        return all_chunks


def run_ingestion_pipeline(pdf_directory: str) -> Dict:
    """
    Convenience function to run the complete ingestion pipeline.

    Args:
        pdf_directory: Directory containing PDF files

    Returns:
        Pipeline results and statistics
    """
    pipeline = IngestionPipeline()
    return pipeline.ingest_batch(pdf_directory)


if __name__ == "__main__":
    # Example usage
    pipeline = IngestionPipeline()

    # Ingest all PDFs from Agri_docs
    results = pipeline.ingest_batch("./Agri_docs")

    print("\n=== INGESTION SUMMARY ===")
    print(f"Total files: {results['total_files']}")
    print(f"Processed: {results['processed_files']}")
    print(f"Failed: {results['failed_files']}")
    print(f"Total chunks created: {results['total_chunks']}")

    print("\n=== DOCUMENT SUMMARY ===")
    for doc in results["documents"]:
        print(f"{doc['filename']}: {doc['status']}", end="")
        if "chunks" in doc:
            print(f" ({doc['chunks']} chunks)")
        else:
            print()

    # Load and display sample chunks
    all_chunks = pipeline.get_all_chunks()
    if all_chunks:
        print(f"\n=== SAMPLE CHUNK ===")
        sample_chunk = all_chunks[0]
        print(f"Filename: {sample_chunk.get('filename')}")
        print(f"Crop: {sample_chunk.get('crops')}")
        print(f"Language: {sample_chunk.get('language')}")
        print(f"Content preview: {sample_chunk.get('content')[:200]}...")
