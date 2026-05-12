"""
PDF Parser for extracting and cleaning text from agricultural documents.

This module handles:
- PDF text extraction using PyMuPDF (fitz)
- Header/footer removal
- OCR artifact cleaning
- Whitespace normalization
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from utils.logger import setup_logger

logger = setup_logger(__name__)


class PDFParser:
    """Extract and clean text from PDF documents."""

    def __init__(self):
        """Initialize PDF parser with cleaning patterns."""
        # Patterns for common OCR artifacts and headers/footers
        self.ocr_patterns = [
            (r"[\|`]", ""),  # Remove common OCR artifacts
            (r"~+", ""),  # Remove repeated tildes
            (r"\^+", ""),  # Remove carets
            (r"={3,}", ""),  # Remove multiple equal signs
            (r"-{4,}", ""),  # Remove long dashes
        ]

        # Common header/footer indicators
        self.header_footer_keywords = [
            r"page \d+",
            r"p\. \d+",
            r"©.*",
            r"www\.",
            r"email:",
            r"contact:",
            r"^chapter \d+",
        ]

    def extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        """
        Extract text from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text or None if extraction fails
        """
        try:
            pdf_document = fitz.open(pdf_path)
            text = ""

            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                page_text = page.get_text()
                text += f"\n--- Page {page_num + 1} ---\n{page_text}"

            pdf_document.close()
            logger.info(f"Successfully extracted text from {pdf_path}")
            return text

        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            return None

    def remove_ocr_artifacts(self, text: str) -> str:
        """
        Remove common OCR artifacts from extracted text.

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        for pattern, replacement in self.ocr_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def remove_headers_footers(self, text: str) -> str:
        """
        Remove common headers and footers from text.

        Args:
            text: Text to clean

        Returns:
            Text without headers/footers
        """
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            # Skip lines that match header/footer patterns
            is_header_footer = any(
                re.search(pattern, line, re.IGNORECASE) for pattern in self.header_footer_keywords
            )

            if not is_header_footer:
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace in text.

        Args:
            text: Text to normalize

        Returns:
            Text with normalized whitespace
        """
        # Replace multiple newlines with double newline
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Replace multiple spaces with single space
        text = re.sub(r" {2,}", " ", text)

        # Remove trailing whitespace from lines
        lines = text.split("\n")
        lines = [line.rstrip() for line in lines]
        text = "\n".join(lines)

        return text.strip()

    def clean_text(self, text: str) -> str:
        """
        Apply all cleaning operations to extracted text.

        Args:
            text: Raw extracted text

        Returns:
            Fully cleaned text
        """
        if not text:
            return ""

        # Apply cleaning in sequence
        text = self.remove_ocr_artifacts(text)
        text = self.remove_headers_footers(text)
        text = self.normalize_whitespace(text)

        return text

    def parse_pdf(self, pdf_path: str) -> Optional[Dict[str, str]]:
        """
        Parse a PDF file and return cleaned text with metadata.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary with 'text' and 'filename' keys, or None if parsing fails
        """
        # Extract text
        raw_text = self.extract_text_from_pdf(pdf_path)
        if raw_text is None:
            return None

        # Clean text
        cleaned_text = self.clean_text(raw_text)

        if not cleaned_text:
            logger.warning(f"No text extracted from {pdf_path}")
            return None

        return {
            "text": cleaned_text,
            "filename": Path(pdf_path).name,
            "path": str(pdf_path),
        }

    def batch_parse_pdfs(
        self, pdf_dir: str, file_pattern: str = "*.pdf"
    ) -> List[Dict[str, str]]:
        """
        Parse all PDFs in a directory.

        Args:
            pdf_dir: Directory containing PDF files
            file_pattern: Pattern to match PDF files (default: *.pdf)

        Returns:
            List of parsed document dictionaries
        """
        pdf_path = Path(pdf_dir)

        if not pdf_path.exists():
            logger.error(f"PDF directory not found: {pdf_dir}")
            return []

        pdf_files = list(pdf_path.glob(file_pattern))

        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_dir}")
            return []

        logger.info(f"Found {len(pdf_files)} PDF files to process")

        parsed_documents = []
        for pdf_file in pdf_files:
            logger.info(f"Processing {pdf_file.name}...")
            parsed = self.parse_pdf(str(pdf_file))

            if parsed:
                parsed_documents.append(parsed)

        logger.info(f"Successfully parsed {len(parsed_documents)}/{len(pdf_files)} PDFs")
        return parsed_documents


def parse_pdfs_from_directory(pdf_directory: str) -> List[Dict[str, str]]:
    """
    Convenience function to parse all PDFs from a directory.

    Args:
        pdf_directory: Path to directory containing PDFs

    Returns:
        List of parsed documents
    """
    parser = PDFParser()
    return parser.batch_parse_pdfs(pdf_directory)


if __name__ == "__main__":
    # Example usage
    parser = PDFParser()

    # Parse a single PDF
    result = parser.parse_pdf("sample.pdf")
    if result:
        print(f"Parsed: {result['filename']}")
        print(f"Text length: {len(result['text'])} characters")

    # Parse all PDFs in a directory
    # documents = parse_pdfs_from_directory("./data/raw")
