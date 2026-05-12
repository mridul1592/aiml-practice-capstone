"""
Smart chunking for agricultural documents.

Implements semantic chunking that:
- Respects document structure (sections, headers)
- Maintains chunk size (300-500 tokens)
- Preserves metadata across chunks
- Handles multilingual content
"""

import re
from typing import Dict, List, Optional, Tuple

from utils.logger import setup_logger

logger = setup_logger(__name__)

# Approximate token counting (rough estimates for different languages)
CHARS_PER_TOKEN = {
    "en": 4.0,  # English: ~4 chars per token
    "hi": 2.0,  # Hindi: ~2 chars per token (Devanagari)
    "pa": 2.0,  # Punjabi: ~2 chars per token (Gurmukhi)
}


class DocumentChunker:
    """Chunk documents while preserving semantic meaning and metadata."""

    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 100,
        language: str = "en",
    ):
        """
        Initialize document chunker.

        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Token overlap between chunks
            language: Language code for character-to-token conversion
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.language = language
        self.chars_per_token = CHARS_PER_TOKEN.get(language, 4.0)

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Args:
            text: Text to count

        Returns:
            Estimated token count
        """
        return int(len(text) / self.chars_per_token)

    def get_section_boundaries(self, text: str) -> List[Tuple[int, str]]:
        """
        Identify section boundaries in the text.

        Detects:
        - Markdown-style headers (# ## ###)
        - Numbered sections (1. 2. etc)
        - All-caps headers
        - Pattern-based sections

        Args:
            text: Text to analyze

        Returns:
            List of (position, header_text) tuples
        """
        boundaries = []

        # Patterns for different header styles
        patterns = [
            (r"^#{1,3}\s+(.+)$", re.MULTILINE),  # Markdown headers
            (r"^(\d+\.)\s+([A-Z][^\n]+)$", re.MULTILINE),  # Numbered sections
            (r"^([A-Z][A-Z\s]{3,})$", re.MULTILINE),  # ALL CAPS headers
            (r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*):$", re.MULTILINE),  # Title case with colon
        ]

        for pattern, flags in patterns:
            for match in re.finditer(pattern, text, flags):
                boundaries.append((match.start(), match.group()))

        # Remove duplicates and sort
        boundaries = list(set(boundaries))
        boundaries.sort(key=lambda x: x[0])

        return boundaries

    def chunk_by_sections(self, text: str) -> List[Dict[str, str]]:
        """
        Chunk text by semantic sections.

        Args:
            text: Text to chunk

        Returns:
            List of chunk dictionaries
        """
        boundaries = self.get_section_boundaries(text)

        if not boundaries:
            # No clear sections found, use simple chunking
            return self._chunk_by_size(text)

        chunks = []
        current_pos = 0

        for i, (boundary_pos, header_text) in enumerate(boundaries):
            # Get text until next boundary
            if i + 1 < len(boundaries):
                next_boundary = boundaries[i + 1][0]
            else:
                next_boundary = len(text)

            section_text = text[boundary_pos:next_boundary].strip()

            if not section_text:
                continue

            # If section is too large, sub-chunk it
            if self.estimate_tokens(section_text) > self.chunk_size * 1.5:
                sub_chunks = self._chunk_by_size(section_text, preserve_header=header_text)
                chunks.extend(sub_chunks)
            else:
                chunks.append(
                    {
                        "content": section_text,
                        "section": header_text,
                        "tokens": self.estimate_tokens(section_text),
                    }
                )

        return chunks

    def _chunk_by_size(
        self, text: str, preserve_header: str = ""
    ) -> List[Dict[str, str]]:
        """
        Chunk text by size when semantic boundaries aren't suitable.

        Args:
            text: Text to chunk
            preserve_header: Optional header to preserve in each chunk

        Returns:
            List of chunk dictionaries
        """
        chunks = []
        target_char_size = int(self.chunk_size * self.chars_per_token)
        overlap_char_size = int(self.chunk_overlap * self.chars_per_token)

        # Split into sentences for better boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text)

        current_chunk = ""
        current_header = preserve_header

        for sentence in sentences:
            # Add header if we're starting a new chunk
            if not current_chunk and current_header:
                current_chunk = f"{current_header}\n\n"

            test_chunk = current_chunk + sentence + " "

            # If adding this sentence exceeds target size, save and start new
            if len(test_chunk) > target_char_size and current_chunk:
                chunks.append(
                    {
                        "content": current_chunk.strip(),
                        "section": current_header,
                        "tokens": self.estimate_tokens(current_chunk),
                    }
                )

                # Create overlap: take last part of previous chunk
                overlap_text = current_chunk[-(overlap_char_size - 100) :]
                if current_header:
                    current_chunk = f"{current_header}\n\n{overlap_text}\n\n{sentence} "
                else:
                    current_chunk = f"{overlap_text}\n\n{sentence} "
            else:
                current_chunk = test_chunk

        # Add remaining chunk
        if current_chunk.strip():
            chunks.append(
                {
                    "content": current_chunk.strip(),
                    "section": current_header,
                    "tokens": self.estimate_tokens(current_chunk),
                }
            )

        return chunks

    def chunk_document(self, text: str) -> List[Dict[str, str]]:
        """
        Chunk a document intelligently.

        Args:
            text: Document text

        Returns:
            List of chunks with metadata
        """
        if not text:
            logger.warning("Empty text provided for chunking")
            return []

        logger.info(
            f"Chunking document ({len(text)} chars, ~{self.estimate_tokens(text)} tokens)"
        )

        # Try section-based chunking first
        chunks = self.chunk_by_sections(text)

        # Validate chunks
        valid_chunks = [c for c in chunks if c["tokens"] > 0]

        logger.info(
            f"Created {len(valid_chunks)} chunks "
            f"(avg: {sum(c['tokens'] for c in valid_chunks) // len(valid_chunks) if valid_chunks else 0} tokens)"
        )

        return valid_chunks

    def add_metadata_to_chunks(
        self, chunks: List[Dict[str, str]], document_metadata: Dict
    ) -> List[Dict]:
        """
        Add document metadata to each chunk.

        Args:
            chunks: List of chunks
            document_metadata: Document-level metadata

        Returns:
            Chunks with added metadata
        """
        enriched_chunks = []

        for chunk_idx, chunk in enumerate(chunks):
            enriched_chunk = {
                **chunk,
                **document_metadata,
                "chunk_id": f"{document_metadata.get('filename', 'unknown')}_{chunk_idx}",
                "chunk_index": chunk_idx,
            }
            enriched_chunks.append(enriched_chunk)

        return enriched_chunks


def chunk_document(
    text: str,
    language: str = "en",
    chunk_size: int = 400,
    chunk_overlap: int = 100,
    metadata: Optional[Dict] = None,
) -> List[Dict]:
    """
    Convenience function to chunk a document.

    Args:
        text: Document text
        language: Language code
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks
        metadata: Optional metadata to add to chunks

    Returns:
        List of chunks with metadata
    """
    chunker = DocumentChunker(chunk_size, chunk_overlap, language)
    chunks = chunker.chunk_document(text)

    if metadata:
        chunks = chunker.add_metadata_to_chunks(chunks, metadata)

    return chunks


if __name__ == "__main__":
    # Example usage
    sample_text = """
    # Wheat Cultivation Guide

    ## Section 1: Soil Preparation
    Prepare the soil by removing weeds and adding organic matter.
    The soil should be well-draining and rich in nutrients.

    ## Section 2: Planting
    Plant wheat seeds at a depth of 5-7 cm.
    Ensure proper spacing between seeds.

    ## Section 3: Disease Management
    Common diseases include leaf spot and blast.
    Use appropriate pesticides as recommended by agricultural experts.
    """

    chunker = DocumentChunker(chunk_size=200, language="en")
    chunks = chunker.chunk_document(sample_text)

    metadata = {"filename": "wheat_guide.pdf", "crop": "wheat", "language": "en"}
    enriched_chunks = chunker.add_metadata_to_chunks(chunks, metadata)

    for chunk in enriched_chunks:
        print(f"\n--- Chunk {chunk['chunk_index']} ---")
        print(f"Section: {chunk['section']}")
        print(f"Tokens: {chunk['tokens']}")
        print(f"Content: {chunk['content'][:100]}...")
