"""
Integration layer between audio processing and RAG system.

Handles:
- Audio input → RAG pipeline workflow
- Language preservation from audio
- Response formatting for audio queries
"""

from typing import Dict, Optional

from audio.audio_processor import AudioProcessor
from rag.rag_orchestrator import RAGPipeline
from utils.logger import setup_logger

logger = setup_logger(__name__)


class AudioRAGHandler:
    """Handle audio input to RAG pipeline."""

    def __init__(self, audio_model_size: str = "base"):
        """
        Initialize audio-RAG handler.

        Args:
            audio_model_size: Whisper model size (base, small, medium, large)
        """
        self.audio_processor = AudioProcessor(model_size=audio_model_size)
        self.rag_pipeline = RAGPipeline()

        logger.info("AudioRAGHandler initialized")

    def process_audio_query(
        self,
        audio_path: str,
        k: int = 5,
        similarity_threshold: Optional[float] = None,
        temperature: float = 0.7,
        preserve_language: bool = True,
    ) -> Dict:
        """
        Process audio file and execute RAG query.

        Args:
            audio_path: Path to audio file
            k: Number of context chunks to retrieve
            similarity_threshold: Minimum similarity threshold
            temperature: LLM temperature
            preserve_language: Whether to respond in audio's language

        Returns:
            Dictionary with audio transcription, detected language, RAG response

        Raises:
            FileNotFoundError: If audio file not found
            ValueError: If format not supported
            Exception: If RAG query fails
        """
        logger.info(f"Processing audio query: {audio_path}")

        try:
            # Step 1: Transcribe audio
            logger.info("Step 1: Transcribing audio...")
            transcription_result = self.audio_processor.transcribe(audio_path)

            query_text = transcription_result["text"]
            detected_language = transcription_result["language"]
            detected_language_code = self._map_language_code(detected_language)

            logger.info(
                f"Audio transcribed: {len(query_text)} characters, "
                f"language: {detected_language}"
            )

            # Step 2: Execute RAG query
            logger.info("Step 2: Executing RAG query...")
            response_language = (
                detected_language_code if preserve_language else None
            )

            rag_result = self.rag_pipeline.query(
                query_text,
                k=k,
                similarity_threshold=similarity_threshold,
                language=response_language,
                temperature=temperature,
            )

            # Step 3: Combine results
            combined_result = {
                "audio_file": audio_path,
                "transcription": query_text,
                "detected_audio_language": detected_language,
                "detected_language_code": detected_language_code,
                "audio_confidence": transcription_result.get("confidence", 0),
                "audio_duration_seconds": transcription_result.get("duration", 0),
                "query": query_text,
                "response": rag_result["response"],
                "response_language": rag_result.get("language", "en"),
                "rag_confidence": rag_result["confidence"],
                "num_context_chunks": rag_result["num_context_chunks"],
                "sources": rag_result["sources"],
            }

            logger.info("Audio query processing completed successfully")
            return combined_result

        except Exception as e:
            logger.error(f"Audio query processing failed: {str(e)}")
            raise

    def _map_language_code(self, whisper_language: str) -> str:
        """
        Map Whisper language code to RAG system language code.

        Args:
            whisper_language: Language code from Whisper

        Returns:
            RAG system language code (en, hi, pa, ta, te, or, kn, mr, ml, bn)
        """
        language_map = {
            "english": "en",
            "hindi": "hi",
            "punjabi": "pa",
            "tamil": "ta",
            "telugu": "te",
            "odia": "or",
            "kannada": "kn",
            "marathi": "mr",
            "malayalam": "ml",
            "bengali": "bn",
        }

        whisper_lang_lower = whisper_language.lower()
        return language_map.get(whisper_lang_lower, whisper_lang_lower[:2])

    def get_supported_audio_formats(self) -> list:
        """Get list of supported audio formats."""
        return list(self.audio_processor.SUPPORTED_FORMATS)

    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages."""
        return self.audio_processor.LANGUAGE_MAPPING.copy()


if __name__ == "__main__":
    try:
        print("Testing AudioRAGHandler...")
        handler = AudioRAGHandler()
        print("✓ AudioRAGHandler initialized successfully")
        print(f"✓ Supported audio formats: {handler.get_supported_audio_formats()}")
        print(f"✓ Supported languages: {handler.get_supported_languages()}")

    except Exception as e:
        print(f"✗ Error: {str(e)}")
