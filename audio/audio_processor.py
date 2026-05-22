"""
Audio processing for RAG system.

Handles:
- Audio file input validation
- Speech-to-text transcription using Whisper
- Automatic language detection from audio
- Support for multiple audio formats
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

from utils.logger import setup_logger

logger = setup_logger(__name__)


class AudioProcessor:
    """Process audio files for RAG queries."""

    SUPPORTED_FORMATS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac"}
    LANGUAGE_MAPPING = {
        "en": "English",
        "hi": "Hindi",
        "pa": "Punjabi",
    }

    def __init__(self, model_size: str = "base"):
        """
        Initialize audio processor with Whisper model.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)

        Raises:
            ImportError: If whisper not installed
        """
        try:
            import whisper

            self.whisper = whisper
        except ImportError:
            raise ImportError(
                "whisper library required for audio processing. "
                "Install with: pip install openai-whisper"
            )

        self.model_size = model_size
        self.model = None
        logger.info(f"AudioProcessor initialized with model size: {model_size}")

    def _load_model(self) -> None:
        """Load Whisper model if not already loaded."""
        if self.model is None:
            logger.info(f"Loading Whisper model: {self.model_size}")
            self.model = self.whisper.load_model(self.model_size)
            logger.info("Whisper model loaded successfully")

    def validate_audio_file(self, audio_path: str) -> bool:
        """
        Validate audio file exists and is supported format.

        Args:
            audio_path: Path to audio file

        Returns:
            True if valid, raises exception otherwise

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If format not supported
        """
        path = Path(audio_path)

        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported audio format: {path.suffix}. "
                f"Supported: {', '.join(self.SUPPORTED_FORMATS)}"
            )

        logger.info(f"Audio file validated: {path.name}")
        return True

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> Dict:
        """
        Transcribe audio file to text using Whisper.

        Args:
            audio_path: Path to audio file
            language: Language code (en, hi, pa) or None for auto-detect

        Returns:
            Dictionary with transcription, language, and confidence

        Raises:
            FileNotFoundError: If audio file not found
            ValueError: If format not supported
            Exception: If transcription fails
        """
        # Validate audio file
        self.validate_audio_file(audio_path)

        # Load model if needed
        self._load_model()

        logger.info(f"Transcribing audio: {audio_path}")

        try:
            # Transcribe with language parameter if specified
            if language and language in self.LANGUAGE_MAPPING:
                logger.info(f"Transcribing with language hint: {language}")
                result = self.model.transcribe(
                    audio_path, language=language, verbose=False
                )
            else:
                logger.info("Transcribing with automatic language detection")
                result = self.model.transcribe(audio_path, verbose=False)

            # Extract results
            text = result.get("text", "").strip()
            detected_language = result.get("language", "en")

            logger.info(
                f"Transcription complete: {len(text)} characters, "
                f"language: {detected_language}"
            )

            return {
                "text": text,
                "language": detected_language,
                "confidence": result.get("avg_logprob", 0),
                "duration": result.get("duration", 0),
            }

        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise

    def transcribe_with_language_detection(
        self, audio_path: str
    ) -> Tuple[str, str]:
        """
        Transcribe audio and detect language.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (transcribed_text, detected_language_code)
        """
        result = self.transcribe(audio_path)

        # Map Whisper language codes to our codes
        language_map = {
            "english": "en",
            "hindi": "hi",
            "punjabi": "pa",
        }

        detected_lang = result["language"].lower()
        language_code = language_map.get(detected_lang, detected_lang[:2])

        return result["text"], language_code

    def get_language_name(self, language_code: str) -> str:
        """
        Get human-readable language name from code.

        Args:
            language_code: Language code (en, hi, pa)

        Returns:
            Language name
        """
        return self.LANGUAGE_MAPPING.get(language_code, "Unknown")


if __name__ == "__main__":
    # Example usage
    try:
        print("Testing AudioProcessor...")
        processor = AudioProcessor(model_size="base")

        # Create a sample audio test
        print("AudioProcessor ready for audio processing")
        print(f"Supported formats: {', '.join(processor.SUPPORTED_FORMATS)}")

    except Exception as e:
        print(f"Error: {str(e)}")
