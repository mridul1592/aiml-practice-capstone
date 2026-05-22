"""
Configuration management for the Agriculture RAG system.

This module loads configuration from environment variables and provides
a centralized config object for the entire application.
"""

import os
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # LLM Configuration
    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "orca2"
    openai_api_key: str = ""

    # Embedding Configuration
    embedding_model: str = "sentence-transformers/multilingual-e5-small"
    device: str = "cuda"
    embedding_dimension: int = 384

    # Vector Store Configuration
    faiss_index_path: str = "./data/embeddings/faiss_index.bin"
    metadata_index_path: str = "./data/embeddings/metadata.json"

    # RAG Configuration
    chunk_size: int = 200
    chunk_overlap: int = 40
    similarity_threshold: float = 0.5
    top_k_retrieval: int = 5
    supported_languages: List[str] = ["en", "hi", "pa"]
    language_detect_threshold: float = 0.5

    # FastAPI Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = True

    # Streamlit Configuration
    streamlit_port: int = 8501

    # Logging Configuration
    log_level: str = "INFO"
    log_file: str = "./logs/app.log"

    # Data Paths
    data_raw_path: str = "./data/raw"
    data_processed_path: str = "./data/processed"

    class Config:
        """Pydantic settings configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.data_raw_path,
            self.data_processed_path,
            os.path.dirname(self.faiss_index_path),
            os.path.dirname(self.log_file),
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
settings.ensure_directories()
