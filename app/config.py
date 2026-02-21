"""Centralized configuration using pydantic-settings."""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    # LLM Configuration
    llm_provider: str = "openai"
    model_name: str = "gpt-3.5-turbo"

    # Embedding Configuration
    embedding_provider: str = "sentence_transformers"
    embedding_model: str = "all-MiniLM-L6-v2"

    # Vector Store Configuration
    vector_store_type: str = "chroma"
    vector_store_persist_dir: str = "./data/vectorstore"

    # Chunking Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval Configuration
    top_k: int = 5

    # API Keys
    openai_api_key: str = None

    # Data Directory
    data_dir: str = "./data"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
