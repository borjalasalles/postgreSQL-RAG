import logging
import os
from datetime import timedelta
from functools import lru_cache
from typing import Optional
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

current_file = Path(__file__)  # app/config/settings.py
services_dir = current_file.parent.parent / 'services'  # app/services/
env_path = services_dir / '.env'
load_dotenv(dotenv_path=env_path)

def setup_logging():
    """Configure basic logging for the application."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


class LLMSettings(BaseModel):
    """Base settings for Language Model configurations."""

    temperature: float = 0.0
    max_tokens: Optional[int] = None
    max_retries: int = 3


class AnthropicSettings(LLMSettings):
    """Anthropic-specific settings extending LLMSettings."""

    api_key: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    default_model: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"))
    temperature: float = Field(default_factory=lambda: float(os.getenv("ANTHROPIC_TEMPERATURE", "0.2")))
    max_tokens: int = Field(default_factory=lambda: int(os.getenv("ANTHROPIC_MAX_TOKENS", "1000")))
    timeout: int = Field(default_factory=lambda: int(os.getenv("ANTHROPIC_TIMEOUT", "300")))


class EmbeddingSettings(BaseModel):
    """Settings for embedding models."""
    
    model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_dimensions: int = Field(default=384)  # For all-MiniLM-L6-v2


class DatabaseSettings(BaseModel):
    """Database connection settings."""

    service_url: str = Field(default_factory=lambda: os.getenv("TIMESCALE_SERVICE_URL"))


class VectorStoreSettings(BaseModel):
    """Settings for the VectorStore."""

    table_name: str = "embeddings"
    time_partition_interval: timedelta = timedelta(days=7)


class Settings(BaseModel):
    """Main settings class combining all sub-settings."""

    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)


@lru_cache()
def get_settings() -> Settings:
    """Create and return a cached instance of the Settings."""
    settings = Settings()
    setup_logging()
    return settings