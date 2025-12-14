"""Configuration management for the RAG backend."""

import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Qdrant configuration
    qdrant_url: str = Field(default="", alias="QDRANT_URL")
    qdrant_api_key: str = Field(default="", alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="textbook_embeddings", alias="QDRANT_COLLECTION")

    # Groq configuration
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.1-8b-instant", alias="GROQ_MODEL")

    # Neon PostgreSQL configuration
    database_url: str = Field(default="", alias="DATABASE_URL")

    # Application configuration
    environment: str = Field(default="development", alias="ENVIRONMENT")
    cors_origins: str = Field(default="http://localhost:3000", alias="CORS_ORIGINS")

    # Rate limiting
    rate_limit_requests: int = Field(default=100, alias="RATE_LIMIT_REQUESTS")
    rate_limit_window_seconds: int = Field(default=3600, alias="RATE_LIMIT_WINDOW_SECONDS")

    # Embedding configuration
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL"
    )
    embedding_dimension: int = Field(default=384, alias="EMBEDDING_DIMENSION")

    # Retrieval configuration
    retrieval_top_k: int = Field(default=3, alias="RETRIEVAL_TOP_K")
    confidence_high_threshold: float = Field(default=0.7, alias="CONFIDENCE_HIGH_THRESHOLD")
    confidence_low_threshold: float = Field(default=0.4, alias="CONFIDENCE_LOW_THRESHOLD")

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.cors_origins.split(",")]

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
