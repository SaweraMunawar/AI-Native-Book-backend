"""Embedding service using sentence-transformers MiniLM-L6-v2."""

from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

from src.config import get_settings

settings = get_settings()


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load and cache the embedding model."""
    return SentenceTransformer(settings.embedding_model)


def embed_text(text: str) -> List[float]:
    """Generate embedding vector for a single text.

    Args:
        text: Input text to embed

    Returns:
        384-dimensional embedding vector
    """
    model = get_embedding_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate embedding vectors for multiple texts.

    Args:
        texts: List of input texts to embed

    Returns:
        List of 384-dimensional embedding vectors
    """
    model = get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()
