"""Retrieval service using Qdrant vector search."""

from typing import List, Optional
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from src.config import get_settings
from src.services.embeddings import embed_text

settings = get_settings()


@dataclass
class RetrievalResult:
    """Single retrieval result with metadata."""
    chunk_text: str
    chapter_slug: str
    section_id: Optional[str]
    chunk_index: int
    score: float
    start_char: int
    end_char: int


# Chapter title mapping
CHAPTER_TITLES = {
    "intro": "Introduction to Physical AI",
    "humanoid-basics": "Basics of Humanoid Robotics",
    "ros2-fundamentals": "ROS 2 Fundamentals",
    "digital-twin": "Digital Twin Simulation",
    "vla-systems": "Vision-Language-Action Systems",
    "capstone": "Capstone Project",
}


def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client instance."""
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )


def search_similar(
    query: str,
    top_k: int = None,
    chapter_filter: Optional[str] = None,
) -> List[RetrievalResult]:
    """Search for similar text chunks using vector similarity.

    Args:
        query: Search query text
        top_k: Number of results to return (default from settings)
        chapter_filter: Optional chapter slug to filter results

    Returns:
        List of retrieval results sorted by score (descending)
    """
    if top_k is None:
        top_k = settings.retrieval_top_k

    # Embed query
    query_vector = embed_text(query)

    # Build filter if chapter specified
    search_filter = None
    if chapter_filter:
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="chapter_slug",
                    match=MatchValue(value=chapter_filter),
                )
            ]
        )

    # Search Qdrant
    client = get_qdrant_client()
    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=top_k,
        query_filter=search_filter,
        with_payload=True,
    ).points

    # Convert to RetrievalResult objects
    retrieval_results = []
    for hit in results:
        payload = hit.payload or {}
        retrieval_results.append(
            RetrievalResult(
                chunk_text=payload.get("chunk_text", ""),
                chapter_slug=payload.get("chapter_slug", "unknown"),
                section_id=payload.get("section_id"),
                chunk_index=payload.get("chunk_index", 0),
                score=hit.score,
                start_char=payload.get("start_char", 0),
                end_char=payload.get("end_char", 0),
            )
        )

    return retrieval_results


def get_confidence_level(top_score: float) -> str:
    """Determine confidence level based on top retrieval score.

    Args:
        top_score: Highest similarity score from retrieval

    Returns:
        Confidence level: 'high', 'medium', or 'low'
    """
    if top_score >= settings.confidence_high_threshold:
        return "high"
    elif top_score >= settings.confidence_low_threshold:
        return "medium"
    else:
        return "low"


def get_chapter_title(chapter_slug: str) -> str:
    """Get human-readable chapter title from slug."""
    return CHAPTER_TITLES.get(chapter_slug, chapter_slug.replace("-", " ").title())
