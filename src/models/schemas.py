"""Pydantic models for API request/response schemas."""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ConfidenceLevel(str, Enum):
    """Confidence level for RAG responses."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class HealthStatus(str, Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class DependencyStatus(str, Enum):
    """Dependency status."""
    UP = "up"
    DOWN = "down"


class ErrorCode(str, Enum):
    """Machine-readable error codes."""
    INVALID_REQUEST = "INVALID_REQUEST"
    MESSAGE_TOO_LONG = "MESSAGE_TOO_LONG"
    SELECTED_TEXT_TOO_LONG = "SELECTED_TEXT_TOO_LONG"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


# Request Models

class ChatRequest(BaseModel):
    """Request body for /chat endpoint."""
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User's question or message"
    )
    session_id: Optional[UUID] = Field(
        default=None,
        description="Optional session ID for conversation continuity"
    )


class ContextualChatRequest(BaseModel):
    """Request body for /chat/context endpoint."""
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User's question about the selected text"
    )
    selected_text: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Text selected by user from textbook content"
    )
    chapter_slug: Optional[str] = Field(
        default=None,
        description="Chapter where text was selected"
    )
    session_id: Optional[UUID] = Field(
        default=None,
        description="Optional session ID for conversation continuity"
    )


# Response Models

class Source(BaseModel):
    """Citation source in a chat response."""
    chapter_slug: str = Field(..., description="URL-safe chapter identifier")
    chapter_title: str = Field(..., description="Human-readable chapter title")
    section_id: Optional[str] = Field(
        default=None,
        description="Section identifier (chapter_slug#heading_slug)"
    )
    section_title: Optional[str] = Field(
        default=None,
        description="Section heading text"
    )
    excerpt: str = Field(
        ...,
        max_length=200,
        description="Relevant excerpt from the source"
    )
    score: float = Field(
        ...,
        ge=-1,
        le=1,
        description="Semantic similarity score (-1 to 1 for cosine similarity)"
    )


class ChatResponse(BaseModel):
    """Response body for chat endpoints."""
    message_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this response"
    )
    session_id: UUID = Field(
        default_factory=uuid4,
        description="Session ID (new if not provided in request)"
    )
    answer: str = Field(
        ...,
        description="AI-generated answer based on textbook content"
    )
    confidence: ConfidenceLevel = Field(
        ...,
        description="Confidence level based on retrieval similarity scores"
    )
    sources: list[Source] = Field(
        default_factory=list,
        description="Citations to textbook sections used for the answer"
    )
    disclaimer: Optional[str] = Field(
        default=None,
        description="Optional disclaimer for medium/low confidence responses"
    )


class DependencyHealth(BaseModel):
    """Health status of individual dependencies."""
    qdrant: DependencyStatus = DependencyStatus.DOWN
    groq: DependencyStatus = DependencyStatus.DOWN
    neon: DependencyStatus = DependencyStatus.DOWN


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""
    status: HealthStatus = Field(..., description="Overall API health status")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Health check timestamp"
    )
    dependencies: Optional[DependencyHealth] = Field(
        default=None,
        description="Individual dependency status"
    )


class ErrorResponse(BaseModel):
    """Error response body."""
    error: str = Field(..., description="Human-readable error message")
    code: ErrorCode = Field(..., description="Machine-readable error code")
    details: Optional[dict] = Field(
        default=None,
        description="Additional error details"
    )
