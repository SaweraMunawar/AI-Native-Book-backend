"""Chat API endpoints."""

from uuid import uuid4
from typing import Optional

from fastapi import APIRouter, HTTPException, status

from src.config import get_settings
from src.models.schemas import (
    ChatRequest,
    ChatResponse,
    ContextualChatRequest,
    ConfidenceLevel,
    ErrorCode,
    ErrorResponse,
    Source,
)
from src.services.retrieval import (
    search_similar,
    get_confidence_level,
    get_chapter_title,
)
from src.services.generation import (
    generate_response,
    generate_low_confidence_response,
)

router = APIRouter(prefix="/chat", tags=["Chat"])
settings = get_settings()


def build_sources(retrieval_results) -> list[Source]:
    """Convert retrieval results to Source objects."""
    sources = []
    for result in retrieval_results:
        # Extract section title from section_id
        section_title = None
        if result.section_id and "#" in result.section_id:
            section_title = result.section_id.split("#")[1].replace("-", " ").title()

        sources.append(
            Source(
                chapter_slug=result.chapter_slug,
                chapter_title=get_chapter_title(result.chapter_slug),
                section_id=result.section_id,
                section_title=section_title,
                excerpt=result.chunk_text[:200] if len(result.chunk_text) > 200 else result.chunk_text,
                score=round(result.score, 3),
            )
        )
    return sources


@router.post(
    "",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    summary="Send a chat message",
    description="Send a user message and receive an AI response based on textbook content.",
)
async def send_chat_message(request: ChatRequest) -> ChatResponse:
    """Handle chat message and generate response.

    1. Retrieve relevant chunks from vector database
    2. Determine confidence level from retrieval scores
    3. Generate response using LLM with retrieved context
    4. Return response with citations
    """
    try:
        # Retrieve relevant content
        results = search_similar(request.message)

        # Determine confidence
        top_score = results[0].score if results else 0
        confidence = get_confidence_level(top_score)

        # Generate response based on confidence
        if confidence == "low":
            answer = generate_low_confidence_response(request.message)
            sources = []
            disclaimer = "This topic may not be covered in the textbook."
        else:
            answer = generate_response(request.message, results)
            sources = build_sources(results)
            disclaimer = None

            if confidence == "medium":
                disclaimer = "Based on limited context from the textbook. The answer may be incomplete."

        return ChatResponse(
            message_id=uuid4(),
            session_id=request.session_id or uuid4(),
            answer=answer,
            confidence=ConfidenceLevel(confidence),
            sources=sources,
            disclaimer=disclaimer,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Failed to process chat request",
                "code": ErrorCode.INTERNAL_ERROR.value,
                "details": {"message": str(e)},
            },
        )


@router.post(
    "/context",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Send a message with selected text context",
    description="Send a user message with pre-selected text from the textbook.",
)
async def send_contextual_message(request: ContextualChatRequest) -> ChatResponse:
    """Handle contextual chat message with selected text.

    1. Use selected text as additional context
    2. Optionally filter retrieval by chapter
    3. Generate response with both selected text and retrieved context
    """
    try:
        # Retrieve relevant content (optionally filtered by chapter)
        results = search_similar(
            request.message,
            chapter_filter=request.chapter_slug,
        )

        # Determine confidence
        top_score = results[0].score if results else 0
        confidence = get_confidence_level(top_score)

        # Generate response with selected text context
        if confidence == "low":
            # Even with low retrieval confidence, we have the selected text
            answer = generate_response(
                request.message,
                results,
                selected_text=request.selected_text,
            )
            sources = build_sources(results) if results else []
            disclaimer = "Response based primarily on the selected text."
            confidence = "medium"  # Upgrade since we have context
        else:
            answer = generate_response(
                request.message,
                results,
                selected_text=request.selected_text,
            )
            sources = build_sources(results)
            disclaimer = None

            if confidence == "medium":
                disclaimer = "Based on the selected text and limited textbook context."

        return ChatResponse(
            message_id=uuid4(),
            session_id=request.session_id or uuid4(),
            answer=answer,
            confidence=ConfidenceLevel(confidence),
            sources=sources,
            disclaimer=disclaimer,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Failed to process contextual chat request",
                "code": ErrorCode.INTERNAL_ERROR.value,
                "details": {"message": str(e)},
            },
        )
