"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from datetime import datetime
import hashlib
from typing import Optional

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import get_settings
from src.api import health, chat
from src.models.schemas import ErrorCode, ErrorResponse

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    print(f"Starting RAG Chatbot API in {settings.environment} mode")
    yield
    # Shutdown
    print("Shutting down RAG Chatbot API")


# Create FastAPI application
app = FastAPI(
    title="Textbook RAG Chatbot API",
    description="RAG-powered chatbot API for Physical AI & Humanoid Robotics textbook",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Rate limiting storage (in-memory for development, use Redis/Neon in production)
rate_limit_store: dict[str, dict] = {}


def get_client_hash(request: Request) -> str:
    """Get hashed client identifier for rate limiting."""
    client_ip = request.client.host if request.client else "unknown"
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    identifier = forwarded_for.split(",")[0].strip() if forwarded_for else client_ip
    return hashlib.sha256(identifier.encode()).hexdigest()


def check_rate_limit(client_hash: str) -> tuple[bool, Optional[int]]:
    """Check if client has exceeded rate limit.

    Returns:
        Tuple of (is_allowed, retry_after_seconds)
    """
    now = datetime.utcnow()

    if client_hash not in rate_limit_store:
        rate_limit_store[client_hash] = {
            "count": 1,
            "window_start": now,
        }
        return True, None

    client_data = rate_limit_store[client_hash]
    window_start = client_data["window_start"]
    elapsed_seconds = (now - window_start).total_seconds()

    # Reset window if expired
    if elapsed_seconds >= settings.rate_limit_window_seconds:
        rate_limit_store[client_hash] = {
            "count": 1,
            "window_start": now,
        }
        return True, None

    # Check if limit exceeded
    if client_data["count"] >= settings.rate_limit_requests:
        retry_after = int(settings.rate_limit_window_seconds - elapsed_seconds)
        return False, retry_after

    # Increment count
    client_data["count"] += 1
    return True, None


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware for chat endpoints."""
    # Only rate limit chat endpoints
    if request.url.path.startswith("/chat"):
        client_hash = get_client_hash(request)
        is_allowed, retry_after = check_rate_limit(client_hash)

        if not is_allowed:
            error = ErrorResponse(
                error="Rate limit exceeded. Please try again later.",
                code=ErrorCode.RATE_LIMIT_EXCEEDED,
                details={"retry_after": retry_after},
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content=error.model_dump(),
                headers={"Retry-After": str(retry_after)},
            )

    return await call_next(request)


# Register routers
app.include_router(health.router)
app.include_router(chat.router)


# Root endpoint
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirect to docs."""
    return {"message": "Textbook RAG Chatbot API", "docs": "/docs"}
