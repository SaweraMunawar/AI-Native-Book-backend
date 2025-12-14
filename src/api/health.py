"""Health check endpoint."""

from datetime import datetime
import httpx
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from src.config import get_settings
from src.models.schemas import (
    DependencyHealth,
    DependencyStatus,
    HealthResponse,
    HealthStatus,
)

router = APIRouter(tags=["System"])
settings = get_settings()


async def check_qdrant() -> DependencyStatus:
    """Check Qdrant connectivity."""
    if not settings.qdrant_url or not settings.qdrant_api_key:
        return DependencyStatus.DOWN
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"{settings.qdrant_url}/collections",
                headers={"api-key": settings.qdrant_api_key}
            )
            return DependencyStatus.UP if response.status_code == 200 else DependencyStatus.DOWN
    except Exception:
        return DependencyStatus.DOWN


async def check_groq() -> DependencyStatus:
    """Check Groq API connectivity."""
    if not settings.groq_api_key:
        return DependencyStatus.DOWN
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {settings.groq_api_key}"}
            )
            return DependencyStatus.UP if response.status_code == 200 else DependencyStatus.DOWN
    except Exception:
        return DependencyStatus.DOWN


async def check_neon() -> DependencyStatus:
    """Check Neon PostgreSQL connectivity."""
    if not settings.database_url:
        return DependencyStatus.DOWN
    try:
        import asyncpg
        conn = await asyncpg.connect(settings.database_url, timeout=5)
        await conn.execute("SELECT 1")
        await conn.close()
        return DependencyStatus.UP
    except Exception:
        return DependencyStatus.DOWN


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={
        200: {"description": "API is healthy"},
        503: {"description": "API is unhealthy"},
    },
    summary="Health check endpoint",
    description="Returns API health status and dependency connectivity",
)
async def health_check() -> HealthResponse | JSONResponse:
    """Check health of all dependencies and return overall status."""
    # Check all dependencies
    qdrant_status = await check_qdrant()
    groq_status = await check_groq()
    neon_status = await check_neon()

    dependencies = DependencyHealth(
        qdrant=qdrant_status,
        groq=groq_status,
        neon=neon_status,
    )

    # Determine overall status
    all_up = all([
        qdrant_status == DependencyStatus.UP,
        groq_status == DependencyStatus.UP,
        neon_status == DependencyStatus.UP,
    ])

    # Core dependencies (Qdrant and Groq are required for chat)
    core_up = (
        qdrant_status == DependencyStatus.UP and
        groq_status == DependencyStatus.UP
    )

    if all_up:
        overall_status = HealthStatus.HEALTHY
    elif core_up:
        overall_status = HealthStatus.DEGRADED
    else:
        overall_status = HealthStatus.UNHEALTHY

    response = HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        dependencies=dependencies,
    )

    if overall_status == HealthStatus.UNHEALTHY:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=response.model_dump(mode="json"),
        )

    return response
