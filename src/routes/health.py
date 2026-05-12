"""GET /api/v1/health — System liveness check."""

from fastapi import APIRouter, Request
from src.routes.schema.schemas import HealthResponse
from src.helpers.config import get_settings

router   = APIRouter()
settings = get_settings()


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health(request: Request):
    return HealthResponse(
        status="ok",
        llm_provider=settings.LLM_PROVIDER,
        vector_db=settings.VECTOR_DB_PROVIDER,
    )
