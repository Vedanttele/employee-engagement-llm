"""Health check and readiness endpoints."""

import time
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["health"])

_START_TIME = time.time()


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    from employee_engagement import __version__
    return HealthResponse(
        status="ok",
        version=__version__,
        uptime_seconds=round(time.time() - _START_TIME, 1),
    )


@router.get("/ready")
def readiness() -> dict:
    """Kubernetes readiness probe — returns 200 when the app is ready to serve."""
    return {"ready": True}
