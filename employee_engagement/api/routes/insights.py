"""Insights generation endpoints."""

import structlog
from fastapi import APIRouter, Depends, HTTPException, status

from employee_engagement.data.schemas import AnalyzedResponse, InsightReport
from employee_engagement.insights.generator import InsightsGenerator
from employee_engagement.api.deps import get_insights_generator

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/insights", tags=["insights"])


class InsightsRequest(BaseModel):
    analyzed_responses: list[AnalyzedResponse]


from pydantic import BaseModel


class InsightsRequest(BaseModel):
    analyzed_responses: list[AnalyzedResponse]


@router.post("/generate", response_model=InsightReport)
def generate_insights(
    request: InsightsRequest,
    generator: InsightsGenerator = Depends(get_insights_generator),
) -> InsightReport:
    """Generate an executive insight report from a set of analyzed survey responses."""
    if len(request.analyzed_responses) < 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least 1 analyzed response is required",
        )

    log.info("insights_generate_start", n=len(request.analyzed_responses))
    try:
        report = generator.generate(request.analyzed_responses)
    except Exception as exc:
        log.error("insights_generate_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Insights generation failed: {exc}",
        )

    log.info("insights_generate_done")
    return report
