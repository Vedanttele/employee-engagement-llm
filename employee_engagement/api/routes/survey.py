"""Survey analysis endpoints — the core pipeline exposed as an API."""

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from employee_engagement.data.schemas import (
    SurveyResponse,
    AnalyzedResponse,
    SentimentLabel,
    ActionSignal,
    BurnoutRiskLevel,
)
from employee_engagement.data.validator import validate_survey_batch
from employee_engagement.api.deps import (
    get_pii_anonymizer,
    get_sentiment_service,
    get_burnout_detector,
    get_topic_service,
)
from employee_engagement.data.pii import PIIAnonymizer
from employee_engagement.services.sentiment import SentimentService
from employee_engagement.services.burnout import BurnoutDetector
from employee_engagement.services.topics import TopicService

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/survey", tags=["survey"])


class AnalyzeRequest(BaseModel):
    responses: list[dict]
    run_topic_modeling: bool = False


class ValidationResponse(BaseModel):
    valid_count: int
    invalid_count: int
    pass_rate: float
    errors: list[str]


class AnalyzeResponse(BaseModel):
    analyzed: list[AnalyzedResponse]
    validation_errors: list[str]
    total_processed: int


@router.post("/validate", response_model=ValidationResponse)
def validate_responses(request: AnalyzeRequest) -> ValidationResponse:
    """Validate raw survey data without running full analysis."""
    result = validate_survey_batch(request.responses)
    return ValidationResponse(
        valid_count=len(result.valid),
        invalid_count=len(result.invalid),
        pass_rate=round(result.pass_rate, 4),
        errors=result.errors[:50],  # cap error list in response
    )


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_responses(
    request: AnalyzeRequest,
    pii: PIIAnonymizer = Depends(get_pii_anonymizer),
    sentiment_svc: SentimentService = Depends(get_sentiment_service),
    burnout_svc: BurnoutDetector = Depends(get_burnout_detector),
    topic_svc: TopicService = Depends(get_topic_service),
) -> AnalyzeResponse:
    """
    Full pipeline: validate → anonymize → sentiment → burnout → topics.
    Returns enriched AnalyzedResponse for each valid input.
    """
    # 1. Validate
    validation = validate_survey_batch(request.responses)
    if not validation.valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": "No valid survey responses", "errors": validation.errors[:20]},
        )

    log.info("survey_analyze_start", n_valid=len(validation.valid))

    # 2. PII anonymization
    anonymized_map: dict[str, str] = {}
    for survey in validation.valid:
        result = pii.anonymize(survey.text, language=survey.language or "en")
        anonymized_map[survey.response_id] = result.anonymized

    # 3. Sentiment analysis
    text_pairs = [
        (s.response_id, anonymized_map[s.response_id]) for s in validation.valid
    ]
    sentiments = sentiment_svc.analyze_batch(text_pairs)
    sentiment_by_id = {s.response_id: s for s in sentiments}

    # 4. Burnout scoring
    burnout_by_id = {
        b.response_id: b
        for b in burnout_svc.score_batch(sentiments)
    }

    # 5. Topic modeling (optional — expensive for small batches)
    topic_by_id: dict[str, object] = {}
    if request.run_topic_modeling and len(validation.valid) >= 10:
        try:
            texts = [anonymized_map[s.response_id] for s in validation.valid]
            assignments = topic_svc.fit_transform(texts)
            for survey, assignment in zip(validation.valid, assignments):
                assignment.response_id = survey.response_id
                topic_by_id[survey.response_id] = assignment
        except Exception as exc:
            log.warning("topic_modeling_skipped", reason=str(exc))

    # 6. Merge into AnalyzedResponse
    analyzed: list[AnalyzedResponse] = []
    for survey in validation.valid:
        rid = survey.response_id
        sa = sentiment_by_id.get(rid)
        br = burnout_by_id.get(rid)
        ta = topic_by_id.get(rid)

        if not sa or not br:
            continue

        analyzed.append(AnalyzedResponse(
            response_id=rid,
            anonymized_text=anonymized_map[rid],
            department=survey.department,
            language_detected=sa.language_detected,
            sentiment=sa.sentiment,
            sentiment_confidence=sa.sentiment_confidence,
            primary_theme=sa.primary_theme,
            secondary_themes=sa.secondary_themes,
            key_phrases=sa.key_phrases,
            summary_en=sa.summary_en,
            emotion_tags=sa.emotion_tags,
            action_signal=sa.action_signal,
            topic_id=ta.topic_id if ta else -1,
            topic_label=ta.topic_label if ta else "N/A",
            topic_keywords=ta.topic_keywords if ta else [],
            burnout_risk_score=br.risk_score,
            burnout_risk_level=br.risk_level,
            burnout_risk_factors=br.risk_factors,
        ))

    log.info("survey_analyze_done", n_analyzed=len(analyzed))
    return AnalyzeResponse(
        analyzed=analyzed,
        validation_errors=validation.errors,
        total_processed=len(analyzed),
    )
