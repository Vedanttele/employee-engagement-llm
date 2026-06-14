"""Pydantic models for survey data — shared across all pipeline stages."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class SentimentLabel(str, Enum):
    very_positive = "very_positive"
    positive = "positive"
    neutral = "neutral"
    negative = "negative"
    very_negative = "very_negative"


class ActionSignal(str, Enum):
    urgent_action = "urgent_action"
    monitor = "monitor"
    positive_share = "positive_share"
    no_action = "no_action"


class BurnoutRiskLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class SurveyResponse(BaseModel):
    """Raw inbound survey response — what arrives at the API."""

    response_id: str = Field(..., description="Unique identifier for this response")
    text: str = Field(..., min_length=5, max_length=5000, description="Free-text survey answer")
    department: Optional[str] = Field(None, description="Employee department")
    language: Optional[str] = Field(None, description="Declared ISO-639-1 language code")
    employee_id: Optional[str] = Field(None, description="Optional pseudonymous employee ID")
    survey_theme: Optional[str] = Field(None, description="Survey theme/question category")

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Survey response text cannot be blank")
        return v.strip()


class AnonymizedSurveyResponse(BaseModel):
    """Survey response after PII detection and anonymization."""

    response_id: str
    original_text: str
    anonymized_text: str
    pii_detected: bool
    pii_types: list[str] = Field(default_factory=list)
    department: Optional[str] = None
    language: Optional[str] = None
    survey_theme: Optional[str] = None


class SentimentAnalysis(BaseModel):
    """Output from the sentiment service for a single response."""

    response_id: str
    sentiment: SentimentLabel
    sentiment_confidence: float = Field(..., ge=0.0, le=1.0)
    primary_theme: str
    secondary_themes: list[str] = Field(default_factory=list)
    key_phrases: list[str] = Field(default_factory=list)
    summary_en: str
    emotion_tags: list[str] = Field(default_factory=list)
    action_signal: ActionSignal
    language_detected: str


class BurnoutRisk(BaseModel):
    """Output from the burnout risk detection service."""

    response_id: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: BurnoutRiskLevel
    risk_factors: list[str] = Field(default_factory=list)
    protective_factors: list[str] = Field(default_factory=list)


class TopicAssignment(BaseModel):
    """Output from the topic extraction service."""

    response_id: str
    topic_id: int
    topic_label: str
    topic_keywords: list[str] = Field(default_factory=list)
    topic_confidence: float = Field(..., ge=0.0, le=1.0)


class AnalyzedResponse(BaseModel):
    """Fully enriched response — output of the complete pipeline."""

    response_id: str
    anonymized_text: str
    department: Optional[str] = None
    language_detected: str

    # Sentiment
    sentiment: SentimentLabel
    sentiment_confidence: float
    primary_theme: str
    secondary_themes: list[str]
    key_phrases: list[str]
    summary_en: str
    emotion_tags: list[str]
    action_signal: ActionSignal

    # Topics
    topic_id: int
    topic_label: str
    topic_keywords: list[str]

    # Burnout
    burnout_risk_score: float
    burnout_risk_level: BurnoutRiskLevel
    burnout_risk_factors: list[str]


class InsightReport(BaseModel):
    """LLM-generated insight report for a cohort of responses."""

    total_responses: int
    sentiment_distribution: dict[str, int]
    top_themes: list[str]
    burnout_risk_summary: dict[str, int]
    executive_summary: str
    key_findings: list[str]
    recommended_actions: list[dict]
    rag_references: list[str] = Field(default_factory=list)
