"""
Burnout risk detection service.

Scoring logic:
  - Sentiment score (40%): very_negative → 1.0, very_positive → 0.0
  - Theme risk (30%): high-risk themes (wellbeing, workload) elevate score
  - Emotion tags (20%): burnout-correlated emotions increase score
  - Action signal (10%): urgent_action adds weight
"""

import structlog
from employee_engagement.config import BURNOUT_RISK_THRESHOLDS
from employee_engagement.data.schemas import (
    BurnoutRisk,
    BurnoutRiskLevel,
    SentimentLabel,
    ActionSignal,
    SentimentAnalysis,
)

log = structlog.get_logger(__name__)

_SENTIMENT_SCORES: dict[str, float] = {
    "very_positive": 0.0,
    "positive": 0.15,
    "neutral": 0.35,
    "negative": 0.65,
    "very_negative": 1.0,
}

_HIGH_RISK_THEMES = {"wellbeing_safety", "workload_balance", "management_leadership"}
_MEDIUM_RISK_THEMES = {"recognition_rewards", "career_growth", "communication"}

_BURNOUT_EMOTIONS = {
    "exhausted", "overwhelmed", "frustrated", "burned out", "stressed",
    "undervalued", "ignored", "hopeless", "anxious", "disengaged",
}

_PROTECTIVE_EMOTIONS = {
    "supported", "motivated", "appreciated", "engaged", "hopeful",
    "satisfied", "valued", "energized",
}


def _risk_level(score: float) -> BurnoutRiskLevel:
    if score <= BURNOUT_RISK_THRESHOLDS["low"]:
        return BurnoutRiskLevel.low
    if score <= BURNOUT_RISK_THRESHOLDS["medium"]:
        return BurnoutRiskLevel.medium
    if score <= BURNOUT_RISK_THRESHOLDS["high"]:
        return BurnoutRiskLevel.high
    return BurnoutRiskLevel.critical


class BurnoutDetector:
    """Deterministic burnout risk scorer — no external calls required."""

    def score(self, analysis: SentimentAnalysis) -> BurnoutRisk:
        """Score burnout risk from a SentimentAnalysis result."""
        # 1. Sentiment component (40%)
        sentiment_score = _SENTIMENT_SCORES.get(analysis.sentiment.value, 0.35)

        # 2. Theme component (30%)
        themes = {analysis.primary_theme} | set(analysis.secondary_themes)
        if themes & _HIGH_RISK_THEMES:
            theme_score = 0.9
        elif themes & _MEDIUM_RISK_THEMES:
            theme_score = 0.5
        else:
            theme_score = 0.2

        # 3. Emotion component (20%)
        emotion_lower = {e.lower() for e in analysis.emotion_tags}
        n_burnout = len(emotion_lower & _BURNOUT_EMOTIONS)
        n_protective = len(emotion_lower & _PROTECTIVE_EMOTIONS)
        emotion_score = min(1.0, n_burnout * 0.35) - min(0.2, n_protective * 0.1)
        emotion_score = max(0.0, emotion_score)

        # 4. Action signal component (10%)
        action_score = 0.9 if analysis.action_signal == ActionSignal.urgent_action else 0.0

        final_score = (
            0.40 * sentiment_score
            + 0.30 * theme_score
            + 0.20 * emotion_score
            + 0.10 * action_score
        )
        final_score = round(min(1.0, max(0.0, final_score)), 4)

        risk_factors = self._risk_factors(analysis, sentiment_score, theme_score, emotion_score)
        protective = self._protective_factors(analysis, emotion_lower)

        return BurnoutRisk(
            response_id=analysis.response_id,
            risk_score=final_score,
            risk_level=_risk_level(final_score),
            risk_factors=risk_factors,
            protective_factors=protective,
        )

    def score_batch(self, analyses: list[SentimentAnalysis]) -> list[BurnoutRisk]:
        return [self.score(a) for a in analyses]

    @staticmethod
    def _risk_factors(
        analysis: SentimentAnalysis,
        sentiment_score: float,
        theme_score: float,
        emotion_score: float,
    ) -> list[str]:
        factors: list[str] = []
        if sentiment_score >= 0.65:
            factors.append(f"Negative sentiment: {analysis.sentiment.value}")
        if analysis.primary_theme in _HIGH_RISK_THEMES:
            factors.append(f"High-risk theme: {analysis.primary_theme}")
        if emotion_score >= 0.35:
            factors.append("Burnout-correlated emotions detected")
        if analysis.action_signal == ActionSignal.urgent_action:
            factors.append("Urgent action signal")
        return factors

    @staticmethod
    def _protective_factors(analysis: SentimentAnalysis, emotion_lower: set) -> list[str]:
        factors: list[str] = []
        if analysis.sentiment in (SentimentLabel.positive, SentimentLabel.very_positive):
            factors.append("Positive sentiment baseline")
        if emotion_lower & _PROTECTIVE_EMOTIONS:
            factors.append(f"Protective emotions: {', '.join(emotion_lower & _PROTECTIVE_EMOTIONS)}")
        return factors
