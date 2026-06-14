"""Tests for the burnout detection service (no external API calls)."""

import pytest
from employee_engagement.services.burnout import BurnoutDetector
from employee_engagement.data.schemas import (
    SentimentAnalysis,
    SentimentLabel,
    ActionSignal,
    BurnoutRiskLevel,
)


def _make_analysis(
    response_id: str = "r1",
    sentiment: SentimentLabel = SentimentLabel.neutral,
    primary_theme: str = "communication",
    emotion_tags: list[str] | None = None,
    action_signal: ActionSignal = ActionSignal.no_action,
) -> SentimentAnalysis:
    return SentimentAnalysis(
        response_id=response_id,
        sentiment=sentiment,
        sentiment_confidence=0.9,
        primary_theme=primary_theme,
        secondary_themes=[],
        key_phrases=[],
        summary_en="Test summary",
        emotion_tags=emotion_tags or [],
        action_signal=action_signal,
        language_detected="en",
    )


def test_very_positive_is_low_risk():
    detector = BurnoutDetector()
    analysis = _make_analysis(
        sentiment=SentimentLabel.very_positive,
        primary_theme="team_collaboration",
        emotion_tags=["happy", "motivated"],
        action_signal=ActionSignal.positive_share,
    )
    result = detector.score(analysis)
    assert result.risk_level == BurnoutRiskLevel.low
    assert result.risk_score < 0.3


def test_very_negative_wellbeing_is_critical():
    detector = BurnoutDetector()
    analysis = _make_analysis(
        sentiment=SentimentLabel.very_negative,
        primary_theme="wellbeing_safety",
        emotion_tags=["exhausted", "overwhelmed", "burned out"],
        action_signal=ActionSignal.urgent_action,
    )
    result = detector.score(analysis)
    assert result.risk_level in (BurnoutRiskLevel.high, BurnoutRiskLevel.critical)
    assert result.risk_score >= 0.7


def test_score_in_valid_range():
    detector = BurnoutDetector()
    for sentiment in SentimentLabel:
        analysis = _make_analysis(sentiment=sentiment)
        result = detector.score(analysis)
        assert 0.0 <= result.risk_score <= 1.0


def test_urgent_action_elevates_score():
    detector = BurnoutDetector()
    without_urgent = _make_analysis(
        sentiment=SentimentLabel.negative,
        action_signal=ActionSignal.monitor,
    )
    with_urgent = _make_analysis(
        sentiment=SentimentLabel.negative,
        action_signal=ActionSignal.urgent_action,
    )
    assert detector.score(with_urgent).risk_score > detector.score(without_urgent).risk_score
