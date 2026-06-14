"""Claude-powered batch sentiment and theme analysis service."""

import json
import structlog
from tenacity import retry, stop_after_attempt, wait_fixed

import anthropic

from employee_engagement.config import get_settings, ENGAGEMENT_THEMES
from employee_engagement.data.schemas import SentimentAnalysis, SentimentLabel, ActionSignal

log = structlog.get_logger(__name__)

_SYSTEM_PROMPT = """You are an expert HR analytics AI analyzing employee engagement survey responses.

For EACH response return a JSON object with:
{
  "id": <original index>,
  "sentiment": one of ["very_positive","positive","neutral","negative","very_negative"],
  "sentiment_confidence": float 0.0-1.0,
  "primary_theme": one of the theme keys provided,
  "secondary_themes": list of up to 2 theme keys,
  "key_phrases": list of 2-4 key phrases in original language,
  "summary_en": 1-sentence English summary,
  "emotion_tags": list of 1-3 emotions,
  "action_signal": one of ["urgent_action","monitor","positive_share","no_action"],
  "language_detected": ISO-639-1 code
}

Action signal rules:
- urgent_action: very_negative + wellbeing/safety themes, or intent-to-leave signals
- monitor: negative or mixed signals
- positive_share: very_positive responses worth sharing
- no_action: neutral/routine

Return ONLY a valid JSON array. No markdown."""


class SentimentService:
    """Batch sentiment analysis via Claude. Thread-safe; client is per-instance."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.claude_model
        self._max_tokens = settings.max_tokens
        self._batch_size = settings.analysis_batch_size

    def analyze_batch(self, texts: list[tuple[str, str]]) -> list[SentimentAnalysis]:
        """
        Analyze a batch of (response_id, text) pairs.
        Returns SentimentAnalysis for each input in the same order.
        """
        results: list[SentimentAnalysis] = []
        theme_list = "\n".join(f"  {k}: {v}" for k, v in ENGAGEMENT_THEMES.items())

        for chunk_start in range(0, len(texts), self._batch_size):
            chunk = texts[chunk_start : chunk_start + self._batch_size]
            batch_input = [
                {"id": i, "text": text, "response_id": rid}
                for i, (rid, text) in enumerate(chunk)
            ]
            try:
                raw = self._call_claude(json.dumps(batch_input, ensure_ascii=False), theme_list)
                parsed = json.loads(raw)
                for item, (rid, _) in zip(parsed, chunk):
                    results.append(self._to_schema(rid, item))
            except Exception as exc:
                log.error("sentiment_batch_failed", error=str(exc), batch_size=len(chunk))
                for rid, _ in chunk:
                    results.append(self._fallback(rid))

        return results

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _call_claude(self, batch_json: str, theme_list: str) -> str:
        message = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens * 3,
            system=_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Available themes:\n{theme_list}\n\nResponses:\n{batch_json}",
            }],
        )
        return message.content[0].text

    @staticmethod
    def _to_schema(response_id: str, raw: dict) -> SentimentAnalysis:
        return SentimentAnalysis(
            response_id=response_id,
            sentiment=SentimentLabel(raw.get("sentiment", "neutral")),
            sentiment_confidence=float(raw.get("sentiment_confidence", 0.5)),
            primary_theme=raw.get("primary_theme", "communication"),
            secondary_themes=raw.get("secondary_themes", []),
            key_phrases=raw.get("key_phrases", []),
            summary_en=raw.get("summary_en", ""),
            emotion_tags=raw.get("emotion_tags", []),
            action_signal=ActionSignal(raw.get("action_signal", "no_action")),
            language_detected=raw.get("language_detected", "en"),
        )

    @staticmethod
    def _fallback(response_id: str) -> SentimentAnalysis:
        return SentimentAnalysis(
            response_id=response_id,
            sentiment=SentimentLabel.neutral,
            sentiment_confidence=0.0,
            primary_theme="communication",
            secondary_themes=[],
            key_phrases=[],
            summary_en="Analysis failed.",
            emotion_tags=[],
            action_signal=ActionSignal.no_action,
            language_detected="unknown",
        )
