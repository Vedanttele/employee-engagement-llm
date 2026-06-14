"""PII detection and anonymization using Microsoft Presidio with regex fallback."""

import re
import structlog
from dataclasses import dataclass, field

log = structlog.get_logger(__name__)


@dataclass
class PIIResult:
    original: str
    anonymized: str
    detected_types: list[str] = field(default_factory=list)

    @property
    def pii_detected(self) -> bool:
        return bool(self.detected_types)


# Regex fallback patterns for when Presidio/spaCy is unavailable
_REGEX_PATTERNS: dict[str, re.Pattern] = {
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "PHONE": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}\b"),
    "PERSON": re.compile(r"\b(?:Mr\.|Ms\.|Mrs\.|Dr\.|Prof\.)\s+[A-Z][a-z]+(?: [A-Z][a-z]+)?\b"),
}

_REPLACEMENT_MAP = {
    "EMAIL": "[EMAIL]",
    "PHONE": "[PHONE]",
    "PERSON": "[PERSON]",
    "LOCATION": "[LOCATION]",
    "ORGANIZATION": "[ORGANIZATION]",
    "DATE_TIME": "[DATE]",
    "NRP": "[NRP]",  # Nationality/Religion/Political group
}


class PIIAnonymizer:
    """
    Anonymizes PII from survey text.
    Uses Presidio (with spaCy en_core_web_lg) when available,
    falls back to regex patterns for email/phone/titled names.
    """

    def __init__(self) -> None:
        self._presidio_available = False
        self._analyzer = None
        self._anonymizer = None
        self._try_init_presidio()

    def _try_init_presidio(self) -> None:
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine

            self._analyzer = AnalyzerEngine()
            self._anonymizer = AnonymizerEngine()
            self._presidio_available = True
            log.info("presidio_initialized", backend="presidio")
        except Exception as exc:
            log.warning("presidio_unavailable", reason=str(exc), fallback="regex")

    def anonymize(self, text: str, language: str = "en") -> PIIResult:
        """Anonymize a single text. Language code used for Presidio NLP engine."""
        if not text or not text.strip():
            return PIIResult(original=text, anonymized=text)

        if self._presidio_available:
            return self._presidio_anonymize(text, language)
        return self._regex_anonymize(text)

    def anonymize_batch(self, texts: list[str], language: str = "en") -> list[PIIResult]:
        return [self.anonymize(t, language) for t in texts]

    def _presidio_anonymize(self, text: str, language: str) -> PIIResult:
        try:
            # Presidio only supports English NLP natively; fall back to regex for others
            nlp_lang = "en" if language not in ("en",) else language
            results = self._analyzer.analyze(text=text, language=nlp_lang)
            detected_types = list({r.entity_type for r in results})

            if not results:
                return PIIResult(original=text, anonymized=text)

            from presidio_anonymizer.entities import OperatorConfig

            operators = {
                t: OperatorConfig("replace", {"new_value": _REPLACEMENT_MAP.get(t, f"[{t}]")})
                for t in detected_types
            }
            anonymized = self._anonymizer.anonymize(
                text=text, analyzer_results=results, operators=operators
            )
            return PIIResult(
                original=text,
                anonymized=anonymized.text,
                detected_types=detected_types,
            )
        except Exception as exc:
            log.error("presidio_anonymize_failed", error=str(exc))
            return self._regex_anonymize(text)

    def _regex_anonymize(self, text: str) -> PIIResult:
        anonymized = text
        detected: list[str] = []
        for entity_type, pattern in _REGEX_PATTERNS.items():
            replaced = pattern.sub(_REPLACEMENT_MAP[entity_type], anonymized)
            if replaced != anonymized:
                detected.append(entity_type)
                anonymized = replaced
        return PIIResult(original=text, anonymized=anonymized, detected_types=detected)
