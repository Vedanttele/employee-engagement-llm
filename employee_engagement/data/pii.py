"""PII detection and anonymization using Microsoft Presidio with regex fallback."""

import re
import structlog
from dataclasses import dataclass, field

log = structlog.get_logger(__name__)

# Only anonymize entities that are genuinely sensitive in HR survey context.
# DATE_TIME and URL are intentionally excluded — survey text legitimately references
# time periods ("this quarter", "last year") and links, which must not be destroyed.
_ALLOWED_ENTITIES = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "IBAN_CODE", "CREDIT_CARD"]

# Canonical replacement labels (same whether Presidio or regex path)
_REPLACEMENT_MAP = {
    "EMAIL_ADDRESS": "[EMAIL]",
    "PHONE_NUMBER": "[PHONE]",
    "PERSON": "[PERSON]",
    "IBAN_CODE": "[IBAN]",
    "CREDIT_CARD": "[CREDIT_CARD]",
    # regex fallback keys
    "EMAIL": "[EMAIL]",
    "PHONE": "[PHONE]",
}

# Regex fallback patterns for when Presidio/spaCy is unavailable
_REGEX_PATTERNS: dict[str, re.Pattern] = {
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "PHONE": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}\b"),
    "PERSON": re.compile(r"\b(?:Mr\.|Ms\.|Mrs\.|Dr\.|Prof\.)\s+[A-Z][a-z]+(?: [A-Z][a-z]+)?\b"),
}


# ── Module-level singleton — initialized once at import, shared across all calls ──
_presidio_analyzer = None
_presidio_anonymizer = None
_presidio_available = False


def _init_presidio() -> None:
    global _presidio_analyzer, _presidio_anonymizer, _presidio_available
    if _presidio_analyzer is not None:
        return
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine

        _presidio_analyzer = AnalyzerEngine()
        _presidio_anonymizer = AnonymizerEngine()
        _presidio_available = True
        log.info("presidio_initialized", backend="presidio")
    except Exception as exc:
        log.warning("presidio_unavailable", reason=str(exc), fallback="regex")


# Eagerly initialize at import time so the first call isn't slow
_init_presidio()


@dataclass
class PIIResult:
    original: str
    anonymized: str
    detected_types: list[str] = field(default_factory=list)

    @property
    def pii_detected(self) -> bool:
        return bool(self.detected_types)


class PIIAnonymizer:
    """
    Anonymizes PII from survey text.
    Uses module-level Presidio singleton (initialized once) when available,
    falls back to regex patterns for email/phone/titled names.
    Restricted to PERSON, EMAIL_ADDRESS, PHONE_NUMBER, IBAN_CODE, CREDIT_CARD —
    DATE_TIME and URL are intentionally left intact.
    """

    def anonymize(self, text: str, language: str = "en") -> PIIResult:
        if not text or not text.strip():
            return PIIResult(original=text, anonymized=text)
        if _presidio_available:
            return self._presidio_anonymize(text, language)
        return self._regex_anonymize(text)

    def anonymize_batch(self, texts: list[str], language: str = "en") -> list[PIIResult]:
        return [self.anonymize(t, language) for t in texts]

    def _presidio_anonymize(self, text: str, language: str) -> PIIResult:
        try:
            nlp_lang = "en" if language not in ("en",) else language
            results = _presidio_analyzer.analyze(
                text=text,
                language=nlp_lang,
                entities=_ALLOWED_ENTITIES,
            )
            detected_types = list({r.entity_type for r in results})

            if not results:
                return PIIResult(original=text, anonymized=text)

            from presidio_anonymizer.entities import OperatorConfig

            operators = {
                entity_type: OperatorConfig(
                    "replace",
                    {"new_value": _REPLACEMENT_MAP.get(entity_type, f"[{entity_type}]")},
                )
                for entity_type in detected_types
            }
            anonymized = _presidio_anonymizer.anonymize(
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
