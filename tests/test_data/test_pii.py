"""Tests for PII anonymization (regex fallback path — no Presidio required)."""

from employee_engagement.data.pii import PIIAnonymizer


def test_email_anonymized():
    anon = PIIAnonymizer()
    result = anon.anonymize("Contact me at john.doe@company.com for feedback.")
    assert "john.doe@company.com" not in result.anonymized
    assert "[EMAIL]" in result.anonymized
    assert result.pii_detected


def test_phone_anonymized():
    anon = PIIAnonymizer()
    result = anon.anonymize("Call me at +1 800-555-1234 any time.")
    assert result.pii_detected or "[PHONE]" in result.anonymized


def test_clean_text_unchanged():
    anon = PIIAnonymizer()
    text = "The team collaboration has improved significantly this quarter."
    result = anon.anonymize(text)
    assert result.anonymized == text
    assert not result.pii_detected


def test_empty_text():
    anon = PIIAnonymizer()
    result = anon.anonymize("")
    assert result.anonymized == ""
    assert not result.pii_detected
