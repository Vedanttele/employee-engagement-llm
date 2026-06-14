"""Tests for data validation."""

from employee_engagement.data.validator import validate_survey_batch


def test_valid_records_pass(sample_responses):
    result = validate_survey_batch(sample_responses)
    assert len(result.valid) == 3
    assert len(result.invalid) == 0
    assert result.pass_rate == 1.0


def test_missing_text_is_invalid():
    result = validate_survey_batch([{"response_id": "x1", "text": ""}])
    assert len(result.invalid) == 1
    assert len(result.errors) > 0


def test_missing_response_id_is_invalid():
    result = validate_survey_batch([{"text": "Valid text here"}])
    assert len(result.invalid) == 1


def test_mixed_valid_invalid():
    records = [
        {"response_id": "ok", "text": "This is a valid response"},
        {"response_id": "bad", "text": ""},
    ]
    result = validate_survey_batch(records)
    assert result.pass_rate == 0.5
