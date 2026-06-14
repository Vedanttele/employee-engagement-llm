"""Data validation — validate raw survey input before processing."""

from pydantic import ValidationError
from .schemas import SurveyResponse


class ValidationResult:
    __slots__ = ("valid", "invalid", "errors")

    def __init__(self) -> None:
        self.valid: list[SurveyResponse] = []
        self.invalid: list[dict] = []
        self.errors: list[str] = []

    @property
    def pass_rate(self) -> float:
        total = len(self.valid) + len(self.invalid)
        return len(self.valid) / total if total else 0.0


def validate_survey_batch(raw_records: list[dict]) -> ValidationResult:
    """
    Validate a batch of raw survey records against SurveyResponse schema.
    Returns ValidationResult with valid/invalid splits and error details.
    """
    result = ValidationResult()

    for i, record in enumerate(raw_records):
        try:
            result.valid.append(SurveyResponse.model_validate(record))
        except ValidationError as exc:
            result.invalid.append(record)
            for err in exc.errors():
                result.errors.append(
                    f"Record {i} [{record.get('response_id', '?')}]: "
                    f"{err['loc']} — {err['msg']}"
                )

    return result
