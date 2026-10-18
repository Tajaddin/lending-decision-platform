"""Request and response models. Field names and meanings are the data contract in ldp.schema."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ldp.schema import CATEGORICAL, NUMERIC

# Allowed codes per categorical input (HMDA LAR codes). Anything else is a 422, not an unknown category.
CODES: dict[str, tuple[str, ...]] = {
    "loan_type": ("1", "2", "3", "4"),
    "loan_purpose": ("1", "31", "32"),
    "occupancy_type": ("1", "2", "3"),
    "construction_method": ("1", "2"),
    "total_units": ("1", "2", "3", "4"),
    # Scoring-model codes: 1-9 since 2018, newer models (codes 11+) appear from 2023; 10 = no co-applicant.
    "applicant_credit_score_type": (*(str(i) for i in range(1, 16)), "1111"),
    "co_applicant_credit_score_type": (*(str(i) for i in range(1, 16)), "1111"),
    "submission_of_application": ("1", "2", "3", "1111"),
    "preapproval": ("1", "2"),
    "conforming_loan_limit": ("C", "NC", "U", "NA"),
}
assert set(CODES) == set(CATEGORICAL), "every categorical input needs its allowed codes"


class Monitoring(BaseModel):
    """Demographic fields collected for HMDA reporting. Used only to monitor fairness; never sent to the model."""

    model_config = ConfigDict(extra="forbid")

    derived_race: str | None = Field(default=None, max_length=64)
    derived_ethnicity: str | None = Field(default=None, max_length=64)
    derived_sex: str | None = Field(default=None, max_length=32)
    applicant_age_above_62: Literal["Yes", "No", "NA"] | None = None


class Application(BaseModel):
    model_config = ConfigDict(extra="forbid")

    application_id: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9._:-]+$")
    loan_amount: float = Field(gt=0, le=50_000_000, description="USD")
    income: float | None = Field(default=None, ge=-1_000, le=100_000, description="thousands of USD")
    property_value: float | None = Field(default=None, gt=0, le=100_000_000, description="USD")
    debt_to_income_ratio: float | None = Field(default=None, ge=0, le=100, description="percent")
    loan_term: float | None = Field(default=None, ge=1, le=600, description="months")
    loan_type: str
    loan_purpose: str
    occupancy_type: str
    construction_method: str
    total_units: str
    applicant_credit_score_type: str
    co_applicant_credit_score_type: str
    submission_of_application: str
    preapproval: str
    conforming_loan_limit: str
    monitoring: Monitoring | None = None

    @field_validator(*CATEGORICAL)
    @classmethod
    def _known_code(cls, value: str, info) -> str:
        allowed = CODES[info.field_name]
        if value not in allowed:
            raise ValueError(f"must be one of {', '.join(allowed)}")
        return value

    @field_validator(*NUMERIC)
    @classmethod
    def _finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("must be a finite number")
        return value

    def model_inputs(self) -> dict[str, float | str]:
        """What TF Serving receives: missing numbers become NaN, which the model imputes and flags."""
        out: dict[str, float | str] = {}
        for name in NUMERIC:
            value = getattr(self, name)
            out[name] = float("nan") if value is None else float(value)
        for name in CATEGORICAL:
            out[name] = getattr(self, name)
        return out


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    applications: list[Application] = Field(min_length=1, max_length=64)


class ReasonOut(BaseModel):
    reason: str
    features: list[str]
    contribution: float


class Decision(BaseModel):
    application_id: str
    decision: Literal["refer", "approve_path"]
    probability: float
    threshold: float
    reasons: list[ReasonOut]
    model_version: str


class DecisionResponse(BaseModel):
    decisions: list[Decision]
