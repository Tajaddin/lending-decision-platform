"""Declarative data-quality checks on silver, evaluated in one aggregation pass per year.

A critical failure stops the pipeline (the model is never trained on that data); a warning is reported and
the run continues. Thresholds are written next to each check so the report explains itself.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from ldp.log import get_logger
from ldp.schema import DECISION_ACTIONS, DENIED_ACTION, SENTINELS

log = get_logger(__name__)

Severity = Literal["critical", "warning"]

VALID_STATES = frozenset(
    [
        "AL",
        "AK",
        "AZ",
        "AR",
        "CA",
        "CO",
        "CT",
        "DE",
        "DC",
        "FL",
        "GA",
        "HI",
        "ID",
        "IL",
        "IN",
        "IA",
        "KS",
        "KY",
        "LA",
        "ME",
        "MD",
        "MA",
        "MI",
        "MN",
        "MS",
        "MO",
        "MT",
        "NE",
        "NV",
        "NH",
        "NJ",
        "NM",
        "NY",
        "NC",
        "ND",
        "OH",
        "OK",
        "OR",
        "PA",
        "RI",
        "SC",
        "SD",
        "TN",
        "TX",
        "UT",
        "VT",
        "VA",
        "WA",
        "WV",
        "WI",
        "WY",
        "PR",
        "GU",
        "VI",
        "AS",
        "MP",
    ]
)


def _rate(condition: Column, among: Column | None = None) -> Column:
    hit = F.when(condition, 1).otherwise(0)
    if among is None:
        return F.avg(hit)
    return F.sum(F.when(among, hit)) / F.greatest(F.sum(F.when(among, 1).otherwise(0)), F.lit(1))


@dataclass(frozen=True)
class Check:
    name: str
    severity: Severity
    description: str
    threshold: str
    metric: Column = field(repr=False, compare=False)
    passes: Callable[[float], bool] = field(repr=False, compare=False)


@dataclass
class CheckResult:
    name: str
    severity: Severity
    description: str
    threshold: str
    value: float
    passed: bool


@dataclass
class QualityReport:
    year: int
    rows: int
    generated_at: str
    results: list[CheckResult]

    @property
    def critical_failures(self) -> list[CheckResult]:
        return [r for r in self.results if r.severity == "critical" and not r.passed]

    @property
    def warnings(self) -> list[CheckResult]:
        return [r for r in self.results if r.severity == "warning" and not r.passed]

    def to_dict(self) -> dict[str, object]:
        return {
            "year": self.year,
            "rows": self.rows,
            "generated_at": self.generated_at,
            "critical_failures": len(self.critical_failures),
            "warnings": len(self.warnings),
            "results": [asdict(r) for r in self.results],
        }


def silver_checks(bronze_rows: int | None = None) -> list[Check]:
    decision = F.col("action_taken").isin(*DECISION_ACTIONS)
    dti_raw = F.trim(F.col("debt_to_income_ratio_raw"))
    checks = [
        Check(
            "action_taken_valid",
            "critical",
            "action_taken is one of the 8 HMDA codes",
            "= 1.0",
            _rate(F.col("action_taken").isin(*[str(i) for i in range(1, 9)])),
            lambda v: v == 1.0,
        ),
        Check(
            "loan_amount_present",
            "critical",
            "loan_amount is reported",
            ">= 0.999",
            _rate(F.col("loan_amount").isNotNull()),
            lambda v: v >= 0.999,
        ),
        Check(
            "loan_amount_positive",
            "critical",
            "reported loan_amount is above zero",
            "= 1.0",
            _rate(F.col("loan_amount") > 0, F.col("loan_amount").isNotNull()),
            lambda v: v == 1.0,
        ),
        Check(
            "lei_format",
            "critical",
            "lei is a 20-character Legal Entity Identifier",
            "= 1.0",
            _rate(F.length("lei") == 20),
            lambda v: v == 1.0,
        ),
        Check(
            "dti_parser_coverage",
            "critical",
            "every non-sentinel debt_to_income_ratio string parses",
            "= 1.0",
            _rate(F.col("debt_to_income_ratio").isNotNull(), dti_raw.isNotNull() & ~dti_raw.isin(*sorted(SENTINELS))),
            lambda v: v == 1.0,
        ),
        Check(
            "income_present_in_decisions",
            "warning",
            "income reported on decided applications",
            ">= 0.85",
            _rate(F.col("income").isNotNull(), decision),
            lambda v: v >= 0.85,
        ),
        Check(
            "property_value_present_in_decisions",
            "warning",
            "property_value reported on decided applications",
            ">= 0.85",
            _rate(F.col("property_value").isNotNull(), decision),
            lambda v: v >= 0.85,
        ),
        Check(
            "income_plausible",
            "warning",
            "reported income between -$1M and $100M (thousands)",
            ">= 0.999",
            _rate(F.col("income").between(-1000, 100000), F.col("income").isNotNull()),
            lambda v: v >= 0.999,
        ),
        Check(
            "cltv_plausible",
            "warning",
            "reported CLTV in (0, 300] percent",
            ">= 0.99",
            _rate(
                F.col("combined_loan_to_value_ratio").between(0.001, 300.0),
                F.col("combined_loan_to_value_ratio").isNotNull(),
            ),
            lambda v: v >= 0.99,
        ),
        Check(
            "state_code_known",
            "warning",
            "state_code is a US state or territory",
            ">= 0.99",
            _rate(F.col("state_code").isin(*sorted(VALID_STATES))),
            lambda v: v >= 0.99,
        ),
        Check(
            "denial_rate_in_decisions",
            "warning",
            "share of decided applications denied",
            "between 0.05 and 0.50",
            _rate(F.col("action_taken") == DENIED_ACTION, decision),
            lambda v: 0.05 <= v <= 0.50,
        ),
        Check(
            "derived_race_present",
            "warning",
            "derived_race populated (Race Not Available counts as present)",
            "= 1.0",
            _rate(F.col("derived_race") != "missing"),
            lambda v: v == 1.0,
        ),
    ]
    if bronze_rows is not None:
        checks.append(
            Check(
                "rows_reconcile_with_bronze",
                "critical",
                f"silver row count equals bronze ({bronze_rows:,})",
                f"= {bronze_rows}",
                F.count(F.lit(1)).cast("double"),
                lambda v: int(v) == bronze_rows,
            )
        )
    return checks


def run_checks(df: DataFrame, year: int, checks: list[Check]) -> QualityReport:
    row = df.agg(F.count(F.lit(1)).alias("__rows"), *[c.metric.alias(c.name) for c in checks]).collect()[0]
    rows = int(row["__rows"])
    results = []
    for check in checks:
        raw = row[check.name]
        value = float(raw) if raw is not None else float("nan")
        passed = rows > 0 and raw is not None and check.passes(value)
        results.append(CheckResult(check.name, check.severity, check.description, check.threshold, value, passed))
    report = QualityReport(
        year=year, rows=rows, generated_at=datetime.now(UTC).isoformat(timespec="seconds"), results=results
    )
    log.info(
        "quality_checked", year=year, rows=rows, critical=len(report.critical_failures), warnings=len(report.warnings)
    )
    return report


class DataQualityError(RuntimeError):
    pass


def write_report(reports: list[QualityReport], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "years": [r.to_dict() for r in reports],
        "critical_failures": sum(len(r.critical_failures) for r in reports),
        "warnings": sum(len(r.warnings) for r in reports),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def enforce(reports: list[QualityReport]) -> None:
    failed = [(r.year, c.name, c.value) for r in reports for c in r.critical_failures]
    if failed:
        raise DataQualityError(f"critical data-quality failures: {failed}")
