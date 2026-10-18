from __future__ import annotations

import math

import pytest

from ldp import schema
from ldp.schema import parse_code, parse_dti, parse_number


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("<20%", 10.0),
        ("20%-<30%", 25.0),
        ("30%-<36%", 33.0),
        ("36", 36.0),
        ("49", 49.0),
        ("50%-60%", 55.0),
        (">60%", 65.0),
        (" 42 ", 42.0),
        ("NA", None),
        ("Exempt", None),
        ("", None),
        (None, None),
        ("140", None),  # out of range is treated as missing, not clipped
        ("abc", None),
    ],
)
def test_parse_dti(raw: str | None, expected: float | None) -> None:
    assert parse_dti(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("255000", 255000.0),
        ("-12", -12.0),
        ("1111", 1111.0),
        ("97.5", 97.5),
        ("NA", None),
        ("Exempt", None),
        ("", None),
        (None, None),
        ("inf", None),
        ("nan", None),
        ("12k", None),
    ],
)
def test_parse_number(raw: str | None, expected: float | None) -> None:
    got = parse_number(raw)
    assert got == expected or (expected is None and got is None)


def test_parse_number_rejects_non_finite() -> None:
    assert parse_number("-Infinity") is None
    assert all(parse_number(v) is None or math.isfinite(parse_number(v) or 0.0) for v in ("1e400", "5"))


def test_parse_code_keeps_codes_and_marks_missing() -> None:
    assert parse_code("1111") == "1111"
    assert parse_code(" 31 ") == "31"
    assert parse_code("") == "missing"
    assert parse_code(None) == "missing"


def test_normalize_column_handles_data_browser_headers() -> None:
    assert schema.normalize_column("co-applicant_credit_score_type") == "co_applicant_credit_score_type"
    assert schema.normalize_column("AUS-1") == "aus_1"
    assert schema.normalize_column("derived_msa-md") == "derived_msa_md"


def test_features_never_include_protected_geographic_or_post_decision_fields() -> None:
    forbidden = (
        set(schema.PROTECTED)
        | set(schema.TRACT_DEMOGRAPHICS)
        | set(schema.GEOGRAPHY)
        | set(schema.POST_DECISION)
        | set(schema.EXCLUDED_FOR_LEAKAGE)
        | {"lei"}
    )
    assert not forbidden & set(schema.FEATURE_NAMES)


def test_feature_names_are_unique_and_split_by_kind() -> None:
    assert len(set(schema.FEATURE_NAMES)) == len(schema.FEATURE_NAMES)
    assert all(f.kind == "numeric" for f in schema.NUMERIC_FEATURES)
    assert all(f.kind == "categorical" for f in schema.CATEGORICAL_FEATURES)
    assert [g.name for g in schema.ATTRIBUTION_GROUPS][: len(schema.NUMERIC)] == list(schema.NUMERIC)


def test_every_silver_column_comes_from_the_lar_header(fixture_csv: dict[int, str]) -> None:
    with open(fixture_csv[2023], encoding="utf-8") as fh:
        header = {schema.normalize_column(c) for c in fh.readline().strip().split(",")}
    silver = set(schema.SILVER_NUMERIC) | set(schema.SILVER_CODES) | set(schema.SILVER_TEXT)
    assert silver <= header, silver - header
    assert set(schema.NOT_APPLICABLE) <= header
