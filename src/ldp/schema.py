"""The data contract, in one place.

Every layer reads its column lists from here: the Spark jobs (silver typing, the model population, the
gold table), the TensorFlow model (its inputs), the API (its request schema) and the monitor (what it
compares). Changing a feature means changing this file, and the tests check the layers still agree.

Source: FFIEC/CFPB public HMDA LAR data fields,
https://ffiec.cfpb.gov/documentation/publications/loan-level-datasets/lar-data-fields
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# HMDA writes these instead of a number. Coded fields use "1111" for Exempt instead; that stays a category
# (parse_code), because in a numeric field 1111 can be a real value (income of $1.111M).
SENTINELS = frozenset({"", "NA", "Exempt"})


def normalize_column(name: str) -> str:
    """Raw LAR headers use hyphens ("aus-1", "co-applicant_age"); silver and gold use snake_case."""
    return name.strip().lower().replace("-", "_")


# --- Decoders. silver.py has a Spark SQL version of each; tests/test_schema_parity.py keeps them identical.

DTI_BINS: dict[str, float] = {
    # Ranged bins are mapped to their midpoint; 36..49 are reported as exact whole percents.
    "<20%": 10.0,
    "20%-<30%": 25.0,
    "30%-<36%": 33.0,
    "50%-60%": 55.0,
    ">60%": 65.0,
}


def parse_dti(raw: str | None) -> float | None:
    """Debt-to-income ratio as a percent, or None when HMDA reports NA/Exempt."""
    if raw is None:
        return None
    value = raw.strip()
    if value in SENTINELS:
        return None
    if value in DTI_BINS:
        return DTI_BINS[value]
    try:
        number = float(value)
    except ValueError:
        return None
    return number if 0.0 <= number <= 100.0 else None


def parse_number(raw: str | None) -> float | None:
    """Numeric LAR field (loan_amount, income, combined_loan_to_value_ratio, ...), None for sentinels."""
    if raw is None:
        return None
    value = raw.strip()
    if value in SENTINELS:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def parse_code(raw: str | None) -> str:
    """Coded LAR field kept as a category string. Missing becomes "missing", a category of its own."""
    if raw is None:
        return "missing"
    value = raw.strip()
    return value if value else "missing"


# --- What the model sees.


@dataclass(frozen=True)
class Feature:
    name: str
    kind: str  # "numeric" or "categorical"
    description: str
    reason: str  # adverse-action reason this feature maps to; "" when it may not be cited as a reason


NUMERIC_FEATURES: tuple[Feature, ...] = (
    Feature(
        "loan_amount",
        "numeric",
        "Loan amount in USD (HMDA rounds to the $10,000 midpoint)",
        "Amount of credit requested is too high for the income or collateral",
    ),
    Feature(
        "income",
        "numeric",
        "Gross annual income relied on, in thousands of USD",
        "Income insufficient for the amount of credit requested",
    ),
    Feature(
        "property_value",
        "numeric",
        "Property value in USD (rounded to the $10,000 midpoint)",
        "Value or type of collateral not sufficient",
    ),
    Feature(
        "debt_to_income_ratio",
        "numeric",
        "Debt-to-income ratio, percent (bins mapped to midpoints)",
        "Excessive obligations in relation to income",
    ),
    Feature("loan_term", "numeric", "Loan term in months", ""),
)

# Ratios the model derives inside its graph from the inputs above (training and serving share the code).
# combined_loan_to_value_ratio is NOT an input: lenders leave it NA on about half of denials because they
# stop before computing it, so its missingness predicts the label (see reports/leakage-audit.json).
DERIVED_FEATURES: tuple[Feature, ...] = (
    Feature("loan_to_value", "derived", "loan_amount / property_value", "Value or type of collateral not sufficient"),
    Feature(
        "loan_to_income",
        "derived",
        "loan_amount / (income * 1000)",
        "Income insufficient for the amount of credit requested",
    ),
)

CATEGORICAL_FEATURES: tuple[Feature, ...] = (
    Feature("loan_type", "categorical", "1 conventional, 2 FHA, 3 VA, 4 USDA", ""),
    Feature("loan_purpose", "categorical", "1 purchase, 31 refinance, 32 cash-out refinance", ""),
    Feature("occupancy_type", "categorical", "1 principal residence, 2 second home, 3 investment", ""),
    Feature(
        "construction_method",
        "categorical",
        "1 site-built, 2 manufactured home",
        "Value or type of collateral not sufficient",
    ),
    Feature("total_units", "categorical", "Dwelling units, 1 to 4", "Value or type of collateral not sufficient"),
    Feature("applicant_credit_score_type", "categorical", "Credit scoring model used for the applicant", ""),
    Feature(
        "co_applicant_credit_score_type",
        "categorical",
        "Credit scoring model for the co-applicant; 10 means no co-applicant",
        "",
    ),
    Feature("submission_of_application", "categorical", "Submitted directly to the institution or not", ""),
    Feature("initially_payable_to_institution", "categorical", "Loan initially payable to the institution", ""),
    Feature("preapproval", "categorical", "1 preapproval requested, 2 not requested", ""),
    Feature(
        "conforming_loan_limit",
        "categorical",
        "C conforming, NC nonconforming, U undetermined",
        "Amount of credit requested is too high for the income or collateral",
    ),
)

FEATURES: tuple[Feature, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
NUMERIC = tuple(f.name for f in NUMERIC_FEATURES)
CATEGORICAL = tuple(f.name for f in CATEGORICAL_FEATURES)
FEATURE_NAMES = NUMERIC + CATEGORICAL
# Attribution groups the explain signature reports, in this order: every input plus the derived ratios.
ATTRIBUTION_GROUPS: tuple[Feature, ...] = NUMERIC_FEATURES + DERIVED_FEATURES + CATEGORICAL_FEATURES

# Fields the leakage audit screens besides the features: the ones excluded for leaking.
EXCLUDED_FOR_LEAKAGE = ("combined_loan_to_value_ratio", "aus_1")

# Never model inputs. Protected bases and tract demographics are kept so the fair-lending audit can use
# them; geography and the lender are kept for grouping. Using any of them as a feature is how a model
# rebuilds race from location (redlining), so a test fails if one enters FEATURE_NAMES.
PROTECTED = ("derived_race", "derived_ethnicity", "derived_sex", "applicant_age", "applicant_age_above_62")
TRACT_DEMOGRAPHICS = (
    "tract_minority_population_percent",
    "tract_to_msa_income_percentage",
    "ffiec_msa_md_median_family_income",
    "tract_population",
    "tract_owner_occupied_units",
    "tract_one_to_four_family_homes",
    "tract_median_age_of_housing_units",
)
GEOGRAPHY = ("state_code", "county_code", "census_tract", "derived_msa_md")
IDENTIFIERS = ("lei", "activity_year")

# Populated only after a decision, or only for originated loans: they would leak the label.
# The leakage audit measures them; this list is what the audit is expected to flag.
POST_DECISION = (
    "interest_rate",
    "rate_spread",
    "total_loan_costs",
    "total_points_and_fees",
    "origination_charges",
    "discount_points",
    "lender_credits",
    "hoepa_status",
    "purchaser_type",
    "intro_rate_period",
    "prepayment_penalty_term",
    "denial_reason_1",
    "denial_reason_2",
    "denial_reason_3",
    "denial_reason_4",
)

# --- The model population: closed-end, first-lien, 1-4 unit home purchase and refinance applications
# that reached a credit decision (originated, approved but not accepted, or denied).
DECISION_ACTIONS = ("1", "2", "3")
DENIED_ACTION = "3"
POPULATION_PURPOSES = ("1", "31", "32")
POPULATION_UNITS = ("1", "2", "3", "4")

LABEL = "denied"

# Codes that mean "not applicable" or "exempt" for a coded field. The leakage audit treats them as missing
# when it compares how often a field is filled in for approved and denied applications.
NOT_APPLICABLE: dict[str, frozenset[str]] = {
    "aus_1": frozenset({"6", "1111"}),
    "applicant_credit_score_type": frozenset({"9", "1111"}),
    "co_applicant_credit_score_type": frozenset({"9", "1111"}),
    "submission_of_application": frozenset({"3", "1111"}),
    "initially_payable_to_institution": frozenset({"3", "1111"}),
    "hoepa_status": frozenset({"3"}),
    "purchaser_type": frozenset({"0"}),
    "denial_reason_1": frozenset({"10"}),
    "conforming_loan_limit": frozenset({"NA", "U"}),
}

# Silver columns: what the typing step produces.
SILVER_NUMERIC = (
    *NUMERIC,
    "combined_loan_to_value_ratio",
    "interest_rate",
    "rate_spread",
    "total_loan_costs",
    "total_points_and_fees",
    "origination_charges",
    "discount_points",
    "lender_credits",
    "intro_rate_period",
    "prepayment_penalty_term",
    "tract_minority_population_percent",
    "tract_to_msa_income_percentage",
    "ffiec_msa_md_median_family_income",
    "tract_population",
    "tract_owner_occupied_units",
    "tract_one_to_four_family_homes",
    "tract_median_age_of_housing_units",
)
SILVER_CODES = (
    "action_taken",
    "lien_status",
    "reverse_mortgage",
    "open_end_line_of_credit",
    "business_or_commercial_purpose",
    "hoepa_status",
    "purchaser_type",
    "denial_reason_1",
    "denial_reason_2",
    "denial_reason_3",
    "denial_reason_4",
    "aus_1",
    *CATEGORICAL,
)
SILVER_TEXT = (
    "lei",
    "state_code",
    "county_code",
    "census_tract",
    "derived_msa_md",
    *PROTECTED,
    "derived_dwelling_category",
    "derived_loan_product_type",
)
