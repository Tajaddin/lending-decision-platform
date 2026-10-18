"""Leakage audit: is a field known before the credit decision, or is it a trace of the decision itself?

Two screens per candidate field, computed on the training year only:

1. Availability gap. How often the field is filled in for approved vs denied applications. HMDA fields like
   CLTV or the AUS result are left NA when a lender stops early, so "missing" alone predicts a denial.
2. Univariate AUC. How well the field alone separates the classes (numeric value, its missing flag, or a
   category's denial rate estimated on one half of the data and scored on the other). Fields that only exist
   after the decision (interest rate, denial reasons, purchaser) score near 1.

A flagged field must not be a model input; `assert_features_clean` enforces that in the pipeline and in CI.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ldp.schema import EXCLUDED_FOR_LEAKAGE, FEATURE_NAMES, LABEL, NOT_APPLICABLE, POST_DECISION

AVAILABILITY_GAP_LIMIT = 0.10
UNIVARIATE_AUC_LIMIT = 0.80
# Below this many rows per class the screens are noise (a 40-denial sample flags fields at random), so the
# audit is reported as underpowered and not enforced.
MIN_ROWS_PER_CLASS = 1_000
MISSING_CODES = frozenset({"missing", "NA", "Exempt"})


def candidates() -> list[str]:
    return list(dict.fromkeys([*FEATURE_NAMES, *EXCLUDED_FOR_LEAKAGE, *POST_DECISION]))


def auc(y: np.ndarray, score: np.ndarray) -> float:
    """Rank-based ROC AUC (ties averaged), symmetric: max(AUC, 1 - AUC)."""
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    pos = int(y.sum())
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = pd.Series(score).rank(method="average").to_numpy()
    value = (ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)
    return float(max(value, 1.0 - value))


def is_available(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.notna()
    codes = MISSING_CODES | NOT_APPLICABLE.get(name, frozenset())
    return series.notna() & ~series.astype(str).isin(codes)


@dataclass
class FieldAudit:
    field: str
    kind: str
    available_approved: float
    available_denied: float
    availability_gap: float
    auc_value: float
    auc_missing_flag: float
    flagged: bool
    reasons: list[str]
    is_feature: bool


def audit_field(df: pd.DataFrame, name: str, rng: np.random.Generator) -> FieldAudit:
    y = df[LABEL].to_numpy()
    s = df[name]
    available = is_available(s, name)
    approved, denied = y == 0, y == 1
    a_app = float(available[approved].mean()) if approved.any() else float("nan")
    a_den = float(available[denied].mean()) if denied.any() else float("nan")
    gap = abs(a_app - a_den)
    auc_missing = auc(y, (~available).astype(float).to_numpy())
    if pd.api.types.is_numeric_dtype(s):
        kind = "numeric"
        filled = s.fillna(s.median() if available.any() else 0.0).to_numpy()
        auc_value = auc(y, filled)
    else:
        kind = "categorical"
        half = rng.random(len(df)) < 0.5
        rates = df.loc[half].groupby(name)[LABEL].mean()
        prior = float(y[half].mean())
        encoded = s.map(rates).fillna(prior).to_numpy()
        auc_value = auc(y[~half], encoded[~half])
    reasons = []
    if gap > AVAILABILITY_GAP_LIMIT:
        reasons.append(f"filled in for {a_app:.0%} of approvals but {a_den:.0%} of denials")
    if auc_value > UNIVARIATE_AUC_LIMIT:
        reasons.append(f"separates the classes alone (AUC {auc_value:.3f})")
    return FieldAudit(
        name, kind, a_app, a_den, gap, auc_value, auc_missing, bool(reasons), reasons, name in FEATURE_NAMES
    )


def run_audit(df: pd.DataFrame, seed: int = 7) -> list[FieldAudit]:
    rng = np.random.default_rng(seed)
    return [audit_field(df, name, rng) for name in candidates() if name in df.columns]


class LeakageError(RuntimeError):
    pass


def is_powered(df: pd.DataFrame) -> bool:
    counts = df[LABEL].value_counts()
    return bool(min(counts.get(0, 0), counts.get(1, 0)) >= MIN_ROWS_PER_CLASS)


def assert_features_clean(results: list[FieldAudit]) -> None:
    leaking = [r.field for r in results if r.flagged and r.is_feature]
    if leaking:
        raise LeakageError(f"model inputs flagged by the leakage audit: {leaking}")


def write_audit(results: list[FieldAudit], rows: int, year: int, path: Path, enforced: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "year": year,
        "rows": rows,
        "enforced": enforced,
        "thresholds": {
            "availability_gap": AVAILABILITY_GAP_LIMIT,
            "univariate_auc": UNIVARIATE_AUC_LIMIT,
            "min_rows_per_class": MIN_ROWS_PER_CLASS,
        },
        "flagged": [r.field for r in results if r.flagged],
        "fields": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
