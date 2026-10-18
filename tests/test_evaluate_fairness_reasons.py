from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss, roc_auc_score

from ldp.reasons import top_reasons
from ldp.training import evaluate, fairness


@pytest.fixture
def scored() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    y = (rng.random(20_000) < 0.25).astype(np.int8)
    p = np.clip(0.25 + 0.3 * (y - 0.25) + rng.normal(0, 0.15, y.size), 0.001, 0.999)
    return y, p


def test_auc_ks_brier_match_references(scored) -> None:
    y, p = scored
    assert evaluate.auc(y, p) == pytest.approx(roc_auc_score(y, p), abs=1e-12)
    assert evaluate.brier(y, p) == pytest.approx(brier_score_loss(y, p), abs=1e-12)
    from scipy.stats import ks_2samp

    assert evaluate.ks_statistic(y, p) == pytest.approx(ks_2samp(p[y == 1], p[y == 0]).statistic, abs=1e-9)


def test_bootstrap_ci_brackets_the_estimate(scored) -> None:
    y, p = scored
    lo, hi = evaluate.bootstrap_auc_ci(y, p, n_boot=50, sample=5_000)
    assert lo < evaluate.auc(y, p) < hi
    assert hi - lo < 0.05


def test_ece_is_zero_for_perfect_calibration() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 200_000)
    y = (rng.random(p.size) < p).astype(int)
    assert evaluate.ece(y, p) < 0.01
    assert evaluate.ece(y, np.clip(p + 0.2, 0, 1)) > 0.1


def test_threshold_for_recall_and_operating_point(scored) -> None:
    y, p = scored
    t = evaluate.threshold_for_recall(y, p, 0.8)
    op = evaluate.operating_point(y, p, t)
    assert op["recall_of_denials"] == pytest.approx(0.8, abs=0.001)
    assert 0 < op["precision"] <= 1


def test_psi_is_zero_for_the_same_population_and_grows_with_shift() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, 50_000)
    ref = evaluate.reference_bins(base, "numeric")
    assert evaluate.psi(np.array(ref["expected"]), evaluate.histogram(rng.normal(0, 1, 50_000), ref)) < 0.01
    assert evaluate.psi(np.array(ref["expected"]), evaluate.histogram(rng.normal(0.8, 1, 50_000), ref)) > 0.25
    cat = evaluate.reference_bins(np.array(["a"] * 70 + ["b"] * 30), "categorical")
    assert evaluate.psi(np.array(cat["expected"]), evaluate.histogram(np.array(["a"] * 30 + ["b"] * 70), cat)) > 0.25


def test_missing_values_get_their_own_bin() -> None:
    ref = evaluate.reference_bins(np.array([1.0, 2.0, 3.0, np.nan]), "numeric")
    counts = evaluate.histogram(np.array([np.nan, np.nan, 2.0]), ref)
    assert counts[-1] == 2


def test_fairness_audit_compares_model_with_lenders() -> None:
    rng = np.random.default_rng(4)
    n = 40_000
    race = rng.choice(["White", "Black or African American", "Asian"], n, p=[0.7, 0.2, 0.1])
    base = np.where(race == "Black or African American", 0.35, 0.2)
    denied = (rng.random(n) < base).astype(int)
    df = pd.DataFrame({
        "derived_race": race, "derived_ethnicity": "Not Hispanic or Latino", "derived_sex": "Male",
        "applicant_age_above_62": "No", "denied": denied,
        "probability": np.clip(base + rng.normal(0, 0.05, n), 0, 1),
    })
    report = fairness.audit(df, threshold=0.3)
    black = next(g for g in report["dimensions"]["race"]["groups"] if g["group"] == "Black or African American")
    assert black["observed_air"] == pytest.approx(0.65 / 0.8, abs=0.03)
    assert black["model_air"] < 0.8
    assert "race:Black or African American" in report["flags"]
    tiny = next(g for g in report["dimensions"]["race"]["groups"] if g["group"] == "American Indian or Alaska Native")
    assert tiny["n"] == 0 and "fewer than" in tiny["note"]


GROUPS = [
    {"name": "debt_to_income_ratio", "reason": "Excessive obligations in relation to income"},
    {"name": "property_value", "reason": "Value or type of collateral not sufficient"},
    {"name": "loan_to_value", "reason": "Value or type of collateral not sufficient"},
    {"name": "loan_type", "reason": ""},
    {"name": "income", "reason": "Income insufficient for the amount of credit requested"},
]


def test_reasons_rank_positive_contributions_and_merge_shared_reasons() -> None:
    reasons = top_reasons([0.4, 0.3, 0.2, 2.0, -0.5], GROUPS)
    assert [r.reason for r in reasons] == ["Value or type of collateral not sufficient",
                                          "Excessive obligations in relation to income"]
    assert reasons[0].features == ("property_value", "loan_to_value")
    assert reasons[0].contribution == pytest.approx(0.5)


def test_reasons_skip_codes_that_are_not_reasons_and_cap_at_four() -> None:
    assert top_reasons([0.0, 0.0, 0.0, 3.0, 0.0], GROUPS) == []
    many = [{"name": f"f{i}", "reason": f"r{i}"} for i in range(6)]
    assert len(top_reasons([1.0] * 6, many)) == 4
    with pytest.raises(ValueError, match="attributions"):
        top_reasons([1.0], GROUPS)
