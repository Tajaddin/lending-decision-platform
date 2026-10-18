from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from ldp.audit import LeakageError, assert_features_clean, auc, is_powered, run_audit


def test_auc_matches_sklearn_and_is_symmetric() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    s = y * 0.5 + rng.normal(size=2000)
    assert auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    assert auc(y, -s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    assert np.isnan(auc(np.zeros(5), np.arange(5)))


def _frame(n: int = 4000, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    denied = (rng.random(n) < 0.3).astype(int)
    df = pd.DataFrame(
        {
            "denied": denied,
            "income": rng.lognormal(4.5, 0.5, n),
            "debt_to_income_ratio": np.where(rng.random(n) < 0.01, np.nan, rng.uniform(10, 60, n)),
            "loan_type": rng.choice(["1", "2", "3"], n),
            # post-decision: only originated loans get a rate
            "interest_rate": np.where(denied == 1, np.nan, rng.uniform(5, 8, n)),
            # reported for most approvals, half of denials
            "combined_loan_to_value_ratio": np.where(
                (denied == 1) & (rng.random(n) < 0.5), np.nan, rng.uniform(50, 100, n)
            ),
            "denial_reason_1": np.where(denied == 1, rng.choice(["1", "3"], n), "10"),
        }
    )
    return df


def test_audit_flags_post_decision_and_missingness_leaks_only() -> None:
    results = {r.field: r for r in run_audit(_frame())}
    assert results["interest_rate"].flagged
    assert results["combined_loan_to_value_ratio"].flagged
    assert results["denial_reason_1"].flagged
    assert not results["income"].flagged
    assert not results["debt_to_income_ratio"].flagged
    assert not results["loan_type"].flagged
    assert results["combined_loan_to_value_ratio"].availability_gap > 0.4
    assert_features_clean(list(results.values()))  # the flagged ones are not features


def test_small_samples_are_underpowered() -> None:
    assert is_powered(_frame(n=6000))
    assert not is_powered(_frame(n=500))


def test_a_flagged_feature_stops_the_pipeline() -> None:
    df = _frame()
    df["income"] = np.where(df["denied"] == 1, np.nan, df["income"])  # income now leaks through missingness
    with pytest.raises(LeakageError, match="income"):
        assert_features_clean(run_audit(df))
