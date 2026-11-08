"""Less discriminatory alternative search: reweighing, the equalized-odds penalty, batch alignment and the choice."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from ldp.schema import CATEGORICAL, LABEL, NUMERIC
from ldp.training import fairness, lda, model_card


def test_reweighing_makes_the_label_independent_of_the_cell():
    rng = np.random.default_rng(0)
    cells = rng.choice(np.array(["a", "b", "c"], dtype=object), 20_000, p=[0.6, 0.3, 0.1])
    rate = np.where(cells == "a", 0.1, np.where(cells == "b", 0.3, 0.5))
    y = (rng.random(20_000) < rate).astype(int)
    w = lda.reweighing_weights(cells, y)
    assert w.sum() == pytest.approx(len(y), rel=1e-5)
    overall = y.mean()
    for c in ("a", "b", "c"):
        m = cells == c
        assert np.sum(w[m] * y[m]) / np.sum(w[m]) == pytest.approx(overall, abs=1e-6)
        assert w[m].sum() == pytest.approx(m.sum(), rel=1e-5)  # each cell keeps its size


def _table(n: int) -> pa.Table:
    rng = np.random.default_rng(1)
    cols: dict[str, object] = {name: rng.uniform(1, 100, n) for name in NUMERIC}
    for name in CATEGORICAL:
        cols[name] = rng.choice(["1", "2"], n).astype(object)
    cols[LABEL] = rng.integers(0, 2, n).astype(np.int32)
    cols["row_id"] = np.arange(n, dtype=np.float64)
    cols["applicant_age_above_62"] = rng.choice(["Yes", "No", "NA"], n).astype(object)
    cols["derived_race"] = rng.choice(["White", "Black or African American", "Asian"], n).astype(object)
    cols["derived_sex"] = rng.choice(["Male", "Female", "Joint"], n).astype(object)
    return pa.table(cols)


@pytest.mark.parametrize("as_weights", [True, False])
def test_aux_stream_keeps_each_row_with_its_label(as_weights):
    table = _table(1000)
    ids = table["row_id"].to_numpy()
    aux = ids.reshape(-1, 1) if not as_weights else ids
    seen = []
    for batch in lda.AuxStream(table, aux, as_weights=as_weights, batch_size=128, seed=3):
        x, y = batch[0], batch[1]
        rid = batch[2] if as_weights else y[:, 1]
        label = y if as_weights else y[:, 0]
        expected = table[LABEL].to_numpy()[rid.astype(int)]
        np.testing.assert_array_equal(label, expected)
        np.testing.assert_allclose(x["loan_amount"], table["loan_amount"].to_numpy()[rid.astype(int)], rtol=1e-6)
        seen.append(rid)
    assert sorted(np.concatenate(seen).astype(int).tolist()) == list(range(1000))


def test_membership_columns_follow_the_comparisons():
    table = _table(500)
    m = lda.membership(table)
    assert m.shape == (500, 2 * len(lda.PENALISED))
    race = table["derived_race"].to_numpy(zero_copy_only=False)
    np.testing.assert_array_equal(m[:, 2], race == "Black or African American")
    np.testing.assert_array_equal(m[:, 3], race == "White")
    assert not np.any((m[:, 0] == 1) & (m[:, 1] == 1))  # a row is never in a group and its reference


def _audit_frame(n: int = 40_000, shift: float = 0.0, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "derived_race": rng.choice(["White", "Black or African American", "Asian"], n, p=[0.7, 0.2, 0.1]),
            "derived_ethnicity": rng.choice(["Not Hispanic or Latino", "Hispanic or Latino"], n),
            "derived_sex": rng.choice(["Male", "Female"], n),
            "applicant_age_above_62": rng.choice(["No", "Yes"], n, p=[0.8, 0.2]),
        }
    )
    risk = rng.normal(-1.5, 1.0, n) + shift * (df["applicant_age_above_62"] == "Yes")
    df["denied"] = (rng.random(n) < 1 / (1 + np.exp(-risk))).astype(int)
    df["probability"] = 1 / (1 + np.exp(-(risk + rng.normal(0, 0.5, n))))
    return df


@pytest.mark.parametrize("shift", [0.0, 1.5])
def test_audit_summary_shortfall_is_positive_exactly_for_flagged_groups(shift):
    report = fairness.audit(_audit_frame(shift=shift), threshold=0.2)
    summary = lda.audit_summary(report)
    flagged = {f"{g['dimension']}:{g['group']}" for g in summary["groups"] if g["shortfall"] > 0}
    assert flagged == set(report["flags"])
    assert summary["passes_audit"] == (not report["flags"])
    assert summary["worst_shortfall"] == max(g["shortfall"] for g in summary["groups"])


def _candidate(name: str, auc: float, passes: bool, shortfall: float) -> dict:
    return {"name": name, "valid_2024": {"auc": auc, "passes_audit": passes, "worst_shortfall": shortfall}}


def test_choose_takes_the_best_auc_that_passes_else_the_smallest_shortfall():
    passing = [_candidate("a", 0.88, False, 0.10), _candidate("b", 0.85, True, -0.01), _candidate("c", 0.86, True, 0)]
    chosen = lda.choose(passing)
    assert chosen["name"] == "c"
    assert chosen["passes_2024_audit"] is True
    failing = [_candidate("a", 0.88, False, 0.10), _candidate("b", 0.80, False, 0.03)]
    chosen = lda.choose(failing)
    assert chosen["name"] == "b"
    assert chosen["passes_2024_audit"] is False


@pytest.mark.tf
def test_equalized_odds_penalty_matches_a_hand_computation():
    tf = pytest.importorskip("tensorflow")
    y = np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=np.float32)
    logit = np.array([1.0, 2.0, 0.0, 3.0, 1.0, -1.0, 2.0, 0.5], dtype=np.float32)
    g = np.array([1, 1, 0, 1, 0, 0, 0, 0], dtype=np.float32)
    r = np.array([0, 0, 1, 0, 1, 1, 1, 0], dtype=np.float32)
    members = np.column_stack([g, r])
    # approved: group mean (1 + 2) / 2 = 1.5, reference mean (0 - 1) / 2 = -0.5 -> gap 2
    # denied:   group mean 3, reference mean (1 + 2) / 2 = 1.5 -> gap 1.5
    value = lda.equalized_odds_penalty(tf.constant(y), tf.constant(logit), tf.constant(members), k=1)
    assert float(value) == pytest.approx(2.0**2 + 1.5**2)
    # no denied group rows: that cell contributes nothing (and no NaN)
    g_no_denied = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    value = lda.equalized_odds_penalty(
        tf.constant(y), tf.constant(logit), tf.constant(np.column_stack([g_no_denied, r])), k=1
    )
    assert float(value) == pytest.approx(2.0**2)
    # split halves: with every row repeated, the even and odd halves are the same batch, so gap x gap = gap^2
    twice = [tf.constant(np.repeat(a, 2, axis=0)) for a in (y, logit, members)]
    value = lda.equalized_odds_penalty(*twice, k=1, split=True)
    assert float(value) == pytest.approx(2.0**2 + 1.5**2)


def test_split_estimate_is_unbiased_where_the_square_of_a_small_sample_gap_is_not():
    tf = pytest.importorskip("tensorflow")
    rng = np.random.default_rng(5)
    plain, split = [], []
    for _ in range(300):  # 10 group rows per batch, no true gap
        y = np.zeros(200, dtype=np.float32)
        logit = rng.normal(size=200).astype(np.float32)
        g = np.zeros(200, dtype=np.float32)
        g[rng.choice(200, 10, replace=False)] = 1
        members = tf.constant(np.column_stack([g, 1 - g]))
        plain.append(float(lda.equalized_odds_penalty(tf.constant(y), tf.constant(logit), members, k=1)))
        split.append(float(lda.equalized_odds_penalty(tf.constant(y), tf.constant(logit), members, k=1, split=True)))
    assert np.mean(plain) > 0.08  # about Var(mean of 10) + Var(mean of 190) = 0.105
    assert abs(np.mean(split)) < 0.03


@pytest.mark.tf
def test_penalised_loss_with_lambda_zero_is_binary_cross_entropy():
    tf = pytest.importorskip("tensorflow")
    keras = pytest.importorskip("keras")
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 64).astype(np.float32)
    logit = rng.normal(size=(64, 1)).astype(np.float32)
    members = rng.integers(0, 2, (64, 2 * len(lda.PENALISED))).astype(np.float32)
    y_true = tf.constant(np.column_stack([y, members]))
    got = float(lda.penalised_loss(0.0)(y_true, tf.constant(logit)))
    want = float(keras.losses.BinaryCrossentropy(from_logits=True)(y.reshape(-1, 1), logit))
    assert got == pytest.approx(want, rel=1e-5)
    assert float(lda.penalised_loss(1.0)(y_true, tf.constant(logit))) > got


def _summary(model_air: float, passes: bool) -> dict:
    groups = [
        {
            "dimension": key.split(":")[0],
            "group": key.split(":")[1],
            "reference": "ref",
            "n": 5000,
            "observed_air": 0.9,
            "model_air": model_air,
            "shortfall": 0.88 - model_air,
        }
        for key, _ in model_card.LDA_COLUMNS
    ]
    flags = [] if passes else [f"{g['dimension']}:{g['group']}" for g in groups]
    return {"groups": groups, "flags": flags, "passes_audit": passes, "worst_shortfall": 0.88 - model_air}


def test_model_card_renders_the_search_and_says_whether_the_choice_passes(tmp_path):
    assert model_card._lda_section(tmp_path) == []  # no search report, no section
    base = {"auc": 0.87, "referral_rate": 0.3}
    report = {
        "protocol": {"selection": f"fairness.audit on all of 2024; {lda.SELECTION_RULE}"},
        "reference": {"name": "v1", "valid_2024": {**base, **_summary(0.75, False)}},
        "candidates": [{"name": "eo", "valid_2024": {**base, "auc": 0.86, **_summary(0.90, True)}}],
        "operating_rule_sensitivity_v1": [
            {
                "recall_target": 0.8,
                "referral_rate": 0.3,
                "flags": ["age:Yes"],
                "model_air": {k: 0.75 for k, _ in model_card.LDA_COLUMNS},
            }
        ],
        "chosen": {"name": "eo", "passes_2024_audit": True, "rule": "highest 2024 AUC among passing variants"},
    }
    (tmp_path / "lda-search.json").write_text(json.dumps(report))
    text = "\n".join(model_card._lda_section(tmp_path))
    assert "| eo | 0.8600 | 30.0% | 0.900 |" in text
    assert "has not been scored on 2025 yet" in text
    assert "has chosen `eo` on 2024; 2025 is not scored yet" in model_card._lda_status(tmp_path)

    report["holdout_2025"] = {
        "scored_at": "2026-10-18T00:00:00+00:00",
        "variant": "eo",
        "auc": 0.85,
        "auc_ci95": [0.84, 0.86],
        "referral_rate": 0.31,
        "recall_of_denials": 0.8,
        **_summary(0.78, False),
        "v1": {"auc": 0.87, "referral_rate": 0.33, **_summary(0.75, False)},
    }
    (tmp_path / "lda-search.json").write_text(json.dumps(report))
    text = "\n".join(model_card._lda_section(tmp_path))
    assert "**It does not pass the audit**: 5 group(s) flagged" in text
    assert "| age | Yes (ref) | 0.900 | 0.750 | 0.780 |" in text
    assert "found none that passes on 2025: `eo` passed on 2024 but flags 5" in model_card._lda_status(tmp_path)
    assert model_card._lda_status(tmp_path / "missing") == ""
    assert "the chosen variant passes 0" in text
    assert "at or above parity" not in text

    # a group flagged while at parity with its reference gets the explanation
    report["holdout_2025"].update(_summary(1.01, False))
    (tmp_path / "lda-search.json").write_text(json.dumps(report))
    text = "\n".join(model_card._lda_section(tmp_path))
    assert "`age:Yes` is flagged at an AIR of 1.010, at or above parity" in text
