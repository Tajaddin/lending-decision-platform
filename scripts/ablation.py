"""Drop-one-input ablation and a small less-discriminatory-alternative (LDA) search.

1. Where does the skill come from? A leaky input shows up as one feature whose removal collapses the AUC.
2. Does any single input drive the fairness gaps? For each variant, the adverse impact ratio (AIR) of the
   "not referred" outcome at the same operating rule (threshold catching 80% of 2024 denials).

Uses XGBoost (fast, same inputs as the TensorFlow model) on a 1M-row sample of 2023, early-stopped on 2024,
scored on 1M of 2025.

    LDP_TRAINING_DIR=~/data/hmda/training python scripts/ablation.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ldp.config import get_settings  # noqa: E402
from ldp.schema import CATEGORICAL, NUMERIC  # noqa: E402
from ldp.training.baselines import fit_xgboost, predict_logit, tree_design  # noqa: E402
from ldp.training.data import labels, load_split  # noqa: E402
from ldp.training.evaluate import auc, threshold_for_recall  # noqa: E402
from ldp.training.model import FeatureStats, sigmoid  # noqa: E402
from ldp.training.train import sample_table  # noqa: E402

GROUPS = {
    **{n: [n] for n in NUMERIC if n not in ("loan_amount", "property_value", "income")},
    # Amount, value and income also feed the two ratios; drop the ratio with its inputs.
    "loan_amount (and both ratios)": ["loan_amount", "loan_to_value", "loan_to_income"],
    "property_value (and loan_to_value)": ["property_value", "loan_to_value"],
    "income (and loan_to_income)": ["income", "loan_to_income"],
    **{n: [n] for n in CATEGORICAL},
}


def main() -> None:
    settings = get_settings()
    root = Path(settings.training_dir).expanduser()
    train = sample_table(load_split(root / "train"), 1_000_000, seed=1)
    valid = sample_table(load_split(root / "valid"), 500_000, seed=2)
    test = sample_table(load_split(root / "test"), 1_000_000, seed=3)
    stats = FeatureStats.fit(train)
    x_tr, x_va, x_te = (tree_design(t, stats) for t in (train, valid, test))
    y_tr, y_va, y_te = labels(train), labels(valid), labels(test)
    groups = {
        "age 62+ vs under 62": (test["applicant_age_above_62"].to_numpy(zero_copy_only=False).astype(str), "Yes", "No"),
        "Black vs White": (
            test["derived_race"].to_numpy(zero_copy_only=False).astype(str),
            "Black or African American",
            "White",
        ),
        "Female vs Male": (test["derived_sex"].to_numpy(zero_copy_only=False).astype(str), "Female", "Male"),
    }

    def score(cols: list[str]) -> dict[str, float]:
        model = fit_xgboost(x_tr[cols], y_tr, x_va[cols], y_va)
        p_va = sigmoid(predict_logit(model, x_va[cols]))
        p_te = sigmoid(predict_logit(model, x_te[cols]))
        passed = p_te < threshold_for_recall(y_va, p_va, 0.80)
        out = {"test_auc": auc(y_te, p_te), "referral_rate": float(1 - passed.mean())}
        for label, (col, g, ref) in groups.items():
            out[f"air {label}"] = float(passed[col == g].mean() / passed[col == ref].mean())
        return out

    start = time.perf_counter()
    all_cols = list(x_tr.columns)
    full = score(all_cols)
    print("all inputs", {k: round(v, 4) for k, v in full.items()}, flush=True)
    observed = {}
    for label, (col, g, ref) in groups.items():
        approved = y_te == 0
        observed[f"air {label}"] = float(approved[col == g].mean() / approved[col == ref].mean())
    rows = []
    for name, drop in GROUPS.items():
        r = score([c for c in all_cols if c not in drop])
        r = {"dropped": name, **r, "auc_drop": full["test_auc"] - r["test_auc"]}
        rows.append(r)
        print(name, {k: round(v, 4) for k, v in r.items() if k != "dropped"}, flush=True)
    rows.sort(key=lambda r: -r["auc_drop"])
    out = {
        "model": "xgboost, 1M-row 2023 sample, early-stopped on 500K of 2024, scored on 1M of 2025",
        "operating_rule": "refer when p >= threshold catching 80% of 2024 denials (per variant)",
        "all_inputs": full,
        "lenders_observed": observed,
        "seconds": round(time.perf_counter() - start, 1),
        "ablations": rows,
    }
    path = Path(settings.reports_dir) / "ablation.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
