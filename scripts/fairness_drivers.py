"""Which inputs drive the model's gap between groups? Mean integrated-gradient attributions by group.

For each flagged comparison (group vs its reference) on the 2025 hold-out, this samples 20,000 applications per
side, runs the SavedModel's `explain` signature, and reports the mean attribution of every input. Attributions
sum to logit - reference logit, so the per-input differences add up to the difference in mean logit between the
two groups: the inputs with the largest differences are where the gap comes from.

    LDP_TRAINING_DIR=~/data/hmda/training python scripts/fairness_drivers.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.compute as pc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ldp.config import get_settings  # noqa: E402
from ldp.training.data import load_split, to_numpy_batch  # noqa: E402
from ldp.training.train import sample_table  # noqa: E402

COMPARISONS = [
    ("age", "applicant_age_above_62", "Yes", "No"),
    ("race", "derived_race", "Black or African American", "White"),
    ("sex", "derived_sex", "Female", "Male"),
]


def mean_attributions(explain, table, n: int, seed: int) -> tuple[np.ndarray, float, dict[str, float]]:
    import tensorflow as tf

    part = sample_table(table, n, seed)
    total = None
    logits = []
    for start in range(0, part.num_rows, 1024):
        batch = {k: tf.constant(v) for k, v in to_numpy_batch(part.slice(start, 1024)).items()}
        out = explain(**batch)
        a = out["attributions"].numpy().sum(axis=0)
        total = a if total is None else total + a
        logits.append(out["logit"].numpy())
    profile = {
        "median_income_k": float(np.nanmedian(part["income"].to_numpy(zero_copy_only=False))),
        "median_loan_amount": float(np.nanmedian(part["loan_amount"].to_numpy(zero_copy_only=False))),
        "median_dti": float(np.nanmedian(part["debt_to_income_ratio"].to_numpy(zero_copy_only=False))),
        "share_refinance": float(
            pc.mean(pc.is_in(part["loan_purpose"], value_set=__import__("pyarrow").array(["31", "32"]))).as_py()
        ),
        "observed_denial_rate": float(pc.mean(part["denied"]).as_py()),
    }
    return total / part.num_rows, float(np.concatenate(logits).mean()), profile


def main() -> None:
    import tensorflow as tf

    settings = get_settings()
    model_dir = Path(settings.models_dir) / settings.model_name / "1"
    meta = json.loads((model_dir / "metadata.json").read_text())
    names = [g["name"] for g in meta["attribution_groups"]]
    explain = tf.saved_model.load(str(model_dir)).signatures["explain"]
    test = load_split(Path(settings.training_dir).expanduser() / "test")
    out = {"model_version": meta["version"], "split": "test 2025", "comparisons": []}
    for dim, column, group, reference in COMPARISONS:
        g_tab = test.filter(pc.equal(test[column], group))
        r_tab = test.filter(pc.equal(test[column], reference))
        g_attr, g_logit, g_prof = mean_attributions(explain, g_tab, 20_000, 1)
        r_attr, r_logit, r_prof = mean_attributions(explain, r_tab, 20_000, 2)
        diff = g_attr - r_attr
        order = np.argsort(-np.abs(diff))
        out["comparisons"].append(
            {
                "dimension": dim,
                "group": group,
                "reference": reference,
                "mean_logit_gap": g_logit - r_logit,
                "attribution_gap_sum": float(diff.sum()),
                "drivers": [{"input": names[i], "gap": float(diff[i])} for i in order[:6]],
                "group_profile": g_prof,
                "reference_profile": r_prof,
            }
        )
        print(
            dim,
            group,
            "logit gap",
            round(g_logit - r_logit, 3),
            [(names[i], round(float(diff[i]), 3)) for i in order[:4]],
            flush=True,
        )
    path = Path(settings.reports_dir) / "fairness-drivers.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
