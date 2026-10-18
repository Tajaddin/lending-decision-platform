"""Fair-lending audit: adverse impact ratios by protected group, for the model and for the lenders.

The favourable outcome is "not referred" (the model) or "approved" (the lenders' actual decision). The adverse
impact ratio (AIR) of a group is its favourable rate divided by the reference group's; values below 0.80 (the
four-fifths rule of thumb used in disparate-impact screening) are flagged. The model learns from the lenders'
decisions, so the useful comparison is model AIR against observed AIR: a model that widens the gap fails.

Groups follow HMDA's derived fields. "Not available" and joint applications are reported but are not
compared with a reference group.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ldp.training.evaluate import auc

AIR_LIMIT = 0.80
MIN_GROUP = 1_000

DIMENSIONS: dict[str, dict[str, Any]] = {
    "race": {
        "column": "derived_race",
        "reference": "White",
        "groups": [
            "Black or African American",
            "Asian",
            "American Indian or Alaska Native",
            "Native Hawaiian or Other Pacific Islander",
            "2 or more minority races",
        ],
    },
    "ethnicity": {
        "column": "derived_ethnicity",
        "reference": "Not Hispanic or Latino",
        "groups": ["Hispanic or Latino"],
    },
    "sex": {"column": "derived_sex", "reference": "Male", "groups": ["Female"]},
    "age": {"column": "applicant_age_above_62", "reference": "No", "groups": ["Yes"]},
}


def group_stats(df: pd.DataFrame, mask: np.ndarray, threshold: float) -> dict[str, float]:
    y = df["denied"].to_numpy()[mask]
    p = df["probability"].to_numpy()[mask]
    out = {
        "n": int(mask.sum()),
        "observed_approval_rate": float(1 - y.mean()),
        "model_pass_rate": float((p < threshold).mean()),
        "mean_predicted": float(p.mean()),
        "observed_denial_rate": float(y.mean()),
    }
    if 0 < y.sum() < len(y):
        out["auc"] = auc(y, p)
    return out


def audit(df: pd.DataFrame, threshold: float) -> dict[str, Any]:
    """df needs columns: denied, probability and the protected columns in DIMENSIONS."""
    report: dict[str, Any] = {"air_limit": AIR_LIMIT, "threshold": threshold, "dimensions": {}}
    flags = []
    for dim, spec in DIMENSIONS.items():
        col = df[spec["column"]].astype(str).to_numpy()
        ref_mask = col == spec["reference"]
        ref = group_stats(df, ref_mask, threshold)
        rows = []
        for group in spec["groups"]:
            mask = col == group
            if mask.sum() < MIN_GROUP:
                rows.append({"group": group, "n": int(mask.sum()), "note": f"fewer than {MIN_GROUP} applications"})
                continue
            g = group_stats(df, mask, threshold)
            g["group"] = group
            g["model_air"] = g["model_pass_rate"] / ref["model_pass_rate"]
            g["observed_air"] = g["observed_approval_rate"] / ref["observed_approval_rate"]
            g["model_minus_observed_air"] = g["model_air"] - g["observed_air"]
            g["calibration_gap"] = g["mean_predicted"] - g["observed_denial_rate"]
            g["flag_model_air_below_limit"] = g["model_air"] < AIR_LIMIT
            g["flag_model_widens_gap"] = g["model_air"] < g["observed_air"] - 0.02
            if g["flag_model_air_below_limit"] or g["flag_model_widens_gap"]:
                flags.append(f"{dim}:{group}")
            rows.append(g)
        report["dimensions"][dim] = {"reference": {"group": spec["reference"], **ref}, "groups": rows}
    report["flags"] = flags
    return report
