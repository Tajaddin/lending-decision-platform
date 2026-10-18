"""Model evaluation: discrimination, calibration, operating point and population stability.

All metrics are plain numpy so the numbers are easy to audit. AUC confidence intervals come from a
bootstrap over applications.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ldp.audit import auc as symmetric_auc


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """Directional ROC AUC (not symmetrised): P(score of a denial > score of an approval)."""
    y = np.asarray(y, dtype=np.int8)
    pos = int(y.sum())
    neg = len(y) - pos
    ranks = pd.Series(np.asarray(p, dtype=np.float64)).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def bootstrap_auc_ci(
    y: np.ndarray, p: np.ndarray, n_boot: int = 200, sample: int = 400_000, seed: int = 11
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    size = min(n, sample)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size)
        stats.append(auc(y[idx], p[idx]))
    # With a subsample the interval is for a sample of `size`; widen it to the full n.
    lo, hi = np.percentile(stats, [2.5, 97.5])
    centre = auc(y, p)
    scale = np.sqrt(size / n)
    return float(centre - (centre - lo) * scale), float(centre + (hi - centre) * scale)


def ks_statistic(y: np.ndarray, p: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance between the score distributions of denials and approvals."""
    order = np.argsort(p)
    y_sorted = np.asarray(y)[order]
    cum_pos = np.cumsum(y_sorted) / max(y_sorted.sum(), 1)
    cum_neg = np.cumsum(1 - y_sorted) / max((1 - y_sorted).sum(), 1)
    return float(np.max(np.abs(cum_pos - cum_neg)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    """Equal-frequency bins of predicted probability: mean prediction vs observed rate."""
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    q[0], q[-1] = -np.inf, np.inf
    idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.any():
            rows.append(
                {
                    "bin": b,
                    "n": int(m.sum()),
                    "mean_predicted": float(p[m].mean()),
                    "observed_rate": float(np.asarray(y)[m].mean()),
                }
            )
    return rows


def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error with equal-width bins."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(float(p[m].mean()) - float(np.asarray(y)[m].mean()))
    return float(total)


def operating_point(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    refer = p >= threshold
    tp = float(np.sum(refer & (y == 1)))
    return {
        "threshold": float(threshold),
        "referral_rate": float(refer.mean()),
        "recall_of_denials": tp / max(float(np.sum(y == 1)), 1.0),
        "precision": tp / max(float(refer.sum()), 1.0),
    }


def threshold_for_recall(y: np.ndarray, p: np.ndarray, recall: float) -> float:
    """Smallest score threshold that still refers `recall` of the denials (chosen on validation)."""
    scores = np.sort(p[np.asarray(y) == 1])[::-1]
    k = max(int(np.ceil(recall * len(scores))) - 1, 0)
    return float(scores[k])


def summary(y: np.ndarray, p: np.ndarray, threshold: float, with_ci: bool = True) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=np.float64)
    out: dict[str, Any] = {
        "rows": len(y),
        "denial_rate": float(y.mean()),
        "auc": auc(y, p),
        "ks": ks_statistic(y, p),
        "brier": brier(y, p),
        "log_loss": log_loss(y, p),
        "ece": ece(y, p),
        "mean_predicted": float(p.mean()),
        "operating_point": operating_point(y, p, threshold),
        "calibration": calibration_table(y, p),
    }
    if with_ci:
        lo, hi = bootstrap_auc_ci(y, p)
        out["auc_ci95"] = [lo, hi]
    return out


# --- Population stability: the monitoring reference and the 2023 -> 2025 shift.


def reference_bins(values: np.ndarray, kind: str, bins: int = 10) -> dict[str, Any]:
    if kind == "categorical":
        s = pd.Series(values).astype(str)
        share = s.value_counts(normalize=True)
        top = share.head(20)
        return {
            "kind": kind,
            "categories": top.index.tolist(),
            "expected": [*top.to_numpy().tolist(), float(1.0 - top.sum())],
        }
    v = np.asarray(values, dtype=np.float64)
    present = v[np.isfinite(v)]
    edges = np.unique(np.quantile(present, np.linspace(0, 1, bins + 1))[1:-1]).tolist() if present.size else []
    counts = histogram(v, {"kind": "numeric", "edges": edges})
    return {"kind": "numeric", "edges": edges, "expected": (counts / max(counts.sum(), 1)).tolist()}


def histogram(values: np.ndarray, ref: dict[str, Any]) -> np.ndarray:
    """Counts in the reference's bins; the last bin is missing (numeric) or other (categorical)."""
    if ref["kind"] == "categorical":
        s = pd.Series(values).astype(str)
        cats = ref["categories"]
        counts = [int((s == c).sum()) for c in cats]
        return np.array([*counts, len(s) - sum(counts)], dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(v)
    idx = np.searchsorted(np.asarray(ref["edges"]), v[finite], side="right")
    binned = np.bincount(idx, minlength=len(ref["edges"]) + 1).astype(np.float64)
    return np.append(binned, float((~finite).sum()))


def psi(expected: np.ndarray, actual_counts: np.ndarray, floor: float = 1e-4) -> float:
    """Population stability index. Rule of thumb: < 0.10 stable, 0.10-0.25 moderate, > 0.25 major shift."""
    e = np.clip(np.asarray(expected, dtype=np.float64), floor, None)
    a = np.clip(actual_counts / max(actual_counts.sum(), 1.0), floor, None)
    return float(np.sum((a - e) * np.log(a / e)))


def univariate_auc(y: np.ndarray, values: np.ndarray) -> float:
    return symmetric_auc(y, values)
