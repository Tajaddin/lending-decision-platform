"""Less discriminatory alternative (LDA) search: the v1 model trained with protected attributes shaping the
training objective, never the inputs (ADR 0010).

Each variant is a change to the v1 recipe's loss and nothing else:

  reweighing         Kamiran & Calders (2012): weight each application by P(cell) P(y) / P(cell, y), where the
                     cell is age 62+ x race x sex. In the weighted data the denial rate is the same in every cell.
  equalized odds     binary cross-entropy + lambda * sum of squared gaps between each group and its reference,
                     among approved and among denied applications separately (Hardt et al. 2016 ask for equal
                     referral rates in both). Two relaxations: gaps in mean logit, and gaps in a smooth referral
                     rate at each batch's own 80%-recall threshold. With equal referral rates among approved and
                     among denied applications, a group's AIR is at least the lenders' own whenever its denial
                     rate is higher: (1 - f - (t - f) d_g) / (1 - f - (t - f) d_r) >= (1 - d_g) / (1 - d_r) for
                     d_g > d_r, recall t and false-referral rate f with f <= t < 1. For d_g < d_r it reverses:
                     the AIR stays at or above 1 but falls below the lenders' own, toward parity.

The deployed model has the same inputs either way: the attributes only enter the loss, through sample weights or
through the extra label columns of each batch.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from ldp.training.data import labels, to_numpy_batch
from ldp.training.fairness import AIR_LIMIT, GAP_TOLERANCE

# (column, group, reference) pairs whose gaps the equalized-odds penalty closes: the three large flagged groups.
PENALISED: tuple[tuple[str, str, str], ...] = (
    ("applicant_age_above_62", "Yes", "No"),
    ("derived_race", "Black or African American", "White"),
    ("derived_sex", "Female", "Male"),
)
# The referral-rate penalty adds American Indian or Alaska Native applicants, still flagged after the mean-logit
# runs. They are about 30 rows of a 4,096-row batch: the square of a gap estimated from so few rows is biased up by
# its variance, so these runs use the split-halves estimate (see equalized_odds_penalty). Native Hawaiian or Other
# Pacific Islander applicants (about 8 rows a batch) are left out: in a 2024 pilot their noise made training
# unstable, and they pass the audit once the other gaps narrow.
PENALISED_RATE: tuple[tuple[str, str, str], ...] = (
    *PENALISED,
    ("derived_race", "American Indian or Alaska Native", "White"),
)
CELL_COLUMNS = ("applicant_age_above_62", "derived_race", "derived_sex")


def _column(table: pa.Table, name: str) -> np.ndarray:
    return np.asarray(table[name].to_pandas().astype("string").fillna("missing").to_numpy(), dtype=object)


def protected_cells(table: pa.Table, columns: tuple[str, ...] = CELL_COLUMNS) -> np.ndarray:
    cols = [_column(table, c) for c in columns]
    return np.array(["|".join(parts) for parts in zip(*cols, strict=True)], dtype=object)


def reweighing_weights(cells: np.ndarray, y: np.ndarray) -> np.ndarray:
    """w(c, y) = P(c) P(y) / P(c, y). The weights sum to n, and the label is independent of the cell under them."""
    df = pd.DataFrame({"c": cells, "y": np.asarray(y).astype(int)})
    n = len(df)
    p_c = df.groupby("c")["y"].transform("size") / n
    p_y = df.groupby("y")["c"].transform("size") / n
    p_cy = df.groupby(["c", "y"])["y"].transform("size") / n
    return (p_c * p_y / p_cy).to_numpy(np.float32)


def membership(table: pa.Table, comparisons: tuple[tuple[str, str, str], ...] = PENALISED) -> np.ndarray:
    """(n, 2K) 0/1 matrix: for each comparison, a column for the group and one for its reference."""
    cols = []
    for column, group, reference in comparisons:
        values = _column(table, column)
        cols += [values == group, values == reference]
    return np.column_stack(cols).astype(np.float32)


@dataclass
class AuxStream:
    """data.BatchStream plus one per-row array carried through the same shuffle.

    as_weights=True yields (inputs, label, weight); otherwise (inputs, [label, aux columns...]) for a loss that
    needs more than the label. Same seed, same batches as BatchStream.
    """

    table: pa.Table
    aux: np.ndarray
    as_weights: bool
    batch_size: int = 4096
    shuffle: bool = True
    seed: int = 7

    def __post_init__(self) -> None:
        self._epoch = 0

    def __len__(self) -> int:
        return (self.table.num_rows + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[tuple[Any, ...]]:
        n = self.table.num_rows
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        order = rng.permutation(n) if self.shuffle else np.arange(n)
        for start in range(0, n, self.batch_size):
            idx = order[start : start + self.batch_size]
            part = self.table.take(pa.array(idx))
            x, y = to_numpy_batch(part), labels(part)
            if self.as_weights:
                yield x, y, self.aux[idx].astype(np.float32)
            else:
                yield x, np.column_stack([y, self.aux[idx]]).astype(np.float32)


def as_aux_dataset(stream: AuxStream):  # -> tf.data.Dataset
    import tensorflow as tf

    from ldp.training.model import input_signature

    sig: tuple[Any, ...]
    if stream.as_weights:
        sig = (input_signature(), tf.TensorSpec((None,), tf.float32), tf.TensorSpec((None,), tf.float32))
    else:
        sig = (input_signature(), tf.TensorSpec((None, 1 + stream.aux.shape[1]), tf.float32))
    ds_ = tf.data.Dataset.from_generator(lambda: iter(stream), output_signature=sig)
    return ds_.apply(tf.data.experimental.assert_cardinality(len(stream))).prefetch(tf.data.AUTOTUNE)


SOFT_TAU = 0.25  # logit units: width of the smooth step in soft_referral


def soft_referral(y: Any, logit: Any, recall: float = 0.80, tau: float = SOFT_TAU) -> Any:
    """sigmoid((logit - t) / tau), with t the batch's own threshold catching `recall` of its denials (held fixed
    for the gradient). Its mean over a group is a smooth referral rate at the operating point."""
    import tensorflow as tf

    denied = tf.sort(tf.boolean_mask(logit, tf.equal(y, 1.0)))
    n = tf.shape(denied)[0]
    k = tf.clip_by_value(tf.cast(tf.floor((1.0 - recall) * tf.cast(n, tf.float32)), tf.int32), 0, tf.maximum(n - 1, 0))
    t = tf.stop_gradient(tf.cond(n > 0, lambda: denied[k], lambda: tf.constant(0.0)))
    return tf.sigmoid((logit - t) / tau)


def equalized_odds_penalty(y: Any, logit: Any, members: Any, k: int, split: bool = False) -> Any:
    """Sum over the k comparisons and both outcomes of (mean score of the group - mean score of the reference)^2.

    The score is the logit, or `soft_referral` for the penalty at the operating point. members holds the 2k columns from `membership`. An outcome cell with no group or no reference rows in the
    batch contributes nothing. With split=True each squared gap is estimated as gap(even rows) x gap(odd rows):
    the batch is shuffled, so the two halves are independent and the product is unbiased for the squared gap,
    where the square of one small-sample gap is biased up by that gap's variance.
    """
    import tensorflow as tf

    if split:
        even = tf.cast(tf.equal(tf.range(tf.shape(y)[0]) % 2, 0), tf.float32)
        halves = [even, 1.0 - even]
    else:
        halves = [tf.ones_like(y)]
    total = tf.constant(0.0)
    for j in range(k):
        for outcome in (0.0, 1.0):
            o = tf.cast(tf.equal(y, outcome), tf.float32)
            gaps, ok = [], tf.constant(1.0)
            for half in halves:
                g, r = members[:, 2 * j] * o * half, members[:, 2 * j + 1] * o * half
                ng, nr = tf.reduce_sum(g), tf.reduce_sum(r)
                gaps.append(
                    tf.math.divide_no_nan(tf.reduce_sum(g * logit), ng)
                    - tf.math.divide_no_nan(tf.reduce_sum(r * logit), nr)
                )
                ok *= tf.cast((ng > 0) & (nr > 0), tf.float32)
            total += ok * (gaps[0] * gaps[1] if split else tf.square(gaps[0]))
    return total


def penalised_loss(lam: float, k: int = len(PENALISED), split: bool = False, at_threshold: bool = False):
    """Loss for y_true = [label, membership columns]: mean binary cross-entropy + lam * equalized_odds_penalty,
    on the logit or (at_threshold) on the soft referral at the batch's 80%-recall threshold."""
    import keras
    import tensorflow as tf

    def loss(y_true: Any, y_pred: Any) -> Any:
        y, logit = y_true[:, 0], y_pred[:, 0]
        bce = tf.reduce_mean(keras.losses.binary_crossentropy(y_true[:, :1], y_pred, from_logits=True))
        score = soft_referral(y, logit) if at_threshold else logit
        return bce + lam * equalized_odds_penalty(y, score, y_true[:, 1:], k, split)

    return loss


def label_auc(name: str = "auc"):
    """keras AUC on the first label column, so early stopping reads the same val_auc as v1."""
    import keras

    class LabelAUC(keras.metrics.AUC):
        def update_state(self, y_true: Any, y_pred: Any, sample_weight: Any = None) -> Any:
            return super().update_state(y_true[:, :1], y_pred, sample_weight)

    return LabelAUC(name=name, from_logits=True)


# --- Reading an audit and choosing a variant.


def audit_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Every compared group's AIRs, and how far the worst one is from passing (> 0 fails).

    Shortfall of a group = max(AIR_LIMIT - model AIR, lenders' AIR - GAP_TOLERANCE - model AIR): positive exactly
    when fairness.audit flags it.
    """
    rows = []
    for dim, block in report["dimensions"].items():
        for g in block["groups"]:
            if "model_air" not in g:
                continue
            shortfall = max(AIR_LIMIT - g["model_air"], g["observed_air"] - GAP_TOLERANCE - g["model_air"])
            rows.append(
                {
                    "dimension": dim,
                    "group": g["group"],
                    "reference": block["reference"]["group"],
                    "n": g["n"],
                    "observed_air": g["observed_air"],
                    "model_air": g["model_air"],
                    "shortfall": shortfall,
                }
            )
    worst = max(rows, key=lambda r: r["shortfall"])
    return {
        "groups": rows,
        "flags": report["flags"],
        "passes_audit": not report["flags"],
        "worst_group": f"{worst['dimension']}:{worst['group']}",
        "worst_shortfall": worst["shortfall"],
        "min_model_minus_observed_air": min(r["model_air"] - r["observed_air"] for r in rows),
    }


SELECTION_RULE = (
    "among variants that pass the 2024 audit, the highest 2024 AUC; if none passes, the smallest worst shortfall"
)


def choose(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """SELECTION_RULE, fixed before the search ran."""
    passing = [c for c in candidates if c["valid_2024"]["passes_audit"]]
    if passing:
        best = max(passing, key=lambda c: c["valid_2024"]["auc"])
        return {"name": best["name"], "passes_2024_audit": True, "rule": "highest 2024 AUC among passing variants"}
    best = min(candidates, key=lambda c: c["valid_2024"]["worst_shortfall"])
    return {"name": best["name"], "passes_2024_audit": False, "rule": "no variant passes; smallest worst shortfall"}
