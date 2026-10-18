"""The decision model: raw typed inputs in, preprocessing inside the graph, a logit out.

Two Keras models share the work:
  encoder  raw inputs -> one dense vector (imputation, transforms, standardisation, missing flags, embeddings)
  head     dense vector -> denial logit (MLP)

Keeping preprocessing in the encoder means the exported SavedModel accepts exactly what the API receives,
so training and serving cannot drift apart. Splitting encoder and head also gives integrated gradients a
continuous space to integrate over: attributions are computed on the encoded vector and summed back to the
input (or derived ratio) each slice came from.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from ldp.schema import ATTRIBUTION_GROUPS, CATEGORICAL, NUMERIC

MIN_CATEGORY_COUNT = 50  # rarer categories share the out-of-vocabulary bucket
CLIP_Z = 6.0


# --- Transforms. One numpy and one TensorFlow version each; tests check they agree.


def transform_np(name: str, x: np.ndarray) -> np.ndarray:
    if name in ("loan_amount", "property_value"):
        return np.log1p(np.maximum(x, 0.0))
    if name == "income":  # thousands of USD, can be negative
        return np.sign(x) * np.log1p(np.abs(x))
    if name == "debt_to_income_ratio":
        return x / 100.0
    if name == "loan_term":
        return x / 360.0
    raise KeyError(name)


def ratios_np(amount: np.ndarray, value: np.ndarray, income: np.ndarray) -> dict[str, np.ndarray]:
    with np.errstate(divide="ignore", invalid="ignore"):
        ltv = np.where((value > 0) & np.isfinite(amount), np.clip(amount / value, 0.0, 3.0), np.nan)
        lti = np.where(
            (income > 0) & np.isfinite(amount), np.log1p(np.clip(amount / (income * 1000.0), 0.0, 50.0)), np.nan
        )
    return {"loan_to_value": ltv, "loan_to_income": lti}


@dataclass
class FeatureStats:
    medians: dict[str, float] = field(default_factory=dict)
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    vocab: dict[str, list[str]] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> FeatureStats:
        return cls(**json.loads(text))

    @classmethod
    def fit(cls, table: pa.Table) -> FeatureStats:
        stats = cls()
        raw = {}
        for name in NUMERIC:
            values = table[name].to_numpy(zero_copy_only=False).astype(np.float64)
            present = values[np.isfinite(values)]
            stats.medians[name] = float(np.median(present)) if present.size else 0.0
            raw[name] = values
            t = transform_np(name, np.where(np.isfinite(values), values, stats.medians[name]))
            stats.means[name], stats.stds[name] = float(t.mean()), float(t.std() or 1.0)
        for name, values in ratios_np(raw["loan_amount"], raw["property_value"], raw["income"]).items():
            present = values[np.isfinite(values)]
            stats.means[name] = float(present.mean()) if present.size else 0.0
            stats.stds[name] = float(present.std() or 1.0) if present.size else 1.0
        # On small samples (tests, a single state) the floor scales down so no vocabulary ends up empty.
        floor = min(MIN_CATEGORY_COUNT, max(1, table.num_rows // 200))
        for name in CATEGORICAL:
            counts = pc.value_counts(pc.fill_null(pc.cast(table[name], pa.string()), "missing")).to_pylist()
            stats.vocab[name] = sorted(c["values"] for c in counts if c["counts"] >= floor)
        return stats


def embedding_dim(cardinality: int) -> int:
    return int(min(8, max(2, round(1.6 * cardinality**0.56))))


def group_widths(stats: FeatureStats) -> list[int]:
    """Width of each attribution group's slice of the encoded vector, in ATTRIBUTION_GROUPS order."""
    widths = []
    for g in ATTRIBUTION_GROUPS:
        if g.kind in ("numeric", "derived"):
            widths.append(2)  # standardised value + missing flag
        else:
            widths.append(embedding_dim(len(stats.vocab[g.name]) + 1))
    return widths


def segment_matrix(stats: FeatureStats) -> np.ndarray:
    widths = group_widths(stats)
    seg = np.zeros((sum(widths), len(widths)), dtype=np.float32)
    start = 0
    for j, w in enumerate(widths):
        seg[start : start + w, j] = 1.0
        start += w
    return seg


def input_signature() -> dict[str, Any]:
    import tensorflow as tf

    spec = {n: tf.TensorSpec(shape=(None,), dtype=tf.float32, name=n) for n in NUMERIC}
    spec.update({n: tf.TensorSpec(shape=(None,), dtype=tf.string, name=n) for n in CATEGORICAL})
    return spec


def build(stats: FeatureStats, hidden: tuple[int, ...] = (256, 128, 64), dropout: float = 0.2):
    """Return (full, encoder, head) Keras models."""
    import keras
    import tensorflow as tf

    class NumericBlock(keras.layers.Layer):
        """Impute, transform, standardise and flag each numeric input and the two derived ratios."""

        def __init__(self, stats_json: str, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.stats_json = stats_json
            self.s = FeatureStats.from_json(stats_json)

        def _z(self, name: str, t: tf.Tensor) -> tf.Tensor:
            z = (t - self.s.means[name]) / self.s.stds[name]
            return tf.clip_by_value(z, -CLIP_Z, CLIP_Z)

        def call(self, inputs: dict[str, tf.Tensor]) -> tf.Tensor:
            cols = []
            for name in NUMERIC:
                x = tf.cast(inputs[name], tf.float32)
                missing = tf.math.logical_not(tf.math.is_finite(x))
                filled = tf.where(missing, tf.constant(self.s.medians[name], tf.float32), x)
                if name in ("loan_amount", "property_value"):
                    t = tf.math.log1p(tf.maximum(filled, 0.0))
                elif name == "income":
                    t = tf.sign(filled) * tf.math.log1p(tf.abs(filled))
                elif name == "debt_to_income_ratio":
                    t = filled / 100.0
                else:
                    t = filled / 360.0
                cols += [self._z(name, t), tf.cast(missing, tf.float32)]
            amount = tf.cast(inputs["loan_amount"], tf.float32)
            value = tf.cast(inputs["property_value"], tf.float32)
            income = tf.cast(inputs["income"], tf.float32)
            ok_ltv = tf.math.logical_and(value > 0, tf.math.is_finite(amount))
            ltv = tf.clip_by_value(tf.math.divide_no_nan(amount, value), 0.0, 3.0)
            ok_lti = tf.math.logical_and(income > 0, tf.math.is_finite(amount))
            lti = tf.math.log1p(tf.clip_by_value(tf.math.divide_no_nan(amount, income * 1000.0), 0.0, 50.0))
            for name, ok, v in (("loan_to_value", ok_ltv, ltv), ("loan_to_income", ok_lti, lti)):
                z = tf.where(ok, self._z(name, v), tf.zeros_like(v))
                cols += [z, tf.cast(tf.math.logical_not(ok), tf.float32)]
            return tf.stack(cols, axis=-1)

        def get_config(self) -> dict[str, Any]:
            return {**super().get_config(), "stats_json": self.stats_json}

    inputs = {n: keras.Input(shape=(), dtype="float32", name=n) for n in NUMERIC}
    inputs.update({n: keras.Input(shape=(), dtype="string", name=n) for n in CATEGORICAL})

    numeric = NumericBlock(stats.to_json(), name="numeric_block")({n: inputs[n] for n in NUMERIC})
    # numeric_block emits numeric then derived groups, which is ATTRIBUTION_GROUPS order.
    parts = [numeric]
    for name in CATEGORICAL:
        vocab = stats.vocab[name]
        idx = keras.layers.StringLookup(vocabulary=vocab, num_oov_indices=1, name=f"lookup_{name}")(inputs[name])
        parts.append(keras.layers.Embedding(len(vocab) + 1, embedding_dim(len(vocab) + 1), name=f"embed_{name}")(idx))
    encoded = keras.layers.Concatenate(name="encoded")(parts)
    encoder = keras.Model(inputs, encoded, name="encoder")

    width = int(sum(group_widths(stats)))
    z_in = keras.Input(shape=(width,), name="encoded_in")
    h = z_in
    for i, units in enumerate(hidden):
        h = keras.layers.Dense(units, activation="relu", name=f"dense_{i}")(h)
        h = keras.layers.Dropout(dropout, name=f"dropout_{i}")(h)
    logit = keras.layers.Dense(1, name="logit")(h)
    head = keras.Model(z_in, logit, name="head")

    full = keras.Model(inputs, head(encoder(inputs)), name="decision_model")
    return full, encoder, head


def save_stats(stats: FeatureStats, path: Path) -> None:
    path.write_text(stats.to_json() + "\n", encoding="utf-8")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def logit(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))
