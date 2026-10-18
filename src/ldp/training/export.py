"""Export the trained model as a TF Serving SavedModel with two signatures.

serving_default  inputs -> calibrated probability of denial and the raw logit
explain          inputs -> the same, plus integrated-gradient attributions per ATTRIBUTION_GROUPS entry

Integrated gradients (Sundararajan et al., 2017) run on the encoded vector: the path goes from a reference
applicant (the mean encoded vector of approved training applications) to the applicant, and the attributions
add up to logit(applicant) - logit(reference). Positive attribution pushes toward denial; the API turns the
largest positive ones into adverse-action reasons.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ldp.schema import ATTRIBUTION_GROUPS
from ldp.training.model import FeatureStats, input_signature, segment_matrix

IG_STEPS = 64


@dataclass
class Calibration:
    """Platt scaling fitted on the validation year: p = sigmoid(a * logit + b)."""

    a: float = 1.0
    b: float = 0.0


@dataclass
class ModelMetadata:
    model_name: str
    version: int
    trained_at: str
    threshold: float
    threshold_rule: str
    calibration: Calibration
    attribution_groups: list[dict[str, str]]
    baseline_logit: float
    train_rows: int
    train_year: int
    valid_year: int
    gold_delta_version: int | None
    reference_bins: dict[str, Any]
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_saved_model(
    encoder: Any, head: Any, stats: FeatureStats, baseline: np.ndarray, calibration: Calibration, out_dir: Path
) -> None:
    import keras
    import tensorflow as tf

    seg = tf.constant(segment_matrix(stats))
    base = tf.constant(baseline.astype(np.float32))
    a = tf.constant(calibration.a, tf.float32)
    b = tf.constant(calibration.b, tf.float32)

    def serve(inputs: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        logit = head(encoder(inputs, training=False), training=False)[:, 0]
        return {"probability": tf.sigmoid(a * logit + b), "logit": logit}

    def explain(inputs: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        e = encoder(inputs, training=False)
        diff = e - base[None, :]
        alphas = tf.linspace(0.0, 1.0, IG_STEPS + 1)
        path = base[None, None, :] + alphas[:, None, None] * diff[None, :, :]
        flat = tf.reshape(path, [-1, tf.shape(e)[1]])
        with tf.GradientTape() as tape:
            tape.watch(flat)
            out = head(flat, training=False)
        grads = tf.reshape(tape.gradient(out, flat), [IG_STEPS + 1, -1, tf.shape(e)[1]])
        avg = tf.reduce_mean((grads[:-1] + grads[1:]) / 2.0, axis=0)  # trapezoid rule over the path
        attributions = tf.matmul(diff * avg, seg)
        logit = head(e, training=False)[:, 0]
        base_logit = head(base[None, :], training=False)[0, 0]
        return {
            "probability": tf.sigmoid(a * logit + b),
            "logit": logit,
            "attributions": attributions,
            "baseline_logit": tf.fill(tf.shape(logit), base_logit),
        }

    archive = keras.export.ExportArchive()
    archive.track(encoder)
    archive.track(head)
    spec = input_signature()
    archive.add_endpoint(name="serving_default", fn=serve, input_signature=[spec])
    archive.add_endpoint(name="explain", fn=explain, input_signature=[spec])
    archive.write_out(str(out_dir), verbose=False)


def write_metadata(meta: ModelMetadata, out_dir: Path) -> None:
    (out_dir / "metadata.json").write_text(json.dumps(meta.to_dict(), indent=2) + "\n", encoding="utf-8")


def attribution_groups() -> list[dict[str, str]]:
    return [
        {"name": g.name, "kind": g.kind, "reason": g.reason, "description": g.description} for g in ATTRIBUTION_GROUPS
    ]
