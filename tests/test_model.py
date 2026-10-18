"""The TensorFlow model: in-graph preprocessing, export signatures, and integrated-gradient completeness."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from ldp.schema import ATTRIBUTION_GROUPS, CATEGORICAL, LABEL, NUMERIC

tf = pytest.importorskip("tensorflow")
pytestmark = pytest.mark.tf


def synthetic_table(n: int = 3000, seed: int = 0) -> pa.Table:
    rng = np.random.default_rng(seed)
    dti = rng.uniform(10, 65, n)
    income = rng.lognormal(4.6, 0.5, n)
    amount = rng.lognormal(12.4, 0.5, n)
    value = amount / rng.uniform(0.5, 1.0, n)
    logit = -2.0 + 0.08 * (dti - 36) + 0.6 * (amount / (income * 1000) > 4)
    denied = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.int32)
    income[rng.random(n) < 0.02] = np.nan
    cols = {
        "loan_amount": amount,
        "income": income,
        "property_value": value,
        "debt_to_income_ratio": dti,
        "loan_term": rng.choice([180.0, 360.0], n),
    }
    for name in CATEGORICAL:
        cols[name] = rng.choice(["1", "2", "3"], n).astype(object)
    cols[LABEL] = denied
    return pa.table(cols)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    from ldp.training.data import BatchStream, as_dataset
    from ldp.training.model import FeatureStats, build

    table = synthetic_table()
    stats = FeatureStats.fit(table)
    full, encoder, head = build(stats, hidden=(16,), dropout=0.0)
    full.compile(optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
    full.fit(as_dataset(BatchStream(table, batch_size=256)), epochs=2, verbose=0)
    return table, stats, encoder, head


def test_numpy_and_graph_transforms_agree(trained) -> None:
    from ldp.training.data import to_numpy_batch
    from ldp.training.model import ratios_np, transform_np

    table, stats, encoder, _ = trained
    batch = to_numpy_batch(table.slice(0, 64))
    encoded = encoder(batch).numpy()
    for i, name in enumerate(NUMERIC):
        x = batch[name].astype(np.float64)
        filled = np.where(np.isfinite(x), x, stats.medians[name])
        z = np.clip((transform_np(name, filled) - stats.means[name]) / stats.stds[name], -6, 6)
        np.testing.assert_allclose(encoded[:, 2 * i], z, rtol=1e-4, atol=1e-4)
        np.testing.assert_array_equal(encoded[:, 2 * i + 1], (~np.isfinite(x)).astype(np.float32))
    ratios = ratios_np(
        batch["loan_amount"].astype(np.float64),
        batch["property_value"].astype(np.float64),
        batch["income"].astype(np.float64),
    )
    j = 2 * len(NUMERIC)
    ltv = ratios["loan_to_value"]
    expected = np.where(np.isfinite(ltv), (ltv - stats.means["loan_to_value"]) / stats.stds["loan_to_value"], 0.0)
    np.testing.assert_allclose(encoded[:, j], expected, rtol=1e-4, atol=1e-4)


def test_export_signatures_and_attribution_completeness(trained, tmp_path: Path) -> None:
    from ldp.training.data import to_numpy_batch
    from ldp.training.export import Calibration, write_saved_model

    table, stats, encoder, head = trained
    baseline = encoder(to_numpy_batch(table.slice(0, 500))).numpy().mean(axis=0)
    write_saved_model(encoder, head, stats, baseline, Calibration(a=1.1, b=-0.2), tmp_path / "1")

    loaded = tf.saved_model.load(str(tmp_path / "1"))
    assert set(loaded.signatures) == {"serving_default", "explain"}
    names = set(loaded.signatures["serving_default"].structured_input_signature[1])
    assert names == set(NUMERIC) | set(CATEGORICAL)

    batch = {k: tf.constant(v) for k, v in to_numpy_batch(table.slice(100, 20)).items()}
    served = loaded.signatures["serving_default"](**batch)
    explained = loaded.signatures["explain"](**batch)
    np.testing.assert_allclose(served["probability"].numpy(), explained["probability"].numpy(), rtol=1e-5)
    np.testing.assert_allclose(
        served["probability"].numpy(), 1 / (1 + np.exp(-(1.1 * served["logit"].numpy() - 0.2))), rtol=1e-5
    )
    attributions = explained["attributions"].numpy()
    assert attributions.shape == (20, len(ATTRIBUTION_GROUPS))
    gap = explained["logit"].numpy() - explained["baseline_logit"].numpy()
    # Completeness: attributions add up to the logit difference from the reference applicant, up to the
    # Riemann-sum error of a finite path (ReLU kinks); 64 steps keeps it within 0.02 + 5% of the gap.
    assert np.all(np.abs(attributions.sum(axis=1) - gap) <= 0.02 + 0.05 * np.abs(gap))


def test_unknown_categories_and_missing_numbers_are_served(trained, tmp_path: Path) -> None:
    from ldp.training.export import Calibration, write_saved_model

    _, stats, encoder, head = trained
    write_saved_model(
        encoder,
        head,
        stats,
        np.zeros(sum(1 for _ in [0]) or 1, dtype=np.float32).repeat(encoder.output_shape[-1]),
        Calibration(),
        tmp_path / "1",
    )
    loaded = tf.saved_model.load(str(tmp_path / "1"))
    request = {n: tf.constant([float("nan")], tf.float32) for n in NUMERIC}
    request.update({n: tf.constant([b"never-seen"]) for n in CATEGORICAL})
    p = loaded.signatures["serving_default"](**request)["probability"].numpy()
    assert p.shape == (1,) and 0.0 < p[0] < 1.0
