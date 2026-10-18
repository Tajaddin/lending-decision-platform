"""Train, calibrate, evaluate, audit and export the decision model.

Data use, strictly by year:
  2023 (train)  fit the feature statistics, the networks and the baselines
  2024 (valid)  early stopping, Platt calibration, the referral threshold, XGBoost early stopping
  2025 (test)   touched once, after every choice above is frozen: the reported numbers
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from ldp.config import Settings
from ldp.log import get_logger
from ldp.reasons import HMDA_DENIAL_REASONS, top_reasons
from ldp.schema import CATEGORICAL, FEATURE_NAMES, LABEL, NUMERIC
from ldp.training import evaluate, fairness
from ldp.training.baselines import fit_logistic, fit_xgboost, linear_design, predict_logit, tree_design
from ldp.training.data import BatchStream, as_dataset, labels, load_split, to_numpy_batch
from ldp.training.export import Calibration, ModelMetadata, attribution_groups, write_metadata, write_saved_model
from ldp.training.model import FeatureStats, build, save_stats, sigmoid

log = get_logger(__name__)

TARGET_RECALL = 0.80  # refer enough applications to catch 80% of what lenders denied (set on 2024)


def fit_platt(logits: np.ndarray, y: np.ndarray) -> Calibration:
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(C=1e6, max_iter=200).fit(logits.reshape(-1, 1), y)
    return Calibration(a=float(lr.coef_[0][0]), b=float(lr.intercept_[0]))


def predict_logits(full: Any, table: pa.Table, batch_size: int = 16384) -> np.ndarray:
    stream = BatchStream(table, batch_size=batch_size, shuffle=False)
    out = full.predict(as_dataset(stream).map(lambda x, y: x), verbose=0)
    return np.asarray(out, dtype=np.float64).reshape(-1)


def sample_table(table: pa.Table, n: int, seed: int) -> pa.Table:
    if table.num_rows <= n:
        return table
    idx = np.sort(np.random.default_rng(seed).choice(table.num_rows, n, replace=False))
    return table.take(pa.array(idx))


def reason_agreement(
    explain_fn: Any, groups: list[dict[str, str]], table: pa.Table, n: int = 20_000, seed: int = 5
) -> dict[str, Any]:
    """On denied 2025 applications whose lender-reported first reason is debt-to-income (1) or collateral (4),
    how often is that reason among the model's top reasons, and how often is it the top one?"""
    import tensorflow as tf

    denied = table.filter(pc.equal(table[LABEL], 1))
    codes = denied["denial_reason_1"].to_numpy(zero_copy_only=False).astype(str)
    keep = np.isin(codes, list(HMDA_DENIAL_REASONS))
    subset = sample_table(denied.filter(pa.array(keep)), n, seed)
    codes = subset["denial_reason_1"].to_numpy(zero_copy_only=False).astype(str)
    hits_top1 = hits_any = 0
    per_code: dict[str, dict[str, int]] = {c: {"n": 0, "top1": 0, "any": 0} for c in HMDA_DENIAL_REASONS}
    for start in range(0, subset.num_rows, 512):
        part = subset.slice(start, 512)
        batch = {k: tf.constant(v) for k, v in to_numpy_batch(part).items()}
        attributions = explain_fn(**batch)["attributions"].numpy()
        for row, code in zip(attributions, codes[start : start + 512], strict=True):
            reasons = [r.reason for r in top_reasons(row.tolist(), groups)]
            want = HMDA_DENIAL_REASONS[code]
            top1 = bool(reasons) and reasons[0] == want
            anyhit = want in reasons
            hits_top1 += top1
            hits_any += anyhit
            per_code[code]["n"] += 1
            per_code[code]["top1"] += top1
            per_code[code]["any"] += anyhit
    total = max(subset.num_rows, 1)
    return {
        "applications": subset.num_rows,
        "lender_reason_is_top_model_reason": hits_top1 / total,
        "lender_reason_in_model_top_4": hits_any / total,
        "by_lender_reason": {HMDA_DENIAL_REASONS[c]: v for c, v in per_code.items()},
    }


def train_and_evaluate(
    settings: Settings, epochs: int = 12, batch_size: int = 4096, version: int = 1, skip_baselines: bool = False
) -> dict[str, Any]:
    import keras
    import tensorflow as tf

    keras.utils.set_random_seed(7)
    t0 = time.perf_counter()
    root = Path(settings.training_dir)
    train, valid, test = (load_split(root / s) for s in ("train", "valid", "test"))
    log.info("data_loaded", train=train.num_rows, valid=valid.num_rows, test=test.num_rows)
    y_train, y_valid, y_test = labels(train), labels(valid), labels(test)

    stats = FeatureStats.fit(train)
    full, encoder, head = build(stats)
    full.compile(
        optimizer=keras.optimizers.Adam(learning_rate=2e-3),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.AUC(name="auc", from_logits=True)],
    )
    valid_es = sample_table(valid, 1_000_000, seed=3)  # epoch-end check on a fixed 1M sample of 2024
    history = full.fit(
        as_dataset(BatchStream(train, batch_size=batch_size)),
        validation_data=as_dataset(BatchStream(valid_es, batch_size=16384, shuffle=False)),
        epochs=epochs,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=2, restore_best_weights=True)],
        verbose=2,
    )
    train_seconds = time.perf_counter() - t0

    logit_valid, logit_test = predict_logits(full, valid), predict_logits(full, test)
    calibration = fit_platt(logit_valid, y_valid)
    p_valid = sigmoid(calibration.a * logit_valid + calibration.b)
    p_test = sigmoid(calibration.a * logit_test + calibration.b)
    threshold = evaluate.threshold_for_recall(y_valid, p_valid, TARGET_RECALL)

    metrics: dict[str, Any] = {
        "tensorflow": {
            "valid": evaluate.summary(y_valid, p_valid, threshold),
            "test": evaluate.summary(y_test, p_test, threshold),
            "test_uncalibrated_ece": evaluate.ece(y_test, sigmoid(logit_test)),
            "epochs_run": len(history.history["loss"]),
            "best_val_auc_sample": float(max(history.history["val_auc"])),
            "train_seconds": round(train_seconds, 1),
        }
    }

    if not skip_baselines:
        x_lin = linear_design(train, stats)
        lr = fit_logistic(x_lin, y_train)
        del x_lin
        lr_valid, lr_test = (
            predict_logit(lr, linear_design(valid, stats)),
            predict_logit(lr, linear_design(test, stats)),
        )
        x_tree_train, x_tree_valid = tree_design(train, stats), tree_design(valid, stats)
        t1 = time.perf_counter()
        xgb = fit_xgboost(x_tree_train, y_train, x_tree_valid, y_valid)
        xgb_seconds = time.perf_counter() - t1
        xgb_valid, xgb_test = predict_logit(xgb, x_tree_valid), predict_logit(xgb, tree_design(test, stats))
        for name, lv, lt, fitted in (
            ("logistic_regression", lr_valid, lr_test, lr),
            ("xgboost", xgb_valid, xgb_test, xgb),
        ):
            cal = fit_platt(lv, y_valid)
            pv, pt = sigmoid(cal.a * lv + cal.b), sigmoid(cal.a * lt + cal.b)
            thr = evaluate.threshold_for_recall(y_valid, pv, TARGET_RECALL)
            metrics[name] = {
                "valid": evaluate.summary(y_valid, pv, thr),
                "test": evaluate.summary(y_test, pt, thr),
                "params": fitted.params,
            }
        metrics["xgboost"]["train_seconds"] = round(xgb_seconds, 1)

    # Fair-lending audit on the 2025 hold-out at the operating threshold.
    protected = test.select(["derived_race", "derived_ethnicity", "derived_sex", "applicant_age_above_62"]).to_pandas()
    fair_df = pd.concat([protected, pd.DataFrame({"denied": y_test.astype(int), "probability": p_test})], axis=1)
    fairness_report = fairness.audit(fair_df, threshold)

    # Population stability 2023 -> 2024 -> 2025, and the reference bins the live monitor compares against.
    reference: dict[str, Any] = {}
    psi_report: dict[str, Any] = {"features": {}}
    for name in FEATURE_NAMES:
        kind = "numeric" if name in NUMERIC else "categorical"
        tr = train[name].to_numpy(zero_copy_only=False)
        if kind == "categorical":
            tr = tr.astype(str)
        ref = evaluate.reference_bins(tr, kind)
        reference[name] = ref
        psi_report["features"][name] = {
            split: evaluate.psi(
                np.asarray(ref["expected"]),
                evaluate.histogram(
                    (
                        tbl[name].to_numpy(zero_copy_only=False).astype(str)
                        if kind == "categorical"
                        else tbl[name].to_numpy(zero_copy_only=False)
                    ),
                    ref,
                ),
            )
            for split, tbl in (("valid_2024", valid), ("test_2025", test))
        }
    train_sample = sample_table(train, 1_000_000, seed=9)
    p_train_sample = sigmoid(calibration.a * predict_logits(full, train_sample) + calibration.b)
    reference["score"] = evaluate.reference_bins(p_train_sample, "numeric")
    psi_report["score"] = {
        "valid_2024": evaluate.psi(
            np.asarray(reference["score"]["expected"]), evaluate.histogram(p_valid, reference["score"])
        ),
        "test_2025": evaluate.psi(
            np.asarray(reference["score"]["expected"]), evaluate.histogram(p_test, reference["score"])
        ),
    }

    # Integrated-gradient reference: the average encoded approved applicant in 2023.
    approved = train_sample.filter(pc.equal(train_sample[LABEL], 0))
    baseline = encoder.predict(
        as_dataset(BatchStream(approved, batch_size=16384, shuffle=False)).map(lambda x, y: x), verbose=0
    ).mean(axis=0)

    out_dir = Path(settings.models_dir) / settings.model_name / str(version)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    write_saved_model(encoder, head, stats, baseline, calibration, out_dir)
    save_stats(stats, out_dir / "feature_stats.json")
    loaded = tf.saved_model.load(str(out_dir))
    groups = attribution_groups()
    agreement = reason_agreement(loaded.signatures["explain"], groups, test)

    dataset = json.loads((Path(settings.reports_dir) / "dataset.json").read_text(encoding="utf-8"))
    meta = ModelMetadata(
        model_name=settings.model_name,
        version=version,
        trained_at=datetime.now(UTC).isoformat(timespec="seconds"),
        threshold=threshold,
        threshold_rule=f"refer when p(denial) >= threshold; threshold set on {settings.valid_year} to catch "
        f"{TARGET_RECALL:.0%} of denials",
        calibration=calibration,
        attribution_groups=groups,
        baseline_logit=float(head(baseline[None, :]).numpy()[0, 0]),
        train_rows=train.num_rows,
        train_year=settings.train_year,
        valid_year=settings.valid_year,
        gold_delta_version=dataset["splits"][0]["gold_delta_version"],
        reference_bins=reference,
        metrics={"test_auc": metrics["tensorflow"]["test"]["auc"], "test_ece": metrics["tensorflow"]["test"]["ece"]},
    )
    write_metadata(meta, out_dir)

    reports = Path(settings.reports_dir)
    payload = {
        "generated_at": meta.trained_at,
        "splits": {"train": settings.train_year, "valid": settings.valid_year, "test": settings.test_year},
        "features": list(FEATURE_NAMES),
        "categorical": list(CATEGORICAL),
        "calibration": asdict(calibration),
        "threshold": threshold,
        "target_recall": TARGET_RECALL,
        "models": metrics,
        "reason_agreement": agreement,
        "training_history": {k: [float(v) for v in vs] for k, vs in history.history.items()},
    }
    (reports / "model-metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (reports / "fairness.json").write_text(json.dumps(fairness_report, indent=2) + "\n", encoding="utf-8")
    (reports / "psi.json").write_text(json.dumps(psi_report, indent=2) + "\n", encoding="utf-8")
    log.info("model_exported", path=str(out_dir), test_auc=metrics["tensorflow"]["test"]["auc"])
    return payload
