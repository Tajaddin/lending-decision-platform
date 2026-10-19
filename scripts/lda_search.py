"""Less discriminatory alternative (LDA) search for model v1 (issue #27).

    LDP_TRAINING_DIR=~/data/hmda/training python scripts/lda_search.py search    # 2023 and 2024 only
    LDP_TRAINING_DIR=~/data/hmda/training python scripts/lda_search.py holdout   # 2025, once, chosen variant only

`search` trains each variant with the v1 recipe (same network, seed, optimiser and batch order; early stopping on
the same fixed 1M sample of 2024; Platt calibration and the 80%-recall threshold on all of 2024), audits it on all
of 2024 with `fairness.audit`, and picks one with `lda.choose`. 2025 is not loaded. `holdout` scores the chosen
variant on 2025 and refuses to run a second time.

Candidates: reweighing; the mean-logit equalized-odds penalty at several strengths, where lambda = 0 is the v1
recipe re-run as a control for the training path; and the referral-rate penalty at the operating point. The first
search (reweighing and the mean-logit runs) showed the mean-logit penalty levelling off for age and leaving American
Indian or Alaska Native applicants flagged (they were not in it), so the third variant was added, still on 2024
only. A pilot on 1M-row samples of 2023 and 2024 set its grid and its groups: lambda 8 collapsed training, and
with Pacific Islander applicants in the penalty training was unstable even at lambda 1.
The committed v1 SavedModel is scored on 2024 as the reference, with its 2024 audit at other recall targets as a
diagnostic: a single cut-off is the only threshold ADR 0006 allows.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ldp.config import get_settings  # noqa: E402
from ldp.training import evaluate, fairness, lda  # noqa: E402
from ldp.training.data import BatchStream, as_dataset, labels, load_split, to_numpy_batch  # noqa: E402
from ldp.training.model import FeatureStats, build, save_stats, sigmoid  # noqa: E402
from ldp.training.train import TARGET_RECALL, fit_platt, predict_logits, sample_table  # noqa: E402

MEAN_LOGIT_LAMBDAS = (0.0, 0.03, 0.1, 0.3, 1.0)
REFERRAL_RATE_LAMBDAS = (0.3, 1.0, 3.0)
RECALL_TARGETS = (0.70, 0.75, 0.80, 0.85)
PROTECTED_COLUMNS = ["derived_race", "derived_ethnicity", "derived_sex", "applicant_age_above_62"]


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def audit_frame(table: pa.Table, p: np.ndarray) -> Any:
    df = table.select(PROTECTED_COLUMNS).to_pandas()
    df["denied"] = labels(table).astype(int)
    df["probability"] = p
    return df


def assess(table: pa.Table, p: np.ndarray, threshold: float) -> dict[str, Any]:
    y = labels(table)
    s = evaluate.summary(y, p, threshold, with_ci=False)
    op = s["operating_point"]
    out = {
        "rows": table.num_rows,
        "auc": s["auc"],
        "ece": s["ece"],
        "threshold": threshold,
        "referral_rate": op["referral_rate"],
        "recall_of_denials": op["recall_of_denials"],
        "precision": op["precision"],
    }
    out.update(lda.audit_summary(fairness.audit(audit_frame(table, p), threshold)))
    return out


def score_saved_model(model_dir: Path, table: pa.Table, batch: int = 16384) -> np.ndarray:
    import tensorflow as tf

    serve = tf.saved_model.load(str(model_dir)).signatures["serving_default"]
    out = []
    for start in range(0, table.num_rows, batch):
        inputs = {k: tf.constant(v) for k, v in to_numpy_batch(table.slice(start, batch)).items()}
        out.append(serve(**inputs)["probability"].numpy())
    return np.concatenate(out).astype(np.float64)


def train_variant(
    stats: FeatureStats,
    train: pa.Table,
    valid_es: pa.Table,
    weights: np.ndarray | None = None,
    members: tuple[np.ndarray, np.ndarray] | None = None,
    penalised: Any = None,
) -> tuple[Any, Any]:
    """The v1 recipe from train.py with sample weights (reweighing) or the penalised loss (members given)."""
    import keras

    keras.backend.clear_session()  # the previous variant's weights are saved; free its graph
    keras.utils.set_random_seed(7)
    full, _, _ = build(stats)
    if weights is not None:
        train_ds = lda.as_aux_dataset(lda.AuxStream(train, weights, as_weights=True))
        valid_ds = as_dataset(BatchStream(valid_es, batch_size=16384, shuffle=False))
        loss: Any = keras.losses.BinaryCrossentropy(from_logits=True)
        metric: Any = keras.metrics.AUC(name="auc", from_logits=True)
    else:
        assert members is not None
        assert penalised is not None
        train_ds = lda.as_aux_dataset(lda.AuxStream(train, members[0], as_weights=False))
        valid_ds = lda.as_aux_dataset(
            lda.AuxStream(valid_es, members[1], as_weights=False, batch_size=16384, shuffle=False)
        )
        loss = penalised
        metric = lda.label_auc()
    full.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-3), loss=loss, metrics=[metric])
    history = full.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=12,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=2, restore_best_weights=True)],
        verbose=2,
    )
    return full, history


def air_by_group(summary: dict[str, Any]) -> dict[str, float]:
    return {f"{g['dimension']}:{g['group']}": g["model_air"] for g in summary["groups"]}


def search(settings: Any, work_dir: Path) -> None:
    start = time.perf_counter()
    root = Path(settings.training_dir).expanduser()
    train, valid = load_split(root / "train"), load_split(root / "valid")
    y_train, y_valid = labels(train), labels(valid)
    stats = FeatureStats.fit(train)
    work_dir.mkdir(parents=True, exist_ok=True)
    save_stats(stats, work_dir / "feature_stats.json")
    valid_es = sample_table(valid, 1_000_000, seed=3)  # the sample v1 early-stopped on

    # Reference: the committed v1 model on 2024, at its own threshold and at other recall targets.
    model_dir = Path(settings.models_dir) / settings.model_name / "1"
    meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    p_v1 = score_saved_model(model_dir, valid)
    reference = {"name": "v1 (committed SavedModel)", "valid_2024": assess(valid, p_v1, meta["threshold"])}
    print("v1", {k: reference["valid_2024"][k] for k in ("auc", "referral_rate", "worst_group", "worst_shortfall")})
    sensitivity = []
    for target in RECALL_TARGETS:
        thr = evaluate.threshold_for_recall(y_valid, p_v1, target)
        summary = lda.audit_summary(fairness.audit(audit_frame(valid, p_v1), thr))
        op = evaluate.operating_point(y_valid, p_v1, thr)
        sensitivity.append(
            {
                "recall_target": target,
                "threshold": thr,
                "referral_rate": op["referral_rate"],
                "recall_of_denials": op["recall_of_denials"],
                "flags": summary["flags"],
                "worst_group": summary["worst_group"],
                "worst_shortfall": summary["worst_shortfall"],
                "model_air": air_by_group(summary),
            }
        )

    variants: list[dict[str, Any]] = [
        {
            "name": "reweighing",
            "method": "Kamiran & Calders (2012) reweighing over age 62+ x race x sex cells",
            "params": {"cells": list(lda.CELL_COLUMNS)},
        }
    ]
    variants += [
        {
            "name": "v1 recipe re-run (lambda=0)" if lam == 0 else f"mean-logit penalty, lambda={lam:g}",
            "method": "cross-entropy + lambda x squared group gaps in mean logit, among approved and among denied",
            "params": {
                "penalty": "mean_logit",
                "lambda": lam,
                "comparisons": [list(c) for c in lda.PENALISED],
                "estimator": "batch mean",
            },
        }
        for lam in MEAN_LOGIT_LAMBDAS
    ]
    variants += [
        {
            "name": f"referral-rate penalty, lambda={lam:g}",
            "method": "cross-entropy + lambda x squared group gaps in soft referral rate at the batch's own "
            f"{TARGET_RECALL:.0%}-recall threshold, among approved and among denied",
            "params": {
                "penalty": "referral_rate",
                "lambda": lam,
                "comparisons": [list(c) for c in lda.PENALISED_RATE],
                "estimator": "split halves",
                "tau": lda.SOFT_TAU,
            },
        }
        for lam in REFERRAL_RATE_LAMBDAS
    ]
    members = {
        "mean_logit": (lda.membership(train, lda.PENALISED), lda.membership(valid_es, lda.PENALISED)),
        "referral_rate": (lda.membership(train, lda.PENALISED_RATE), lda.membership(valid_es, lda.PENALISED_RATE)),
    }
    candidates = []
    for v in variants:
        t0 = time.perf_counter()
        params = v["params"]
        if v["name"] == "reweighing":
            weights = lda.reweighing_weights(lda.protected_cells(train), y_train)
            params["weight_range"] = [float(weights.min()), float(weights.max())]
            full, history = train_variant(stats, train, valid_es, weights=weights)
        else:
            rate = params["penalty"] == "referral_rate"
            loss = lda.penalised_loss(params["lambda"], len(params["comparisons"]), split=rate, at_threshold=rate)
            full, history = train_variant(stats, train, valid_es, members=members[params["penalty"]], penalised=loss)
        train_seconds = time.perf_counter() - t0
        logit_valid = predict_logits(full, valid)
        calibration = fit_platt(logit_valid, y_valid)
        p_valid = sigmoid(calibration.a * logit_valid + calibration.b)
        threshold = evaluate.threshold_for_recall(y_valid, p_valid, TARGET_RECALL)
        weights_path = work_dir / (re.sub(r"[^a-z0-9.]+", "-", v["name"].lower()).strip("-") + ".weights.h5")
        full.save_weights(str(weights_path))
        candidate = {
            **v,
            "epochs_run": len(history.history["loss"]),
            "best_val_auc_sample": float(max(history.history["val_auc"])),
            "train_seconds": round(train_seconds, 1),
            "calibration": {"a": calibration.a, "b": calibration.b},
            "weights_file": weights_path.name,
            "valid_2024": assess(valid, p_valid, threshold),
        }
        candidates.append(candidate)
        print(
            v["name"],
            {k: candidate["valid_2024"][k] for k in ("auc", "referral_rate", "worst_group", "worst_shortfall")},
        )
        print("   ", {k: round(a, 3) for k, a in air_by_group(candidate["valid_2024"]).items()}, flush=True)

    chosen = lda.choose(candidates)
    out = {
        "issue": 27,
        "generated_at": now(),
        "protocol": {
            "train": f"all of {settings.train_year} ({train.num_rows:,} applications), v1 recipe",
            "early_stopping": f"the fixed 1,000,000-row sample of {settings.valid_year} v1 used (seed 3), val AUC",
            "calibration_and_threshold": f"Platt scaling and the {TARGET_RECALL:.0%}-recall threshold on all of "
            f"{settings.valid_year} ({valid.num_rows:,} applications)",
            "selection": f"fairness.audit on all of {settings.valid_year}; {lda.SELECTION_RULE}",
            "holdout": f"{settings.test_year} scored once, for the chosen variant only (`lda_search.py holdout`)",
            "inputs": "the 15 v1 inputs; protected attributes enter only the loss (ADR 0010)",
        },
        "reference": reference,
        "operating_rule_sensitivity_v1": sensitivity,
        "candidates": candidates,
        "chosen": chosen,
        "seconds": round(time.perf_counter() - start, 1),
    }
    path = Path(settings.reports_dir) / "lda-search.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("chosen", chosen, f"wrote {path}")


def holdout(settings: Any, work_dir: Path, force: bool) -> None:
    path = Path(settings.reports_dir) / "lda-search.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if "holdout_2025" in report and not force:
        sys.exit(f"{settings.test_year} was already scored for this search at {report['holdout_2025']['scored_at']}")
    chosen = next(c for c in report["candidates"] if c["name"] == report["chosen"]["name"])
    stats = FeatureStats.from_json((work_dir / "feature_stats.json").read_text(encoding="utf-8"))
    full, _, _ = build(stats)
    full.load_weights(str(work_dir / chosen["weights_file"]))
    test = load_split(Path(settings.training_dir).expanduser() / "test")
    y = labels(test)
    cal = chosen["calibration"]
    p = sigmoid(cal["a"] * predict_logits(full, test) + cal["b"])
    threshold = chosen["valid_2024"]["threshold"]
    s = evaluate.summary(y, p, threshold)
    summary = lda.audit_summary(fairness.audit(audit_frame(test, p), threshold))

    reports = Path(settings.reports_dir)
    v1_metrics = json.loads((reports / "model-metrics.json").read_text(encoding="utf-8"))["models"]["tensorflow"]
    v1_fair = json.loads((reports / "fairness.json").read_text(encoding="utf-8"))
    report["holdout_2025"] = {
        "scored_at": now(),
        "variant": chosen["name"],
        "rows": test.num_rows,
        "auc": s["auc"],
        "auc_ci95": s["auc_ci95"],
        "ks": s["ks"],
        "brier": s["brier"],
        "ece": s["ece"],
        "threshold": threshold,
        **{k: s["operating_point"][k] for k in ("referral_rate", "recall_of_denials", "precision")},
        **summary,
        "v1": {
            "source": "reports/model-metrics.json and reports/fairness.json (committed, not re-scored)",
            "auc": v1_metrics["test"]["auc"],
            "referral_rate": v1_metrics["test"]["operating_point"]["referral_rate"],
            **lda.audit_summary(v1_fair),
        },
    }
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(chosen["name"], {k: report["holdout_2025"][k] for k in ("auc", "referral_rate", "flags", "worst_shortfall")})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("phase", choices=["search", "holdout"])
    parser.add_argument("--work-dir", type=Path, default=None, help="weights of every variant (default: <data>/lda)")
    parser.add_argument("--force", action="store_true", help="score 2025 again (the report keeps the new time)")
    args = parser.parse_args()
    settings = get_settings()
    work_dir = args.work_dir or Path(settings.training_dir).expanduser().parent / "lda"
    if args.phase == "search":
        search(settings, work_dir)
    else:
        holdout(settings, work_dir, args.force)


if __name__ == "__main__":
    main()
