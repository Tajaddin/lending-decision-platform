"""Baselines on the same inputs and split: logistic regression and gradient-boosted trees (XGBoost).

Both see exactly the information the TensorFlow model sees: the numeric inputs, the two derived ratios and
the categorical inputs, with the same vocabularies. Each is calibrated on the validation year the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from ldp.schema import CATEGORICAL, NUMERIC
from ldp.training.model import FeatureStats, ratios_np, transform_np


def numeric_frame(table: pa.Table) -> dict[str, np.ndarray]:
    raw = {n: table[n].to_numpy(zero_copy_only=False).astype(np.float64) for n in NUMERIC}
    raw.update(ratios_np(raw["loan_amount"], raw["property_value"], raw["income"]))
    return raw


def categorical_frame(table: pa.Table, stats: FeatureStats) -> pd.DataFrame:
    out = {}
    for name in CATEGORICAL:
        values = pd.Series(table[name].to_pandas()).astype("string").fillna("missing")
        vocab = stats.vocab[name]
        out[name] = pd.Categorical(values.where(values.isin(vocab), "__oov__"), categories=[*vocab, "__oov__"])
    return pd.DataFrame(out)


def linear_design(table: pa.Table, stats: FeatureStats) -> np.ndarray:
    """The same standardisation and missing flags as the model's numeric block, plus one-hot categories."""
    raw = numeric_frame(table)
    cols = []
    for name in NUMERIC:
        x = raw[name]
        missing = ~np.isfinite(x)
        t = transform_np(name, np.where(missing, stats.medians[name], x))
        cols += [np.clip((t - stats.means[name]) / stats.stds[name], -6, 6), missing.astype(np.float64)]
    for name in ("loan_to_value", "loan_to_income"):
        x = raw[name]
        ok = np.isfinite(x)
        cols += [np.where(ok, (x - stats.means[name]) / stats.stds[name], 0.0), (~ok).astype(np.float64)]
    dense = np.column_stack(cols).astype(np.float32)
    onehot = pd.get_dummies(categorical_frame(table, stats), dtype=np.float32).to_numpy()
    return np.hstack([dense, onehot])


def tree_design(table: pa.Table, stats: FeatureStats) -> pd.DataFrame:
    df = pd.DataFrame(numeric_frame(table))
    return pd.concat([df, categorical_frame(table, stats)], axis=1)


@dataclass
class FittedBaseline:
    name: str
    model: Any
    params: dict[str, Any]


def fit_logistic(x_train: np.ndarray, y_train: np.ndarray) -> FittedBaseline:
    from sklearn.linear_model import LogisticRegression

    params = {"C": 1.0, "max_iter": 300, "solver": "lbfgs"}
    model = LogisticRegression(**params).fit(x_train, y_train)
    return FittedBaseline("logistic_regression", model, params)


def fit_xgboost(x_train: pd.DataFrame, y_train: np.ndarray, x_valid: pd.DataFrame, y_valid: np.ndarray,
                n_jobs: int = -1) -> FittedBaseline:
    import xgboost as xgb

    params = {
        "n_estimators": 2000, "learning_rate": 0.08, "max_depth": 8, "min_child_weight": 50, "subsample": 0.8,
        "colsample_bytree": 0.8, "tree_method": "hist", "enable_categorical": True, "max_cat_to_onehot": 1,
        "eval_metric": "auc", "early_stopping_rounds": 50, "n_jobs": n_jobs, "random_state": 7,
    }
    model = xgb.XGBClassifier(**params).fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
    params["best_iteration"] = int(model.best_iteration)
    return FittedBaseline("xgboost", model, params)


def predict_logit(fitted: FittedBaseline, x: Any) -> np.ndarray:
    if fitted.name == "logistic_regression":
        return fitted.model.decision_function(x).astype(np.float64)
    return fitted.model.predict(x, output_margin=True).astype(np.float64)
