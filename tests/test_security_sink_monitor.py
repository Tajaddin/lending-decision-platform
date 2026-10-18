from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ldp.monitoring.monitor import compute, export
from ldp.schema import CATEGORICAL, FEATURE_NAMES, NUMERIC
from ldp.serving.security import KeyStore, RateLimiter
from ldp.streaming.sink import to_row
from ldp.training.evaluate import reference_bins


def test_keystore_matches_only_exact_keys() -> None:
    store = KeyStore({"k" * 20: "branch", "j" * 20: "batch"})
    assert store.client_for("k" * 20) == "branch"
    assert store.client_for("k" * 19) is None
    assert store.client_for("") is None
    assert store.client_for(None) is None


def test_rate_limiter_refills_over_time() -> None:
    limiter = RateLimiter(per_minute=60)
    assert limiter.allow("a", cost=60, now=0.0) == (True, 0.0)
    allowed, wait = limiter.allow("a", now=0.0)
    assert not allowed and wait == pytest.approx(1.0)
    assert limiter.allow("a", now=1.0)[0]
    assert limiter.allow("b", now=0.0)[0]  # buckets are per client


EVENT = {
    "event_id": "e1",
    "application_id": "A-1",
    "client": "branch",
    "decided_at": "2026-10-18T07:00:00.000+00:00",
    "model_version": "1",
    "probability": 0.7,
    "decision": "refer",
    "threshold": 0.4,
    "reasons": "[]",
    "loan_amount": 1.0,
    "not_a_column": "dropped",
}


def test_sink_keeps_schema_columns_and_rejects_poison() -> None:
    row = to_row(json.dumps(EVENT).encode())
    assert "not_a_column" not in row and row["probability"] == 0.7
    with pytest.raises(ValueError, match="missing fields"):
        to_row(json.dumps({**EVENT, "decision": None}).encode())
    with pytest.raises(ValueError, match="JSON object"):
        to_row(b"[1, 2]")
    with pytest.raises(json.JSONDecodeError):
        to_row(b"not json")


def _frame(n: int, seed: int, shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {name: rng.lognormal(4 + shift, 1, n) for name in NUMERIC}
    data.update({name: rng.choice(["1", "2"], n) for name in CATEGORICAL})
    data["probability"] = np.clip(rng.beta(2, 6, n) + shift / 4, 0, 1)
    data["derived_race"] = rng.choice(["White", "Black or African American"], n, p=[0.8, 0.2])
    return pd.DataFrame(data)


def test_monitor_finds_no_drift_on_the_reference_population_and_finds_a_shift() -> None:
    ref_df = _frame(50_000, 1)
    reference = {
        n: reference_bins(
            ref_df[n].astype(str).to_numpy() if n in CATEGORICAL else ref_df[n].to_numpy(),
            "categorical" if n in CATEGORICAL else "numeric",
        )
        for n in FEATURE_NAMES
    }
    reference["score"] = reference_bins(ref_df["probability"].to_numpy(), "numeric")
    same = compute(_frame(20_000, 2), reference, threshold=0.4)
    assert max(same["psi"].values()) < 0.02 and same["score_psi"] < 0.02
    shifted = compute(_frame(20_000, 3, shift=0.8), reference, threshold=0.4)
    assert shifted["psi"]["loan_amount"] > 0.25 and shifted["score_psi"] > 0.10
    assert "race:Black or African American" in same["air"]
    assert same["air"]["race:Black or African American"] == pytest.approx(1.0, abs=0.08)
    export(same)  # sets the gauges without error
