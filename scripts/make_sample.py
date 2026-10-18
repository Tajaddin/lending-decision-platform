"""Commit-sized samples of the 2025 hold-out (public HMDA rows) for the quickstart, replay and k6.

    python scripts/make_sample.py ~/data/hmda/training/test

Writes data/sample/test/part-0.parquet (5,000 applications with their HMDA monitoring fields) and
bench/k6/applications.json (1,000 requests in the API's JSON shape).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ldp.monitoring.replay import MONITORING  # noqa: E402
from ldp.schema import CATEGORICAL, LABEL, NUMERIC  # noqa: E402


def main(split_dir: str) -> None:
    table = ds.dataset(split_dir, format="parquet").to_table(columns=[*NUMERIC, *CATEGORICAL, LABEL, *MONITORING])
    idx = np.sort(np.random.default_rng(2025).choice(table.num_rows, 5000, replace=False))
    sample = table.take(idx)
    out = ROOT / "data" / "sample" / "test"
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(sample, out / "part-0.parquet", compression="zstd")

    from ldp.monitoring.replay import load_applications

    apps = load_applications(out, 1000, seed=7)
    for app in apps:
        app.pop("monitoring", None)
        app.pop("application_id", None)
    k6 = ROOT / "bench" / "k6" / "applications.json"
    k6.parent.mkdir(parents=True, exist_ok=True)
    k6.write_text(json.dumps(apps) + "\n", encoding="utf-8", newline="\n")
    print(f"{out / 'part-0.parquet'}: {sample.num_rows} rows; {k6}: {len(apps)} requests")


if __name__ == "__main__":
    main(sys.argv[1])
