"""Core-scaling benchmark for the silver stage: same year, same input, local[n] for several n.

Each run writes to a scratch Delta path, so the real silver table is untouched. One untimed warm-up run
first, so JVM start-up and the Delta jar resolution are not charged to the first core count. Each core count
then gets `repeats` timed runs (default 1: a 1-core run of a 13.5M-row year takes minutes).
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from ldp.config import Settings
from ldp.log import get_logger
from ldp.pipeline import host_info, write_json

log = get_logger(__name__)


def bench_silver(settings: Settings, year: int, cores: list[int], repeats: int = 1) -> dict[str, object]:
    from pyspark.sql import SparkSession

    from ldp.spark.session import get_spark
    from ldp.spark.silver import write_silver

    bronze = settings.lake("bronze")
    scratch = f"{settings.lake_root}/_bench_silver"
    runs = []
    for i, n in enumerate([max(cores), *cores]):  # warm-up on the most cores: same JVM work, shortest wait
        active = SparkSession.getActiveSession()
        if active is not None:
            active.stop()
        spark = get_spark(f"ldp-bench-{n}", master=f"local[{n}]")
        spark.conf.set("spark.sql.shuffle.partitions", str(max(4, n * 2)))
        seconds = []
        for _ in range(1 if i == 0 else repeats):
            shutil.rmtree(scratch, ignore_errors=True)
            start = time.perf_counter()
            rows = write_silver(spark, bronze, scratch, year)
            seconds.append(round(time.perf_counter() - start, 2))
        if i == 0:
            log.info("bench_warmup_done", cores=n, seconds=seconds[0])
            continue
        best = min(seconds)
        runs.append({"cores": n, "seconds": seconds, "best_s": best, "rows": rows, "rows_per_s": round(rows / best)})
        log.info("bench_run", cores=n, best_s=best)
    shutil.rmtree(scratch, ignore_errors=True)
    base = runs[0]["best_s"]
    for r in runs:
        r["speedup"] = round(float(base) / float(r["best_s"]), 2)  # type: ignore[arg-type]
    result = {
        "benchmark": "silver stage (typing and decoding one year of bronze into Delta)",
        "repeats_per_core_count": repeats,
        "year": year,
        "host": host_info(),
        "runs": runs,
    }
    write_json(result, Path(settings.reports_dir) / "spark-bench.json")
    return result
