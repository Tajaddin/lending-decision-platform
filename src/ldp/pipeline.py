"""The batch data pipeline end to end, with a timing for every stage (the benchmark reads these)."""

from __future__ import annotations

import json
import os
import platform
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ldp.config import Settings
from ldp.log import get_logger
from ldp.schema import LABEL

log = get_logger(__name__)


def raw_csv(raw_dir: str | Path, year: int) -> str:
    return f"{str(raw_dir).rstrip('/')}/{year}_public_lar_csv.csv"


@dataclass
class StageTimer:
    timings: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = round(time.perf_counter() - start, 2)
            log.info("stage_done", stage=name, seconds=self.timings[name])


def run_pipeline(
    settings: Settings,
    years: list[int],
    raw_dir: str | Path,
    master: str | None = None,
    audit_sample_rows: int = 600_000,
) -> dict[str, Any]:
    from ldp.audit import assert_features_clean, candidates, is_powered, run_audit, write_audit
    from ldp.spark import bronze, gold, quality, silver
    from ldp.spark.session import get_spark

    timer = StageTimer()
    spark = get_spark("ldp-pipeline", master=master)
    bronze_path, silver_path, gold_path = settings.lake("bronze"), settings.lake("silver"), settings.lake("gold")
    result: dict[str, Any] = {"years": years, "bronze": {}, "silver": {}}

    for year in years:
        with timer.stage(f"bronze_{year}"):
            b = bronze.write_bronze(spark, raw_csv(raw_dir, year), bronze_path, year)
            result["bronze"][year] = b.rows
        with timer.stage(f"silver_{year}"):
            result["silver"][year] = silver.write_silver(spark, bronze_path, silver_path, year)

    reports = []
    with timer.stage("quality"):
        from pyspark.sql import functions as F

        silver_df = spark.read.format("delta").load(silver_path)
        for year in years:
            checks = quality.silver_checks(bronze_rows=result["bronze"][year])
            reports.append(quality.run_checks(silver_df.where(F.col("activity_year") == year), year, checks))
        quality.write_report(reports, settings.reports_dir / "data-quality.json")
        quality.enforce(reports)

    with timer.stage("gold"):
        result["gold"] = gold.write_gold(spark, silver_path, gold_path, years)

    with timer.stage("leakage_audit"):
        from pyspark.sql import functions as F

        train = spark.read.format("delta").load(gold_path).where(F.col("activity_year") == settings.train_year)
        n = train.count()
        fraction = min(1.0, audit_sample_rows / max(n, 1))
        cols = [c for c in candidates() if c in train.columns]
        sample = train.select(*cols, LABEL).sample(fraction=fraction, seed=7).toPandas()
        audit = run_audit(sample)
        powered = is_powered(sample)
        write_audit(
            audit,
            rows=len(sample),
            year=settings.train_year,
            path=settings.reports_dir / "leakage-audit.json",
            enforced=powered,
        )
        result["leakage_flagged"] = [a.field for a in audit if a.flagged]
        result["leakage_audit_enforced"] = powered
        if powered:
            assert_features_clean(audit)
        else:
            log.warning("leakage_audit_underpowered", rows=len(sample))

    with timer.stage("export"):
        splits = {"train": settings.train_year, "valid": settings.valid_year, "test": settings.test_year}
        splits = {k: v for k, v in splits.items() if v in years}
        exports = gold.export_splits(spark, gold_path, str(settings.training_dir), splits)
        gold.write_manifest(exports, settings.reports_dir / "dataset.json")

    result["timings_s"] = timer.timings
    result["spark"] = {
        "master": spark.sparkContext.master,
        "version": spark.version,
        "default_parallelism": spark.sparkContext.defaultParallelism,
    }
    return result


def host_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpus": os.cpu_count(),
    }
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            info["mem_total_gb"] = round(int(fh.readline().split()[1]) / 1024 / 1024, 1)
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            info["cpu_model"] = next((ln.split(":", 1)[1].strip() for ln in fh if ln.startswith("model name")), "")
    except OSError:
        pass
    return info


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"generated_at": datetime.now(UTC).isoformat(timespec="seconds"), **payload}
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - start
