"""Gold: the model population, the label, and the out-of-time training exports.

Population: closed-end, first-lien, 1-4 unit, non-business, non-reverse home purchase and refinance
applications that reached a credit decision. Label: denied (action_taken 3) vs approved (1 originated,
2 approved but not accepted). Withdrawn, incomplete and purchased loans have no decision and are left out.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ldp.log import get_logger
from ldp.schema import (
    DECISION_ACTIONS,
    DENIED_ACTION,
    EXCLUDED_FOR_LEAKAGE,
    FEATURE_NAMES,
    GEOGRAPHY,
    LABEL,
    POPULATION_PURPOSES,
    POPULATION_UNITS,
    POST_DECISION,
    PROTECTED,
    TRACT_DEMOGRAPHICS,
)

log = get_logger(__name__)

# What the training export carries: features, label, and what the fairness audit and the reports need.
EXPORT_COLUMNS = (
    *FEATURE_NAMES,
    LABEL,
    *PROTECTED,
    "tract_minority_population_percent",
    "lei",
    "state_code",
    "activity_year",
    "record_hash",
)


def population(silver: DataFrame) -> DataFrame:
    return silver.where(
        F.col("action_taken").isin(*DECISION_ACTIONS)
        & (F.col("lien_status") == "1")
        & F.col("loan_purpose").isin(*POPULATION_PURPOSES)
        & (F.col("open_end_line_of_credit") == "2")
        & (F.col("reverse_mortgage") == "2")
        & (F.col("business_or_commercial_purpose") == "2")
        & F.col("total_units").isin(*POPULATION_UNITS)
    ).withColumn(LABEL, (F.col("action_taken") == DENIED_ACTION).cast("int"))


def gold_columns() -> list[str]:
    keep = [
        *FEATURE_NAMES,
        LABEL,
        "action_taken",
        *EXCLUDED_FOR_LEAKAGE,
        *POST_DECISION,
        *PROTECTED,
        *TRACT_DEMOGRAPHICS,
        *GEOGRAPHY,
        "lei",
        "activity_year",
        "record_hash",
    ]
    return list(dict.fromkeys(keep))  # de-duplicate, keep order


def write_gold(spark: SparkSession, silver_path: str, gold_path: str, years: list[int]) -> dict[int, dict[str, float]]:
    silver = spark.read.format("delta").load(silver_path).where(F.col("activity_year").isin(*years))
    gold = population(silver).select(*gold_columns())
    gold.write.format("delta").mode("overwrite").partitionBy("activity_year").save(gold_path)
    stats = (
        spark.read.format("delta")
        .load(gold_path)
        .groupBy("activity_year")
        .agg(F.count(F.lit(1)).alias("rows"), F.avg(LABEL).alias("denial_rate"))
        .collect()
    )
    by_year = {int(r["activity_year"]): {"rows": int(r["rows"]), "denial_rate": float(r["denial_rate"])} for r in stats}
    log.info("gold_written", years=by_year)
    return by_year


@dataclass
class SplitExport:
    split: str
    year: int
    rows: int
    denial_rate: float
    path: str
    gold_delta_version: int


def export_splits(
    spark: SparkSession, gold_path: str, out_dir: str, splits: dict[str, int], files_per_split: int = 8
) -> list[SplitExport]:
    """Write Parquet per split for TensorFlow. The gold Delta version goes into the manifest, so the model
    card can name the exact table version the model was trained on (Delta time travel can rebuild it)."""
    version = int(DeltaTable.forPath(spark, gold_path).history(1).select("version").collect()[0][0])
    gold = spark.read.format("delta").option("versionAsOf", version).load(gold_path)
    exports = []
    for split, year in splits.items():
        part = gold.where(F.col("activity_year") == year).select(*EXPORT_COLUMNS)
        path = f"{out_dir.rstrip('/')}/{split}"
        part.repartition(files_per_split).write.mode("overwrite").parquet(path)
        agg = spark.read.parquet(path).agg(F.count(F.lit(1)).alias("n"), F.avg(LABEL).alias("p")).collect()[0]
        exports.append(SplitExport(split, year, int(agg["n"]), float(agg["p"] or 0.0), path, version))
        log.info("split_exported", split=split, year=year, rows=int(agg["n"]))
    return exports


def write_manifest(exports: list[SplitExport], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"splits": [asdict(e) for e in exports]}, indent=2) + "\n", encoding="utf-8")
