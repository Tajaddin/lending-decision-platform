"""Bronze: the LAR CSV exactly as published, one Delta partition per year, with file lineage.

Every column stays a string. Re-running a year overwrites only that year's partition (replaceWhere), so a
corrected file from the CFPB can be reloaded without touching the other years.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ldp.log import get_logger
from ldp.schema import normalize_column

log = get_logger(__name__)


@dataclass(frozen=True)
class BronzeResult:
    year: int
    source_file: str
    rows: int
    delta_version: int


def read_lar_csv(spark: SparkSession, csv_path: str) -> DataFrame:
    df = (
        spark.read.option("header", True)
        .option("inferSchema", False)
        .option("mode", "FAILFAST")  # a malformed row stops the load instead of becoming nulls
        .option("enforceSchema", False)
        .csv(csv_path)
    )
    # The national snapshot uses snake_case already; data-browser extracts use hyphens. Normalise both.
    return df.toDF(*[normalize_column(c) for c in df.columns])


def write_bronze(spark: SparkSession, csv_path: str, bronze_path: str, year: int) -> BronzeResult:
    df = read_lar_csv(spark, csv_path)
    if "activity_year" not in df.columns:
        raise ValueError(f"{csv_path} has no activity_year column; is it a LAR file?")
    source = PurePosixPath(csv_path.replace("\\", "/")).name
    df = df.withColumn("_source_file", F.lit(source)).withColumn("_ingested_at", F.current_timestamp())

    # A file whose rows are not all from the year it is loaded as would corrupt the partition.
    years = [r[0] for r in df.select("activity_year").distinct().limit(3).collect()]
    if years != [str(year)]:
        raise ValueError(f"{source}: activity_year values {years}, expected only {year}")

    (
        df.write.format("delta")
        .mode("overwrite")
        .option("replaceWhere", f"activity_year = '{year}'")
        .partitionBy("activity_year")
        .save(bronze_path)
    )
    table = DeltaTable.forPath(spark, bronze_path)
    last = table.history(1).select("version", "operationMetrics").collect()[0]
    rows = int(last["operationMetrics"].get("numOutputRows", "0"))
    log.info("bronze_written", year=year, source=source, rows=rows, version=last["version"])
    return BronzeResult(year=year, source_file=source, rows=rows, delta_version=int(last["version"]))
