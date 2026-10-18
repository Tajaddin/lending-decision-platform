"""Where a layer lives: a Delta path (laptop, CI) or a Unity Catalog table (Databricks).

LDP_LAKE_ROOT=data/lake                  -> data/lake/silver          (path-based Delta)
LDP_LAKE_ROOT=uc:workspace.lending       -> workspace.lending.silver  (managed table, saveAsTable)

The jobs call these helpers instead of .save()/.load(), so the same code runs in both places.
"""

from __future__ import annotations

from typing import Any

from pyspark.sql import DataFrame, SparkSession

UC_PREFIX = "uc:"


def is_table(target: str) -> bool:
    return target.startswith(UC_PREFIX)


def name(target: str) -> str:
    return target[len(UC_PREFIX) :] if is_table(target) else target


def read(spark: SparkSession, target: str, version: int | None = None) -> DataFrame:
    if is_table(target):
        if version is not None:
            return spark.read.option("versionAsOf", version).table(name(target))
        return spark.read.table(name(target))
    reader = spark.read.format("delta")
    if version is not None:
        reader = reader.option("versionAsOf", version)
    return reader.load(target)


def write(df: DataFrame, target: str, replace_where: str | None = None, partition_by: str | None = None) -> None:
    w = df.write.format("delta").mode("overwrite")
    if replace_where:
        w = w.option("replaceWhere", replace_where)
    if partition_by:
        w = w.partitionBy(partition_by)
    if is_table(target):
        w.saveAsTable(name(target))
    else:
        w.save(target)


def delta(spark: SparkSession, target: str) -> Any:
    from delta.tables import DeltaTable

    if is_table(target):
        return DeltaTable.forName(spark, name(target))
    return DeltaTable.forPath(spark, target)


def last_operation(spark: SparkSession, target: str) -> dict[str, Any]:
    row = delta(spark, target).history(1).select("version", "operationMetrics").collect()[0]
    return {"version": int(row["version"]), "metrics": dict(row["operationMetrics"] or {})}
