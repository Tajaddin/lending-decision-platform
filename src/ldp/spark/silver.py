"""Silver: typed, decoded LAR records.

Each decoder here is the Spark SQL twin of a Python function in ldp.schema (the API uses the Python one).
tests/test_silver.py runs both on the same raw strings and requires identical output, which is what keeps
training and serving from parsing a field two different ways.
"""

from __future__ import annotations

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F

from ldp.log import get_logger
from ldp.schema import DTI_BINS, SENTINELS, SILVER_CODES, SILVER_NUMERIC, SILVER_TEXT
from ldp.spark import lake

log = get_logger(__name__)

_SENTINELS = sorted(SENTINELS)


def _trimmed(name: str) -> Column:
    return F.trim(F.col(name))


def number(name: str) -> Column:
    """parse_number: sentinels and unparseable text become null; infinities too."""
    raw = _trimmed(name)
    value = F.expr(f"try_cast(trim(`{name}`) AS DOUBLE)")
    finite = ~F.isnan(value) & (F.abs(value) != F.lit(float("inf")))
    return F.when(raw.isNull() | raw.isin(*_SENTINELS), F.lit(None).cast("double")).when(finite, value).alias(name)


def dti(name: str = "debt_to_income_ratio") -> Column:
    """parse_dti: bins to midpoints, exact percents kept when in [0, 100]."""
    raw = _trimmed(name)
    value = F.expr(f"try_cast(trim(`{name}`) AS DOUBLE)")
    expr = F.when(raw.isNull() | raw.isin(*_SENTINELS), F.lit(None).cast("double"))
    for label, midpoint in DTI_BINS.items():
        expr = expr.when(raw == F.lit(label), F.lit(midpoint))
    return expr.when(value.between(0.0, 100.0), value).alias(name)


def code(name: str) -> Column:
    """parse_code: a category string; null or blank becomes "missing"."""
    raw = _trimmed(name)
    return F.when(raw.isNull() | (raw == F.lit("")), F.lit("missing")).otherwise(raw).alias(name)


def text(name: str) -> Column:
    raw = _trimmed(name)
    return F.when(raw.isNull() | (raw == F.lit("")), F.lit("missing")).otherwise(raw).alias(name)


def to_silver(bronze: DataFrame) -> DataFrame:
    numeric = [dti() if c == "debt_to_income_ratio" else number(c) for c in SILVER_NUMERIC]
    raw_columns = [c for c in bronze.columns if not c.startswith("_")]
    return bronze.select(
        F.col("activity_year").cast("int").alias("activity_year"),
        F.xxhash64(*[F.coalesce(F.col(c), F.lit("")) for c in raw_columns]).alias("record_hash"),
        *numeric,
        F.col("debt_to_income_ratio").alias("debt_to_income_ratio_raw"),
        *[code(c) for c in SILVER_CODES],
        *[text(c) for c in SILVER_TEXT],
        F.col("_source_file"),
    )


def write_silver(spark: SparkSession, bronze_path: str, silver_path: str, year: int) -> int:
    bronze = lake.read(spark, bronze_path).where(F.col("activity_year") == F.lit(str(year)))
    lake.write(to_silver(bronze), silver_path, replace_where=f"activity_year = {year}", partition_by="activity_year")
    rows = int(lake.last_operation(spark, silver_path)["metrics"].get("numOutputRows", "0"))
    log.info("silver_written", year=year, rows=rows)
    return rows
