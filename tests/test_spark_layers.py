"""Bronze, silver, quality and gold on real LAR rows (tests/fixtures, cut by scripts/make_fixtures.py)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ldp import schema
from ldp.schema import parse_dti, parse_number

pytestmark = pytest.mark.spark


def _raw_values(path: str, column: str) -> list[str]:
    with open(path, encoding="utf-8", newline="") as fh:
        return [row[column] for row in csv.DictReader(fh)]


def test_bronze_keeps_every_row_as_text_and_is_idempotent(spark, fixture_csv, tmp_path: Path) -> None:
    from ldp.spark.bronze import write_bronze

    bronze = str(tmp_path / "bronze")
    expected = len(_raw_values(fixture_csv[2023], "lei"))
    first = write_bronze(spark, fixture_csv[2023], bronze, 2023)
    second = write_bronze(spark, fixture_csv[2023], bronze, 2023)  # reload of the same year replaces it
    write_bronze(spark, fixture_csv[2024], bronze, 2024)
    df = spark.read.format("delta").load(bronze)
    assert first.rows == second.rows == expected
    assert df.where("activity_year = '2023'").count() == expected
    assert {f.dataType.simpleString() for f in df.schema.fields if not f.name.startswith("_")} == {"string"}
    assert df.select("_source_file").distinct().count() == 2


def test_bronze_refuses_a_file_from_another_year(spark, fixture_csv, tmp_path: Path) -> None:
    from ldp.spark.bronze import write_bronze

    with pytest.raises(ValueError, match="expected only 2024"):
        write_bronze(spark, fixture_csv[2023], str(tmp_path / "bronze"), 2024)


@pytest.fixture(scope="module")
def silver_df(spark, fixture_csv, tmp_path_factory):
    from ldp.spark.bronze import read_lar_csv
    from ldp.spark.silver import to_silver

    raw = read_lar_csv(spark, fixture_csv[2023])
    from pyspark.sql import functions as F

    return to_silver(raw.withColumn("_source_file", F.lit("lar_2023.csv"))).cache()


def test_silver_decoders_match_the_python_parsers(spark, fixture_csv, silver_df) -> None:
    """Training (Spark) and serving (Python) must decode the same string to the same number."""
    raw = _raw_values(fixture_csv[2023], "debt_to_income_ratio")
    rows = silver_df.select("debt_to_income_ratio", "debt_to_income_ratio_raw", "income", "loan_term").collect()
    assert [r["debt_to_income_ratio"] for r in rows] == [parse_dti(v) for v in raw]
    assert [r["income"] for r in rows] == [parse_number(v) for v in _raw_values(fixture_csv[2023], "income")]
    assert [r["loan_term"] for r in rows] == [parse_number(v) for v in _raw_values(fixture_csv[2023], "loan_term")]


def test_silver_decoders_on_edge_values(spark) -> None:
    from ldp.spark.silver import code, dti, number

    values = ["<20%", "36", ">60%", "NA", "Exempt", "", "140", "abc", "1111", "-5", "inf", None]
    df = spark.createDataFrame([(v, v, v) for v in values], "debt_to_income_ratio string, income string, aus_1 string")
    got = df.select(dti(), number("income"), code("aus_1")).collect()
    assert [r[0] for r in got] == [parse_dti(v) for v in values]
    assert [r[1] for r in got] == [parse_number(v) for v in values]
    assert [r[2] for r in got] == [schema.parse_code(v) for v in values]


def test_silver_types(silver_df) -> None:
    types = dict(silver_df.dtypes)
    assert all(types[c] == "double" for c in schema.SILVER_NUMERIC)
    assert all(types[c] == "string" for c in schema.SILVER_CODES)
    assert types["activity_year"] == "int"


def test_quality_checks_pass_on_real_rows_and_catch_corruption(spark, silver_df, tmp_path: Path) -> None:
    from pyspark.sql import functions as F

    from ldp.spark.quality import DataQualityError, enforce, run_checks, silver_checks, write_report

    rows = silver_df.count()
    good = run_checks(silver_df, 2023, silver_checks(bronze_rows=rows))
    assert good.critical_failures == []
    bad_df = silver_df.withColumn("lei", F.lit("SHORT")).withColumn("action_taken", F.lit("9"))
    bad = run_checks(bad_df, 2023, silver_checks(bronze_rows=rows + 1))
    assert {r.name for r in bad.critical_failures} == {"lei_format", "action_taken_valid", "rows_reconcile_with_bronze"}
    with pytest.raises(DataQualityError):
        enforce([good, bad])
    write_report([good], tmp_path / "dq.json")
    assert json.loads((tmp_path / "dq.json").read_text())["critical_failures"] == 0


def test_population_filters_and_label(spark) -> None:
    from ldp.spark.gold import population

    base = {
        "action_taken": "1",
        "lien_status": "1",
        "loan_purpose": "1",
        "open_end_line_of_credit": "2",
        "reverse_mortgage": "2",
        "business_or_commercial_purpose": "2",
        "total_units": "1",
    }
    variants = [
        {},
        {"action_taken": "3"},
        {"action_taken": "2"},  # in: originated, denied, approved not accepted
        {"action_taken": "4"},
        {"action_taken": "6"},  # out: withdrawn, purchased
        {"lien_status": "2"},
        {"loan_purpose": "4"},
        {"open_end_line_of_credit": "1"},
        {"reverse_mortgage": "1111"},
        {"business_or_commercial_purpose": "1"},
        {"total_units": "5-24"},
    ]
    rows = [{**base, **v} for v in variants]
    df = spark.createDataFrame(rows)
    out = population(df).select("action_taken", "denied").collect()
    assert sorted((r["action_taken"], r["denied"]) for r in out) == [("1", 0), ("2", 0), ("3", 1)]


def test_gold_and_split_export(spark, fixture_csv, tmp_path: Path) -> None:
    from ldp.spark.bronze import write_bronze
    from ldp.spark.gold import EXPORT_COLUMNS, export_splits, write_gold, write_manifest
    from ldp.spark.silver import write_silver

    lake = tmp_path / "lake"
    for year in (2023, 2024, 2025):
        write_bronze(spark, fixture_csv[year], str(lake / "bronze"), year)
        write_silver(spark, str(lake / "bronze"), str(lake / "silver"), year)
    stats = write_gold(spark, str(lake / "silver"), str(lake / "gold"), [2023, 2024, 2025])
    assert set(stats) == {2023, 2024, 2025}
    assert all(0.0 < s["denial_rate"] < 1.0 for s in stats.values())
    exports = export_splits(
        spark,
        str(lake / "gold"),
        str(tmp_path / "training"),
        {"train": 2023, "valid": 2024, "test": 2025},
        files_per_split=1,
    )
    assert [e.rows for e in exports] == [stats[y]["rows"] for y in (2023, 2024, 2025)]
    df = spark.read.parquet(exports[0].path)
    assert tuple(df.columns) == EXPORT_COLUMNS
    write_manifest(exports, tmp_path / "dataset.json")
    manifest = json.loads((tmp_path / "dataset.json").read_text())
    assert manifest["splits"][0]["gold_delta_version"] == exports[0].gold_delta_version


def test_spark_info_on_a_local_session_and_without_a_spark_context(spark, monkeypatch) -> None:
    from pyspark.errors import PySparkAttributeError

    from ldp.spark.session import spark_info

    local = spark_info(spark)
    assert local["master"].startswith("local") and local["default_parallelism"] >= 1

    class ConnectSession:  # serverless compute: Spark Connect has no sparkContext
        version = "4.0.0"

        @property
        def sparkContext(self):  # noqa: N802 (PySpark's name)
            raise PySparkAttributeError(
                errorClass="JVM_ATTRIBUTE_NOT_SUPPORTED", messageParameters={"attr_name": "sparkContext"}
            )

    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "client.3")
    assert spark_info(ConnectSession()) == {
        "master": "databricks-serverless",
        "version": "4.0.0",
        "default_parallelism": None,
    }
