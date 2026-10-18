# 0001. PySpark and Delta Lake for the batch layer

Date: 2026-10-18. Status: accepted.

## Context

Three years of the national HMDA register are 37 million rows and 14 GB of CSV (11.5M, 12.2M and 13.5M
rows for 2023, 2024 and 2025). The batch layer has to load them, type them, check them, build the model
population and export training splits, and it has to run the same way on a laptop, in CI and on Databricks.
Corrected HMDA files are republished, so reloading one year must not disturb the others, and a trained model
has to name the exact data it was trained on.

## Decision

- PySpark 4.2 with Delta Lake 4.4. Local mode (`local[*]`) on one machine; on Databricks the existing session
  is used unchanged (`ldp.spark.session.get_spark`).
- Bronze keeps the CSV as text, one Delta partition per `activity_year`; silver and gold are typed Delta
  tables. Every write to a year uses `replaceWhere`, so a year can be reloaded on its own.
- ANSI mode stays on (the Spark 4 default). Sentinel-prone fields go through `try_cast`; any other cast
  failure is a bug and stops the job.
- The training export records the gold table's Delta version; the model metadata carries it, and Delta time
  travel rebuilds the exact training data.

## Alternatives

- DuckDB or Polars on one machine would be faster to start for this size (the author's NOAA pipeline uses
  DuckDB). Rejected here because the target environments are Spark-first (Databricks, Dataproc) and the
  postings this project answers ask for Spark.
- Plain Parquet without Delta: no atomic partition replace, no history, no time travel for lineage.

## Consequences

- JVM start-up and Delta jar resolution add ~10 s to every run; tests share one session.
- Spark needs Java 17+ wherever the tests run (CI installs Temurin 21).
- On Windows, Spark needs Hadoop native helpers to write files; development runs in WSL or Linux.
