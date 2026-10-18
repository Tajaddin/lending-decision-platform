"""One SparkSession factory for laptops, CI and Databricks.

On Databricks the session already exists (serverless or a cluster) and Delta is built in, so this returns it
unchanged. Everywhere else it starts a local session with the open-source Delta jars.
"""

from __future__ import annotations

import os

from pyspark.sql import SparkSession

from ldp.config import get_settings


def on_databricks() -> bool:
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def get_spark(app_name: str = "ldp", master: str | None = None) -> SparkSession:
    if on_databricks():
        return SparkSession.builder.getOrCreate()
    from delta import configure_spark_with_delta_pip

    settings = get_settings()
    builder = (
        SparkSession.builder.appName(app_name)
        .master(master or settings.spark_master)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.driver.memory", settings.spark_driver_memory)
        .config("spark.sql.shuffle.partitions", str(settings.spark_shuffle_partitions))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
        # Spark 4 runs in ANSI mode: a bad cast raises instead of returning null. The jobs use try_cast where
        # HMDA sentinels can appear, so any other cast failure is a real bug and should stop the job.
        .config("spark.sql.ansi.enabled", "true")
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark
