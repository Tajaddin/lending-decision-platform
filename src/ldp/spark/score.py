"""Batch scoring in Spark with the exported SavedModel.

mapInPandas hands each Python worker a stream of Arrow batches; the worker loads the SavedModel once and
keeps it for every batch it scores. The output is a Delta table of probabilities and decisions, for portfolio
reviews and back-testing that should not go through the online API.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ldp.log import get_logger
from ldp.schema import CATEGORICAL, NUMERIC
from ldp.spark import lake

log = get_logger(__name__)

_MODEL: dict[str, Any] = {}

OUTPUT_SCHEMA = "record_hash long, activity_year int, probability double, refer boolean"


def _signature(model_dir: str) -> Any:
    if model_dir not in _MODEL:
        import tensorflow as tf

        _MODEL[model_dir] = tf.saved_model.load(model_dir).signatures["serving_default"]
    return _MODEL[model_dir]


def score_partitions(model_dir: str, threshold: float, batch: int = 8192):
    def score(frames: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        import tensorflow as tf

        fn = _signature(model_dir)
        for frame in frames:
            for start in range(0, len(frame), batch):
                part = frame.iloc[start : start + batch]
                inputs = {n: tf.constant(part[n].to_numpy(dtype=np.float32, na_value=np.nan)) for n in NUMERIC}
                inputs.update(
                    {
                        n: tf.constant(part[n].fillna("missing").astype(str).to_numpy().astype(bytes))
                        for n in CATEGORICAL
                    }
                )
                p = fn(**inputs)["probability"].numpy().astype(np.float64)
                yield pd.DataFrame(
                    {
                        "record_hash": part["record_hash"].to_numpy(),
                        "activity_year": part["activity_year"].to_numpy(),
                        "probability": p,
                        "refer": p >= threshold,
                    }
                )

    return score


def score_year(
    spark: SparkSession, gold_path: str, model_dir: str, out_path: str, year: int, partitions: int = 8
) -> DataFrame:
    meta = json.loads((Path(model_dir) / "metadata.json").read_text(encoding="utf-8"))
    src = (
        lake.read(spark, gold_path)
        .where(F.col("activity_year") == year)
        .select("record_hash", "activity_year", *NUMERIC, *CATEGORICAL)
        .repartition(partitions)
    )
    scored = src.mapInPandas(score_partitions(model_dir, float(meta["threshold"])), schema=OUTPUT_SCHEMA)
    lake.write(scored, out_path, replace_where=f"activity_year = {year}", partition_by="activity_year")
    return lake.read(spark, out_path).where(F.col("activity_year") == year)
