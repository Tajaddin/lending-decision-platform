from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

# Small, fast Spark for tests; must be set before ldp.config caches its settings.
os.environ.setdefault("LDP_SPARK_DRIVER_MEMORY", "1g")
os.environ.setdefault("LDP_SPARK_SHUFFLE_PARTITIONS", "4")

FIXTURES = Path(__file__).parent / "fixtures"


def _java_available() -> bool:
    return bool(os.environ.get("JAVA_HOME")) or shutil.which("java") is not None


@pytest.fixture(scope="session")
def spark() -> Iterator[object]:
    pytest.importorskip("pyspark")
    pytest.importorskip("delta")
    if not _java_available():
        pytest.skip("Spark tests need Java 17+")
    from ldp.spark.session import get_spark

    session = get_spark("ldp-tests", master="local[2]")
    yield session
    session.stop()


@pytest.fixture(scope="session")
def fixture_csv() -> dict[int, str]:
    return {year: str(FIXTURES / f"lar_{year}.csv") for year in (2023, 2024, 2025)}
