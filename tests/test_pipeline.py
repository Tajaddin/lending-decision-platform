"""The whole batch pipeline on the three fixture years, as `ldp pipeline` runs it."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from ldp.config import Settings

pytestmark = [pytest.mark.spark, pytest.mark.tf]


def test_pipeline_end_to_end(spark, fixture_csv, tmp_path: Path) -> None:
    from ldp.pipeline import run_pipeline

    raw = tmp_path / "raw"
    raw.mkdir()
    for year, path in fixture_csv.items():
        shutil.copy(path, raw / f"{year}_public_lar_csv.csv")
    settings = Settings(
        lake_root=str(tmp_path / "lake"), training_dir=tmp_path / "training", reports_dir=tmp_path / "reports"
    )
    result = run_pipeline(
        settings, [2023, 2024, 2025], raw, master="local[2]", audit_sample_rows=100_000, allow_underpowered_audit=True
    )

    assert set(result["timings_s"]) >= {"bronze_2023", "silver_2025", "quality", "gold", "leakage_audit", "export"}
    assert result["bronze"] == result["silver"]
    quality = json.loads((tmp_path / "reports" / "data-quality.json").read_text())
    assert quality["critical_failures"] == 0
    audit = json.loads((tmp_path / "reports" / "leakage-audit.json").read_text())
    assert "interest_rate" in audit["flagged"]
    assert audit["enforced"] is False  # a few hundred fixture rows cannot power the screens
    assert result["leakage_audit_enforced"] is False
    dataset = json.loads((tmp_path / "reports" / "dataset.json").read_text())
    assert [s["split"] for s in dataset["splits"]] == ["train", "valid", "test"]
    assert all((tmp_path / "training" / s).is_dir() for s in ("train", "valid", "test"))

    # Train on the exported fixture splits, score a year in Spark with the SavedModel, render the model card.
    pytest.importorskip("tensorflow")
    from ldp.spark.score import score_year
    from ldp.training.model_card import render
    from ldp.training.train import train_and_evaluate

    settings = settings.model_copy(update={"models_dir": tmp_path / "models"})
    payload = train_and_evaluate(settings, epochs=1, batch_size=128)
    assert set(payload["models"]) == {"tensorflow", "logistic_regression", "xgboost"}
    assert 0.0 <= payload["models"]["tensorflow"]["test"]["auc"] <= 1.0  # 1 epoch on ~250 rows: wiring, not skill
    model_dir = tmp_path / "models" / "ldp" / "1"
    assert (model_dir / "saved_model.pb").exists() and (model_dir / "metadata.json").exists()
    scored = score_year(spark, settings.lake("gold"), str(model_dir), settings.lake("scores"), 2025, partitions=2)
    assert scored.count() == dataset["splits"][2]["rows"]
    card = render(tmp_path / "reports", model_dir)
    assert "## Fair-lending audit" in card and "TensorFlow MLP (this model)" in card


def _audit_only(spark, tmp_path: Path, years: list[int], **options: bool) -> dict:
    """Run only the audit stage on a gold table holding a few rows per given year."""
    from ldp.pipeline import run_pipeline

    settings = Settings(lake_root=str(tmp_path / "lake"), reports_dir=tmp_path / "reports")
    rows = [(year, i % 2, float(i)) for year in years for i in range(20)]
    gold = spark.createDataFrame(rows, "activity_year int, denied int, income double")
    gold.write.format("delta").partitionBy("activity_year").save(settings.lake("gold"))
    return run_pipeline(settings, [2025], tmp_path, master="local[2]", stages=("audit",), **options)


def test_the_audit_stops_on_an_empty_training_year(spark, tmp_path: Path) -> None:
    """Even a run that allows an underpowered audit stops when there is nothing to screen."""
    from ldp.audit import LeakageError

    for run, options in (("strict", {}), ("fixture", {"allow_underpowered_audit": True})):
        with pytest.raises(LeakageError, match="no 2023 rows"):
            _audit_only(spark, tmp_path / run, [2024, 2025], **options)
        assert not (tmp_path / run / "reports" / "leakage-audit.json").exists()


def test_an_underpowered_audit_stops_the_run_unless_allowed(spark, tmp_path: Path) -> None:
    from ldp.audit import LeakageError

    with pytest.raises(LeakageError, match="cannot power"):
        _audit_only(spark, tmp_path / "strict", [2023, 2025])
    assert not (tmp_path / "strict" / "reports" / "leakage-audit.json").exists()
    result = _audit_only(spark, tmp_path / "fixture", [2023, 2025], allow_underpowered_audit=True)
    audit = json.loads((tmp_path / "fixture" / "reports" / "leakage-audit.json").read_text())
    assert result["leakage_audit_enforced"] is False and audit["enforced"] is False and audit["rows"] == 20
