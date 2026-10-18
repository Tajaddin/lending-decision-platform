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
    result = run_pipeline(settings, [2023, 2024, 2025], raw, master="local[2]", audit_sample_rows=100_000)

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
