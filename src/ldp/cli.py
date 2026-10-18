"""Command line: `ldp <command>`. Each command is also a Databricks wheel-task entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ldp.config import get_settings
from ldp.log import configure

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _years(value: str) -> list[int]:
    return [int(y) for y in value.split(",") if y.strip()]


@app.callback()
def main(log_level: str = "INFO") -> None:
    configure(log_level)


@app.command()
def pipeline(
    years: Annotated[str, typer.Option(help="comma-separated activity years")] = "2023,2024,2025",
    raw_dir: Annotated[Path, typer.Option(help="folder with <year>_public_lar_csv.csv files")] = Path("data/raw"),
    master: Annotated[str | None, typer.Option(help="Spark master, e.g. local[8]")] = None,
    stages: Annotated[str, typer.Option(help="comma-separated subset of bronze,silver,quality,gold,audit,export")] = "",
) -> None:
    """bronze -> silver -> quality -> gold -> leakage audit -> training export."""
    from ldp.pipeline import ALL_STAGES, host_info, run_pipeline, write_json

    settings = get_settings()
    chosen = tuple(s.strip() for s in stages.split(",") if s.strip()) or ALL_STAGES
    result = run_pipeline(settings, _years(years), raw_dir, master=master, stages=chosen)
    name = "pipeline-run.json" if chosen == ALL_STAGES else f"pipeline-run-{'-'.join(chosen)}.json"
    write_json({"host": host_info(), "stages": list(chosen), **result}, settings.reports_dir / name)
    typer.echo(f"done: {result['timings_s']}")


@app.command()
def bench_silver(
    year: int = 2025,
    cores: Annotated[str, typer.Option(help="comma-separated core counts")] = "1,2,4,8,16",
    repeats: int = 1,
) -> None:
    """Time the silver stage at several core counts on one year (bronze must exist)."""
    from ldp.bench import bench_silver as run

    settings = get_settings()
    run(settings, year, [int(c) for c in cores.split(",")], repeats=repeats)


@app.command()
def train(
    epochs: int = 12,
    version: Annotated[int, typer.Option(help="model version folder under models/<name>/")] = 1,
    skip_baselines: bool = False,
) -> None:
    """Train, calibrate and evaluate on the out-of-time split; export the SavedModel and reports."""
    from ldp.training.train import train_and_evaluate

    result = train_and_evaluate(get_settings(), epochs=epochs, version=version, skip_baselines=skip_baselines)
    tf_test = result["models"]["tensorflow"]["test"]
    typer.echo(f"test AUC {tf_test['auc']:.4f}  ECE {tf_test['ece']:.4f}  threshold {result['threshold']:.4f}")


@app.command()
def model_card(version: int = 1, out: Path = Path("docs/MODEL-CARD.md")) -> None:
    """Render the model card from reports/*.json and the model's metadata."""
    from ldp.training.model_card import write

    settings = get_settings()
    write(Path(settings.reports_dir), Path(settings.models_dir) / settings.model_name / str(version), out)
    typer.echo(f"wrote {out}")


@app.command()
def score_batch(
    year: int = 2025,
    version: int = 1,
    partitions: int = 8,
    master: Annotated[str | None, typer.Option(help="Spark master")] = None,
) -> None:
    """Score one year of the gold table in Spark with the SavedModel; writes the scores Delta table."""
    import time

    from ldp.pipeline import host_info, write_json
    from ldp.spark.score import score_year
    from ldp.spark.session import get_spark

    settings = get_settings()
    spark = get_spark("ldp-score", master=master)
    model_dir = str((Path(settings.models_dir) / settings.model_name / str(version)).resolve())
    start = time.perf_counter()
    scored = score_year(spark, settings.lake("gold"), model_dir, settings.lake("scores"), year, partitions)
    rows = scored.count()
    seconds = time.perf_counter() - start
    referred = scored.where("refer").count()
    write_json(
        {
            "benchmark": "Spark batch scoring with the SavedModel (mapInPandas)",
            "year": year,
            "rows": rows,
            "referred": referred,
            "seconds": round(seconds, 1),
            "rows_per_s": round(rows / seconds),
            "partitions": partitions,
            "master": spark.sparkContext.master,
            "host": host_info(),
        },
        Path(settings.reports_dir) / "batch-scoring.json",
    )
    typer.echo(f"scored {rows:,} rows in {seconds:.1f}s ({rows / seconds:,.0f}/s), referred {referred:,}")


@app.command()
def download(years: str = "2023,2024,2025", out_dir: Path = Path("data/raw")) -> None:
    """Download and unzip the national snapshot files (wraps scripts/download_hmda.sh)."""
    import subprocess

    script = Path(__file__).resolve().parents[2] / "scripts" / "download_hmda.sh"
    for year in _years(years):
        subprocess.run(["bash", str(script), str(year), str(out_dir)], check=True)


if __name__ == "__main__":
    app()
