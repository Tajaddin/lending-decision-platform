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
def main(
    log_level: str = "INFO",
    lake_root: Annotated[str | None, typer.Option(help="overrides LDP_LAKE_ROOT, e.g. uc:workspace.lending")] = None,
    training_dir: Annotated[str | None, typer.Option(help="overrides LDP_TRAINING_DIR")] = None,
    reports_dir: Annotated[str | None, typer.Option(help="overrides LDP_REPORTS_DIR")] = None,
) -> None:
    """Options before the command override the environment: a Databricks wheel task has parameters, not env."""
    import os

    for name, value in (
        ("LDP_LAKE_ROOT", lake_root),
        ("LDP_TRAINING_DIR", training_dir),
        ("LDP_REPORTS_DIR", reports_dir),
    ):
        if value:
            os.environ[name] = value
    get_settings.cache_clear()
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
def serve(host: str = "0.0.0.0", port: int = 8080, workers: int = 2) -> None:  # noqa: S104 (container)
    """Run the decision API (uvicorn). With more than one worker, metrics use prometheus_client's multiprocess
    mode: each worker writes to PROMETHEUS_MULTIPROC_DIR and /metrics merges them."""
    import os
    import shutil
    import tempfile

    import uvicorn

    if workers > 1:
        metrics_dir = os.environ.setdefault("PROMETHEUS_MULTIPROC_DIR", f"{tempfile.gettempdir()}/ldp-metrics")
        shutil.rmtree(metrics_dir, ignore_errors=True)  # stale files from a previous run would be counted
        os.makedirs(metrics_dir, exist_ok=True)
    uvicorn.run("ldp.serving.app:main", factory=True, host=host, port=port, workers=workers, access_log=False)


@app.command()
def bootstrap_local() -> None:
    """Create the topics, subscription, dataset and table in the local emulators (idempotent)."""
    from ldp.streaming.bq import client, ensure_table
    from ldp.streaming.sink import Sink

    settings = get_settings()
    bq = client(settings)
    table_id = ensure_table(settings, bq)
    Sink(settings, bq, table_id).ensure_subscription()
    typer.echo(f"ready: topic {settings.pubsub_topic}, subscription {settings.pubsub_subscription}, table {table_id}")


@app.command()
def sink(metrics_port: int = 9109) -> None:
    """Stream decision events from Pub/Sub into BigQuery (local stand-in for a BigQuery subscription)."""
    from prometheus_client import start_http_server

    from ldp.streaming.bq import client, ensure_table
    from ldp.streaming.sink import Sink

    settings = get_settings()
    bq = client(settings)
    worker = Sink(settings, bq, ensure_table(settings, bq))
    worker.ensure_subscription()
    start_http_server(metrics_port)
    worker.run_forever()


@app.command()
def monitor(once: bool = False, interval_s: int = 30, port: int = 9108) -> None:
    """Drift (PSI) and fairness (AIR) on recent decisions; a service with /metrics, or one cycle with --once."""
    from ldp.monitoring.monitor import run
    from ldp.serving.app import load_metadata

    settings = get_settings()
    result = run(
        settings,
        load_metadata(Path(settings.models_dir), settings.model_name),
        once=once,
        interval_s=interval_s,
        port=port,
    )
    if once:
        typer.echo(result)


@app.command()
def replay(
    url: str = "http://localhost:8200",
    split: Path = Path("data/sample/test"),
    n: int = 5000,
    rate: float = 50.0,
    batch: int = 1,
    api_key: Annotated[str, typer.Option(envvar="LDP_REPLAY_KEY")] = "",
) -> None:
    """Send real held-out applications to the API at a fixed rate."""
    import asyncio

    from ldp.monitoring.replay import load_applications
    from ldp.monitoring.replay import replay as run_replay

    apps = load_applications(split, n)
    result = asyncio.run(run_replay(url, api_key, apps, rate=rate, batch=batch))
    typer.echo(result)


@app.command()
def download(years: str = "2023,2024,2025", out_dir: Path = Path("data/raw")) -> None:
    """Download and unzip the national snapshot files (wraps scripts/download_hmda.sh)."""
    import subprocess

    script = Path(__file__).resolve().parents[2] / "scripts" / "download_hmda.sh"
    for year in _years(years):
        subprocess.run(["bash", str(script), str(year), str(out_dir)], check=True)


if __name__ == "__main__":
    app()
