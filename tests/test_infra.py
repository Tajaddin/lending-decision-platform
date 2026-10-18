"""Keep the infrastructure definitions in step with the code."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_terraform_bigquery_schema_matches_the_sink_schema() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("export_bq_schema", ROOT / "scripts" / "export_bq_schema.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    committed = (ROOT / "infra" / "terraform" / "schemas" / "decisions.json").read_text(encoding="utf-8")
    assert json.loads(committed) == json.loads(module.render()), "run scripts/export_bq_schema.py"


def _cli_params() -> tuple[set[str], dict[str, set[str]]]:
    import typer.main

    from ldp.cli import app

    group = typer.main.get_command(app)
    global_opts = {o for p in group.params for o in p.opts}
    commands = {name: {o for p in cmd.params for o in p.opts} for name, cmd in group.commands.items()}  # type: ignore[attr-defined]
    return global_opts, commands


def test_every_databricks_task_parameter_is_a_real_cli_option() -> None:
    global_opts, commands = _cli_params()
    job = yaml.safe_load((ROOT / "databricks" / "resources" / "hmda_pipeline.yml").read_text())
    tasks = job["resources"]["jobs"]["hmda_pipeline"]["tasks"]
    assert tasks
    for task in tasks:
        params = task["python_wheel_task"]["parameters"]
        command = next(p for p in params if not p.startswith("--"))
        before, after = params[: params.index(command)], params[params.index(command) + 1 :]
        assert command in commands, command
        assert {p.split("=")[0] for p in before} <= global_opts, (task["task_key"], before)
        assert {p.split("=")[0] for p in after} <= commands[command], (task["task_key"], after)
        stages = next((p.split("=")[1] for p in after if p.startswith("--stages=")), "")
        from ldp.pipeline import ALL_STAGES

        assert set(stages.split(",")) <= set(ALL_STAGES)


def test_the_entry_point_returns_on_success_and_keeps_failure_codes(monkeypatch) -> None:
    import tomllib

    from ldp.cli import run

    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts["ldp"] == "ldp.cli:run", "Databricks wheel tasks call the ldp entry point"
    monkeypatch.setattr("sys.argv", ["ldp", "--help"])
    assert run() is None  # serverless reports even SystemExit(0) as a failed task
    monkeypatch.setattr("sys.argv", ["ldp", "no-such-command"])
    with pytest.raises(SystemExit) as exc:
        run()
    assert exc.value.code == 2


def test_compose_services_and_ports() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    services = compose["services"]
    assert {"tfserving", "api", "pubsub", "bigquery", "sink", "monitor", "prometheus", "grafana"} <= set(services)
    host_ports = [p.split(":")[0] for s in services.values() for p in s.get("ports", [])]
    assert len(host_ports) == len(set(host_ports)), "two services publish the same host port"
    assert "tensorflow/serving:2.21.0" in services["tfserving"]["image"]


@pytest.mark.spark
def test_unity_catalog_style_tables_work_through_the_same_code(spark) -> None:
    from ldp.config import Settings
    from ldp.spark import lake

    settings = Settings(lake_root="uc:default")
    target = settings.lake("ldp_test_table")
    assert target == "uc:default.ldp_test_table"
    df = spark.createDataFrame([(2025, 1), (2024, 2)], "activity_year int, v int")
    lake.write(df, target, partition_by="activity_year")
    lake.write(
        df.where("activity_year = 2025").withColumn("v", df.v * 10),
        target,
        replace_where="activity_year = 2025",
        partition_by="activity_year",
    )
    rows = sorted(tuple(r) for r in lake.read(spark, target).collect())
    assert rows == [(2024, 2), (2025, 10)]
    assert lake.last_operation(spark, target)["version"] == 1
    assert sorted(tuple(r) for r in lake.read(spark, target, version=0).collect()) == [(2024, 2), (2025, 1)]
    spark.sql("DROP TABLE default.ldp_test_table")
