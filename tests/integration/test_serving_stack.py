"""Integration: real TF Serving, real Pub/Sub and BigQuery emulators, started by Testcontainers.

    pytest -m integration

Needs Docker and the committed model in models/ldp/1 (TF Serving loads it from a bind mount).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import numpy as np
import pytest
from pydantic import SecretStr

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
APPS = json.loads((ROOT / "bench" / "k6" / "applications.json").read_text())
KEY = "integration-key-000000001"


def _wait(url: str, ok: callable, timeout: float = 120.0) -> None:  # type: ignore[valid-type]
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=2)
            if ok(r):
                return
            last = r.status_code
        except httpx.HTTPError as exc:
            last = exc
        time.sleep(1)
    raise TimeoutError(f"{url} not ready: {last}")


@pytest.fixture(scope="module")
def tfserving() -> Iterator[str]:
    from testcontainers.core.container import DockerContainer

    if not (MODELS / "ldp" / "1" / "saved_model.pb").exists():
        pytest.skip("no exported model in models/ldp/1")
    container = (
        DockerContainer("tensorflow/serving:2.21.0")
        .with_volume_mapping(str(MODELS), "/models", "ro")
        .with_command("--model_name=ldp --model_base_path=/models/ldp --rest_api_port=8501")
        .with_exposed_ports(8501)
    )
    with container:
        url = f"http://{container.get_container_host_ip()}:{container.get_exposed_port(8501)}"
        _wait(f"{url}/v1/models/ldp", lambda r: r.status_code == 200 and "AVAILABLE" in r.text)
        yield url


@pytest.fixture(scope="module")
def emulators() -> Iterator[dict[str, str]]:
    from testcontainers.core.container import DockerContainer

    pubsub = (
        DockerContainer("gcr.io/google.com/cloudsdktool/google-cloud-cli:589.0.0-emulators")
        .with_command("gcloud beta emulators pubsub start --host-port=0.0.0.0:8681 --project=ldp-it")
        .with_exposed_ports(8681)
    )
    bigquery = (
        DockerContainer("ghcr.io/goccy/bigquery-emulator:0.8.1")
        .with_command("--project=ldp-it --dataset=lending --port=9050")
        .with_exposed_ports(9050)
    )
    with pubsub, bigquery:
        ps = f"{pubsub.get_container_host_ip()}:{pubsub.get_exposed_port(8681)}"
        bq = f"http://{bigquery.get_container_host_ip()}:{bigquery.get_exposed_port(9050)}"
        _wait(f"http://{ps}", lambda r: r.status_code == 200)
        _wait(f"{bq}/bigquery/v2/projects/ldp-it/datasets", lambda r: r.status_code == 200)
        yield {"pubsub": ps, "bigquery": bq}


def test_tfserving_matches_the_savedmodel_and_accepts_nan(tfserving: str) -> None:
    tf = pytest.importorskip("tensorflow")
    import asyncio

    from ldp.schema import CATEGORICAL, NUMERIC
    from ldp.serving.contracts import Application
    from ldp.serving.tfs import TFServingClient

    rows = [Application(**a).model_inputs() for a in APPS[:16]]
    rows[0]["income"] = float("nan")
    rows[1]["property_value"] = float("nan")

    async def call():
        client = TFServingClient(tfserving, "ldp")
        try:
            return await client.explain(rows)
        finally:
            await client.close()

    served = asyncio.run(call())
    local = tf.saved_model.load(str(MODELS / "ldp" / "1")).signatures["explain"]
    inputs = {n: tf.constant(np.array([r[n] for r in rows], dtype=np.float32)) for n in NUMERIC}
    inputs.update({n: tf.constant(np.array([r[n] for r in rows]).astype(bytes)) for n in CATEGORICAL})
    expected = local(**inputs)
    np.testing.assert_allclose(served.probability, expected["probability"].numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(served.attributions, expected["attributions"].numpy(), rtol=1e-4, atol=1e-5)
    assert all(0.0 < p < 1.0 for p in served.probability)


def test_api_to_pubsub_to_bigquery_to_monitor(tfserving: str, emulators: dict[str, str], monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from ldp.config import Settings
    from ldp.monitoring.monitor import compute, fetch_window
    from ldp.serving.app import create_app, load_metadata
    from ldp.streaming.bq import client as bq_client
    from ldp.streaming.bq import ensure_table
    from ldp.streaming.sink import Sink

    monkeypatch.setenv("PUBSUB_EMULATOR_HOST", emulators["pubsub"])
    settings = Settings(
        api_keys=SecretStr(f"it:{KEY}"),
        tfs_url=tfserving,
        models_dir=MODELS,
        gcp_project="ldp-it",
        bq_endpoint=emulators["bigquery"],
        monitor_min_rows=10,
    )
    bq = bq_client(settings)
    table_id = ensure_table(settings, bq)
    sink = Sink(settings, bq, table_id)
    sink.ensure_subscription()

    with TestClient(create_app(settings)) as api:
        for i in range(0, 60, 20):
            batch = [{**a, "monitoring": {"derived_race": "White", "derived_sex": "Female"}} for a in APPS[i : i + 20]]
            r = api.post("/v1/decisions", json={"applications": batch}, headers={"X-API-Key": KEY})
            assert r.status_code == 200, r.text
            assert len(r.json()["decisions"]) == 20
        assert api.get("/ready").json()["ready"] is True
    # The publisher flushes on shutdown (lifespan); now drain the subscription into BigQuery.
    written = 0
    for _ in range(10):
        written += sink.drain_once()
        if written >= 60:
            break
    assert written == 60

    # A poison message goes to the dead-letter topic and does not block the subscription.
    from google.cloud import pubsub_v1

    pub = pubsub_v1.PublisherClient()
    pub.publish(pub.topic_path("ldp-it", settings.pubsub_topic), b"{not json").result(timeout=10)
    assert sink.drain_once() == 0

    df = fetch_window(bq, table_id, hours=1)
    assert len(df) == 60
    meta = load_metadata(MODELS, "ldp")
    result = compute(df, meta["reference_bins"], float(meta["threshold"]), min_group=1)
    assert set(result["psi"]) and 0.0 <= result["referral_rate"] <= 1.0
    os.environ.pop("PUBSUB_EMULATOR_HOST", None)
