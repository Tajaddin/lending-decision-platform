"""The decision API with TF Serving mocked at the HTTP layer (respx) and an in-memory event publisher."""

from __future__ import annotations

import json
import math

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from pydantic import SecretStr

from ldp.config import Settings
from ldp.schema import ATTRIBUTION_GROUPS
from ldp.serving.app import create_app
from ldp.serving.events import MemoryPublisher
from ldp.serving.tfs import TFServingClient

TFS = "http://tfs.test"
KEY = "test-key-branch-app-0001"
GROUPS = [{"name": g.name, "kind": g.kind, "reason": g.reason} for g in ATTRIBUTION_GROUPS]
META = {"version": 3, "threshold": 0.4, "attribution_groups": GROUPS}

APP = {
    "application_id": "A-1",
    "loan_amount": 305000,
    "income": 84,
    "property_value": 355000,
    "debt_to_income_ratio": 44,
    "loan_term": 360,
    "loan_type": "1",
    "loan_purpose": "1",
    "occupancy_type": "1",
    "construction_method": "1",
    "total_units": "1",
    "applicant_credit_score_type": "1",
    "co_applicant_credit_score_type": "10",
    "submission_of_application": "1",
    "preapproval": "2",
    "conforming_loan_limit": "C",
}


def tfs_response(probabilities: list[float]) -> dict:
    attributions = []
    for _ in probabilities:
        row = [0.0] * len(GROUPS)
        row[[g["name"] for g in GROUPS].index("debt_to_income_ratio")] = 0.9
        row[[g["name"] for g in GROUPS].index("loan_to_value")] = 0.3
        row[[g["name"] for g in GROUPS].index("loan_type")] = 1.5  # not a reason; must be skipped
        attributions.append(row)
    return {
        "outputs": {
            "probability": probabilities,
            "logit": [0.0] * len(probabilities),
            "attributions": attributions,
            "baseline_logit": [-1.0] * len(probabilities),
        }
    }


@pytest.fixture
def setup():
    settings = Settings(api_keys=SecretStr(f"branch-app:{KEY}"), rate_limit_per_minute=600, publish_events=False)
    publisher = MemoryPublisher()
    with respx.mock(base_url=TFS, assert_all_called=False) as mock:
        tfs = TFServingClient(TFS, "ldp", client=httpx.AsyncClient())
        app = create_app(settings, tfs=tfs, publisher=publisher, metadata=META)
        with TestClient(app) as client:
            yield client, mock, publisher


def post(client: TestClient, apps: list[dict], key: str | None = KEY):
    headers = {"X-API-Key": key} if key else {}
    return client.post("/v1/decisions", json={"applications": apps}, headers=headers)


def test_refer_with_reasons_and_event(setup) -> None:
    client, mock, publisher = setup
    route = mock.post("/v1/models/ldp:predict").mock(return_value=httpx.Response(200, json=tfs_response([0.71])))
    r = post(client, [{**APP, "monitoring": {"derived_race": "White", "derived_sex": "Female"}}])
    assert r.status_code == 200
    d = r.json()["decisions"][0]
    assert d["decision"] == "refer" and d["probability"] == 0.71 and d["model_version"] == "3"
    assert [x["reason"] for x in d["reasons"]] == [
        "Excessive obligations in relation to income",
        "Value or type of collateral not sufficient",
    ]
    sent = json.loads(route.calls[0].request.content)
    assert sent["signature_name"] == "explain"
    assert "monitoring" not in sent["inputs"] and "derived_race" not in sent["inputs"]  # never sent to the model
    event = publisher.events[0]
    assert event["application_id"] == "A-1" and event["decision"] == "refer"
    assert event["derived_race"] == "White" and event["debt_to_income_ratio"] == 44
    assert json.loads(event["reasons"])[0]["reason"] == "Excessive obligations in relation to income"


def test_approve_path_has_no_reasons_and_missing_numbers_go_as_nan(setup) -> None:
    client, mock, _ = setup
    route = mock.post("/v1/models/ldp:predict").mock(return_value=httpx.Response(200, json=tfs_response([0.1, 0.2])))
    r = post(client, [{**APP, "income": None}, {**APP, "application_id": None}])
    assert r.status_code == 200
    decisions = r.json()["decisions"]
    assert [d["decision"] for d in decisions] == ["approve_path", "approve_path"]
    assert all(d["reasons"] == [] for d in decisions)
    assert len(decisions[1]["application_id"]) == 32  # generated when the client sends none
    body = route.calls[0].request.content.decode()
    assert '"income": [NaN, 84.0]' in body


@pytest.mark.parametrize("key", [None, "wrong-key-xxxxxxxxxxxx"])
def test_requests_without_a_valid_key_are_refused(setup, key) -> None:
    client, _, _ = setup
    r = post(client, [APP], key=key)
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "ApiKey"


@pytest.mark.parametrize(
    "change",
    [
        {"loan_type": "9"},
        {"loan_amount": -5},
        {"debt_to_income_ratio": 140},
        {"unknown_field": 1},
        {"income": math.inf},
        {"application_id": "bad id with spaces"},
    ],
)
def test_invalid_applications_are_rejected(setup, change) -> None:
    client, _, _ = setup
    payload = json.dumps({"applications": [{**APP, **change}]}, allow_nan=True)
    r = client.post("/v1/decisions", content=payload, headers={"X-API-Key": KEY, "content-type": "application/json"})
    assert r.status_code == 422
    assert all("input" not in e for e in r.json()["detail"])  # rejected values are never echoed back


def test_rate_limit_returns_429_with_retry_after() -> None:
    settings = Settings(api_keys=SecretStr(f"branch-app:{KEY}"), rate_limit_per_minute=2, publish_events=False)
    with respx.mock(base_url=TFS) as mock:
        mock.post("/v1/models/ldp:predict").mock(return_value=httpx.Response(200, json=tfs_response([0.1])))
        app = create_app(
            settings,
            tfs=TFServingClient(TFS, "ldp", client=httpx.AsyncClient()),
            publisher=MemoryPublisher(),
            metadata=META,
        )
        with TestClient(app) as client:
            assert post(client, [APP]).status_code == 200
            assert post(client, [APP]).status_code == 200
            r = post(client, [APP])
            assert r.status_code == 429
            assert int(r.headers["retry-after"]) >= 1


def test_model_errors_become_503(setup) -> None:
    client, mock, publisher = setup
    mock.post("/v1/models/ldp:predict").mock(return_value=httpx.Response(500, text="boom"))
    assert post(client, [APP]).status_code == 503
    mock.post("/v1/models/ldp:predict").mock(side_effect=httpx.ConnectError("down"))
    assert post(client, [APP]).status_code == 503
    assert publisher.events == []


def test_oversized_bodies_are_refused(setup) -> None:
    client, _, _ = setup
    r = client.post(
        "/v1/decisions", content=b"x" * (300 * 1024), headers={"X-API-Key": KEY, "content-type": "application/json"}
    )
    assert r.status_code == 413


def test_batches_above_64_are_rejected(setup) -> None:
    client, _, _ = setup
    assert post(client, [APP] * 65).status_code == 422


def test_ready_reflects_the_served_version(setup) -> None:
    client, mock, _ = setup
    mock.get("/v1/models/ldp").mock(
        return_value=httpx.Response(200, json={"model_version_status": [{"version": "3", "state": "AVAILABLE"}]})
    )
    assert client.get("/ready").json() == {"ready": True, "model_version_expected": "3", "model_version_served": "3"}
    mock.get("/v1/models/ldp").mock(
        return_value=httpx.Response(200, json={"model_version_status": [{"version": "2", "state": "AVAILABLE"}]})
    )
    r = client.get("/ready")
    assert r.status_code == 503 and r.json()["model_version_served"] == "2"
    mock.get("/v1/models/ldp").mock(side_effect=httpx.ConnectError("down"))
    assert client.get("/ready").status_code == 503


def test_health_and_metrics(setup) -> None:
    client, mock, _ = setup
    mock.post("/v1/models/ldp:predict").mock(return_value=httpx.Response(200, json=tfs_response([0.9])))
    post(client, [APP])
    assert client.get("/health").json() == {"status": "ok"}
    text = client.get("/metrics").text
    assert 'ldp_decisions_total{decision="refer"}' in text
    assert "ldp_http_request_seconds_bucket" in text


def test_overload_is_shed_with_503_instead_of_queueing() -> None:
    """With max_inflight=1, a second concurrent request is refused at once with Retry-After."""
    import asyncio

    settings = Settings(api_keys=SecretStr(f"branch-app:{KEY}"), max_inflight=1, publish_events=False)

    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.3)
        return httpx.Response(200, json=tfs_response([0.1]))

    async def run() -> list[httpx.Response]:
        with respx.mock(base_url=TFS, assert_all_called=False) as mock:
            mock.post("/v1/models/ldp:predict").mock(side_effect=slow)
            app = create_app(
                settings,
                tfs=TFServingClient(TFS, "ldp", client=httpx.AsyncClient()),
                publisher=MemoryPublisher(),
                metadata=META,
            )
            async with app.router.lifespan_context(app):
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://api") as client:
                    body = {"applications": [APP]}
                    headers = {"X-API-Key": KEY}
                    return list(
                        await asyncio.gather(
                            client.post("/v1/decisions", json=body, headers=headers),
                            client.post("/v1/decisions", json=body, headers=headers),
                        )
                    )

    responses = asyncio.run(run())
    assert sorted(r.status_code for r in responses) == [200, 503]
    shed = next(r for r in responses if r.status_code == 503)
    assert shed.headers["retry-after"] == "1"
    assert shed.json()["detail"] == "overloaded, retry shortly"
