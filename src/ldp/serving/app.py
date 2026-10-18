"""The decision API.

POST /v1/decisions  (X-API-Key)  1-64 applications -> decision, calibrated probability, adverse-action reasons
GET  /health        liveness
GET  /ready         TF Serving has the model loaded and the metadata matches
GET  /metrics       Prometheus

The gateway holds no TensorFlow: TF Serving computes the probability and the integrated-gradient attributions
(`explain` signature); the gateway validates, authenticates, applies the threshold, turns attributions into
reasons, and publishes one event per decision.
"""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest, multiprocess

from ldp.config import Settings
from ldp.log import get_logger
from ldp.reasons import top_reasons
from ldp.schema import FEATURE_NAMES
from ldp.serving.contracts import Decision, DecisionRequest, DecisionResponse, ReasonOut
from ldp.serving.events import NullPublisher, Publisher, PubSubPublisher
from ldp.serving.metrics import DECISIONS, LATENCY, MODEL_VERSION, OVERLOADED, PROBABILITY, RATE_LIMITED, REQUESTS
from ldp.serving.security import KeyStore, RateLimiter
from ldp.serving.tfs import ModelUnavailableError, TFServingClient

log = get_logger(__name__)

MAX_BODY_BYTES = 256 * 1024


def load_metadata(models_dir: Path, model_name: str) -> dict[str, Any]:
    versions = sorted((p for p in (models_dir / model_name).iterdir() if p.name.isdigit()), key=lambda p: int(p.name))
    if not versions:
        raise FileNotFoundError(f"no model versions under {models_dir / model_name}")
    meta: dict[str, Any] = json.loads((versions[-1] / "metadata.json").read_text(encoding="utf-8"))
    return meta


def create_app(
    settings: Settings,
    tfs: TFServingClient | None = None,
    publisher: Publisher | None = None,
    metadata: dict[str, Any] | None = None,
) -> FastAPI:
    meta = metadata or load_metadata(Path(settings.models_dir), settings.model_name)
    keys = KeyStore(settings.client_keys())
    limiter = RateLimiter(settings.rate_limit_per_minute)
    threshold = float(meta["threshold"])
    # Admission control: at most max_inflight model calls per worker; beyond that, answer 503 at once instead of
    # queueing until every request times out (the failure mode the load test found at 800 req/s).
    inflight = {"n": 0}
    groups = meta["attribution_groups"]
    version = str(meta["version"])

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.tfs = tfs or TFServingClient(settings.tfs_url, settings.model_name, settings.tfs_timeout_s)
        if publisher is not None:
            app.state.publisher = publisher
        elif settings.publish_events:
            app.state.publisher = PubSubPublisher(settings.gcp_project, settings.pubsub_topic)
        else:
            app.state.publisher = NullPublisher()
        MODEL_VERSION.labels(version=version).set(1)
        log.info("api_started", model_version=version, threshold=threshold, clients=len(settings.client_keys()))
        yield
        app.state.publisher.close()
        await app.state.tfs.close()

    app = FastAPI(title="Lending decision API", version="1.0", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default 422 echoes the rejected values back: that leaks applicant data into logs and
        # proxies, and a non-finite number (Infinity) cannot even be encoded. Return where and why only.
        errors = [
            {"loc": list(e.get("loc", ())), "msg": e.get("msg", ""), "type": e.get("type", "")} for e in exc.errors()
        ]
        return JSONResponse({"detail": errors}, status_code=422)

    @app.middleware("http")
    async def observe(request: Request, call_next: Any) -> Response:
        route = request.url.path
        if int(request.headers.get("content-length") or 0) > MAX_BODY_BYTES:
            REQUESTS.labels(route=route, status="413").inc()
            return JSONResponse({"detail": "request body too large"}, status_code=413)
        start = time.perf_counter()
        response: Response = await call_next(request)
        if route.startswith("/v1/"):
            LATENCY.labels(route=route).observe(time.perf_counter() - start)
        REQUESTS.labels(route=route, status=str(response.status_code)).inc()
        return response

    def authenticate(x_api_key: str | None = Header(default=None)) -> str:
        client = keys.client_for(x_api_key)
        if client is None:
            raise HTTPException(
                status_code=401, detail="missing or invalid API key", headers={"WWW-Authenticate": "ApiKey"}
            )
        return client

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(request: Request) -> JSONResponse:
        served = await request.app.state.tfs.ready_version()
        ok = served == version
        body = {"ready": ok, "model_version_expected": version, "model_version_served": served}
        return JSONResponse(body, status_code=200 if ok else 503)

    @app.get("/metrics")
    async def metrics() -> Response:
        # With several uvicorn workers each process counts on its own; multiprocess mode merges them so a scrape
        # sees the whole service, not whichever worker answered.
        if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
            registry = CollectorRegistry()
            multiprocess.MultiProcessCollector(registry)
            return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/decisions", response_model=DecisionResponse)
    async def decide(body: DecisionRequest, request: Request, client: str = Depends(authenticate)) -> DecisionResponse:
        allowed, retry_after = limiter.allow(client, cost=float(len(body.applications)))
        if not allowed:
            RATE_LIMITED.labels(client=client).inc()
            raise HTTPException(
                status_code=429, detail="rate limit exceeded", headers={"Retry-After": str(max(1, round(retry_after)))}
            )
        rows = [a.model_inputs() for a in body.applications]
        if inflight["n"] >= settings.max_inflight:
            OVERLOADED.inc()
            raise HTTPException(status_code=503, detail="overloaded, retry shortly", headers={"Retry-After": "1"})
        inflight["n"] += 1
        try:
            prediction = await request.app.state.tfs.explain(rows)
        except ModelUnavailableError as exc:
            log.error("model_unavailable", error=str(exc) or type(exc.__cause__).__name__)
            raise HTTPException(status_code=503, detail="model unavailable") from exc
        finally:
            inflight["n"] -= 1

        decided_at = datetime.now(UTC).isoformat(timespec="milliseconds")
        decisions = []
        for app_in, p, attributions in zip(
            body.applications, prediction.probability, prediction.attributions, strict=True
        ):
            refer = p >= threshold
            reasons = top_reasons(attributions, groups) if refer else []
            app_id = app_in.application_id or uuid.uuid4().hex
            decision = Decision(
                application_id=app_id,
                decision="refer" if refer else "approve_path",
                probability=round(p, 6),
                threshold=threshold,
                model_version=version,
                reasons=[
                    ReasonOut(reason=r.reason, features=list(r.features), contribution=r.contribution) for r in reasons
                ],
            )
            decisions.append(decision)
            DECISIONS.labels(decision=decision.decision).inc()
            PROBABILITY.observe(p)
            event = {
                "event_id": uuid.uuid4().hex,
                "application_id": app_id,
                "client": client,
                "decided_at": decided_at,
                "model_version": version,
                "probability": decision.probability,
                "decision": decision.decision,
                "threshold": threshold,
                "reasons": json.dumps([r.model_dump() for r in decision.reasons]),
                **{
                    k: (None if isinstance(v, float) and math.isnan(v) else v)
                    for k, v in app_in.model_dump(include=set(FEATURE_NAMES)).items()
                },
                **(app_in.monitoring.model_dump() if app_in.monitoring else {}),
            }
            request.app.state.publisher.publish(event)
        log.info("decided", client=client, n=len(decisions), refer=sum(d.decision == "refer" for d in decisions))
        return DecisionResponse(decisions=decisions)

    return app


def main() -> FastAPI:
    """uvicorn factory: `uvicorn ldp.serving.app:main --factory`."""
    from ldp.config import get_settings
    from ldp.log import configure

    configure()
    return create_app(get_settings())
