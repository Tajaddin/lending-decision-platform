"""TF Serving REST client: one predict call per request batch, with a timeout and metrics.

Load testing found that an overload which made clients give up left the connection pool unusable: every later
call waited out the pool timeout and the service stayed at 100% errors until restarted. Two defences: the pool
has explicit limits and its own short timeout, and each call to TF Serving runs in its own task that a client
disconnect cannot cancel half-way (the connection always goes back to the pool).

Every call names a model version. Unversioned, TF Serving answers with the highest version it has loaded, so a
newly uploaded model would score requests while the gateway still applied the old threshold and reason groups
and recorded the old version on every decision.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

import httpx

from ldp.serving.metrics import TFS_ERRORS, TFS_SECONDS


class ModelUnavailableError(RuntimeError):
    pass


@dataclass
class Prediction:
    probability: list[float]
    attributions: list[list[float]]


class TFServingClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        version: int | str,
        timeout_s: float = 2.0,
        client: httpx.AsyncClient | None = None,
        max_connections: int = 64,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.version = str(version)
        self._model_url = f"{self.base_url}/v1/models/{model}/versions/{self.version}"
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_s, pool=0.5),
            limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def status(self) -> dict[str, Any]:
        r = await self._client.get(self._model_url)
        r.raise_for_status()
        return r.json()

    async def ready_version(self) -> str | None:
        """This client's version if TF Serving has it loaded, else None (TF Serving answers 404 for a version it
        does not have)."""
        try:
            versions = (await self.status()).get("model_version_status", [])
        except httpx.HTTPError:
            return None
        loaded = any(v.get("version") == self.version and v.get("state") == "AVAILABLE" for v in versions)
        return self.version if loaded else None

    async def explain(self, rows: list[dict[str, float | str]]) -> Prediction:
        columns = {name: [row[name] for row in rows] for name in rows[0]}
        # TF Serving's JSON parser accepts NaN; json.dumps writes it unquoted (allow_nan defaults to True).
        body = json.dumps({"signature_name": "explain", "inputs": columns})
        start = time.perf_counter()
        call = asyncio.ensure_future(
            self._client.post(
                f"{self._model_url}:predict",
                content=body,
                headers={"content-type": "application/json"},
            )
        )
        try:
            # shield: if the caller is cancelled (client hung up), the call still completes and frees its connection.
            r = await asyncio.shield(call)
        except httpx.HTTPError as exc:
            TFS_ERRORS.labels(kind="transport").inc()
            raise ModelUnavailableError(str(exc)) from exc
        finally:
            TFS_SECONDS.observe(time.perf_counter() - start)
        if r.status_code != 200:
            TFS_ERRORS.labels(kind=str(r.status_code)).inc()
            raise ModelUnavailableError(f"TF Serving answered {r.status_code}: {r.text[:200]}")
        out = r.json()["outputs"]
        return Prediction(
            probability=[float(p) for p in out["probability"]],
            attributions=[[float(v) for v in row] for row in out["attributions"]],
        )
