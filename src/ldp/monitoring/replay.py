"""Replay real applications from a held-out year through the live API at a fixed rate.

This is how the local stack gets realistic traffic: 2025 applications (with their HMDA monitoring fields)
scored by a model whose reference population is 2023, so the drift monitor has a real shift to find.
"""

from __future__ import annotations

import asyncio
import math
import time
from pathlib import Path
from typing import Any

import httpx
import pyarrow.dataset as ds

from ldp.log import get_logger
from ldp.schema import CATEGORICAL, NUMERIC

log = get_logger(__name__)

MONITORING = ("derived_race", "derived_ethnicity", "derived_sex", "applicant_age_above_62")


def load_applications(split_dir: Path, n: int, seed: int = 21) -> list[dict[str, Any]]:
    table = ds.dataset(str(split_dir), format="parquet").to_table(columns=[*NUMERIC, *CATEGORICAL, *MONITORING])
    import numpy as np

    idx = np.random.default_rng(seed).choice(table.num_rows, size=min(n, table.num_rows), replace=False)
    rows = table.take(idx).to_pylist()
    apps = []
    for i, row in enumerate(rows):
        app: dict[str, Any] = {"application_id": f"replay-{seed}-{i}"}
        for name in NUMERIC:
            v = row[name]
            app[name] = None if v is None or (isinstance(v, float) and math.isnan(v)) else v
        for name in CATEGORICAL:
            app[name] = row[name] if row[name] not in (None, "missing") else "NA"
        app["monitoring"] = {k: row[k] for k in MONITORING if row[k] not in (None, "missing")}
        apps.append(app)
    return apps


async def replay(
    base_url: str, api_key: str, apps: list[dict[str, Any]], rate: float, batch: int = 1, concurrency: int = 32
) -> dict[str, Any]:
    sem = asyncio.Semaphore(concurrency)
    status: dict[int, int] = {}
    interval = batch / rate
    async with httpx.AsyncClient(base_url=base_url, headers={"X-API-Key": api_key}, timeout=10) as client:

        async def send(chunk: list[dict[str, Any]]) -> None:
            async with sem:
                try:
                    r = await client.post("/v1/decisions", json={"applications": chunk})
                    status[r.status_code] = status.get(r.status_code, 0) + 1
                except httpx.HTTPError:
                    status[0] = status.get(0, 0) + 1

        start = time.perf_counter()
        tasks = []
        for i in range(0, len(apps), batch):
            target = start + (i // batch) * interval
            delay = target - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(send(apps[i : i + batch])))
        await asyncio.gather(*tasks)
    seconds = time.perf_counter() - start
    log.info("replay_done", applications=len(apps), seconds=round(seconds, 1), status=status)
    return {"applications": len(apps), "seconds": seconds, "status": status}
