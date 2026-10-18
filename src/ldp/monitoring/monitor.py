"""Drift and fairness monitor.

Every cycle it reads the decisions of the last window from BigQuery and compares them with the training
reference stored in the model's metadata:

- PSI of every model input and of the score (bins from 2023; > 0.25 is a major shift)
- referral rate
- adverse impact ratio of the model's favourable outcome by group, when applicants supplied monitoring data

Locally it runs as a service and exposes the numbers at /metrics for Prometheus (alert rules in
observability/alert_rules.yml). In GCP the same code runs as a Cloud Run job (`--once`) on a schedule.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from prometheus_client import Gauge, start_http_server

from ldp.config import Settings
from ldp.log import get_logger
from ldp.schema import CATEGORICAL, FEATURE_NAMES
from ldp.training.evaluate import histogram, psi
from ldp.training.fairness import DIMENSIONS, MIN_GROUP

log = get_logger(__name__)

PSI = Gauge("ldp_feature_psi", "Population stability index vs the training year", ["feature"])
SCORE_PSI = Gauge("ldp_score_psi", "PSI of the predicted probability vs the training year")
AIR = Gauge("ldp_model_air", "Adverse impact ratio of the model's favourable outcome", ["dimension", "group"])
REFERRAL_RATE = Gauge("ldp_referral_rate", "Share of decisions referred to an underwriter")
WINDOW_ROWS = Gauge("ldp_monitor_window_rows", "Decisions in the monitoring window")
LAST_RUN = Gauge("ldp_monitor_last_success_timestamp", "Unix time of the last successful monitoring cycle")


def compute(
    df: pd.DataFrame, reference: dict[str, Any], threshold: float, min_group: int = MIN_GROUP
) -> dict[str, Any]:
    out: dict[str, Any] = {"rows": len(df), "psi": {}, "air": {}}
    for name in FEATURE_NAMES:
        ref = reference[name]
        values = df[name].astype(str).to_numpy() if name in CATEGORICAL else df[name].astype(float).to_numpy()
        out["psi"][name] = psi(np.asarray(ref["expected"]), histogram(values, ref))
    p = df["probability"].astype(float).to_numpy()
    out["score_psi"] = psi(np.asarray(reference["score"]["expected"]), histogram(p, reference["score"]))
    out["referral_rate"] = float((p >= threshold).mean()) if len(p) else 0.0
    passed = p < threshold
    for dim, spec in DIMENSIONS.items():
        if spec["column"] not in df:
            continue
        col = df[spec["column"]].astype(str).to_numpy()
        ref_mask = col == spec["reference"]
        if ref_mask.sum() < min_group:
            continue
        ref_rate = passed[ref_mask].mean()
        for group in spec["groups"]:
            mask = col == group
            if mask.sum() >= min_group and ref_rate > 0:
                out["air"][f"{dim}:{group}"] = float(passed[mask].mean() / ref_rate)
    return out


def export(result: dict[str, Any]) -> None:
    WINDOW_ROWS.set(result["rows"])
    for name, value in result["psi"].items():
        PSI.labels(feature=name).set(value)
    SCORE_PSI.set(result["score_psi"])
    REFERRAL_RATE.set(result["referral_rate"])
    for key, value in result["air"].items():
        dim, group = key.split(":", 1)
        AIR.labels(dimension=dim, group=group).set(value)
    LAST_RUN.set(time.time())


def fetch_window(bq: Any, table_id: str, hours: int) -> pd.DataFrame:
    since = (datetime.now(UTC) - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
    columns = ", ".join(
        ["probability", *FEATURE_NAMES, "derived_race", "derived_ethnicity", "derived_sex", "applicant_age_above_62"]
    )
    sql = f"SELECT {columns} FROM `{table_id}` WHERE decided_at >= TIMESTAMP('{since}')"  # noqa: S608 (constants)
    rows = [dict(r.items()) for r in bq.query(sql).result()]
    return pd.DataFrame(rows, columns=columns.split(", "))


def run(
    settings: Settings, metadata: dict[str, Any], once: bool = False, interval_s: int = 30, port: int = 9108
) -> dict[str, Any] | None:
    from ldp.streaming.bq import client, ensure_table

    bq = client(settings)
    table_id = ensure_table(settings, bq)
    if not once:
        start_http_server(port)
        log.info("monitor_started", port=port, table=table_id, window_hours=settings.monitor_window_hours)
    while True:
        try:
            df = fetch_window(bq, table_id, settings.monitor_window_hours)
            if len(df) >= settings.monitor_min_rows:
                result = compute(df, metadata["reference_bins"], float(metadata["threshold"]))
                export(result)
                worst = max(result["psi"].items(), key=lambda kv: kv[1])
                log.info(
                    "monitor_cycle",
                    rows=result["rows"],
                    score_psi=round(result["score_psi"], 4),
                    worst_feature=worst[0],
                    worst_psi=round(worst[1], 4),
                    air=result["air"],
                )
                if once:
                    return result
            else:
                WINDOW_ROWS.set(len(df))
                log.info("monitor_waiting", rows=len(df), needed=settings.monitor_min_rows)
                if once:
                    return None
        except Exception as exc:  # keep monitoring alive; the staleness alert fires if this persists
            log.error("monitor_error", error=str(exc))
            if once:
                raise
        time.sleep(interval_s)
