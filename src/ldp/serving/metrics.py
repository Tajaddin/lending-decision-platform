"""Prometheus metrics for the gateway (scraped at /metrics)."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("ldp_http_requests_total", "HTTP requests", ["route", "status"])
LATENCY = Histogram(
    "ldp_http_request_seconds",
    "Request latency",
    ["route"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1.0, 2.5),
)
DECISIONS = Counter("ldp_decisions_total", "Decisions returned", ["decision"])
PROBABILITY = Histogram(
    "ldp_decision_probability", "Predicted probability of denial", buckets=tuple(i / 20 for i in range(1, 20))
)
TFS_SECONDS = Histogram(
    "ldp_tfserving_seconds",
    "TF Serving predict latency",
    buckets=(0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
TFS_ERRORS = Counter("ldp_tfserving_errors_total", "TF Serving failures", ["kind"])
EVENTS = Counter("ldp_events_published_total", "Decision events sent to Pub/Sub", ["outcome"])
OVERLOADED = Counter("ldp_overload_rejections_total", "Requests refused because the worker was at max_inflight")
RATE_LIMITED = Counter("ldp_rate_limited_total", "Requests refused by the per-client rate limit", ["client"])
MODEL_VERSION = Gauge("ldp_model_version_info", "Model version being served", ["version"], multiprocess_mode="max")
