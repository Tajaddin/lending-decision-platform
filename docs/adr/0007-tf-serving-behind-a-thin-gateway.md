# 0007. TF Serving behind a thin gateway without TensorFlow

Date: 2026-10-18. Status: accepted.

## Context

The decision service needs authentication, validation, rate limiting, a business threshold, reasons, events and
metrics. Loading TensorFlow into a web process makes a large image and couples model rollout to code rollout.

## Decision

- TF Serving 2.21 serves the SavedModel (REST, columnar `inputs`, `signature_name: explain`). It hot-loads new
  version folders, and the gateway's `/ready` reports which version is served versus the one the gateway's
  metadata expects.
- The gateway (FastAPI) holds the rest: API keys compared in constant time, a per-client token bucket, request
  limits (64 applications, 256 KB), validated HMDA codes, 422 responses that never echo the rejected values, and
  503 (not a guess) when the model is unavailable.
- Missing numbers travel as NaN; TF Serving's JSON parser accepts it and the model imputes and flags it (tested
  against the real container in the integration suite).

## Consequences

- Two processes per instance; on Cloud Run they run as a service with a sidecar, sharing localhost.
- The gateway image carries only the base dependencies (FastAPI, Google clients, pandas for the monitor).
