# 0007. TF Serving behind a thin gateway without TensorFlow

Date: 2026-10-18. Status: accepted.

## Context

The decision service needs authentication, validation, rate limiting, a business threshold, reasons, events and
metrics. Loading TensorFlow into a web process makes a large image and couples model rollout to code rollout.

## Decision

- TF Serving 2.21 serves the SavedModel (REST, columnar `inputs`, `signature_name: explain`). The gateway calls
  the version its metadata comes from (`/versions/<n>:predict`), and on Cloud Run a version policy keeps TF Serving
  from hot-loading a newer folder, so a revision's threshold and model cannot drift apart. `/ready` reports
  whether that version is loaded and is the startup probe. (The version policy and the startup probe were added
  after the 2026-10-18 deploy and have not been applied on Cloud Run yet.)
- The gateway (FastAPI) holds the rest: API keys compared in constant time, a per-client token bucket, request
  limits (64 applications, 256 KB), validated HMDA codes, 422 responses that never echo the rejected values, and
  503 (not a guess) when the model is unavailable.
- Missing numbers travel as NaN; TF Serving's JSON parser accepts it and the model imputes and flags it (tested
  against the real container in the integration suite).

## Consequences

- Two processes per instance; on Cloud Run they run as a service with a sidecar, sharing localhost.
- The gateway image carries only the base dependencies (FastAPI, Google clients, pandas for the monitor).
