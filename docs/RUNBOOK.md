# Runbook

Each alert in `observability/alert_rules.yml` links to a section here.

## High error rate

`LdpHighErrorRate`: more than 2% of `/v1/decisions` calls return 5xx for 5 minutes.

1. `GET /ready` on the API. `model_version_served: null` means TF Serving is down or still loading: see *Model
   unavailable*.
2. API logs (`decided`, `model_unavailable` events) show whether failures are TF Serving or the gateway.
3. Roll back by pointing traffic at the previous Cloud Run revision; the model version is part of the image tag
   and the GCS path, so the pair moves together.

## Latency

`LdpLatencySloBreach`: decision p99 above 250 ms for 10 minutes.

1. Compare `ldp_tfserving_seconds` (model time) with `ldp_http_request_seconds` (total). Most of the request
   is TF Serving's `explain` call (64 integrated-gradient steps per application).
2. Large batches cost more: check request sizes in the logs (`n`).
3. Scale: raise `max_instances` (Terraform) or the uvicorn worker count; the load test in `reports/k6/` shows
   the measured capacity per instance.

## Model unavailable

`LdpModelUnavailable`: more than 5 TF Serving failures in 5 minutes. TF Serving logs say whether the model failed
to load (bad export, missing files in the bucket) or the process restarted. The API answers 503 (not a guess)
while the model is missing.

## Events

`LdpDecisionEventsFailing`: Pub/Sub publishes are failing. Decisions are still served, but the monitor sees a
partial window. Check the publisher's credentials (`roles/pubsub.publisher` on the topic) and the topic.
Messages BigQuery rejects land on `decisions-dead-letter`; inspect them with
`gcloud pubsub subscriptions pull decisions-dead-letter-hold --limit 10`.

## Drift

`LdpFeatureDriftMajor` (an input's PSI > 0.25) or `LdpScoreDrift` (score PSI > 0.10).

1. The dashboard shows which inputs moved. The 2023 -> 2025 shift in `reports/psi.json` is the expected
   background: compare against it before acting.
2. A shift in one input with a stable score usually needs no action; a moving score means the referral volume
   is changing: tell underwriting capacity planning.
3. Retrain on the most recent full year when score PSI stays above 0.25, and re-run the fairness audit before
   promoting.

## Adverse impact

`LdpAdverseImpact`: the model's AIR for a group below 0.80 for 30 minutes. This is a compliance event.

1. Compare with the lenders' observed AIR for the same group in `reports/fairness.json`; the model learns from
   those decisions.
2. Check the window size (`ldp_monitor_window_rows`): small windows are noisy.
3. Escalate to compliance with the dashboard snapshot; do not change the threshold for one group (disparate
   treatment). A model change goes through a less-discriminatory-alternative search and the full audit.

## Monitor

`LdpMonitorStale`: no successful monitoring cycle in 15 minutes. Check the monitor's logs for BigQuery errors
(permissions: `roles/bigquery.dataViewer` and `roles/bigquery.jobUser`) and whether the decisions table exists.
