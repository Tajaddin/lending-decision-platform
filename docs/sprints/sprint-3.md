# Sprint 3 retro: serving and monitoring

Dates: 2026-10-18. Milestone: [Sprint 3 - Serving and monitoring](https://github.com/Tajaddin/lending-decision-platform/milestone/3).

**Goal:** real-time decisions through TF Serving and an authenticated API, decision events through Pub/Sub into
BigQuery, drift and fairness monitoring with alerts, all from one command. **Met.**

## Delivered

| Issue | Points | Result |
|---|---|---|
| #13 decision API | 5 | API keys, validated HMDA codes, per-client rate limit, load shedding, readiness tied to the served model version, reasons from `explain` |
| #14 events through Pub/Sub into BigQuery | 5 | publisher off the request path; sink with dead-lettering and insert ids; Terraform uses a native BigQuery subscription instead |
| #15 integration tests | 3 | real TF Serving, Pub/Sub and BigQuery emulator containers; NaN inputs and attributions match the local SavedModel |
| #16 drift and fairness monitoring | 5 | PSI per input and for the score, AIR by group, 8 alert rules with promtool tests, Grafana dashboard |
| #17 one command | 3 | `make up` starts 8 services with health checks; compose smoke test in CI |
| #18 load test | 2 | k6 (JavaScript) at 50 to 800 req/s, results in `reports/k6/` |

23 of 23 committed points done.

## Measured

- Replaying 5,000 real 2025 applications at 50 req/s: 5,000 answered, 5,001 rows in BigQuery (one earlier smoke
  request), referral rate 33.7%. The live monitor's PSI matched the offline analysis: credit score type 0.186
  live vs 0.174 offline, loan purpose 0.160 vs 0.153.
- k6, one machine, 2 API workers: 0 failures and p99 36.2 ms up to 400 req/s; 600 req/s at p99 229.0 ms with
  0.34% of requests shed; 800 req/s sheds 30.8% fast and the next request is answered normally.

## What went well

- The integration tests caught nothing because the unit tests had pinned the contracts (NaN handling, signature
  names, column order); the first run against real containers passed.

## What did not

- **Overload broke the API permanently.** The first 800 req/s run failed 88% of requests, and afterwards every
  request failed with a 503 after 2 s even at low load, until the API restarted: client timeouts cancelled
  requests mid-call and the connection pool to TF Serving never recovered. Fix: a bounded number of model calls
  per worker with an immediate 503 and `Retry-After` beyond it, explicit pool limits, and the TF Serving call
  shielded from cancellation. A test covers the shedding. (The failing run's result files were overwritten by
  the rerun; its numbers are from the console output.)
- **Metrics were per worker.** With two uvicorn workers, `/metrics` showed whichever process answered. Fixed with
  prometheus_client's multiprocess mode.
- **The BigQuery emulator answers a duplicate create with HTTP 500**, which the Google client retries forever, so
  the bootstrap job hung. Fixed by checking for the dataset and table first.

## Carried into Sprint 4

- The monitor only reports AIR for groups with 1,000+ decisions in the window; at replay volumes that leaves out
  most groups. Production volumes fill the window; the threshold stays.
