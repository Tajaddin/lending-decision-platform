# Cloud deploy retro: GCP and Databricks

Date: 2026-10-18. Issues: [#23](https://github.com/Tajaddin/lending-decision-platform/issues/23) (GCP) and
[#24](https://github.com/Tajaddin/lending-decision-platform/issues/24) (Databricks), left in the Backlog by Sprint 4
because they needed accounts.

**Goal:** run what Sprint 4 only validated. **Met:** 49 resources applied on GCP, and the bundle job ran the full
pipeline on Databricks Free Edition with results identical to the local run.

## Measured

| Where | Result | Source |
|---|---|---|
| Databricks Free Edition, serverless | 441 s for the whole job on the 14 GB of CSV, first attempt (bronze and silver 274 s, quality 7 s, gold 26 s, leakage audit 11 s, export 36 s). Bronze, silver and split row counts, quality results and the 13 leakage flags equal the local run | [`reports/databricks/`](../../reports/databricks) |
| GCP, Cloud Run with TF Serving sidecar | Model v1 served; decisions in 112-169 ms from a home connection; 401 without a key; 5 decisions reached BigQuery within 5 s | [`reports/cloud/smoke.json`](../../reports/cloud/smoke.json) |

## Bugs the real runs found

Each was invisible locally, where the stack runs on a laptop Spark session and emulators:

1. **Budget needs a quota project.** The Budgets API rejects user credentials that carry no quota project.
   The budget now uses an aliased provider with `user_project_override`; doing that for every call would also have
   needed Resource Manager enabled first.
2. **Wheel path.** The job's dependency pointed at `databricks/dist`; `uv build` writes to the repository's
   `dist/`. `bundle validate` passed because it does not build.
3. **No `sparkContext` on serverless.** Serverless is Spark Connect; the run report read master and parallelism
   from `sparkContext` and crashed after bronze and silver had been written. `spark_info()` falls back.
4. **`SystemExit(0)` counted as a failure.** Serverless runs wheel tasks in a kernel that reports any
   `SystemExit` as a failed task, even Click's exit after a successful command, so the job retried finished work.
   The `ldp` entry point now returns on a clean exit and keeps failure codes.

Fixes 2-4 have tests; all four are in the PR that closes #23 and #24.

## Not done

- Cloud Run load test (the latencies are from one home connection).
- Keyless deploys from GitHub Actions: the identity pool and deployer account exist, the repository variables
  `GCP_WIF_PROVIDER` and `GCP_DEPLOYER_SA` are not set, so pushes do not deploy.
