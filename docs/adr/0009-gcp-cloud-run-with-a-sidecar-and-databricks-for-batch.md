# 0009. Cloud Run with a TF Serving sidecar on GCP; Databricks for the batch jobs

Date: 2026-10-18. Status: accepted (Terraform validated, not applied: needs a GCP project with billing).

## Context

The online side must cost nothing when idle and fit the GCP free tier; the batch side needs Spark at a scale a
laptop handles but a team would run on a managed platform.

## Decision

- **Cloud Run** service with two containers: the gateway (ingress) and TF Serving as a sidecar on localhost. The
  model folder is mounted read-only from a Cloud Storage bucket (versioned; old versions expire). Min instances 0,
  max 3. API keys come from Secret Manager; the secret's value is never in Terraform state.
- **Monitor** as a Cloud Run job triggered hourly by Cloud Scheduler with its own service account.
- **Least privilege**: one service account per workload; the Pub/Sub service agent may write only the decisions
  table and the dead-letter topic.
- **Keyless CI**: Workload Identity Federation for GitHub Actions, limited to this repository's `main` branch.
- **Cost guard rails**: an optional billing budget alert ($5 by default) and an uptime/5xx alert policy.
- **Databricks**: an Asset Bundle runs the same `ldp pipeline` stages as serverless wheel tasks, writing Unity
  Catalog tables (`LDP_LAKE_ROOT=uc:workspace.lending`) instead of Delta paths. CI validates the bundle against
  the CLI's JSON schema and checks that every task parameter is a real CLI option.

## Consequences

- The first request after idle pays a cold start for two containers (TF Serving loading the model).
- Cloud Run accepts unauthenticated requests at the platform layer; authentication is the API key. A production
  deployment would put API Gateway or IAP in front.
- Free Edition limits (serverless only, five concurrent tasks, daily quota) fit a three-year pipeline run but not
  a cluster-sized job.
