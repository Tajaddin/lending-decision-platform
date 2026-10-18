# Sprint 4 retro: cloud and release

Dates: 2026-10-18. Milestone: [Sprint 4 - Cloud and release](https://github.com/Tajaddin/lending-decision-platform/milestone/4).

**Goal:** the same system described as GCP and Databricks infrastructure, checked in CI, with the docs a new
reader needs. **Met for everything that can be checked without a cloud account; nothing was applied.**

## Delivered

| Issue | Points | Result |
|---|---|---|
| #19 Terraform for GCP | 5 | Cloud Run gateway with TF Serving as a sidecar (model mounted from Cloud Storage), Pub/Sub to BigQuery subscription with a dead-letter topic, Secret Manager, one service account per workload, keyless CI through Workload Identity Federation, budget and alert policies |
| #20 Databricks Asset Bundle | 3 | the `ldp pipeline` stages as serverless wheel tasks writing Unity Catalog tables (`LDP_LAKE_ROOT=uc:...`) |
| #21 README, architecture, ADRs | 3 | README results traced to report files, architecture diagram, deploy guide, runbook, ADR 0009 |
| #22 security scanning | 2 | trivy on the Terraform and on both images, tflint, alongside the gitleaks and pip-audit jobs from Sprint 1 |

13 of 13 committed points done.

## Checked (2026-10-18, before the pull request)

- `terraform fmt -check` and `terraform validate` (Terraform 1.16.6, google provider 8.7.0): valid.
- tflint 0.64.0: no warnings. trivy 0.76.0 `config` on `infra/terraform`: 0 high or critical misconfigurations,
  with nothing in `.trivyignore`.
- Both bundle files validate against the JSON schema from Databricks CLI 1.21.0.
- `tests/test_infra.py`: the BigQuery table schema in Terraform equals the one the sink writes, every bundle task
  parameter is a real CLI option, and the pipeline writes and reads Unity Catalog-style table names.

## What went well

- Tying the infrastructure to the code with tests instead of review: if someone adds a field to the decision
  event or renames a CLI option, CI fails before a deploy would.

## What did not

- **Nothing was applied.** There is no GCP project with billing and no Databricks workspace, so every check is
  static. Static checks do not catch IAM propagation, quota or API-enablement errors that appear only at apply
  time. The deploy and the first bundle run stay in the backlog as #23 and #24, labelled `blocked:account`.
- `databricks bundle validate` needs an authenticated workspace, so CI checks the bundle against the CLI's
  published schema instead. That proves the structure, not that the catalog, schema or serverless environment
  exist.
- The provider lock file held only the hash for the platform it was first created on. Running `terraform init`
  on Linux added the Linux hash; it is committed so local runs and CI verify the same provider build.

## Carried forward (Backlog milestone)

- #23 deploy to GCP and #24 run the bundle on Databricks Free Edition, once the accounts exist (steps in
  `docs/DEPLOY.md`).
- #25 the GitHub Projects board needs a token with the `project` scope.
- #27 a joint search for a less discriminatory alternative model; v1 stays "not approved".
