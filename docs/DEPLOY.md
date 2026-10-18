# Deploying to GCP and Databricks

Everything here needs accounts only the owner can create. The local stack (`make up`) needs none of it. Both
were done this way on 2026-10-18; results and the bugs it found are in the
[cloud retro](sprints/cloud-deploy.md).

## GCP (free tier)

1. Create a project with billing (the free trial creates "My First Project" and a billing account), log in and
   create the Terraform state bucket once:
   ```bash
   gcloud auth login && gcloud auth application-default login
   gcloud config set project <project>
   gcloud auth application-default set-quota-project <project>
   gcloud storage buckets create gs://<project>-tfstate --location=us-central1 --uniform-bucket-level-access
   terraform -chdir=infra/terraform init -backend-config="bucket=<project>-tfstate"
   ```
2. Create what the services need before they can start (registry, models bucket, API-key secret):
   ```bash
   VARS="-var project_id=<project> -var billing_account_id=<billing-account-id> -var alert_email=<you>"
   terraform -chdir=infra/terraform apply $VARS -target=google_artifact_registry_repository.images \
     -target=google_storage_bucket.models -target=google_secret_manager_secret_iam_member.api_reads_keys
   ```
3. Add the API keys as the secret's first version (not in Terraform state), upload the model and push the image:
   ```bash
   printf 'branch-app:%s' "$(openssl rand -hex 24)" > ~/.config/ldp/api-keys
   gcloud secrets versions add ldp-api-keys --data-file="$HOME/.config/ldp/api-keys"
   gcloud storage cp -r models/ldp gs://<project>-ldp-models/
   gcloud auth configure-docker us-central1-docker.pkg.dev
   docker build -f docker/app.Dockerfile -t us-central1-docker.pkg.dev/<project>/ldp/app:latest .
   docker push us-central1-docker.pkg.dev/<project>/ldp/app:latest
   ```
4. Apply the rest (Cloud Run service and monitor job, Pub/Sub and BigQuery, identity federation, budget, alerts),
   then check it end to end (writes `reports/cloud/smoke.json`):
   ```bash
   terraform -chdir=infra/terraform apply $VARS
   python scripts/cloud_smoke.py "$(terraform -chdir=infra/terraform output -raw api_url)" \
     --key-file ~/.config/ldp/api-keys --project <project>
   ```
5. For keyless deploys from GitHub Actions, set repository variables `GCP_WIF_PROVIDER` and
   `GCP_DEPLOYER_SA` from `terraform output`. (Not done yet.)

## Databricks Free Edition

1. Sign up for [Free Edition](https://login.databricks.com/?intent=CE_SIGN_UP) and log the CLI in with OAuth (no
   token to store): `databricks auth login --host https://<workspace-url> --profile free`.
2. Create the schema and volume, upload the zipped registers (2 GB instead of 14 GB) and unzip them in the
   workspace with a one-off serverless run:
   ```bash
   export DATABRICKS_CONFIG_PROFILE=free
   databricks schemas create lending workspace
   databricks volumes create workspace lending raw MANAGED
   databricks fs mkdir dbfs:/Volumes/workspace/lending/raw/zips
   for y in 2023 2024 2025; do
     databricks fs cp data/raw/${y}_public_lar_csv.zip dbfs:/Volumes/workspace/lending/raw/zips/
   done
   databricks workspace import /Workspace/Users/<you>/unzip_hmda.py --file databricks/unzip_hmda.py --format AUTO
   databricks jobs submit --json '{"run_name": "unzip-hmda", "tasks": [{"task_key": "unzip",
     "environment_key": "default", "spark_python_task": {"python_file": "/Workspace/Users/<you>/unzip_hmda.py"}}],
     "environments": [{"environment_key": "default", "spec": {"client": "3"}}]}'
   ```
3. Deploy and run:
   ```bash
   cd databricks
   databricks bundle validate -t free
   databricks bundle deploy -t free
   databricks bundle run -t free hmda_pipeline
   ```
   The reports land in `/Volumes/workspace/lending/raw/reports`.

## GitHub Projects board

The sprint issues and milestones exist already. The board needs the `project` scope on the CLI token:

```bash
gh auth refresh -s project
python scripts/agile/sync_backlog.py --board
```
