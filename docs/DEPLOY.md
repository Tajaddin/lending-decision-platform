# Deploying to GCP and Databricks

Everything here needs accounts only the owner can create. The local stack (`make up`) needs none of it.

## GCP (free tier)

1. Create a project and attach a billing account (Always Free resources still need billing enabled):
   [console.cloud.google.com/projectcreate](https://console.cloud.google.com/projectcreate).
2. Install the gcloud CLI and log in: `gcloud auth login && gcloud auth application-default login`.
3. Create the Terraform state bucket once:
   ```bash
   gcloud storage buckets create gs://<project>-tfstate --location=us-central1 --uniform-bucket-level-access
   ```
4. Apply:
   ```bash
   cd infra/terraform
   terraform init -backend-config="bucket=<project>-tfstate"
   terraform apply -var project_id=<project> -var billing_account_id=<billing-account-id> -var alert_email=<you>
   ```
5. Add the API keys as the secret's first version (not in Terraform state):
   ```bash
   printf 'branch-app:%s' "$(openssl rand -hex 24)" | gcloud secrets versions add ldp-api-keys --data-file=-
   ```
6. Upload the model and push the image:
   ```bash
   gcloud storage cp -r models/ldp gs://<project>-ldp-models/
   gcloud auth configure-docker us-central1-docker.pkg.dev
   docker build -f docker/app.Dockerfile -t us-central1-docker.pkg.dev/<project>/ldp/app:latest .
   docker push us-central1-docker.pkg.dev/<project>/ldp/app:latest
   ```
7. For keyless deploys from GitHub Actions, set repository variables `GCP_WIF_PROVIDER` and
   `GCP_DEPLOYER_SA` from `terraform output`.

## Databricks Free Edition

1. Sign up: [databricks.com/try-databricks-free](https://www.databricks.com/try-databricks-free) (Free Edition).
2. In the workspace, create schema `workspace.lending` and volume `workspace.lending.raw`; upload the three
   `<year>_public_lar_csv.csv` files into the volume (or run `scripts/download_hmda.sh` in a notebook if the
   workspace has outbound internet).
3. Create a personal access token (User settings, Developer) and configure the CLI:
   `databricks configure --host https://<workspace-url>`.
4. Deploy and run:
   ```bash
   cd databricks
   databricks bundle validate -t free
   databricks bundle deploy -t free
   databricks bundle run -t free hmda_pipeline
   ```

## GitHub Projects board

The sprint issues and milestones exist already. The board needs the `project` scope on the CLI token:

```bash
gh auth refresh -s project
python scripts/agile/sync_backlog.py --board
```
