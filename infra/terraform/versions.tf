terraform {
  required_version = ">= 1.9"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 8.7"
    }
  }
  # State in a GCS bucket created once by hand (see docs/DEPLOY.md):
  # terraform init -backend-config="bucket=<project>-tfstate"
  backend "gcs" {
    prefix = "lending-decision-platform"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
