output "api_url" {
  value = google_cloud_run_v2_service.api.uri
}

output "models_bucket" {
  value = google_storage_bucket.models.name
}

output "image_repository" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}"
}

output "workload_identity_provider" {
  description = "Value for google-github-actions/auth workload_identity_provider"
  value       = google_iam_workload_identity_pool_provider.github.name
}

output "deployer_service_account" {
  value = google_service_account.deployer.email
}
