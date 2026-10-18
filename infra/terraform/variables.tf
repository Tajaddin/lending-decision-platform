variable "project_id" {
  description = "GCP project that hosts everything"
  type        = string
}

variable "region" {
  description = "Region for Cloud Run, Artifact Registry and the bucket (us-central1 is in the Cloud Storage free tier)"
  type        = string
  default     = "us-central1"
}

variable "github_repository" {
  description = "owner/name of the repository whose Actions may deploy (Workload Identity Federation)"
  type        = string
  default     = "Tajaddin/lending-decision-platform"
}

variable "image_tag" {
  description = "Tag of the app image in Artifact Registry"
  type        = string
  default     = "latest"
}

variable "tfserving_image" {
  description = "TF Serving image (mirrored to Artifact Registry by CI to avoid Docker Hub pulls)"
  type        = string
  default     = "tensorflow/serving:2.21.0"
}

variable "max_instances" {
  description = "Upper bound on Cloud Run instances; also a cost ceiling"
  type        = number
  default     = 3
}

variable "monitor_schedule" {
  description = "Cron schedule for the drift and fairness monitor job"
  type        = string
  default     = "0 * * * *"
}

variable "billing_account_id" {
  description = "Billing account for the budget alert; empty skips the budget"
  type        = string
  default     = ""
}

variable "monthly_budget_usd" {
  description = "Budget alert threshold in USD"
  type        = number
  default     = 5
}

variable "alert_email" {
  description = "Where Cloud Monitoring sends alerts; empty skips the notification channel"
  type        = string
  default     = ""
}
