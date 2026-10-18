# Cloud Run: the API container with TF Serving as a sidecar in the same instance (localhost:8501),
# the model mounted read-only from Cloud Storage. Scales to zero, so idle cost is nil.

resource "google_cloud_run_v2_service" "api" {
  name                = "${local.name}-api"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false

  template {
    service_account                  = google_service_account.api.email
    max_instance_request_concurrency = 40
    timeout                          = "10s"

    scaling {
      min_instance_count = 0
      max_instance_count = var.max_instances
    }

    volumes {
      name = "models"
      gcs {
        bucket    = google_storage_bucket.models.name
        read_only = true
      }
    }

    containers {
      name       = "api"
      image      = local.app_image
      depends_on = ["tfserving"]
      ports {
        container_port = 8080
      }
      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
        cpu_idle = true
      }
      env {
        name  = "LDP_TFS_URL"
        value = "http://localhost:8501"
      }
      env {
        name  = "LDP_GCP_PROJECT"
        value = var.project_id
      }
      env {
        name  = "LDP_MODELS_DIR"
        value = "/models"
      }
      env {
        name = "LDP_API_KEYS"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.api_keys.secret_id
            version = "latest"
          }
        }
      }
      volume_mounts {
        name       = "models"
        mount_path = "/models"
      }
      startup_probe {
        http_get {
          path = "/health"
        }
        period_seconds    = 2
        failure_threshold = 15
      }
      liveness_probe {
        http_get {
          path = "/health"
        }
        period_seconds = 30
      }
    }

    containers {
      name  = "tfserving"
      image = var.tfserving_image
      args  = ["--model_name=ldp", "--model_base_path=/models/ldp", "--rest_api_port=8501", "--port=8500"]
      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
      }
      volume_mounts {
        name       = "models"
        mount_path = "/models"
      }
      startup_probe {
        tcp_socket {
          port = 8501
        }
        period_seconds    = 2
        failure_threshold = 30
      }
    }
  }

  depends_on = [google_project_service.apis, google_secret_manager_secret_iam_member.api_reads_keys]
}

# Callers authenticate with API keys at the application layer, so the service accepts unauthenticated
# requests at the Cloud Run layer. Put it behind API Gateway or IAP for a real deployment.
resource "google_cloud_run_v2_service_iam_member" "public" {
  name     = google_cloud_run_v2_service.api.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# --- Monitor: a Cloud Run job on a schedule -----------------------------------------------------------

resource "google_cloud_run_v2_job" "monitor" {
  name                = "${local.name}-monitor"
  location            = var.region
  deletion_protection = false

  template {
    task_count = 1
    template {
      service_account = google_service_account.monitor.email
      max_retries     = 1
      timeout         = "300s"
      volumes {
        name = "models"
        gcs {
          bucket    = google_storage_bucket.models.name
          read_only = true
        }
      }
      containers {
        image   = local.app_image
        command = ["ldp", "monitor", "--once"]
        env {
          name  = "LDP_GCP_PROJECT"
          value = var.project_id
        }
        env {
          name  = "LDP_MODELS_DIR"
          value = "/models"
        }
        volume_mounts {
          name       = "models"
          mount_path = "/models"
        }
        resources {
          limits = {
            cpu    = "1"
            memory = "512Mi"
          }
        }
      }
    }
  }
  depends_on = [google_project_service.apis]
}

resource "google_cloud_run_v2_job_iam_member" "scheduler_runs_monitor" {
  name     = google_cloud_run_v2_job.monitor.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.scheduler.email}"
}

resource "google_cloud_scheduler_job" "monitor" {
  name      = "${local.name}-monitor"
  region    = var.region
  schedule  = var.monitor_schedule
  time_zone = "Etc/UTC"

  http_target {
    http_method = "POST"
    uri         = "https://run.googleapis.com/v2/projects/${var.project_id}/locations/${var.region}/jobs/${google_cloud_run_v2_job.monitor.name}:run"
    oauth_token {
      service_account_email = google_service_account.scheduler.email
    }
  }
  depends_on = [google_project_service.apis]
}
