# Decision events: Pub/Sub topic -> BigQuery subscription (no code in the path) -> decisions table.
# Messages BigQuery rejects go to a dead-letter topic after 5 attempts.

resource "google_bigquery_dataset" "lending" {
  dataset_id                 = "lending"
  location                   = "US"
  description                = "Decision events and monitoring results"
  delete_contents_on_destroy = false
  depends_on                 = [google_project_service.apis]
}

resource "google_bigquery_table" "decisions" {
  dataset_id          = google_bigquery_dataset.lending.dataset_id
  table_id            = "decisions"
  description         = "One row per decision (schema = ldp.streaming.bq.decisions_schema)"
  schema              = file("${path.module}/schemas/decisions.json")
  deletion_protection = true

  time_partitioning {
    type  = "DAY"
    field = "decided_at"
  }
  clustering = ["decision", "model_version"]
}

resource "google_pubsub_topic" "decisions" {
  name                       = "decisions"
  message_retention_duration = "604800s"
  depends_on                 = [google_project_service.apis]
}

resource "google_pubsub_topic" "decisions_dead_letter" {
  name                       = "decisions-dead-letter"
  message_retention_duration = "604800s"
}

resource "google_pubsub_subscription" "decisions_dead_letter_hold" {
  name                       = "decisions-dead-letter-hold"
  topic                      = google_pubsub_topic.decisions_dead_letter.id
  message_retention_duration = "604800s"
}

locals {
  pubsub_agent = "serviceAccount:service-${data.google_project.this.number}@gcp-sa-pubsub.iam.gserviceaccount.com"
}

resource "google_bigquery_table_iam_member" "pubsub_writes_decisions" {
  dataset_id = google_bigquery_dataset.lending.dataset_id
  table_id   = google_bigquery_table.decisions.table_id
  role       = "roles/bigquery.dataEditor"
  member     = local.pubsub_agent
}

resource "google_pubsub_topic_iam_member" "pubsub_dead_letters" {
  topic  = google_pubsub_topic.decisions_dead_letter.id
  role   = "roles/pubsub.publisher"
  member = local.pubsub_agent
}

resource "google_pubsub_subscription" "decisions_to_bigquery" {
  name  = "decisions-bq"
  topic = google_pubsub_topic.decisions.id

  bigquery_config {
    table               = "${var.project_id}.${google_bigquery_dataset.lending.dataset_id}.${google_bigquery_table.decisions.table_id}"
    use_table_schema    = true
    drop_unknown_fields = true
  }
  dead_letter_policy {
    dead_letter_topic     = google_pubsub_topic.decisions_dead_letter.id
    max_delivery_attempts = 5
  }
  retry_policy {
    minimum_backoff = "10s"
    maximum_backoff = "600s"
  }
  depends_on = [google_bigquery_table_iam_member.pubsub_writes_decisions, google_pubsub_topic_iam_member.pubsub_dead_letters]
}

resource "google_pubsub_subscription_iam_member" "pubsub_acks_dead_lettered" {
  subscription = google_pubsub_subscription.decisions_to_bigquery.id
  role         = "roles/pubsub.subscriber"
  member       = local.pubsub_agent
}

resource "google_pubsub_topic_iam_member" "api_publishes" {
  topic  = google_pubsub_topic.decisions.id
  role   = "roles/pubsub.publisher"
  member = "serviceAccount:${google_service_account.api.email}"
}

resource "google_bigquery_dataset_iam_member" "monitor_reads" {
  dataset_id = google_bigquery_dataset.lending.dataset_id
  role       = "roles/bigquery.dataViewer"
  member     = "serviceAccount:${google_service_account.monitor.email}"
}

resource "google_project_iam_member" "monitor_runs_queries" {
  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.monitor.email}"
}
