"""BigQuery: the decisions table schema and a client that also works against the local emulator."""

from __future__ import annotations

import contextlib
from typing import Any

from ldp.config import Settings
from ldp.schema import CATEGORICAL, NUMERIC

MONITORING_COLUMNS = ("derived_race", "derived_ethnicity", "derived_sex", "applicant_age_above_62")


def decisions_schema() -> list[Any]:
    from google.cloud import bigquery

    f = bigquery.SchemaField
    return [
        f("event_id", "STRING", mode="REQUIRED"),
        f("application_id", "STRING", mode="REQUIRED"),
        f("client", "STRING", mode="REQUIRED"),
        f("decided_at", "TIMESTAMP", mode="REQUIRED"),
        f("model_version", "STRING", mode="REQUIRED"),
        f("probability", "FLOAT64", mode="REQUIRED"),
        f("decision", "STRING", mode="REQUIRED"),
        f("threshold", "FLOAT64", mode="REQUIRED"),
        f("reasons", "STRING"),
        *[f(n, "FLOAT64") for n in NUMERIC],
        *[f(n, "STRING") for n in CATEGORICAL],
        *[f(n, "STRING") for n in MONITORING_COLUMNS],
    ]


def client(settings: Settings) -> Any:
    from google.auth.credentials import AnonymousCredentials
    from google.cloud import bigquery

    if settings.bq_endpoint:
        return bigquery.Client(
            project=settings.gcp_project,
            credentials=AnonymousCredentials(),
            client_options={"api_endpoint": settings.bq_endpoint},
        )
    return bigquery.Client(project=settings.gcp_project)


def ensure_table(settings: Settings, bq: Any) -> str:
    """Create the dataset and table if missing (local emulator; Terraform owns them in GCP).

    Existence is checked first: the BigQuery emulator answers a duplicate create with HTTP 500 instead of 409,
    which the Google client retries forever.
    """
    from google.api_core.exceptions import Conflict, NotFound
    from google.cloud import bigquery

    dataset_id = f"{settings.gcp_project}.{settings.bq_dataset}"
    try:
        bq.get_dataset(dataset_id)
    except NotFound:
        with contextlib.suppress(Conflict):
            bq.create_dataset(bigquery.Dataset(dataset_id))
    table_id = f"{dataset_id}.{settings.bq_table}"
    try:
        bq.get_table(table_id)
    except NotFound:
        with contextlib.suppress(Conflict):
            bq.create_table(bigquery.Table(table_id, schema=decisions_schema()))
    return table_id
