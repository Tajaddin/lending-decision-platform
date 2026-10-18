"""Pull decision events from Pub/Sub and stream them into BigQuery.

At-least-once: a message is acked only after its row is inserted. Each row carries `insertId = event_id`, so
BigQuery's best-effort de-duplication absorbs most redeliveries. A message that cannot be parsed or fails the
schema is a poison message: it is published to the dead-letter topic with the error and acked, so it does not
block the subscription. A failed insert (BigQuery down) nacks the batch for redelivery.
"""

from __future__ import annotations

import contextlib
import json
import time
from typing import Any

from prometheus_client import Counter

from ldp.config import Settings
from ldp.log import get_logger
from ldp.streaming.bq import decisions_schema

log = get_logger(__name__)

SINK_ROWS = Counter("ldp_sink_rows_total", "Decision events written to BigQuery", ["outcome"])

REQUIRED = (
    "event_id",
    "application_id",
    "client",
    "decided_at",
    "model_version",
    "probability",
    "decision",
    "threshold",
)


def to_row(data: bytes) -> dict[str, Any]:
    event = json.loads(data)
    if not isinstance(event, dict):
        raise ValueError("event is not a JSON object")
    missing = [k for k in REQUIRED if event.get(k) in (None, "")]
    if missing:
        raise ValueError(f"missing fields {missing}")
    allowed = {f.name for f in decisions_schema()}
    return {k: v for k, v in event.items() if k in allowed}


class Sink:
    def __init__(self, settings: Settings, bq: Any, table_id: str) -> None:
        from google.cloud import pubsub_v1  # type: ignore[attr-defined]

        self.settings = settings
        self.bq = bq
        self.table_id = table_id
        self.subscriber = pubsub_v1.SubscriberClient()
        self.publisher = pubsub_v1.PublisherClient()
        self.subscription = self.subscriber.subscription_path(settings.gcp_project, settings.pubsub_subscription)
        self.dead_letter = self.publisher.topic_path(settings.gcp_project, settings.pubsub_dead_letter_topic)

    def ensure_subscription(self) -> None:
        from google.api_core.exceptions import AlreadyExists

        topic = self.publisher.topic_path(self.settings.gcp_project, self.settings.pubsub_topic)
        for name in (topic, self.dead_letter):
            with contextlib.suppress(AlreadyExists):
                self.publisher.create_topic(name=name)
        with contextlib.suppress(AlreadyExists):
            self.subscriber.create_subscription(name=self.subscription, topic=topic, ack_deadline_seconds=30)

    def drain_once(self, max_messages: int = 500) -> int:
        """Pull one batch, write it, ack it. Returns the number of rows written."""
        response = self.subscriber.pull(subscription=self.subscription, max_messages=max_messages, timeout=10)
        if not response.received_messages:
            return 0
        rows, row_ids, ack_ids, poison = [], [], [], []
        for msg in response.received_messages:
            try:
                row = to_row(msg.message.data)
                rows.append(row)
                row_ids.append(row["event_id"])
                ack_ids.append(msg.ack_id)
            except (ValueError, json.JSONDecodeError) as exc:
                poison.append((msg, str(exc)))
        for msg, error in poison:
            self.publisher.publish(self.dead_letter, msg.message.data, error=error[:500]).result(timeout=10)
            SINK_ROWS.labels(outcome="dead_letter").inc()
        if poison:
            self.subscriber.acknowledge(subscription=self.subscription, ack_ids=[m.ack_id for m, _ in poison])
        if not rows:
            return 0
        errors = self.bq.insert_rows_json(self.table_id, rows, row_ids=row_ids)
        if errors:
            SINK_ROWS.labels(outcome="insert_error").inc(len(rows))
            log.error("bigquery_insert_failed", errors=str(errors)[:500])
            self.subscriber.modify_ack_deadline(subscription=self.subscription, ack_ids=ack_ids, ack_deadline_seconds=0)
            return 0
        self.subscriber.acknowledge(subscription=self.subscription, ack_ids=ack_ids)
        SINK_ROWS.labels(outcome="ok").inc(len(rows))
        return len(rows)

    def run_forever(self, idle_sleep_s: float = 1.0) -> None:
        log.info("sink_started", subscription=self.subscription, table=self.table_id)
        while True:
            try:
                if self.drain_once() == 0:
                    time.sleep(idle_sleep_s)
            except Exception as exc:  # keep the worker alive; Pub/Sub redelivers what was not acked
                log.error("sink_error", error=str(exc))
                time.sleep(5)
