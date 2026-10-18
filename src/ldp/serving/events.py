"""Decision events: one JSON message per decision on a Pub/Sub topic.

Publishing never blocks a response: the Google client batches messages on a background thread and the
callback records success or failure. In GCP a BigQuery subscription writes the topic straight into the
`decisions` table; locally (the emulator has no BigQuery subscriptions) ldp.streaming.sink does that job.
"""

from __future__ import annotations

import contextlib
import json
from typing import Any, Protocol

from ldp.log import get_logger
from ldp.serving.metrics import EVENTS

log = get_logger(__name__)


class Publisher(Protocol):
    def publish(self, event: dict[str, Any]) -> None: ...

    def close(self) -> None: ...


class NullPublisher:
    def publish(self, event: dict[str, Any]) -> None:
        EVENTS.labels(outcome="disabled").inc()

    def close(self) -> None:
        return None


class MemoryPublisher:
    """For tests: keeps every event."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def publish(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        EVENTS.labels(outcome="ok").inc()

    def close(self) -> None:
        return None


class PubSubPublisher:
    def __init__(self, project: str, topic: str) -> None:
        from google.cloud import pubsub_v1  # type: ignore[attr-defined]

        settings = pubsub_v1.types.BatchSettings(max_messages=500, max_latency=0.05)
        self._client = pubsub_v1.PublisherClient(batch_settings=settings)
        self._topic = self._client.topic_path(project, topic)

    def publish(self, event: dict[str, Any]) -> None:
        data = json.dumps(event, separators=(",", ":"), allow_nan=False).encode()
        future = self._client.publish(self._topic, data, model_version=str(event.get("model_version", "")))
        future.add_done_callback(self._done)

    @staticmethod
    def _done(future: Any) -> None:
        try:
            future.result()
            EVENTS.labels(outcome="ok").inc()
        except Exception as exc:
            EVENTS.labels(outcome="error").inc()
            log.warning("event_publish_failed", error=str(exc))

    def close(self) -> None:
        self._client.stop()


def ensure_topic(project: str, topic: str) -> None:
    """Create the topic if it does not exist (local emulator start-up; Terraform owns it in GCP)."""
    from google.api_core.exceptions import AlreadyExists
    from google.cloud import pubsub_v1  # type: ignore[attr-defined]

    client = pubsub_v1.PublisherClient()
    with contextlib.suppress(AlreadyExists):
        client.create_topic(name=client.topic_path(project, topic))
