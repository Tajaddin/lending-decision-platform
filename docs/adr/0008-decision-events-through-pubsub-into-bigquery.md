# 0008. Decision events through Pub/Sub into BigQuery

Date: 2026-10-18. Status: accepted.

## Context

Drift and fairness monitoring need every decision with its inputs and outcome. Writing to a warehouse inside the
request path adds latency and a failure mode to every decision.

## Decision

- The gateway publishes one JSON event per decision to a Pub/Sub topic. The Google client batches on a background
  thread (500 messages or 50 ms); the response never waits for it, and failures are counted and alerted on.
- In GCP, a **BigQuery subscription** writes the topic into `lending.decisions` (partitioned by day, clustered by
  decision and model version) with no code in between; rejected messages go to a dead-letter topic after five
  attempts.
- The Pub/Sub emulator has no BigQuery subscriptions, so locally `ldp sink` does the same job: it acks only after
  the insert, sends unparseable messages to the dead-letter topic, and uses the event id as BigQuery's insert id
  so redeliveries are de-duplicated (best effort).
- The table schema is generated from one Python definition (`ldp.streaming.bq.decisions_schema`) into the
  Terraform JSON; a test fails if they differ.

## Consequences

- At-least-once delivery: the monitor tolerates rare duplicates.
- The local sink and the managed subscription are different code paths; the integration test covers the local
  one end to end, and the Terraform covers the managed one (not applied without an account).
