# Architecture

```mermaid
flowchart LR
    subgraph batch["Batch: PySpark 4.2 + Delta Lake (local, or Databricks serverless via the bundle)"]
        raw["HMDA snapshot CSVs<br/>2023-2025, 37.3M rows"] --> bronze["bronze<br/>text, per-year partitions"]
        bronze --> silver["silver<br/>typed, decoded"]
        silver --> dq{"quality checks<br/>critical / warning"}
        dq --> gold["gold<br/>model population + label"]
        gold --> audit{"leakage audit<br/>blocks leaking inputs"}
        audit --> export["training splits<br/>2023 / 2024 / 2025 Parquet"]
        gold --> score["batch scoring<br/>mapInPandas + SavedModel"]
    end

    subgraph train["Training: TensorFlow 2.21 / Keras 3"]
        export --> fit["encoder + MLP head<br/>early stop on 2024"]
        fit --> cal["Platt calibration, threshold<br/>on 2024"]
        cal --> evalr["2025 hold-out: metrics,<br/>fairness, PSI, reason check"]
        cal --> sm["SavedModel<br/>serving_default + explain"]
    end

    subgraph online["Online (Docker Compose locally, Cloud Run on GCP)"]
        client["loan system"] -->|"X-API-Key, JSON"| api["FastAPI gateway<br/>validation, auth, rate limit,<br/>threshold, reasons"]
        api -->|"REST :predict explain"| tfs["TF Serving 2.21"]
        sm --> tfs
        api -->|"decision event"| ps[("Pub/Sub<br/>decisions")]
        ps -->|"BigQuery subscription (GCP)<br/>ldp sink (local)"| bq[("BigQuery<br/>lending.decisions")]
        ps -.->|"poison messages"| dlq[("dead-letter topic")]
        bq --> mon["monitor<br/>PSI per input, score PSI,<br/>AIR by group"]
    end

    subgraph obs["Observability"]
        api --> prom["Prometheus<br/>8 alert rules"]
        tfs --> prom
        mon --> prom
        prom --> graf["Grafana dashboard"]
    end
```

## The decisions behind it

| ADR | Decision |
|---|---|
| [0001](adr/0001-spark-and-delta-for-the-batch-layer.md) | PySpark and Delta for the batch layer; one session factory for laptop, CI and Databricks |
| [0002](adr/0002-leakage-audit-gates-the-feature-list.md) | A leakage audit gates the feature list and stops the pipeline |
| [0003](adr/0003-out-of-time-split-by-year.md) | Train 2023, choose on 2024, report on 2025 |
| [0004](adr/0004-preprocessing-inside-the-savedmodel.md) | Preprocessing lives in the SavedModel; serving receives raw typed inputs |
| [0005](adr/0005-integrated-gradients-for-adverse-action-reasons.md) | Integrated gradients for adverse-action reasons, served by TF Serving |
| [0006](adr/0006-protected-attributes-are-for-monitoring-only.md) | Protected attributes and tract demographics are for monitoring only |
| [0007](adr/0007-tf-serving-behind-a-thin-gateway.md) | TF Serving behind a thin gateway without TensorFlow |
| [0008](adr/0008-decision-events-through-pubsub-into-bigquery.md) | Decision events through Pub/Sub into BigQuery |
| [0009](adr/0009-gcp-cloud-run-with-a-sidecar-and-databricks-for-batch.md) | Cloud Run with a TF Serving sidecar on GCP; Databricks for the batch jobs |

## Where the code is

| Path | What |
|---|---|
| `src/ldp/schema.py` | The data contract: decoders, model inputs, excluded fields |
| `src/ldp/spark/` | bronze, silver, quality, gold, batch scoring; `lake.py` hides Delta path vs Unity Catalog table |
| `src/ldp/audit.py` | Leakage audit |
| `src/ldp/training/` | Model, export with signatures, baselines, evaluation, fairness, model card |
| `src/ldp/serving/` | FastAPI gateway, TF Serving client, events, API keys, rate limit |
| `src/ldp/streaming/` | BigQuery schema and the Pub/Sub sink |
| `src/ldp/monitoring/` | Drift and fairness monitor, replay tool |
| `infra/terraform/` | GCP: Cloud Run, Pub/Sub, BigQuery, Cloud Storage, Secret Manager, WIF, budget |
| `databricks/` | Asset Bundle: the pipeline as a serverless job |
| `observability/` | Prometheus config, alert rules with promtool tests, Grafana dashboard |
