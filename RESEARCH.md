# Research behind this project

Written before the code (2026-10-18), then updated with what the first runs taught. Sources are linked.

## What employers in the target list ask for

From 25 saved postings (fintech/lending, healthcare, security, agriculture, developer tools):

- **TensorFlow** (5 postings: Artera, Mosaic, Quora x2, ReliaQuest), **Spark/PySpark** (4: Quora, ReliaQuest,
  Tampa General, Upstart), **GCP** (4: Mosaic, OPSWAT, ReliaQuest, SpruceID), **Databricks** (Wurl).
- Lending employers describe their problems directly: Upstart's underwriting team works on "the models that
  inform credit decisions" with "rigor, reliability, and responsible oversight"; Zest AI's mission is "fair and
  transparent credit" with "inclusive risk models" and "dataset validation". Those words shaped the scope: an
  out-of-time evaluation, a leakage audit, calibration, a fair-lending audit and explainable reasons, not just a
  model with an AUC.

## Trends that matter for this kind of project (2026)

- Portfolios built from tutorials are filtered out; reviewers look for the full lifecycle, deployment and
  monitoring, and a README with the result near the top
  ([letsdatascience](https://letsdatascience.com/blog/the-ml-portfolio-that-actually-gets-you-hired-in-2026),
  [DataExpert](https://www.dataexpert.io/blog/ultimate-guide-ai-engineering-portfolios)).
- Lakehouse practice: open table formats with history and time travel; Spark kept for heavy batch work while
  lighter engines take the rest ([Apache Data Lakehouse Weekly](https://amdatalakehouse.substack.com/p/apache-data-lakehouse-weekly-may-746),
  [RisingWave on Iceberg 2026](https://risingwave.com/blog/apache-iceberg-streaming-2026)). Here: Delta Lake
  with the training data's table version recorded in the model metadata.
- Fair lending and AI underwriting: the CFPB's January 2025 Supervisory Highlights special edition focused on
  AI/ML underwriting, proxies and disparate impact; regulators and advocates expect a search for less
  discriminatory alternatives, and the adverse impact ratio is the common screening metric
  ([Compliance Alliance summary](https://compliancealliance.com/?p=74465),
  [Consumer Federation letter](https://consumerfed.org/news/press-releases/consumer-groups-call-on-cfpb-to-protect-consumers-from-discriminatory-algorithms-used-by-banks-and-other-financial-institutions-to-make-credit-decisions/),
  [FinRegLab disparate impact results](https://oldsite.finreglab.org/summary-of-disparate-impact-usability-results)).
  FinRegLab found that dropping the features that "drive" disparities rarely yields a less discriminatory model
  and usually costs accuracy, which is why this project measures AIR against the lenders' own decisions instead
  of pretending a feature list can make the labels fair.

## How each gap skill is used in production, and the free way to use it here

| Skill | Production use | Free or local mode used |
|---|---|---|
| TensorFlow | Keras 3 on the TF backend; TF Serving for low-latency REST/gRPC serving of SavedModels | TF 2.21 CPU ([release](https://newreleases.io/project/github/tensorflow/tensorflow/release/v2.21.0)); `tensorflow/serving:2.21.0` container |
| Spark / PySpark | Batch ETL on clusters; Delta for ACID tables | Spark 4.2 local mode + delta-spark 4.4 (versions checked on PyPI 2026-10-18) |
| Databricks | Jobs on serverless compute, defined as code with Asset Bundles, Unity Catalog tables | Free Edition: serverless only, 5 concurrent job tasks, daily fair-use quota ([limits](https://docs.databricks.com/aws/en/getting-started/free-edition-limitations)); the bundle is validated offline against `databricks bundle schema` (CLI 1.21.0) |
| GCP | Cloud Run (with sidecars and Cloud Storage volumes), Pub/Sub, BigQuery subscriptions, Workload Identity Federation | Always Free: Cloud Run 2M requests/month, BigQuery 1 TB queries + 10 GB storage, Pub/Sub 10 GB ([summary](https://agentdeals.dev/gcp-free-tier-2026), confirm on Google's page); locally the official Pub/Sub emulator and [goccy/bigquery-emulator](https://github.com/goccy/bigquery-emulator) |

TF Serving's REST API ([docs](https://www.tensorflow.org/tfx/serving/api_rest)) takes columnar `inputs` and a
`signature_name`, which is how one SavedModel serves both `serving_default` and `explain`.

## Dataset

**HMDA loan application register**, national snapshot files 2023, 2024, 2025 from the FFIEC/CFPB
([data browser API](https://ffiec.cfpb.gov/documentation/api/data-browser/),
[field definitions](https://ffiec.cfpb.gov/documentation/publications/loan-level-datasets/lar-data-fields)).
US government data, public domain. 11.5M, 12.2M and 13.5M records (0.6-0.7 GB zipped each, 14 GB as CSV).
Downloaded from `files.ffiec.cfpb.gov/static-data/snapshot/<year>/<year>_public_lar_csv.zip`; the older S3
links return 403.

Why HMDA and not a credit-performance dataset: it is the largest public, license-free lending dataset, it carries
the protected attributes a fair-lending audit needs (as monitoring fields), and it is the data regulators use for
disparity analysis. Its limit is stated everywhere: the label is the lender's decision, not repayment, and it has
no credit score or history. Alternatives considered: Freddie Mac and Fannie Mae loan-level data (performance
labels, but registration required), Lending Club and Home Credit (Kaggle account, unclear licences), UCI
credit default (30,000 rows, too small for a Spark pipeline).

## Findings from the data before modelling

- Value encodings checked on 300,000 raw rows: DTI is binned below 36% and above 49% ("<20%", "20%-<30%",
  "30%-<36%", "50%-60%", ">60%") and exact in between; numeric fields use `NA`/`Exempt`; coded fields use `1111`
  for exempt and codes like 9 or 6 for "not applicable".
- Reporting depends on the outcome. Within the decided population, CLTV and the AUS result are often `NA` on
  denials because lenders stop early; "initially payable to the institution" is "not applicable" when no loan is
  made. The leakage audit (ADR 0002) turned this into a rule that stops the pipeline.
