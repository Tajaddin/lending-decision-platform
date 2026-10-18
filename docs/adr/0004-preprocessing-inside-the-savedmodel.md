# 0004. Preprocessing lives inside the SavedModel

Date: 2026-10-18. Status: accepted.

## Context

The model needs imputation, log transforms, standardisation, missing-value flags, two derived ratios and
category lookups. If the API re-implements any of that in Python, the training and serving versions drift
apart, and the failure is silent: the model keeps answering, just wrongly.

## Decision

- The Keras model is split into an **encoder** (all preprocessing, from raw typed inputs to one dense vector)
  and a **head** (an MLP from that vector to a logit). Both are exported in one SavedModel.
- The SavedModel's inputs are exactly the API's fields: five numbers (NaN for missing) and ten category codes as
  strings. Unknown categories go to an out-of-vocabulary bucket; rare categories (fewer than 50 training rows)
  share it.
- Spark's silver layer and the API's request model decode HMDA values with twin implementations of the same
  functions in `ldp.schema`, and a test runs both on the same raw strings.
- Training statistics (medians, means, standard deviations, vocabularies) are fitted on 2023 only and saved next
  to the model (`feature_stats.json`).

## Consequences

- The API image has no TensorFlow; TF Serving does all numeric work.
- Batch scoring in Spark loads the same SavedModel, so batch and online scores agree by construction.
- A change to a transform means a new model version, never a code change in the API.
