# 0010. Protected attributes may shape the training loss, never the inputs

Date: 2026-10-18. Status: accepted. Amends [0006](0006-protected-attributes-are-for-monitoring-only.md).

## Context

Model v1 fails its own fair-lending audit (`reports/fairness.json`), and dropping one input at a time cannot fix the
age and race gaps together (`reports/ablation.json`). A search for a less discriminatory alternative (issue #27)
needs to know who is in which group while it trains: reweighing needs each application's group to set its weight,
and a fairness penalty needs it to measure the gap it penalises. Under ADR 0006 the attributes were for monitoring
only, so neither was possible.

## Decision

- The race, sex and age fields may enter the **training objective** of an alternative model: as sample weights
  (reweighing) or as extra label columns that only the loss reads (the equalized-odds penalty in
  `ldp.training.lda`).
- They still never enter `FEATURE_NAMES`, the SavedModel's inputs, the API request sent to TF Serving, or the
  threshold. A model trained this way scores an application exactly as v1 does: from the 15 inputs alone, with one
  cut-off for everyone.
- Every variant is chosen on the validation year (2024) and must pass the same audit as v1; 2025 is scored once,
  for the chosen variant only (`scripts/lda_search.py`).

## Consequences

- The deployed model still learns from legitimate inputs only, but its weights were fitted to narrow group gaps.
  Whether a lender may use a model trained this way is a legal question for compliance counsel, not one this
  repository answers; the model card records how each variant was trained so that review has what it needs.
- Group membership is needed at training time, so the training export keeps the protected fields (it already did,
  for the audit).

## Result (2026-10-18, `reports/lda-search.json`)

The referral-rate penalty chosen on 2024 passes all five groups v1 fails on 2025, but flags Asian applicants (AIR
1.002 vs the lenders' 1.036) at AUC 0.8449 (v1 0.8698). No model trained this way is approved, exported or served;
the model card has the full table.
