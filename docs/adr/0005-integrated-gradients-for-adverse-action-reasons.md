# 0005. Integrated gradients for adverse-action reasons, served by TF Serving

Date: 2026-10-18. Status: accepted.

## Context

Under ECOA and Regulation B a lender that takes adverse action must state the principal reasons. A referral is
not an adverse action by itself, but the underwriter needs the same reasons to act on it, and they must come
from the model, not from a separate rule set that can disagree with it.

## Decision

- The SavedModel has a second signature, `explain`, that returns integrated-gradient attributions
  (Sundararajan et al., 2017) next to the probability. The path runs in the encoded space from a reference
  applicant (the mean encoded vector of approved 2023 applications) to the applicant, with 64 trapezoid steps;
  slices of the encoded vector are summed back to the input they came from.
- Completeness is tested: attributions add up to the logit difference from the reference within 0.02 + 5%.
- Only inputs that map to a reason a lender could state produce reasons (debt-to-income, collateral, income
  relative to the amount); program and channel codes do not. Inputs that share a reason are merged; at most four
  reasons are returned (`ldp.reasons`).
- The reasons are checked against what lenders reported: on denied 2025 applications whose first HMDA denial
  reason is debt-to-income or collateral, how often the model names the same reason (`reports/model-metrics.json`).

## Alternatives

- SHAP on a tree model: well established, but it would need the model in the API process or a second service.
- Fixed rules: easy to explain, but they can contradict the score.

## Consequences

- One TF Serving call returns both the score and the reasons; `explain` costs more than `serving_default`
  (65 forward and backward passes of the small head per application), measured in the load test.
- HMDA has no credit-history fields, so reasons about credit history can never appear; the model card says so.
