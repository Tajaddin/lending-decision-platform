# 0002. A leakage audit gates the feature list

Date: 2026-10-18. Status: accepted.

## Context

HMDA records what happened to an application, not only what was known when it was decided. Some fields are
filled in only for originated loans (interest rate, fees, purchaser, HOEPA status) or only for denials
(denial reasons). Subtler: fields like the combined loan-to-value ratio or the automated underwriting result
are left `NA` when a lender stops processing early, which is far more common on denials. A model trained on
such fields learns "this file was never finished", not "this applicant is risky", and its hold-out score
looks excellent for the wrong reason.

## Decision

Every pipeline run screens each candidate field on a sample of about 600,000 applications from the training year:

1. **Availability gap**: share filled in among approvals minus among denials (HMDA's not-applicable codes
   count as missing). Flag above 0.10.
2. **Univariate AUC**: the field alone (value, missing flag, or a category's denial rate estimated on one half
   and scored on the other). Flag above 0.80.

A flagged field that is also a model input stops the pipeline (`LeakageError`) before the training export.
Below 1,000 rows per class the screens are noise, so the report says `enforced: false` instead.

## Result on 2023 (committed in `reports/leakage-audit.json`)

The first full run stopped on `initially_payable_to_institution`: filled in for 99.7% of approvals and 87.4%
of denials (599,846 sampled applications), because HMDA records "not applicable" when no loan is made. It was
removed from the inputs, as were the combined loan-to-value ratio (95.5% vs 73.1%) and the AUS result (87.5% vs
53.0%). Loan-to-value is
derived inside the model from loan amount and property value instead, which are reported at close rates.
Post-decision fields flag as expected (interest rate available for 99.9% vs 0.0%, denial reason AUC 1.000).

## Consequences

- The model gives up some apparent accuracy in exchange for numbers that hold at decision time.
- Property value (gap 0.065) and debt-to-income (0.044) pass but still carry a little missingness signal;
  they are watched in the audit every run.
- Adding a feature means passing the audit; the test suite also fails if a protected, geographic or
  post-decision field enters `FEATURE_NAMES`.
