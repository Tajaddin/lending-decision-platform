# 0006. Protected attributes and tract demographics are for monitoring only

Date: 2026-10-18. Status: accepted.

## Context

HMDA records race, ethnicity, sex and age, and each census tract's minority share and income level. Using any of
them to decide credit is disparate treatment (ECOA prohibits it); using neighbourhood demographics rebuilds race
from location (redlining). Yet a fair-lending audit needs exactly these fields.

## Decision

- They flow through bronze, silver and gold, and into the training export, but never into `FEATURE_NAMES`; a test
  fails if one does. Geography (state, county, tract, MSA) and the lender's identifier are excluded too.
- The API accepts them in a separate `monitoring` block that is published with the decision event and never
  sent to TF Serving (tested).
- Fairness is measured as the adverse impact ratio of the model's favourable outcome against a reference group,
  next to the lenders' own approval AIR for the same applications (`reports/fairness.json`), and live by the
  monitor with an alert below 0.80.
- No group-specific thresholds: changing the cut-off by group would itself be disparate treatment.

## Consequences

- The model can still learn proxies from legitimate inputs (loan size, property value). The audit measures the
  outcome rather than assuming the feature list prevents it.
- Live AIR is only as complete as the monitoring data applicants provide.
