# 0003. Out-of-time split by year: train 2023, validate 2024, test 2025

Date: 2026-10-18. Status: accepted.

## Context

A credit decision model is used on applications that arrive after it was trained, in a market that moves
(rates, house prices, product mix). A random split mixes the years and hides that drift. The public HMDA
register has no application date, only the activity year.

## Decision

- Train on 2023, use 2024 for every choice (early stopping, calibration, the referral threshold, XGBoost
  early stopping), and touch 2025 once, after those choices are frozen, for the reported numbers.
- Population stability (PSI) of each input and of the score is reported for 2024 and 2025 against 2023;
  the 2023 bins become the live monitor's reference.

## Consequences

- The reported metrics include a year of drift, so they are lower than a random split would show, and closer
  to what a deployed model would see.
- With yearly granularity there are only three points in time; quarterly monitoring has to come from live
  traffic (the decision events in BigQuery), not from HMDA.
