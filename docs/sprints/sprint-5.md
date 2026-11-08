# Sprint 5 retro: less discriminatory alternative

Dates: 2026-10-18. Milestone: [Sprint 5 - Less discriminatory alternative](https://github.com/Tajaddin/lending-decision-platform/milestone/6).

**Goal:** a joint search for a less discriminatory alternative to model v1, chosen on 2024 and scored once on 2025,
with the result in the model card whether or not it passes. **Met as a search; no alternative is approved.** The
variant chosen on 2024 fixes all five groups v1 fails on 2025, but it newly flags Asian applicants and costs 0.025 AUC.

## Delivered

| Issue | Points | Result |
|---|---|---|
| #27 joint LDA search | 8 | 9 training runs on 2024 (`reports/lda-search.json`); the chosen referral-rate penalty passes 5 of v1's 5 flagged groups on 2025 and flags 1 new one; model card, README and ADR 0010 updated |

8 of 8 committed points done; the story's second acceptance criterion (a variant that passes the whole audit) is
not met and moves to the follow-up story.

## How the search was run

- Every variant is the v1 recipe with one change to the loss: same network, seed, optimiser, batch order, early
  stopping sample, Platt calibration and 80%-recall threshold. The lambda = 0 run reproduces v1's 2024 AUC to every
  printed digit (0.88022395), and a second full run reproduced the first run's six variants exactly, so
  differences between variants come from the loss, not from noise or the pipeline.
- Protected attributes enter only the loss (sample weights or extra label columns), never the inputs or the
  threshold: ADR 0010, which amends ADR 0006.
- Selection used 2024 only. `scripts/lda_search.py holdout` scored 2025 once, for the chosen variant, and refuses to
  run again.

## Measured (2024 unless marked)

- Reweighing barely moves anything: AUC 0.8782, age 62+ AIR 0.754 (v1 0.757), 3 groups still flagged.
- The mean-logit penalty levels off: age 62+ AIR 0.847 at lambda 0.3 and still 0.847 at lambda 1, while the AUC
  falls from 0.8675 to 0.8492. Closing the gap in mean score does not close it at the threshold.
- The referral-rate penalty works where the audit measures: at lambda 1, no group is flagged on 2024 (AUC 0.8586,
  referral rate 38.7%). At lambda 3 it gets worse again (age 62+ flagged, AUC 0.8496).
- **2025, scored once:** AUC 0.8449 (95% CI 0.8444-0.8454) vs v1's 0.8698; referral rate 40.5% vs 32.7%, catching 79.5% of
  denials (v1 80.0%). Age 62+ AIR 0.955 (lenders 0.930, v1 0.761), Black 0.872 (lenders 0.827, v1 0.759). Asian
  applicants: 1.002, lenders 1.036, flagged.

## What went well

- Writing the operating point into the loss. The algebra (equal referral rates among approved and among denied
  applications give every group with a higher denial rate an AIR at least the lenders' own) said which penalty to
  try, and it was the one that worked.
- Two phases with a guard: 2025 could not be peeked at while the search was being tuned.

## What did not

- The first two variants were the textbook ones and both failed for the same reason the ablation did: they move
  averages, and the audit counts who is above a threshold.
- **The Asian flag is structural, not bad luck.** The same algebra reverses for a group whose denial rate is lower
  than its reference's: equalized odds pulls its AIR toward parity, below the lenders' 1.036. The audit flags any
  model AIR more than 0.02 below the lenders' own, even at parity. Changing that rule after seeing 2025 would be
  choosing the test after the result, so it stands; whether it should apply at parity is the audit owner's call.
- Batch penalties cannot carry very small groups: in a pilot, Pacific Islander applicants (about 8 rows of a 4,096
  batch) made training unstable even at lambda 1, and lambda 8 collapsed it. They were left out of the penalty
  (they pass anyway); a constrained method with full-data rate estimates is the fix.
- The cost is real: 24% more referrals for about the same share of denials caught, and precision falls from 43.5% to 34.9%.
  Underwriting capacity decides whether that is acceptable.

## Carried forward (Backlog milestone)

- #31 the audit-owner decision on the parity rule, the advantaged groups in the penalty or a constrained
  method, and the cost. 2025 has now been looked at once for this search; the next variant evaluated on it is a
  second look, and the card has to say so (or wait for the 2026 snapshot).
