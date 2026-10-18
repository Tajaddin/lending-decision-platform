# Sprint 2 retro: model

Dates: 2026-10-18. Milestone: [Sprint 2 - Model](https://github.com/Tajaddin/lending-decision-platform/milestone/2).

**Goal:** a calibrated TensorFlow model that beats the baselines on the 2025 hold-out, with a fair-lending audit,
reason codes and a model card. **Partly met:** the model, calibration, reasons and audit are done; the model ties
XGBoost instead of beating it, and it fails the fair-lending audit.

## Delivered

| Issue | Points | Result |
|---|---|---|
| #8 baselines | 3 | logistic regression AUC 0.806, XGBoost 0.871 on 2025 |
| #9 TensorFlow model, in-graph preprocessing, explain signature | 8 | AUC 0.870, 805 KB SavedModel, attributions complete within 0.02 + 5% (tested) |
| #10 out-of-time evaluation and calibration | 3 | ECE 0.012 before, 0.010 after Platt scaling; score PSI 0.016 on 2025 |
| #11 fair-lending audit and model card | 5 | AIR by race, ethnicity, sex and age vs the lenders' own; card marks the model not approved |
| #12 batch scoring in Spark | 3 | 5,444,487 applications in 19.3 s on 8 cores (282,806 rows/s) |

22 of 22 committed points done.

## What went well

- The reasons hold up against what lenders said: on 20,000 real 2025 denials the lender's stated reason is the
  model's first reason 71.1% of the time, and in its top four 93.0%.
- Training is fast enough to iterate: 5 epochs over 4.6M applications in 76 s on CPU, early-stopped on 2024.

## What did not

- The validation AUC of 0.88 looked too good for data without credit scores, so a drop-one-input ablation checked
  where the skill comes from (`reports/ablation.json`): no single input collapses it (the largest loss is
  debt-to-income, 0.870 to 0.830), so there is no hidden leak.
- The fairness audit fails for applicants aged 62+ (model AIR 0.761 vs the lenders' 0.930) and Black applicants
  (0.759 vs 0.827). Integrated-gradient gaps show why (`reports/fairness-drivers.json`): for age it is mostly
  `loan_purpose` (51% of applicants 62+ refinance vs 28%), for race mostly debt-to-income, loan-to-value and loan
  type. Removing any single input does not fix both gaps (the ablation doubles as that search). The model card
  says "not approved"; a joint search (constraints or reweighting) is the next step.
- XGBoost is 0.0014 AUC ahead of the network. The network is kept for serving (preprocessing and explanations
  in one SavedModel behind TF Serving), and the README says so.

## Carried into Sprint 3

- Serving must use the `explain` signature for every referral, so the reasons shown are the model's own.
