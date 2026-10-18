# Sprint 1 retro: data foundation

Dates: 2026-10-18. Milestone: [Sprint 1 - Data foundation](https://github.com/Tajaddin/lending-decision-platform/milestone/1).

**Goal:** the 2023-2025 HMDA files land in Delta, are typed, quality-checked and audited for leakage, and the model
population exists with an out-of-time split. **Met.**

## Delivered

| Issue | Points | Result |
|---|---|---|
| #1 scaffold, CI, secret scan | 2 | lint, tests with coverage, shellcheck/bats, gitleaks and pip-audit jobs; green on the first push |
| #2 download script | 2 | resume, retries, zip check; 6 bats tests |
| #3 bronze | 3 | 37,256,793 records as text in Delta, per-year `replaceWhere` |
| #4 silver | 5 | typed and decoded; Spark decoders proven identical to the Python ones on real rows |
| #5 quality checks | 3 | 12 checks per year, 0 critical failures; 1 real warning (1.3-2.1% of records have no state) |
| #6 leakage audit, population, split | 5 | 15.1M applications; the gate stopped the first full run on a leaking field |
| #7 benchmark | 2 | full pipeline 384 s; silver stage on 13.5M rows: 553 s on 1 core, 93 s on 16 (5.95x) |

22 of 22 committed points done.

## What went well

- The leakage gate earned its place on day one: the first full run stopped on
  `initially_payable_to_institution` (filled in for 99.7% of approvals vs 87.4% of denials), a leak no one had
  listed. It is now excluded, with the numbers in ADR 0002.
- Checking value encodings on 300,000 raw rows before writing parsers meant the DTI parser covered every bin
  on the first real run (`dti_parser_coverage` = 1.0).

## What did not

- The first full run's stage timings were lost because only the tail of the log was kept. Action taken: runs
  now write the whole log to a file and the JSON report carries the timings.
- The leakage audit flagged three fields at random on the 600-row test fixtures. Action taken: the gate only
  enforces with at least 1,000 rows per class and says `enforced: false` otherwise.
- A gitleaks false positive (a CI cache key) blocked the first push until allowlisted precisely.
- The development machine's Docker engine crashed twice: Windows had run out of commit memory because a
  screen-capture service had leaked 86 GB. Action: check commit before heavy runs.

- Scaling flattens: 2 cores 1.93x, 4 cores 2.70x, 8 cores 4.97x, 16 cores 5.95x. The 2025 bronze partition is
  about 20 Parquet files of ~30 MB, so reads cannot fan out much further, and the write ends in one partition.
  Not worth fixing at this size (the whole pipeline is 6.4 minutes); noted for a cluster run.

## Carried into Sprint 2

- Re-export the splits with `denial_reason_1` so model reasons can be compared with lender-reported reasons
  (done before the sprint closed: `--stages export`).
