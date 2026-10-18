"""Cut the small real-data fixtures the tests use from the full LAR files (public domain, CFPB).

    python scripts/make_fixtures.py ~/data/hmda/raw

Writes tests/fixtures/lar_<year>.csv: every Nth row of each year's file, so the fixtures cover real
encodings (sentinels, DTI bins, codes) without committing gigabytes. Deterministic: same input, same rows.
"""

from __future__ import annotations

import sys
from pathlib import Path

EVERY = 20_000
YEARS = (2023, 2024, 2025)


def main(raw_dir: str) -> None:
    out_dir = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for year in YEARS:
        src = Path(raw_dir).expanduser() / f"{year}_public_lar_csv.csv"
        dst = out_dir / f"lar_{year}.csv"
        kept = 0
        with src.open(encoding="utf-8", newline="") as fin, dst.open("w", encoding="utf-8", newline="") as fout:
            fout.write(fin.readline())
            for i, line in enumerate(fin):
                if i % EVERY == 0:
                    fout.write(line)
                    kept += 1
        print(f"{dst.name}: {kept} rows")


if __name__ == "__main__":
    main(sys.argv[1])
