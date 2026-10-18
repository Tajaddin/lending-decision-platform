#!/usr/bin/env bash
# Download one year of the national HMDA snapshot LAR, verify the zip, and unzip it.
#
#   scripts/download_hmda.sh 2025 data/raw
#
# Resumes a partial download, retries with a delay, and refuses a zip that fails its CRC check.
# Environment (for tests and mirrors): HMDA_BASE_URL, HMDA_ATTEMPTS (default 5), HMDA_RETRY_DELAY (seconds, 5).
set -euo pipefail

usage() {
  echo "usage: $(basename "$0") YEAR OUT_DIR" >&2
  exit 2
}

[[ $# -eq 2 ]] || usage
year=$1
out=$2
[[ $year =~ ^20[0-9]{2}$ ]] || { echo "invalid year: $year" >&2; exit 2; }

base=${HMDA_BASE_URL:-https://files.ffiec.cfpb.gov/static-data/snapshot}
attempts=${HMDA_ATTEMPTS:-5}
delay=${HMDA_RETRY_DELAY:-5}
url="$base/$year/${year}_public_lar_csv.zip"
zip="$out/${year}_public_lar_csv.zip"
csv="$out/${year}_public_lar_csv.csv"

mkdir -p "$out"
if [[ -s $csv ]]; then
  echo "exists: $csv"
  exit 0
fi

for ((i = 1; i <= attempts; i++)); do
  if curl -fsSL --continue-at - -o "$zip" "$url"; then
    break
  fi
  if ((i == attempts)); then
    echo "download failed after $attempts attempts: $url" >&2
    exit 1
  fi
  echo "attempt $i failed, retrying in ${delay}s" >&2
  sleep "$delay"
done

if ! python3 -m zipfile -t "$zip" >/dev/null 2>&1; then
  echo "corrupt zip, removed: $zip" >&2
  rm -f "$zip"
  exit 1
fi
python3 -m zipfile -e "$zip" "$out"
if [[ ! -s $csv ]]; then
  echo "zip did not contain ${year}_public_lar_csv.csv" >&2
  exit 1
fi
echo "ok: $csv ($(wc -l <"$csv") lines)"
