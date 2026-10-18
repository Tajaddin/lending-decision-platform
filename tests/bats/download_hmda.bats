#!/usr/bin/env bats
# Tests for scripts/download_hmda.sh against a local file:// mirror (no network).

setup() {
  SCRIPT="$BATS_TEST_DIRNAME/../../scripts/download_hmda.sh"
  MIRROR="$BATS_TEST_TMPDIR/mirror"
  OUT="$BATS_TEST_TMPDIR/out"
  mkdir -p "$MIRROR/2025"
  printf 'activity_year,lei\n2025,ABCDEFGHIJKLMNOPQRST\n' >"$BATS_TEST_TMPDIR/2025_public_lar_csv.csv"
  (cd "$BATS_TEST_TMPDIR" && python3 -m zipfile -c "$MIRROR/2025/2025_public_lar_csv.zip" 2025_public_lar_csv.csv)
  export HMDA_BASE_URL="file://$MIRROR" HMDA_RETRY_DELAY=0 HMDA_ATTEMPTS=2
}

@test "prints usage and exits 2 without arguments" {
  run bash "$SCRIPT"
  [ "$status" -eq 2 ]
  [[ "$output" == *"usage:"* ]]
}

@test "rejects a malformed year" {
  run bash "$SCRIPT" 25 "$OUT"
  [ "$status" -eq 2 ]
  [[ "$output" == *"invalid year"* ]]
}

@test "downloads, verifies and unzips a year" {
  run bash "$SCRIPT" 2025 "$OUT"
  [ "$status" -eq 0 ]
  [[ "$output" == *"ok: $OUT/2025_public_lar_csv.csv (2 lines)"* ]]
  [ -s "$OUT/2025_public_lar_csv.csv" ]
}

@test "skips a year whose CSV already exists" {
  mkdir -p "$OUT"
  echo "already here" >"$OUT/2025_public_lar_csv.csv"
  run bash "$SCRIPT" 2025 "$OUT"
  [ "$status" -eq 0 ]
  [[ "$output" == *"exists:"* ]]
  [ "$(cat "$OUT/2025_public_lar_csv.csv")" = "already here" ]
}

@test "removes a corrupt zip and fails" {
  echo "not a zip" >"$MIRROR/2025/2025_public_lar_csv.zip"
  run bash "$SCRIPT" 2025 "$OUT"
  [ "$status" -eq 1 ]
  [[ "$output" == *"corrupt zip"* ]]
  [ ! -e "$OUT/2025_public_lar_csv.zip" ]
}

@test "retries and gives up when the file is missing" {
  run bash "$SCRIPT" 2024 "$OUT"
  [ "$status" -eq 1 ]
  [[ "$output" == *"attempt 1 failed"* ]]
  [[ "$output" == *"download failed after 2 attempts"* ]]
}
