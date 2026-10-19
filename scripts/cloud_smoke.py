"""Smoke test the deployed decision API: readiness, decisions with and without a key, and the events in BigQuery.

    python scripts/cloud_smoke.py https://<api>.run.app --key-file ~/.config/ldp/api-keys --project <project-id>

Needs `gcloud` logged in for the BigQuery check. Writes reports/cloud/smoke.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APPLICATION = ROOT / "bench" / "k6" / "one-application.json"


def call(url: str, key: str | None, body: bytes | None = None) -> tuple[int, float, dict | None]:
    req = urllib.request.Request(url, method="POST" if body else "GET", data=body)
    req.add_header("content-type", "application/json")
    if key:
        req.add_header("X-API-Key", key)
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            raw, status = r.read(), r.status
    except urllib.error.HTTPError as e:
        raw, status = e.read(), e.code
    ms = round((time.perf_counter() - start) * 1000, 1)
    return status, ms, json.loads(raw) if raw[:1] == b"{" else None


def bigquery_count(project: str, table: str) -> int:
    sql = f"select count(*) as n from `{project}.{table}`"
    out = subprocess.run(
        ["bq", "query", f"--project_id={project}", "--use_legacy_sql=false", "--format=json", sql],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return int(json.loads(out)[0]["n"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--key-file", type=Path, required=True, help="client:key pairs, comma separated")
    parser.add_argument("--project", required=True)
    parser.add_argument("--table", default="lending.decisions")
    parser.add_argument("--decisions", type=int, default=5)
    args = parser.parse_args()

    url = args.url.rstrip("/")
    key = args.key_file.expanduser().read_text().strip().split(",")[0].split(":", 1)[1]
    body = APPLICATION.read_bytes()

    before = bigquery_count(args.project, args.table)
    ready_status, _, ready = call(f"{url}/ready", None)
    timings, decision = [], None
    for _ in range(args.decisions):
        status, ms, decision = call(f"{url}/v1/decisions", key, body)
        assert status == 200, status
        timings.append(ms)
    unauthorized, _, _ = call(f"{url}/v1/decisions", None, body)

    after, waited = before, 0
    while after < before + args.decisions and waited < 120:  # the Pub/Sub subscription writes within seconds
        time.sleep(5)
        waited += 5
        after = bigquery_count(args.project, args.table)

    report = {
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "url": url,
        "ready": {"status": ready_status, **(ready or {})},
        "decision_ms": timings,
        "decision": decision,
        "status_without_key": unauthorized,
        "bigquery_events": {"table": args.table, "before": before, "after": after, "seconds_waited": waited},
        "note": "client on a home connection in Tampa, FL; API in us-central1; latencies include the network",
    }
    out = ROOT / "reports" / "cloud" / "smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
