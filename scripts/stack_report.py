"""Record what the running stack measured after a replay: rows in BigQuery, the monitor's PSI/AIR, alerts.

    make up && make replay && python scripts/stack_report.py   (writes reports/replay-monitor.json)

Reads the BigQuery emulator's REST API and Prometheus; standard library only.
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

BQ = "http://localhost:9050/bigquery/v2/projects/ldp-local/queries"
PROM = "http://localhost:9095/api/v1"


def _get(url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def prom(query: str) -> list[dict]:
    return _get(f"{PROM}/query?" + urllib.parse.urlencode({"query": query}))["data"]["result"]


def main() -> int:
    rows = _get(
        BQ, {"query": "SELECT decision, COUNT(*) AS n FROM lending.decisions GROUP BY decision", "useLegacySql": False}
    ).get("rows", [])
    decisions = {r["f"][0]["v"]: int(r["f"][1]["v"]) for r in rows}
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "bigquery_decisions": decisions,
        "monitor_window_rows": float(prom("ldp_monitor_window_rows")[0]["value"][1]),
        "referral_rate": float(prom("ldp_referral_rate")[0]["value"][1]),
        "score_psi": float(prom("ldp_score_psi")[0]["value"][1]),
        "feature_psi": {r["metric"]["feature"]: float(r["value"][1]) for r in prom("sort_desc(ldp_feature_psi)")},
        "model_air": {
            f"{r['metric']['dimension']}:{r['metric']['group']}": float(r["value"][1]) for r in prom("ldp_model_air")
        },
        "sink_rows": {
            r["metric"].get("outcome", ""): float(r["value"][1]) for r in prom("sum by (outcome) (ldp_sink_rows_total)")
        },
        "alerts": [
            {"alert": a["labels"]["alertname"], "state": a["state"], "labels": a["labels"]}
            for a in _get(f"{PROM}/alerts")["data"]["alerts"]
        ],
    }
    out = Path(__file__).resolve().parents[1] / "reports" / "replay-monitor.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("bigquery_decisions", "score_psi", "referral_rate")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
