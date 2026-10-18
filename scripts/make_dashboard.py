"""Generate observability/grafana/dashboards/ldp.json (kept as code so panels stay consistent)."""

from __future__ import annotations

import json
from pathlib import Path

DS = {"type": "prometheus", "uid": "prometheus"}


def panel(
    pid: int,
    title: str,
    exprs: list[tuple[str, str]],
    x: int,
    y: int,
    w: int = 12,
    h: int = 8,
    kind: str = "timeseries",
    unit: str = "short",
    thresholds: list[tuple[float, str]] | None = None,
) -> dict:
    steps = [{"color": "green", "value": None}] + [{"color": c, "value": v} for v, c in (thresholds or [])]
    return {
        "id": pid,
        "title": title,
        "type": kind,
        "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [
            {"refId": chr(65 + i), "datasource": DS, "expr": e, "legendFormat": legend}
            for i, (e, legend) in enumerate(exprs)
        ],
        "fieldConfig": {
            "defaults": {"unit": unit, "thresholds": {"mode": "absolute", "steps": steps}},
            "overrides": [],
        },
        "options": {"legend": {"displayMode": "list", "placement": "bottom"}},
    }


def main() -> None:
    panels = [
        panel(
            1,
            "Decision requests per second by status",
            [('sum by (status) (rate(ldp_http_requests_total{route="/v1/decisions"}[1m]))', "{{status}}")],
            0,
            0,
            unit="reqps",
        ),
        panel(
            2,
            "Decision latency (p50 / p95 / p99)",
            [
                (
                    f'histogram_quantile({q}, sum by (le) (rate(ldp_http_request_seconds_bucket{{route="/v1/decisions"}}[5m])))',
                    f"p{int(q * 100)}",
                )
                for q in (0.5, 0.95, 0.99)
            ],
            12,
            0,
            unit="s",
            thresholds=[(0.25, "red")],
        ),
        panel(
            3,
            "Decisions per second",
            [("sum by (decision) (rate(ldp_decisions_total[1m]))", "{{decision}}")],
            0,
            8,
            unit="reqps",
        ),
        panel(
            4,
            "Referral rate (monitor window)",
            [("ldp_referral_rate", "referral rate")],
            12,
            8,
            kind="stat",
            unit="percentunit",
            w=6,
        ),
        panel(5, "Decisions in monitor window", [("ldp_monitor_window_rows", "rows")], 18, 8, kind="stat", w=6),
        panel(
            6,
            "Score PSI vs 2023",
            [("ldp_score_psi", "score PSI")],
            0,
            16,
            w=6,
            kind="stat",
            thresholds=[(0.10, "orange"), (0.25, "red")],
        ),
        panel(
            7,
            "Input PSI vs 2023 (alert above 0.25)",
            [("ldp_feature_psi", "{{feature}}")],
            6,
            16,
            w=18,
            kind="bargauge",
            thresholds=[(0.10, "orange"), (0.25, "red")],
        ),
        panel(
            8,
            "Adverse impact ratio of the model (alert below 0.80)",
            [("ldp_model_air", "{{dimension}}: {{group}}")],
            0,
            24,
            w=12,
            kind="bargauge",
            thresholds=[(0.80, "green")],
        ),
        panel(
            9,
            "TF Serving predict latency p99 (from the gateway)",
            [("histogram_quantile(0.99, sum by (le) (rate(ldp_tfserving_seconds_bucket[5m])))", "p99")],
            12,
            24,
            unit="s",
        ),
        panel(
            10,
            "Decision events: published and written to BigQuery",
            [
                ("sum by (outcome) (rate(ldp_events_published_total[1m]))", "published {{outcome}}"),
                ("sum by (outcome) (rate(ldp_sink_rows_total[1m]))", "sink {{outcome}}"),
            ],
            0,
            32,
            w=24,
            unit="reqps",
        ),
    ]
    # AIR panel: red below 0.80, green above.
    panels[7]["fieldConfig"]["defaults"]["thresholds"]["steps"] = [
        {"color": "red", "value": None},
        {"color": "green", "value": 0.8},
    ]
    dashboard = {
        "uid": "ldp-overview",
        "title": "Lending decisions: service, drift and fairness",
        "schemaVersion": 41,
        "version": 1,
        "refresh": "30s",
        "time": {"from": "now-1h", "to": "now"},
        "panels": panels,
        "tags": ["ldp"],
        "timezone": "utc",
    }
    out = Path(__file__).resolve().parents[1] / "observability" / "grafana" / "dashboards" / "ldp.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dashboard, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
