// Load test for POST /v1/decisions at a fixed arrival rate (k6, JavaScript).
//
//   docker run --rm --network ldp_default -v "$PWD/bench/k6:/scripts" -v "$PWD/reports/k6:/out" \
//     -e BASE_URL=http://api:8080 -e API_KEY=... -e RATE=50 grafana/k6:1.3.0 run /scripts/decisions.js
//
// Each request carries one real 2025 application (bench/k6/applications.json, from scripts/make_sample.py).
import http from "k6/http";
import { check } from "k6";
import { SharedArray } from "k6/data";

const applications = new SharedArray("applications", () => JSON.parse(open("./applications.json")));
const RATE = Number(__ENV.RATE || 50);
const DURATION = __ENV.DURATION || "60s";
const BASE_URL = __ENV.BASE_URL || "http://localhost:8200";

export const options = {
  scenarios: {
    steady: {
      executor: "constant-arrival-rate",
      rate: RATE,
      timeUnit: "1s",
      duration: DURATION,
      preAllocatedVUs: Math.max(20, RATE),
      maxVUs: Math.max(100, RATE * 4),
    },
  },
  summaryTrendStats: ["avg", "min", "med", "p(90)", "p(95)", "p(99)", "max"],
  thresholds: {
    http_req_failed: ["rate<0.01"],
    "http_req_duration{expected_response:true}": ["p(99)<250"],
  },
};

export default function () {
  const application = applications[Math.floor(Math.random() * applications.length)];
  const response = http.post(`${BASE_URL}/v1/decisions`, JSON.stringify({ applications: [application] }), {
    headers: { "Content-Type": "application/json", "X-API-Key": __ENV.API_KEY },
    tags: { name: "decisions" },
  });
  check(response, {
    "status is 200": (r) => r.status === 200,
    "has a decision": (r) => r.status === 200 && r.json("decisions.0.decision") !== undefined,
  });
}

export function handleSummary(data) {
  const m = data.metrics;
  const line = `rate ${RATE}/s: ${m.http_reqs.values.count} requests, failed ${(100 * m.http_req_failed.values.rate).toFixed(2)}%, ` +
    `p50 ${m.http_req_duration.values.med.toFixed(1)} ms, p95 ${m.http_req_duration.values["p(95)"].toFixed(1)} ms, ` +
    `p99 ${m.http_req_duration.values["p(99)"].toFixed(1)} ms\n`;
  return { [`/out/k6-r${RATE}.json`]: JSON.stringify(data, null, 2), stdout: line };
}
