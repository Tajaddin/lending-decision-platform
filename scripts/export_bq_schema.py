"""Write infra/terraform/schemas/decisions.json from ldp.streaming.bq.decisions_schema().

tests/test_infra.py fails if the committed file and the Python schema disagree.
"""

from __future__ import annotations

import json
from pathlib import Path

from ldp.streaming.bq import decisions_schema

OUT = Path(__file__).resolve().parents[1] / "infra" / "terraform" / "schemas" / "decisions.json"


def render() -> str:
    fields = [{"name": f.name, "type": f.field_type, "mode": f.mode} for f in decisions_schema()]
    return json.dumps(fields, indent=2) + "\n"


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(), encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
