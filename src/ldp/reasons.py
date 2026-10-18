"""Turn the model's attributions into adverse-action reasons (ECOA / Regulation B allows up to four).

A positive attribution pushed the application toward denial. Only inputs that map to a reason a lender could
state are eligible (loan program or channel codes are not reasons), several inputs can map to one reason
(property value and loan-to-value are both "collateral"), and each reason is listed once.
This module has no TensorFlow dependency: the API uses it on TF Serving's explain output.
"""

from __future__ import annotations

from dataclasses import dataclass

MAX_REASONS = 4

# HMDA denial_reason codes and the reason text each one corresponds to (used to check the model's reasons
# against what lenders reported on real denials).
HMDA_DENIAL_REASONS = {
    "1": "Excessive obligations in relation to income",
    "4": "Value or type of collateral not sufficient",
}


@dataclass(frozen=True)
class Reason:
    reason: str
    features: tuple[str, ...]
    contribution: float  # logit units


def top_reasons(
    attributions: list[float],
    groups: list[dict[str, str]],
    max_reasons: int = MAX_REASONS,
    min_contribution: float = 0.0,
) -> list[Reason]:
    if len(attributions) != len(groups):
        raise ValueError(f"{len(attributions)} attributions for {len(groups)} groups")
    by_reason: dict[str, list[tuple[str, float]]] = {}
    for value, group in zip(attributions, groups, strict=True):
        if group.get("reason") and value > min_contribution:
            by_reason.setdefault(group["reason"], []).append((group["name"], float(value)))
    ranked = sorted(by_reason.items(), key=lambda kv: -sum(v for _, v in kv[1]))
    return [
        Reason(
            reason=text,
            features=tuple(n for n, _ in sorted(items, key=lambda x: -x[1])),
            contribution=round(sum(v for _, v in items), 4),
        )
        for text, items in ranked[:max_reasons]
    ]
