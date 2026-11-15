"""Data-quality gate for policy-outcome rows.

Reports issues, never raises on bad data, so the CLI ``check`` gate
can map data problems to exit code 2 — distinct from a publish
refusal (exit 1) and from a publish (exit 0).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

REQUIRED_FIELDS = ("item_id", "subgroup", "batch", "gold_label",
                   "annotated_label")


def validate_row(row, index: int = 0) -> List[str]:
    issues: List[str] = []
    if not isinstance(row, dict):
        return [f"row {index}: not an object"]
    for field in REQUIRED_FIELDS:
        if field not in row:
            issues.append(f"row {index}: missing field {field!r}")
    if issues:
        return issues
    if not isinstance(row["item_id"], str) or not row["item_id"]:
        issues.append(f"row {index}: item_id must be a non-empty string")
    if not isinstance(row["subgroup"], str) or not row["subgroup"]:
        issues.append(f"row {index}: subgroup must be a non-empty string")
    batch = row["batch"]
    if isinstance(batch, bool) or batch not in (1, 2):
        issues.append(f"row {index}: batch must be 1 or 2")
    for field in ("gold_label", "annotated_label"):
        value = row[field]
        if isinstance(value, bool) or value not in (0, 1):
            issues.append(f"row {index}: {field} must be 0 or 1")
    return issues


def validate_dataset(rows: Sequence[Dict]) -> List[str]:
    issues: List[str] = []
    if not rows:
        return ["dataset: no rows"]
    seen = set()
    subgroups = set()
    batches = set()
    for i, row in enumerate(rows):
        issues.extend(validate_row(row, i))
        if isinstance(row, dict):
            item_id = row.get("item_id")
            if item_id in seen:
                issues.append(f"row {i}: duplicate item_id {item_id!r}")
            seen.add(item_id)
            if isinstance(row.get("subgroup"), str):
                subgroups.add(row["subgroup"])
            if row.get("batch") in (1, 2):
                batches.add(row["batch"])
        if len(issues) > 50:
            issues.append("dataset: too many issues, truncated")
            break
    if not issues:
        if len(subgroups) < 2:
            issues.append("dataset: fewer than 2 subgroups present")
        if batches != {1, 2}:
            issues.append("dataset: both batches (1 and 2) must be present")
    return issues
