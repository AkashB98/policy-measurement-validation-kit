"""Full report assembly: rates, error estimates, drift, and the
publish gate, in one deterministic document."""

from __future__ import annotations

from typing import Dict, Sequence

from generator import batch_counts, generate_dataset, subgroup_counts
from measure import drift_check, estimate_errors, evaluate_gate, headline_rates


def build_report(rows: Sequence[Dict] | None = None) -> Dict:
    data = list(rows) if rows is not None else generate_dataset("planted")
    return {
        "dataset": {
            "rows": len(data),
            "subgroup_counts": subgroup_counts(data),
            "batch_counts": {str(k): v for k, v in
                             sorted(batch_counts(data).items())},
            "synthetic": True,
            "program": "fictional policy-outcome measurement program",
        },
        "rates": headline_rates(data),
        "errors": estimate_errors(data),
        "drift": drift_check(data),
        "gate": evaluate_gate(data),
    }
