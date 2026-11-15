"""Measurement core: error estimation, Rogan-Gladen correction,
drift, and the publish gate.

The annotator's labels are compared against the human gold labels
(loaded as given — see ``generator.py`` for where gold comes from).
From that comparison:

- **Sensitivity** = TP / (TP + FN): of the items gold calls
  positive, the share the annotator also calls positive.
- **Specificity** = TN / (TN + FP): of the items gold calls
  negative, the share the annotator also calls negative.
- **Error rate** = (FP + FN) / N.

The headline rate is corrected with the **Rogan-Gladen** estimator.
If the annotator's apparent (annotated) prevalence is ``p``, its
sensitivity ``Se`` and specificity ``Sp``, then the prevalence
consistent with those error rates is

    corrected = (p + Sp - 1) / (Se + Sp - 1)

clamped to [0, 1]. The correction assumes the gold labels are truth
and that the estimated Se/Sp describe the whole annotated corpus —
both assumptions are stated, not hidden (see README honest limits).

One algebraic fact is reported, not hidden: when Se/Sp are estimated
on the *same* rows the headline rate is computed from, the pooled
correction recovers the gold rate exactly — it is an identity, not
evidence the correction "works". The non-trivial question is whether
error rates estimated on one batch *transport* to another, so the
report also carries a transport diagnostic: Se/Sp estimated on
batch 1 only, applied to batch 2's naive rate, scored against
batch 2's gold rate. Drift is exactly what makes transport fail,
which is why the gate checks drift separately.

**Drift** is stated plainly: the drift statistic is the absolute
difference between the batch-2 and batch-1 overall error rates.
Per-subgroup drift is reported alongside it.

The **publish gate** compares three quantities against stated
floors and returns PUBLISH or REFUSE with reasons — and nothing
more. There is deliberately no evidence bundle or audit trail here;
that is ``model-governance-evidence-kit`` territory.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

# Publish floors (stated, not tuned per dataset).
MAX_OVERALL_ERROR = 0.10
MAX_SUBGROUP_ERROR = 0.15
MAX_DRIFT = 0.05

DEFAULT_FLOORS = {
    "max_overall_error": MAX_OVERALL_ERROR,
    "max_subgroup_error": MAX_SUBGROUP_ERROR,
    "max_drift": MAX_DRIFT,
}


def _r6(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(float(x), 6)


def confusion(rows: Sequence[Dict]) -> Dict[str, int]:
    tp = fp = tn = fn = 0
    for r in rows:
        gold, ann = r["gold_label"], r["annotated_label"]
        if gold == 1 and ann == 1:
            tp += 1
        elif gold == 0 and ann == 1:
            fp += 1
        elif gold == 0 and ann == 0:
            tn += 1
        else:
            fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def error_stats(rows: Sequence[Dict]) -> Dict:
    """Sensitivity / specificity / error rate for one slice of rows."""
    rows = list(rows)
    c = confusion(rows)
    n = len(rows)
    positives = c["tp"] + c["fn"]
    negatives = c["tn"] + c["fp"]
    sens = c["tp"] / positives if positives else None
    spec = c["tn"] / negatives if negatives else None
    error = (c["fp"] + c["fn"]) / n if n else None
    return {
        "n": n,
        "positives": positives,
        "negatives": negatives,
        "tp": c["tp"], "fp": c["fp"], "tn": c["tn"], "fn": c["fn"],
        "sensitivity": _r6(sens),
        "specificity": _r6(spec),
        "error_rate": _r6(error),
        "accuracy": _r6(1.0 - error) if error is not None else None,
    }


def estimate_errors(rows: Sequence[Dict]) -> Dict:
    """Overall, per-subgroup, and per-batch error estimates."""
    rows = list(rows)
    by_subgroup: Dict[str, Dict] = {}
    for sg in sorted({r["subgroup"] for r in rows}):
        by_subgroup[sg] = error_stats([r for r in rows if r["subgroup"] == sg])
    by_batch: Dict[str, Dict] = {}
    for b in sorted({r["batch"] for r in rows}):
        by_batch[str(b)] = error_stats([r for r in rows if r["batch"] == b])
    by_subgroup_batch: Dict[str, Dict] = {}
    for sg in sorted({r["subgroup"] for r in rows}):
        inner: Dict[str, Dict] = {}
        for b in sorted({r["batch"] for r in rows}):
            cell = [r for r in rows if r["subgroup"] == sg and r["batch"] == b]
            if cell:
                inner[str(b)] = error_stats(cell)
        by_subgroup_batch[sg] = inner
    return {
        "overall": error_stats(rows),
        "by_subgroup": by_subgroup,
        "by_batch": by_batch,
        "by_subgroup_batch": by_subgroup_batch,
    }


def rogan_gladen(apparent: float, sensitivity: float,
                 specificity: float) -> Dict:
    """Rogan-Gladen corrected prevalence from an apparent rate.

    Returns the corrected value clamped to [0, 1] plus the raw
    value and whether clamping happened. Raises ValueError when
    sensitivity + specificity <= 1 (the annotator carries no
    information about the label, so no correction exists).
    """
    denom = sensitivity + specificity - 1.0
    if denom <= 0:
        raise ValueError(
            "sensitivity + specificity must exceed 1 for a "
            "Rogan-Gladen correction (annotator is uninformative)")
    raw = (apparent + specificity - 1.0) / denom
    clamped_value = min(1.0, max(0.0, raw))
    return {
        "apparent_rate": _r6(apparent),
        "sensitivity": _r6(sensitivity),
        "specificity": _r6(specificity),
        "denominator": _r6(denom),
        "corrected_rate_raw": _r6(raw),
        "corrected_rate": _r6(clamped_value),
        "clamped": clamped_value != raw,
    }


def headline_rates(rows: Sequence[Dict]) -> Dict:
    """True vs naive (annotated) vs Rogan-Gladen corrected rates.

    ``transport`` is the non-trivial diagnostic: sensitivity and
    specificity estimated on batch 1 only, applied to batch 2's
    naive rate, scored against batch 2's gold rate.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("no rows")
    overall = error_stats(rows)
    if overall["sensitivity"] is None or overall["specificity"] is None:
        raise ValueError("gold set needs both positive and negative items")
    true_rate = sum(r["gold_label"] for r in rows) / len(rows)
    naive_rate = sum(r["annotated_label"] for r in rows) / len(rows)
    correction = rogan_gladen(naive_rate, overall["sensitivity"],
                              overall["specificity"])

    transport = None
    batch_1 = [r for r in rows if r["batch"] == 1]
    batch_2 = [r for r in rows if r["batch"] == 2]
    if batch_1 and batch_2:
        est = error_stats(batch_1)
        if est["sensitivity"] is not None and est["specificity"] is not None:
            t_true = sum(r["gold_label"] for r in batch_2) / len(batch_2)
            t_naive = sum(r["annotated_label"] for r in batch_2) / len(batch_2)
            t_corr = rogan_gladen(t_naive, est["sensitivity"],
                                  est["specificity"])
            transport = {
                "estimated_on": "batch_1",
                "applied_to": "batch_2",
                "sensitivity": est["sensitivity"],
                "specificity": est["specificity"],
                "true_rate": _r6(t_true),
                "naive_rate": _r6(t_naive),
                "corrected_rate": t_corr["corrected_rate"],
                "naive_abs_error": _r6(abs(t_naive - t_true)),
                "corrected_abs_error": _r6(
                    abs(t_corr["corrected_rate"] - t_true)),
            }
    return {
        "n": len(rows),
        "true_rate": _r6(true_rate),
        "naive_rate": _r6(naive_rate),
        "corrected_rate": correction["corrected_rate"],
        "naive_abs_error": _r6(abs(naive_rate - true_rate)),
        "corrected_abs_error": _r6(abs(correction["corrected_rate"] - true_rate)),
        "correction": correction,
        "transport": transport,
    }


def drift_check(rows: Sequence[Dict]) -> Dict:
    """Batch drift: per-batch error rates + the drift statistic.

    The drift statistic is |error_rate(batch 2) - error_rate(batch 1)|
    on the pooled data; per-subgroup drift uses the same definition
    inside each subgroup.
    """
    rows = list(rows)
    batches = sorted({r["batch"] for r in rows})
    if batches != [1, 2]:
        raise ValueError("drift check needs exactly batches 1 and 2")
    per_batch = {str(b): error_stats([r for r in rows if r["batch"] == b])
                 for b in batches}
    statistic = abs(per_batch["2"]["error_rate"] - per_batch["1"]["error_rate"])
    by_subgroup: Dict[str, Dict] = {}
    for sg in sorted({r["subgroup"] for r in rows}):
        e1 = error_stats([r for r in rows
                          if r["subgroup"] == sg and r["batch"] == 1])
        e2 = error_stats([r for r in rows
                          if r["subgroup"] == sg and r["batch"] == 2])
        by_subgroup[sg] = {
            "batch_1_error_rate": e1["error_rate"],
            "batch_2_error_rate": e2["error_rate"],
            "drift": _r6(abs(e2["error_rate"] - e1["error_rate"])),
        }
    return {
        "per_batch": per_batch,
        "drift_statistic": _r6(statistic),
        "definition": "abs(error_rate_batch_2 - error_rate_batch_1)",
        "by_subgroup": by_subgroup,
    }


def evaluate_gate(rows: Sequence[Dict],
                  floors: Optional[Dict[str, float]] = None) -> Dict:
    """PUBLISH or REFUSE, with reasons. Nothing more.

    Refuses when any of these holds:
    - overall error rate exceeds ``max_overall_error``;
    - any subgroup's error rate exceeds ``max_subgroup_error``;
    - the drift statistic exceeds ``max_drift``;
    - sensitivity/specificity cannot be estimated for a slice the
      gate must judge (no gold positives or negatives there).
    """
    floors = dict(DEFAULT_FLOORS if floors is None else floors)
    rows = list(rows)
    errors = estimate_errors(rows)
    drift = drift_check(rows)
    overall_error = errors["overall"]["error_rate"]

    reasons: List[str] = []
    if overall_error is None or overall_error > floors["max_overall_error"]:
        reasons.append(
            f"overall error rate {overall_error} exceeds floor "
            f"{floors['max_overall_error']}")
    subgroup_errors = {sg: s["error_rate"]
                       for sg, s in errors["by_subgroup"].items()}
    for sg in sorted(subgroup_errors):
        err = subgroup_errors[sg]
        stats = errors["by_subgroup"][sg]
        if stats["sensitivity"] is None or stats["specificity"] is None:
            reasons.append(
                f"subgroup {sg!r}: sensitivity/specificity not estimable "
                f"(positives={stats['positives']}, "
                f"negatives={stats['negatives']})")
        elif err is not None and err > floors["max_subgroup_error"]:
            reasons.append(
                f"subgroup {sg!r} error rate {err} exceeds floor "
                f"{floors['max_subgroup_error']}")
    drift_stat = drift["drift_statistic"]
    if drift_stat is None or drift_stat > floors["max_drift"]:
        reasons.append(
            f"drift statistic {drift_stat} exceeds floor "
            f"{floors['max_drift']}")

    worst_subgroup = None
    if subgroup_errors:
        worst_subgroup = max(
            subgroup_errors,
            key=lambda sg: (subgroup_errors[sg] is not None,
                            subgroup_errors[sg] or 0.0))
    return {
        "verdict": "REFUSE" if reasons else "PUBLISH",
        "reasons": reasons,
        "floors": floors,
        "checks": {
            "overall_error_rate": overall_error,
            "subgroup_error_rates": subgroup_errors,
            "worst_subgroup": worst_subgroup,
            "worst_subgroup_error_rate": (
                subgroup_errors.get(worst_subgroup)
                if worst_subgroup else None),
            "drift_statistic": drift_stat,
        },
    }
