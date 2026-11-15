"""Deterministic synthetic policy-outcome generator.

SYNTHETIC DATA ONLY. Every item is fabricated by a seeded PRNG about
a fictional policy-outcome measurement program. No real person,
post, or company is used anywhere.

One row per item:

  item_id, subgroup, batch, gold_label, annotated_label

- ``gold_label`` — the human gold label (1 = the policy-relevant
  outcome is present, 0 = absent). The gold set is *loaded as given*:
  this kit never builds or adjudicates it. How a trustworthy gold
  set is built — human agreement, adjudication, leakage checks — is
  the sibling repo ``human-feedback-golden-set-kit``'s job.
- ``annotated_label`` — the label produced by a simulated
  LLM-assisted annotator. Its error structure is planted (below), so
  the measurement machinery can be checked against a known answer.
- ``subgroup`` — the slice the item belongs to (``standard``,
  ``dialect``, or ``sarcasm``). ``batch`` is 1 or 2: two annotation
  batches produced at different times, so drift can be checked.

Planted error structure (sensitivity / specificity of the annotator
against gold):

- ``planted`` variant — the headline dataset. The annotator is good
  on ``standard`` and ``dialect`` items, much worse on the planted
  subgroup ``sarcasm``, and slightly worse in batch 2 than batch 1
  (drift). Pooled, the overall error stays under the publish floor
  while the ``sarcasm`` subgroup breaches its own floor — the exact
  failure mode the gate exists to catch.
- ``clean`` variant — the control. One uniformly good annotator
  (sensitivity 0.93, specificity 0.95) on every subgroup and batch,
  so the gate publishes.

True prevalence is 0.30 in every subgroup.
"""

from __future__ import annotations

import csv
import io
import json
import random
from typing import Dict, List, Sequence

DEFAULT_SEED = 42
DEFAULT_N = 2400
TRUE_PREVALENCE = 0.30
PLANTED_SUBGROUP = "sarcasm"

SUBGROUPS = ("standard", "dialect", "sarcasm")
SUBGROUP_WEIGHTS = (0.58, 0.30, 0.12)
BATCHES = (1, 2)
VARIANTS = ("planted", "clean")

ROW_FIELDS = ["item_id", "subgroup", "batch", "gold_label", "annotated_label"]

# (sensitivity, specificity) per variant / batch / subgroup.
_ERROR_MODEL = {
    "planted": {
        1: {"standard": (0.94, 0.96), "dialect": (0.93, 0.95),
            "sarcasm": (0.70, 0.78)},
        2: {"standard": (0.92, 0.94), "dialect": (0.91, 0.93),
            "sarcasm": (0.64, 0.72)},
    },
    "clean": {
        1: {"standard": (0.93, 0.95), "dialect": (0.93, 0.95),
            "sarcasm": (0.93, 0.95)},
        2: {"standard": (0.93, 0.95), "dialect": (0.93, 0.95),
            "sarcasm": (0.93, 0.95)},
    },
}


def annotator_params(variant: str, batch: int, subgroup: str):
    """Return the planted (sensitivity, specificity) for a cell."""
    try:
        return _ERROR_MODEL[variant][batch][subgroup]
    except KeyError as exc:
        raise ValueError(
            f"unknown variant/batch/subgroup: {variant!r}/{batch!r}/"
            f"{subgroup!r}") from exc


def generate_dataset(variant: str = "planted", seed: int = DEFAULT_SEED,
                     n: int = DEFAULT_N) -> List[Dict]:
    """Return synthetic rows, deterministic for ``seed``.

    The first half of the rows are batch 1, the second half batch 2.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant!r} (choose from {VARIANTS})")
    if n < 20 or n % 2:
        raise ValueError("n must be an even number >= 20")
    rng = random.Random(seed)
    cum: List[float] = []
    acc = 0.0
    for w in SUBGROUP_WEIGHTS:
        acc += w
        cum.append(acc)
    rows: List[Dict] = []
    for i in range(n):
        batch = 1 if i < n // 2 else 2
        u = rng.random()
        subgroup = next(sg for sg, c in zip(SUBGROUPS, cum) if u <= c)
        gold = 1 if rng.random() < TRUE_PREVALENCE else 0
        sens, spec = annotator_params(variant, batch, subgroup)
        if gold == 1:
            annotated = 1 if rng.random() < sens else 0
        else:
            annotated = 0 if rng.random() < spec else 1
        rows.append({
            "item_id": f"item_{i + 1:05d}",
            "subgroup": subgroup,
            "batch": batch,
            "gold_label": gold,
            "annotated_label": annotated,
        })
    return rows


def subgroup_counts(rows: Sequence[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["subgroup"]] = counts.get(r["subgroup"], 0) + 1
    return counts


def batch_counts(rows: Sequence[Dict]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for r in rows:
        counts[r["batch"]] = counts.get(r["batch"], 0) + 1
    return counts


def rows_to_jsonl(rows: Sequence[Dict]) -> str:
    return "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)


def rows_to_json(rows: Sequence[Dict]) -> str:
    return json.dumps(list(rows), sort_keys=True) + "\n"


def rows_to_csv(rows: Sequence[Dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=ROW_FIELDS)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    return buf.getvalue()


def _coerce_row(raw: Dict, lineno: int) -> Dict:
    try:
        return {
            "item_id": str(raw["item_id"]),
            "subgroup": str(raw["subgroup"]),
            "batch": int(raw["batch"]),
            "gold_label": int(raw["gold_label"]),
            "annotated_label": int(raw["annotated_label"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"line {lineno}: bad row {raw!r}: {exc}") from exc


def load_rows(path: str) -> List[Dict]:
    """Load rows from .jsonl, .csv, or a JSON array file."""
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    if path.endswith(".csv"):
        reader = csv.DictReader(io.StringIO(text))
        return [_coerce_row(r, i + 2) for i, r in enumerate(reader)]
    stripped = text.strip()
    if stripped.startswith("["):
        data = json.loads(stripped)
        if not isinstance(data, list):
            raise ValueError("JSON input must be an array of rows")
        return [_coerce_row(r, i + 1) for i, r in enumerate(data)]
    rows = []
    for i, line in enumerate(text.splitlines()):
        if line.strip():
            rows.append(_coerce_row(json.loads(line), i + 1))
    return rows


def write_rows(rows: Sequence[Dict], path: str, fmt: str = "jsonl") -> None:
    if fmt == "csv":
        text = rows_to_csv(rows)
    elif fmt == "json":
        text = rows_to_json(rows)
    else:
        text = rows_to_jsonl(rows)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
