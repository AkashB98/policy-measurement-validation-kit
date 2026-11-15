"""Shared row builders for the test suite (not a test module)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def row(item_id="item_00001", subgroup="standard", batch=1,
        gold_label=0, annotated_label=0):
    return {"item_id": item_id, "subgroup": subgroup, "batch": batch,
            "gold_label": gold_label, "annotated_label": annotated_label}


def grid(n_per_cell=100, flip=None):
    """Deterministic hand-built dataset: every subgroup x batch cell
    gets ``n_per_cell`` items, labels copied from gold except where
    ``flip(subgroup, batch, gold, index_in_cell)`` returns True.

    Gold pattern inside a cell: index < n_per_cell // 2 -> positive,
    so each cell is exactly half gold-positive for even counts.
    """
    rows = []
    i = 0
    for subgroup in ("standard", "dialect", "sarcasm"):
        for batch in (1, 2):
            for j in range(n_per_cell):
                i += 1
                gold = 1 if j < n_per_cell // 2 else 0
                ann = gold
                if flip is not None and flip(subgroup, batch, gold, j):
                    ann = 1 - gold
                rows.append(row(f"item_{i:05d}", subgroup, batch, gold, ann))
    return rows
