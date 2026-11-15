"""Golden evals for policy-measurement-validation-kit.

SYNTHETIC DATA ONLY (fictional policy-outcome program). Every eval
is deterministic: run_evals.py writes eval_report.json and the file
must be byte-identical across runs. No timestamps, ports, or temp
paths enter the report - only names, booleans, pinned counts and
hashes.

E3 pins the error statistics and the Rogan-Gladen correction
against hand-computed cases. E4-E6 pin the seeded headline rates,
including the in-sample identity (pooled correction recovers the
gold rate exactly - algebra, pinned as a sanity fact) and the
transport diagnostic (batch-1 error rates applied to batch 2).
E7 is the headline test: the overall error is under its floor while
the planted 'sarcasm' subgroup breaches its own, and the gate
refuses for that reason alone. E10 pins the clean control, which
publishes.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import threading
import urllib.request
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli import EXIT_ERROR, EXIT_FAIL, EXIT_OK, main as cli_main  # noqa: E402
from cli import make_server, reset_state
from generator import (batch_counts, generate_dataset, load_rows,  # noqa: E402
                       rows_to_jsonl, subgroup_counts)
from measure import (drift_check, error_stats, estimate_errors,  # noqa: E402
                     evaluate_gate, headline_rates, rogan_gladen)
from report import build_report  # noqa: E402
from validation import validate_dataset  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIXTURE_PLANTED = os.path.join(ROOT, "fixtures", "sample_planted.jsonl")
FIXTURE_CLEAN = os.path.join(ROOT, "fixtures", "sample_clean.jsonl")
RESULTS = []


def check(name, passed, detail=""):
    RESULTS.append({"name": name, "passed": bool(passed), "detail": detail})
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    return passed


def _run_cli(argv):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return cli_main(argv)


def _grid(n_per_cell, flip=None):
    rows, i = [], 0
    for subgroup in ("standard", "dialect", "sarcasm"):
        for batch in (1, 2):
            for j in range(n_per_cell):
                i += 1
                gold = 1 if j < n_per_cell // 2 else 0
                ann = 1 - gold if (flip and flip(subgroup, batch, gold, j)) else gold
                rows.append({"item_id": f"item_{i:05d}", "subgroup": subgroup,
                             "batch": batch, "gold_label": gold,
                             "annotated_label": ann})
    return rows


def main():
    print("== golden evals: policy-measurement-validation-kit ==")
    planted = generate_dataset("planted")
    clean = generate_dataset("clean")

    # E1: generator determinism (bytes pinned for both variants)
    md5p = hashlib.md5(rows_to_jsonl(planted).encode()).hexdigest()
    md5c = hashlib.md5(rows_to_jsonl(clean).encode()).hexdigest()
    check("E1 generator determinism (seed=42, md5 pinned both variants)",
          planted == generate_dataset("planted")
          and clean == generate_dataset("clean")
          and md5p == "4e719e93a81ea3e8c49e90cc75549c03"
          and md5c == "a54108aadc5c39ff383d59b39419b22c",
          f"planted_md5={md5p} clean_md5={md5c}")

    # E2: shape pinned
    check("E2 shape pinned (2400 rows, subgroups 1348/755/297, batches 1200/1200)",
          len(planted) == 2400
          and subgroup_counts(planted) == {"standard": 1348, "dialect": 755,
                                           "sarcasm": 297}
          and batch_counts(planted) == {1: 1200, 2: 1200}
          and validate_dataset(planted) == [])

    # E3: hand-computed error stats + Rogan-Gladen
    pairs = [(1, 1), (1, 1), (1, 1), (1, 0),
             (0, 0), (0, 0), (0, 0), (0, 0), (0, 1), (0, 1)]
    hand = [{"item_id": f"item_{i + 1:05d}", "subgroup": "standard",
             "batch": 1, "gold_label": g, "annotated_label": a}
            for i, (g, a) in enumerate(pairs)]
    stats = error_stats(hand)
    check("E3 hand-computed (sens 0.75, spec 0.666667, err 0.3; RG 0.34->0.30)",
          stats["sensitivity"] == 0.75 and stats["specificity"] == 0.666667
          and stats["error_rate"] == 0.3
          and rogan_gladen(0.34, 0.90, 0.90)["corrected_rate"] == 0.3
          and rogan_gladen(0.42, 1.0, 1.0)["corrected_rate"] == 0.42)

    # E4: seeded headline rates pinned
    rates = headline_rates(planted)
    check("E4 headline rates pinned (true 0.301667, naive 0.315833, Se 0.895028, Sp 0.934368)",
          rates["true_rate"] == 0.301667 and rates["naive_rate"] == 0.315833
          and rates["correction"]["sensitivity"] == 0.895028
          and rates["correction"]["specificity"] == 0.934368
          and rates["correction"]["denominator"] == 0.829396,
          f"naive={rates['naive_rate']} true={rates['true_rate']}")

    # E5: in-sample identity, pinned as a sanity fact (not a result)
    check("E5 in-sample RG identity (corrected == true exactly; naive err 0.014167)",
          rates["corrected_rate"] == rates["true_rate"]
          and rates["corrected_abs_error"] == 0.0
          and rates["naive_abs_error"] == 0.014167)

    # E6: transport diagnostic pinned (batch-1 Se/Sp -> batch 2)
    t = rates["transport"]
    check("E6 transport pinned (batch2 true 0.300833, corrected 0.290587, err 0.010246 vs naive 0.01)",
          t["true_rate"] == 0.300833 and t["naive_rate"] == 0.310833
          and t["corrected_rate"] == 0.290587
          and t["corrected_abs_error"] == 0.010246
          and t["naive_abs_error"] == 0.01,
          f"transport_corrected_err={t['corrected_abs_error']}")

    # E7: HEADLINE TEST - overall fine, planted subgroup breaches, gate
    # refuses for that reason alone.
    gate = evaluate_gate(planted)
    check("E7 HEADLINE gate (overall 0.0775 < 0.10; sarcasm 0.242424 > 0.15; REFUSE, 1 reason)",
          gate["checks"]["overall_error_rate"] == 0.0775
          and gate["checks"]["overall_error_rate"] < gate["floors"]["max_overall_error"]
          and gate["checks"]["subgroup_error_rates"]["sarcasm"] == 0.242424
          and gate["verdict"] == "REFUSE" and len(gate["reasons"]) == 1
          and "sarcasm" in gate["reasons"][0],
          f"reasons={gate['reasons']}")

    # E8: per-subgroup errors and Se/Sp pinned
    by_sg = estimate_errors(planted)["by_subgroup"]
    check("E8 subgroup errors pinned (standard 0.04451, dialect 0.071523, sarcasm 0.242424)",
          by_sg["standard"]["error_rate"] == 0.04451
          and by_sg["dialect"]["error_rate"] == 0.071523
          and by_sg["sarcasm"]["error_rate"] == 0.242424
          and by_sg["sarcasm"]["sensitivity"] == 0.642857
          and by_sg["sarcasm"]["specificity"] == 0.802817
          and by_sg["standard"]["sensitivity"] == 0.94686)

    # E9: drift pinned
    drift = drift_check(planted)
    check("E9 drift pinned (batch errs 0.075/0.08, statistic 0.005, sarcasm drift 0.021637)",
          drift["per_batch"]["1"]["error_rate"] == 0.075
          and drift["per_batch"]["2"]["error_rate"] == 0.08
          and drift["drift_statistic"] == 0.005
          and drift["by_subgroup"]["sarcasm"]["drift"] == 0.021637
          and drift["by_subgroup"]["dialect"]["drift"] == 0.025166)

    # E10: clean control publishes, checks pinned
    clean_gate = evaluate_gate(clean)
    clean_rates = headline_rates(clean)
    check("E10 clean control PUBLISHES (overall err 0.054583, worst subgroup 0.070707, drift 0.015833)",
          clean_gate["verdict"] == "PUBLISH" and clean_gate["reasons"] == []
          and clean_gate["checks"]["overall_error_rate"] == 0.054583
          and clean_gate["checks"]["worst_subgroup_error_rate"] == 0.070707
          and clean_gate["checks"]["drift_statistic"] == 0.015833
          and clean_rates["naive_rate"] == 0.309583,
          f"clean_verdict={clean_gate['verdict']}")

    # E11: gate hand cases - subgroup-only breach and drift-only breach
    sub_rows = _grid(100, flip=lambda sg, b, g, j: sg == "sarcasm" and j % 10 < 3)
    drift_rows = _grid(100, flip=lambda sg, b, g, j:
                       sg == "standard" and b == 2 and j % 5 == 0)
    g_sub, g_drift = evaluate_gate(sub_rows), evaluate_gate(drift_rows)
    check("E11 gate hand cases (subgroup-only REFUSE at overall 0.10; drift-only REFUSE)",
          g_sub["verdict"] == "REFUSE" and len(g_sub["reasons"]) == 1
          and g_sub["checks"]["overall_error_rate"] == 0.1
          and g_drift["verdict"] == "REFUSE"
          and any("drift" in r for r in g_drift["reasons"])
          and evaluate_gate(_grid(50))["verdict"] == "PUBLISH")

    # E12: CLI check exit codes 1 / 0 / 2 (+ bare check refuses by design)
    import tempfile
    tmp = tempfile.mkdtemp(prefix="policy-evals-")
    planted_path = os.path.join(tmp, "planted.jsonl")
    clean_path = os.path.join(tmp, "clean.jsonl")
    with open(planted_path, "w", encoding="utf-8") as fh:
        fh.write(rows_to_jsonl(planted))
    with open(clean_path, "w", encoding="utf-8") as fh:
        fh.write(rows_to_jsonl(clean))
    codes = (_run_cli(["check", "--input", planted_path]),
             _run_cli(["check", "--input", clean_path]),
             _run_cli(["check"]),
             _run_cli(["check", "--input", "/tmp/no-such-policy-xyz.jsonl"]))
    check("E12 CLI check exit codes 1/0/1/2",
          codes == (EXIT_FAIL, EXIT_OK, EXIT_FAIL, EXIT_ERROR),
          f"codes={codes}")

    # E13: serve round-trip matches local computation
    reset_state(planted)
    server = make_server(0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/report",
                                    timeout=10) as resp:
            served = json.loads(resp.read())
        serve_ok = served == build_report(planted)
    finally:
        server.shutdown()
        server.server_close()
        reset_state()
    check("E13 serve round-trip matches local report", serve_ok)

    # E14: bundled fixtures pinned (files equal the seeded generator output)
    check("E14 fixtures pinned (planted + clean files equal generator output)",
          load_rows(FIXTURE_PLANTED) == planted
          and load_rows(FIXTURE_CLEAN) == clean)

    # E15: full report deterministic and internally consistent
    rep1, rep2 = build_report(), build_report()
    check("E15 report deterministic (REFUSE, sarcasm worst, rates pinned)",
          rep1 == rep2 and rep1["gate"]["verdict"] == "REFUSE"
          and rep1["gate"]["checks"]["worst_subgroup"] == "sarcasm"
          and rep1["rates"]["corrected_rate"] == 0.301667
          and rep1["drift"]["drift_statistic"] == 0.005)

    passed = sum(1 for r in RESULTS if r["passed"])
    total_n = len(RESULTS)
    summary_doc = {"passed": passed, "total": total_n,
                   "all_passed": passed == total_n, "evals": RESULTS}
    with open(os.path.join(HERE, "eval_report.json"), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(summary_doc, sort_keys=True) + "\n")
    print(f"\n{passed}/{total_n} evals passed -> evals/eval_report.json")
    return 0 if passed == total_n else 1


if __name__ == "__main__":
    sys.exit(main())
