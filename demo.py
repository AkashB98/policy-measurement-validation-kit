"""End-to-end demo: generate -> validate -> rates + Rogan-Gladen ->
subgroup errors -> drift -> publish gate (planted refuses via its
subgroup, clean publishes) -> check -> serve.

SYNTHETIC DATA ONLY (fictional policy-outcome program). Runs fully
offline; the serve round-trip is loopback-only on an ephemeral port.
Every stage asserts.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli import EXIT_OK, main as cli_main, make_server, reset_state  # noqa: E402
from generator import generate_dataset, rows_to_jsonl  # noqa: E402
from measure import (MAX_DRIFT, MAX_OVERALL_ERROR, MAX_SUBGROUP_ERROR,  # noqa: E402
                     drift_check, estimate_errors, evaluate_gate,
                     headline_rates)
from report import build_report  # noqa: E402
from validation import validate_dataset  # noqa: E402


def _request(port: int, path: str):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method="GET")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status, json.loads(resp.read())


def main() -> None:
    print("== 1. generate synthetic datasets (seed=42, 2400 items each) ==")
    planted = generate_dataset("planted")
    clean = generate_dataset("clean")
    assert planted == generate_dataset("planted"), "generator must be deterministic"
    assert len(planted) == 2400 and len(clean) == 2400
    assert validate_dataset(planted) == [] and validate_dataset(clean) == []
    print("   2400 items x 2 variants, deterministic, valid")

    print("\n== 2. headline rates: naive vs Rogan-Gladen corrected vs true ==")
    rates = headline_rates(planted)
    assert rates["naive_abs_error"] > 0, rates
    # In-sample identity (see measure.py): pooled Se/Sp estimated on
    # the same rows make the correction recover the gold rate exactly.
    assert rates["corrected_rate"] == rates["true_rate"], rates
    transport = rates["transport"]
    assert transport is not None
    assert transport["corrected_abs_error"] <= 0.02, transport
    print(f"   planted: true={rates['true_rate']} naive={rates['naive_rate']} "
          f"corrected={rates['corrected_rate']} "
          f"(naive err {rates['naive_abs_error']})")
    print(f"   transport (batch-1 Se/Sp -> batch 2): corrected err "
          f"{transport['corrected_abs_error']} vs naive err "
          f"{transport['naive_abs_error']} — the honest, non-trivial number")

    print("\n== 3. HEADLINE: overall error fine, one subgroup breaches ==")
    errors = estimate_errors(planted)
    gate = evaluate_gate(planted)
    overall_err = errors["overall"]["error_rate"]
    sarcasm_err = errors["by_subgroup"]["sarcasm"]["error_rate"]
    assert overall_err < MAX_OVERALL_ERROR, overall_err
    assert sarcasm_err > MAX_SUBGROUP_ERROR, sarcasm_err
    assert gate["verdict"] == "REFUSE", gate
    assert len(gate["reasons"]) == 1 and "sarcasm" in gate["reasons"][0], gate
    print(f"   overall error={overall_err} (floor {MAX_OVERALL_ERROR}) — fine; "
          f"sarcasm error={sarcasm_err} (floor {MAX_SUBGROUP_ERROR}) — breach")
    print(f"   gate REFUSES for exactly one reason: {gate['reasons'][0]}")

    print("\n== 4. drift: present per subgroup, under the pooled floor ==")
    drift = drift_check(planted)
    assert drift["drift_statistic"] < MAX_DRIFT, drift
    assert drift["by_subgroup"]["sarcasm"]["drift"] > 0, drift
    print(f"   batch errors: 1={drift['per_batch']['1']['error_rate']} "
          f"2={drift['per_batch']['2']['error_rate']} "
          f"drift={drift['drift_statistic']} (floor {MAX_DRIFT}); "
          f"sarcasm drift={drift['by_subgroup']['sarcasm']['drift']}")

    print("\n== 5. clean control: the gate publishes ==")
    clean_gate = evaluate_gate(clean)
    assert clean_gate["verdict"] == "PUBLISH", clean_gate
    assert clean_gate["reasons"] == []
    print(f"   clean overall error="
          f"{clean_gate['checks']['overall_error_rate']}, worst subgroup "
          f"{clean_gate['checks']['worst_subgroup']}="
          f"{clean_gate['checks']['worst_subgroup_error_rate']}, "
          f"drift={clean_gate['checks']['drift_statistic']} -> PUBLISH")

    print("\n== 6. CLI check gate: 1 planted / 0 clean / 2 bad input ==")
    import tempfile
    tmp = tempfile.mkdtemp(prefix="policy-demo-")
    planted_path = os.path.join(tmp, "planted.jsonl")
    clean_path = os.path.join(tmp, "clean.jsonl")
    with open(planted_path, "w", encoding="utf-8") as fh:
        fh.write(rows_to_jsonl(planted))
    with open(clean_path, "w", encoding="utf-8") as fh:
        fh.write(rows_to_jsonl(clean))
    code_refuse = cli_main(["check", "--input", planted_path])
    code_ok = cli_main(["check", "--input", clean_path])
    code_err = cli_main(["check", "--input", "/tmp/no-such-policy-xyz.jsonl"])
    print(f"   planted={code_refuse} clean={code_ok} missing-file={code_err}")
    assert (code_refuse, code_ok, code_err) == (1, EXIT_OK, 2)

    print("\n== 7. live localhost serve round-trip ==")
    reset_state(planted)
    server = make_server(0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, health = _request(port, "/health")
        assert status == 200 and health == {"status": "ok"}
        status, served = _request(port, "/report")
        assert status == 200 and served == build_report(planted), "serve mismatch"
        print("   GET /report matches local computation; GET /health ok")
    finally:
        server.shutdown()
        server.server_close()
        reset_state()

    print("\nDemo OK: generate, rates + correction, subgroup headline, "
          "drift, gate (refuse + publish), check and serve verified.")


if __name__ == "__main__":
    main()
