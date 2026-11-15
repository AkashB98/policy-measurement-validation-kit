"""CLI: generate | validate | report | check | demo | serve.

SYNTHETIC DATA ONLY - every dataset handled here is fabricated
sample data about a fictional policy-outcome program.

Exit codes:
  check : 0 = gate PUBLISH, 1 = gate REFUSE (the seeded planted
          dataset refuses by design, via its planted subgroup),
          2 = usage / IO / data-quality error
  others: 0 = success, 2 = usage / IO error
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generator import (DEFAULT_N, DEFAULT_SEED, VARIANTS,  # noqa: E402
                       generate_dataset, load_rows, write_rows)
from report import build_report  # noqa: E402
from validation import validate_dataset  # noqa: E402

EXIT_OK, EXIT_FAIL, EXIT_ERROR = 0, 1, 2


def canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True)


def _load_or_error(path: str) -> Optional[List[Dict]]:
    try:
        return load_rows(path)
    except FileNotFoundError:
        print(f"error: file not found: {path}", file=sys.stderr)
    except (ValueError, OSError) as exc:
        print(f"error: cannot read {path}: {exc}", file=sys.stderr)
    return None


def _rows_for(args) -> Optional[List[Dict]]:
    """Rows from --input, or the seeded dataset for --variant."""
    if getattr(args, "input", None):
        return _load_or_error(args.input)
    return generate_dataset(getattr(args, "variant", None) or "planted",
                            seed=getattr(args, "seed", DEFAULT_SEED),
                            n=getattr(args, "n", DEFAULT_N))


def _validated_rows(args):
    rows = _rows_for(args)
    if rows is None:
        return None, EXIT_ERROR
    issues = validate_dataset(rows)
    if issues:
        for issue in issues[:20]:
            print(f"data-quality: {issue}", file=sys.stderr)
        return None, EXIT_ERROR
    return rows, EXIT_OK


def _write_out(args, text: str, summary: str) -> int:
    if getattr(args, "out", None):
        try:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(text)
        except OSError as exc:
            print(f"error: cannot write {args.out}: {exc}", file=sys.stderr)
            return EXIT_ERROR
        print(summary)
    else:
        print(text, end="")
    return EXIT_OK


def cmd_generate(args) -> int:
    rows = generate_dataset(args.variant, seed=args.seed, n=args.n)
    if args.out:
        fmt = "csv" if args.out.endswith(".csv") else (
            "json" if args.out.endswith(".json") else "jsonl")
        try:
            write_rows(rows, args.out, fmt=fmt)
        except OSError as exc:
            print(f"error: cannot write {args.out}: {exc}", file=sys.stderr)
            return EXIT_ERROR
        print(f"generated {len(rows)} synthetic rows "
              f"(variant={args.variant}, seed={args.seed}) -> {args.out}")
        return EXIT_OK
    from generator import rows_to_jsonl
    print(rows_to_jsonl(rows), end="")
    return EXIT_OK


def cmd_validate(args) -> int:
    rows = _load_or_error(args.input)
    if rows is None:
        return EXIT_ERROR
    issues = validate_dataset(rows)
    if issues:
        for issue in issues[:20]:
            print(f"data-quality: {issue}")
        print(f"validate FAIL: {len(issues)} issue(s)")
        return EXIT_ERROR
    print(f"validate OK: {len(rows)} rows")
    return EXIT_OK


def cmd_report(args) -> int:
    rows, code = _validated_rows(args)
    if rows is None:
        return code
    try:
        report = build_report(rows)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    return _write_out(args, canonical(report) + "\n", f"report -> {args.out}")


def cmd_check(args) -> int:
    rows = _rows_for(args)
    if rows is None:
        return EXIT_ERROR
    issues = validate_dataset(rows)
    if issues:
        for issue in issues[:20]:
            print(f"data-quality: {issue}")
        print(f"check FAIL (data quality): {len(issues)} issue(s)")
        return EXIT_ERROR
    try:
        report = build_report(rows)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    gate = report["gate"]
    rates = report["rates"]
    checks = gate["checks"]
    print(f"verdict={gate['verdict']} "
          f"true_rate={rates['true_rate']} "
          f"naive_rate={rates['naive_rate']} "
          f"corrected_rate={rates['corrected_rate']} "
          f"overall_error={checks['overall_error_rate']} "
          f"worst_subgroup={checks['worst_subgroup']} "
          f"worst_subgroup_error={checks['worst_subgroup_error_rate']} "
          f"drift={checks['drift_statistic']}")
    if gate["verdict"] == "REFUSE":
        for reason in gate["reasons"]:
            print(f"refusal: {reason}")
        print("check FAIL (gate refused)")
        return EXIT_FAIL
    print("check PASS")
    return EXIT_OK


def cmd_demo(args) -> int:
    import demo as demo_mod
    try:
        demo_mod.main()
    except AssertionError as exc:
        print(f"demo FAILED: {exc}", file=sys.stderr)
        return EXIT_FAIL
    return EXIT_OK


# ------------------------------------------------------------------- serve

class _State:
    def __init__(self):
        self.rows: List[Dict] = generate_dataset("planted")


_STATE = _State()


def get_state() -> _State:
    return _STATE


def reset_state(rows: Optional[List[Dict]] = None) -> None:
    _STATE.rows = list(rows) if rows is not None else generate_dataset("planted")


class _Handler(BaseHTTPRequestHandler):
    def _send_json(self, code: int, payload: Any) -> None:
        body = (canonical(payload) + "\n").encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            return self._send_json(200, {"status": "ok"})
        if self.path == "/report":
            try:
                return self._send_json(200, build_report(_STATE.rows))
            except ValueError as exc:
                return self._send_json(400, {"error": str(exc)})
        return self._send_json(404, {"error": "unknown endpoint"})

    def log_message(self, *a: Any) -> None:
        pass


def make_server(port: int) -> HTTPServer:
    return HTTPServer(("127.0.0.1", port), _Handler)


def cmd_serve(args) -> int:
    rows, code = _validated_rows(args)
    if rows is None:
        return code
    reset_state(rows)
    server = make_server(args.port)
    print(f"serving on http://127.0.0.1:{args.port}  "
          f"(GET /report, GET /health)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="policy-measurement-validation-kit",
        description="Policy-outcome measurement validation: Rogan-Gladen "
                    "correction, subgroup error and drift checks, and a "
                    "publish gate (synthetic data only).")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate a synthetic dataset")
    g.add_argument("--variant", choices=VARIANTS, default="planted")
    g.add_argument("--seed", type=int, default=DEFAULT_SEED)
    g.add_argument("--n", type=int, default=DEFAULT_N)
    g.add_argument("--out", default=None)
    g.set_defaults(func=cmd_generate)

    v = sub.add_parser("validate", help="data-quality validation of a file")
    v.add_argument("--input", required=True)
    v.set_defaults(func=cmd_validate)

    for name, fn, helptext in (
            ("report", cmd_report, "full validation report, one document"),
            ("check", cmd_check, "CI gate: exit 0 publish / 1 refuse / 2 error")):
        sp = sub.add_parser(name, help=helptext)
        sp.add_argument("--input", default=None,
                        help="dataset file (default: seeded dataset)")
        sp.add_argument("--variant", choices=VARIANTS, default=None,
                        help="seeded variant when --input is not given")
        sp.add_argument("--seed", type=int, default=DEFAULT_SEED)
        sp.add_argument("--n", type=int, default=DEFAULT_N)
        if name == "report":
            sp.add_argument("--out", default=None)
        sp.set_defaults(func=fn)

    sub.add_parser("demo", help="run the end-to-end demo").set_defaults(func=cmd_demo)

    sv = sub.add_parser("serve", help="localhost JSON API")
    sv.add_argument("--port", type=int, default=8765)
    sv.add_argument("--input", default=None)
    sv.add_argument("--variant", choices=VARIANTS, default=None)
    sv.add_argument("--seed", type=int, default=DEFAULT_SEED)
    sv.add_argument("--n", type=int, default=DEFAULT_N)
    sv.set_defaults(func=cmd_serve)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
