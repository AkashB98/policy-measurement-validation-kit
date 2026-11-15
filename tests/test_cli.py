import io
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.request
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli import EXIT_ERROR, EXIT_FAIL, EXIT_OK, main as cli_main  # noqa: E402
from cli import make_server, reset_state
from generator import generate_dataset, rows_to_jsonl  # noqa: E402
from report import build_report  # noqa: E402


def _run(argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli_main(argv)
    return code, out.getvalue(), err.getvalue()


class TestCli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="policy-cli-")
        self.planted_path = os.path.join(self.tmp, "planted.jsonl")
        self.clean_path = os.path.join(self.tmp, "clean.jsonl")
        with open(self.planted_path, "w", encoding="utf-8") as fh:
            fh.write(rows_to_jsonl(generate_dataset("planted")))
        with open(self.clean_path, "w", encoding="utf-8") as fh:
            fh.write(rows_to_jsonl(generate_dataset("clean")))

    def test_generate_to_file(self):
        out_path = os.path.join(self.tmp, "gen.jsonl")
        code, out, _ = _run(["generate", "--variant", "clean", "--out", out_path])
        self.assertEqual(code, EXIT_OK)
        self.assertIn("2400", out)
        with open(out_path, encoding="utf-8") as fh:
            self.assertEqual(len(fh.readlines()), 2400)

    def test_validate_codes(self):
        code, out, _ = _run(["validate", "--input", self.planted_path])
        self.assertEqual(code, EXIT_OK)
        self.assertIn("validate OK", out)
        code, _, _ = _run(["validate", "--input", "/tmp/no-such-policy-xyz.jsonl"])
        self.assertEqual(code, EXIT_ERROR)

    def test_report_matches_local(self):
        code, out, _ = _run(["report", "--input", self.planted_path])
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(json.loads(out), build_report(generate_dataset("planted")))

    def test_report_default_variant_clean_publishes(self):
        code, out, _ = _run(["report", "--variant", "clean"])
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(json.loads(out)["gate"]["verdict"], "PUBLISH")

    def test_check_exit_codes(self):
        code, out, _ = _run(["check", "--input", self.planted_path])
        self.assertEqual(code, EXIT_FAIL)
        self.assertIn("REFUSE", out)
        self.assertIn("sarcasm", out)
        code, out, _ = _run(["check", "--input", self.clean_path])
        self.assertEqual(code, EXIT_OK)
        self.assertIn("PUBLISH", out)
        code, _, _ = _run(["check", "--input", "/tmp/no-such-policy-xyz.jsonl"])
        self.assertEqual(code, EXIT_ERROR)

    def test_check_default_seeded_refuses_by_design(self):
        code, _, _ = _run(["check"])
        self.assertEqual(code, EXIT_FAIL)

    def test_check_bad_data_is_error_not_refusal(self):
        bad_path = os.path.join(self.tmp, "bad.jsonl")
        with open(bad_path, "w", encoding="utf-8") as fh:
            fh.write('{"item_id": "x"}\n')
        code, _, _ = _run(["check", "--input", bad_path])
        self.assertEqual(code, EXIT_ERROR)

    def test_serve_round_trip(self):
        reset_state(generate_dataset("planted"))
        server = make_server(0)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=10) as resp:
                self.assertEqual(json.loads(resp.read()), {"status": "ok"})
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/report", timeout=10) as resp:
                served = json.loads(resp.read())
            self.assertEqual(served, build_report(generate_dataset("planted")))
        finally:
            server.shutdown()
            server.server_close()
            reset_state()


if __name__ == "__main__":
    unittest.main()
