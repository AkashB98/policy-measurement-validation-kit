import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator import generate_dataset  # noqa: E402
from report import build_report  # noqa: E402


class TestReport(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(build_report(), build_report())

    def test_default_is_planted_and_refuses(self):
        report = build_report()
        self.assertEqual(report["dataset"]["rows"], 2400)
        self.assertEqual(report["gate"]["verdict"], "REFUSE")
        self.assertEqual(report["rates"]["true_rate"], 0.301667)

    def test_clean_rows_publish(self):
        report = build_report(generate_dataset("clean"))
        self.assertEqual(report["gate"]["verdict"], "PUBLISH")
        self.assertEqual(report["errors"]["overall"]["error_rate"], 0.054583)

    def test_structure(self):
        report = build_report()
        self.assertEqual(set(report),
                         {"dataset", "rates", "errors", "drift", "gate"})
        self.assertEqual(report["dataset"]["batch_counts"],
                         {"1": 1200, "2": 1200})
        self.assertTrue(report["dataset"]["synthetic"])


if __name__ == "__main__":
    unittest.main()
