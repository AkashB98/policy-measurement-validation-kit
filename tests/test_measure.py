import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import grid, row  # noqa: E402
from measure import (DEFAULT_FLOORS, confusion, drift_check,  # noqa: E402
                     error_stats, estimate_errors, evaluate_gate,
                     headline_rates, rogan_gladen)


def _hand_rows():
    # 10 rows, hand-counted: TP=3, FN=1, TN=4, FP=2.
    pairs = [(1, 1), (1, 1), (1, 1), (1, 0),
             (0, 0), (0, 0), (0, 0), (0, 0), (0, 1), (0, 1)]
    return [row(f"item_{i + 1:05d}", "standard", 1, g, a)
            for i, (g, a) in enumerate(pairs)]


class TestErrorStats(unittest.TestCase):
    def test_confusion_hand_counted(self):
        self.assertEqual(confusion(_hand_rows()),
                         {"tp": 3, "fp": 2, "tn": 4, "fn": 1})

    def test_sens_spec_error_hand_counted(self):
        stats = error_stats(_hand_rows())
        self.assertEqual(stats["sensitivity"], 0.75)       # 3/4
        self.assertEqual(stats["specificity"], round(4 / 6, 6))
        self.assertEqual(stats["error_rate"], 0.3)         # 3/10
        self.assertEqual(stats["positives"], 4)
        self.assertEqual(stats["negatives"], 6)

    def test_perfect_annotator(self):
        rows = grid(20)
        stats = error_stats(rows)
        self.assertEqual((stats["sensitivity"], stats["specificity"],
                          stats["error_rate"]), (1.0, 1.0, 0.0))

    def test_no_positives_sensitivity_none(self):
        rows = [row(f"item_{i:05d}", "standard", 1, 0, 0) for i in range(5)]
        stats = error_stats(rows)
        self.assertIsNone(stats["sensitivity"])
        self.assertEqual(stats["specificity"], 1.0)

    def test_estimate_errors_structure(self):
        est = estimate_errors(grid(10))
        self.assertEqual(set(est["by_subgroup"]),
                         {"standard", "dialect", "sarcasm"})
        self.assertEqual(set(est["by_batch"]), {"1", "2"})
        self.assertEqual(est["overall"]["n"], 60)


class TestRoganGladen(unittest.TestCase):
    def test_hand_computed(self):
        # (0.34 + 0.90 - 1) / (0.90 + 0.90 - 1) = 0.24 / 0.80 = 0.30
        out = rogan_gladen(0.34, 0.90, 0.90)
        self.assertEqual(out["corrected_rate"], 0.3)
        self.assertFalse(out["clamped"])

    def test_perfect_annotator_is_identity(self):
        self.assertEqual(rogan_gladen(0.42, 1.0, 1.0)["corrected_rate"], 0.42)

    def test_clamps_above_one(self):
        out = rogan_gladen(0.99, 0.60, 0.60)
        self.assertEqual(out["corrected_rate"], 1.0)
        self.assertTrue(out["clamped"])

    def test_clamps_below_zero(self):
        out = rogan_gladen(0.01, 0.60, 0.60)
        self.assertEqual(out["corrected_rate"], 0.0)
        self.assertTrue(out["clamped"])

    def test_uninformative_annotator_raises(self):
        with self.assertRaises(ValueError):
            rogan_gladen(0.30, 0.50, 0.50)
        with self.assertRaises(ValueError):
            rogan_gladen(0.30, 0.40, 0.50)


class TestHeadlineRates(unittest.TestCase):
    def test_in_sample_identity_on_seeded_data(self):
        from generator import generate_dataset
        rates = headline_rates(generate_dataset("planted"))
        self.assertEqual(rates["true_rate"], 0.301667)
        self.assertEqual(rates["naive_rate"], 0.315833)
        # Identity (see measure.py docstring), pinned as a sanity fact.
        self.assertEqual(rates["corrected_rate"], rates["true_rate"])
        self.assertEqual(rates["corrected_abs_error"], 0.0)
        self.assertEqual(rates["naive_abs_error"], 0.014167)

    def test_transport_block_present(self):
        from generator import generate_dataset
        transport = headline_rates(generate_dataset("planted"))["transport"]
        self.assertEqual(transport["estimated_on"], "batch_1")
        self.assertEqual(transport["applied_to"], "batch_2")
        self.assertEqual(transport["true_rate"], 0.300833)
        self.assertEqual(transport["corrected_rate"], 0.290587)


class TestDrift(unittest.TestCase):
    def test_zero_drift_on_symmetric_grid(self):
        drift = drift_check(grid(50))
        self.assertEqual(drift["drift_statistic"], 0.0)

    def test_hand_drift(self):
        # Batch 2 flips every 5th item in each cell; batch 1 is clean.
        rows = grid(100, flip=lambda sg, b, g, j: b == 2 and j % 5 == 0)
        drift = drift_check(rows)
        self.assertEqual(drift["per_batch"]["1"]["error_rate"], 0.0)
        self.assertEqual(drift["per_batch"]["2"]["error_rate"], 0.2)
        self.assertEqual(drift["drift_statistic"], 0.2)

    def test_requires_both_batches(self):
        rows = [row(f"item_{i:05d}", "standard", 1, 0, 0) for i in range(4)]
        with self.assertRaises(ValueError):
            drift_check(rows)


class TestGate(unittest.TestCase):
    def test_clean_grid_publishes(self):
        gate = evaluate_gate(grid(50))
        self.assertEqual(gate["verdict"], "PUBLISH")
        self.assertEqual(gate["reasons"], [])
        self.assertEqual(gate["floors"], DEFAULT_FLOORS)

    def test_headline_subgroup_breach_refuses(self):
        # HEADLINE TEST: sarcasm flips 30% of its items in both batches;
        # every other subgroup is perfect. Pooled overall error is
        # 0.30 * (200/600) = 0.10... sarcasm is 1/3 of rows, so overall
        # error = 0.30/3 = 0.10 — at the floor, not above it — while
        # the subgroup error 0.30 breaches its 0.15 floor.
        rows = grid(100, flip=lambda sg, b, g, j: sg == "sarcasm" and j % 10 < 3)
        gate = evaluate_gate(rows)
        self.assertEqual(gate["checks"]["overall_error_rate"], 0.1)
        self.assertEqual(gate["checks"]["subgroup_error_rates"]["sarcasm"], 0.3)
        self.assertEqual(gate["verdict"], "REFUSE")
        self.assertEqual(len(gate["reasons"]), 1)
        self.assertIn("sarcasm", gate["reasons"][0])

    def test_overall_breach_refuses(self):
        rows = grid(100, flip=lambda sg, b, g, j: j % 5 == 0)  # 20% everywhere
        gate = evaluate_gate(rows)
        self.assertEqual(gate["verdict"], "REFUSE")
        self.assertTrue(any("overall" in r for r in gate["reasons"]))

    def test_drift_breach_refuses(self):
        # Batch 2 flips 20% in standard only: pooled drift = 0.20/3.
        rows = grid(100, flip=lambda sg, b, g, j:
                    sg == "standard" and b == 2 and j % 5 == 0)
        gate = evaluate_gate(rows)
        self.assertEqual(gate["verdict"], "REFUSE")
        self.assertTrue(any("drift" in r for r in gate["reasons"]))

    def test_custom_floors(self):
        rows = grid(100, flip=lambda sg, b, g, j: sg == "sarcasm" and j % 10 < 3)
        gate = evaluate_gate(rows, floors={"max_overall_error": 0.10,
                                           "max_subgroup_error": 0.35,
                                           "max_drift": 0.05})
        self.assertEqual(gate["verdict"], "PUBLISH")

    def test_seeded_planted_refuses_via_sarcasm_only(self):
        from generator import generate_dataset
        gate = evaluate_gate(generate_dataset("planted"))
        self.assertEqual(gate["verdict"], "REFUSE")
        self.assertEqual(len(gate["reasons"]), 1)
        self.assertIn("sarcasm", gate["reasons"][0])
        self.assertEqual(gate["checks"]["worst_subgroup"], "sarcasm")
        self.assertEqual(gate["checks"]["worst_subgroup_error_rate"], 0.242424)
        self.assertEqual(gate["checks"]["overall_error_rate"], 0.0775)

    def test_seeded_clean_publishes(self):
        from generator import generate_dataset
        gate = evaluate_gate(generate_dataset("clean"))
        self.assertEqual(gate["verdict"], "PUBLISH")


if __name__ == "__main__":
    unittest.main()
