import hashlib
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator import (BATCHES, DEFAULT_N, SUBGROUPS, batch_counts,  # noqa: E402
                       generate_dataset, load_rows, rows_to_jsonl,
                       subgroup_counts, write_rows)


class TestGenerator(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(generate_dataset("planted"), generate_dataset("planted"))
        self.assertEqual(generate_dataset("clean"), generate_dataset("clean"))

    def test_shape(self):
        rows = generate_dataset("planted")
        self.assertEqual(len(rows), DEFAULT_N)
        self.assertEqual(batch_counts(rows), {1: 1200, 2: 1200})
        counts = subgroup_counts(rows)
        self.assertEqual(set(counts), set(SUBGROUPS))
        self.assertEqual(sum(counts.values()), DEFAULT_N)

    def test_pinned_subgroup_counts(self):
        self.assertEqual(subgroup_counts(generate_dataset("planted")),
                         {"standard": 1348, "dialect": 755, "sarcasm": 297})

    def test_variants_share_gold_differ_in_annotations(self):
        planted = generate_dataset("planted")
        clean = generate_dataset("clean")
        # Same seed path: ids, subgroups, batches and gold labels align;
        # only the annotator's labels may differ.
        for p, c in zip(planted, clean):
            self.assertEqual(p["item_id"], c["item_id"])
            self.assertEqual(p["subgroup"], c["subgroup"])
            self.assertEqual(p["batch"], c["batch"])
            self.assertEqual(p["gold_label"], c["gold_label"])
        self.assertNotEqual([r["annotated_label"] for r in planted],
                            [r["annotated_label"] for r in clean])

    def test_labels_binary_and_fields(self):
        for r in generate_dataset("planted"):
            self.assertIn(r["gold_label"], (0, 1))
            self.assertIn(r["annotated_label"], (0, 1))
            self.assertIn(r["batch"], BATCHES)
            self.assertIn(r["subgroup"], SUBGROUPS)

    def test_seed_changes_data(self):
        self.assertNotEqual(generate_dataset("planted", seed=1),
                            generate_dataset("planted", seed=2))

    def test_unknown_variant_and_bad_n(self):
        with self.assertRaises(ValueError):
            generate_dataset("nope")
        with self.assertRaises(ValueError):
            generate_dataset("planted", n=21)  # odd
        with self.assertRaises(ValueError):
            generate_dataset("planted", n=10)  # too small

    def test_jsonl_md5_pinned(self):
        text = rows_to_jsonl(generate_dataset("planted"))
        self.assertEqual(hashlib.md5(text.encode()).hexdigest(),
                         "4e719e93a81ea3e8c49e90cc75549c03")

    def test_write_load_round_trip(self):
        import tempfile
        rows = generate_dataset("clean", n=100)
        for fmt, suffix in (("jsonl", ".jsonl"), ("json", ".json"),
                            ("csv", ".csv")):
            path = os.path.join(tempfile.mkdtemp(), f"data{suffix}")
            write_rows(rows, path, fmt=fmt)
            self.assertEqual(load_rows(path), rows)


if __name__ == "__main__":
    unittest.main()
