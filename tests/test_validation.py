import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import grid, row  # noqa: E402
from validation import validate_dataset, validate_row  # noqa: E402


class TestValidation(unittest.TestCase):
    def test_clean_row(self):
        self.assertEqual(validate_row(row()), [])

    def test_generated_datasets_clean(self):
        from generator import generate_dataset
        self.assertEqual(validate_dataset(generate_dataset("planted")), [])
        self.assertEqual(validate_dataset(generate_dataset("clean")), [])

    def test_hand_grid_clean(self):
        self.assertEqual(validate_dataset(grid(10)), [])

    def test_missing_field(self):
        r = row()
        del r["gold_label"]
        self.assertTrue(any("gold_label" in i for i in validate_row(r)))

    def test_label_domain(self):
        self.assertTrue(validate_row(row(gold_label=2)))
        self.assertTrue(validate_row(row(annotated_label=-1)))
        self.assertTrue(validate_row(row(gold_label=True)))  # bool is not 0/1
        self.assertEqual(validate_row(row(gold_label=1, annotated_label=1)), [])

    def test_batch_domain(self):
        self.assertTrue(validate_row(row(batch=3)))
        self.assertTrue(validate_row(row(batch=True)))
        self.assertEqual(validate_row(row(batch=2)), [])

    def test_empty_strings(self):
        self.assertTrue(validate_row(row(item_id="")))
        self.assertTrue(validate_row(row(subgroup="")))

    def test_duplicate_item_flagged(self):
        rows = grid(10)
        rows.append(dict(rows[0]))
        self.assertTrue(any("duplicate" in i for i in validate_dataset(rows)))

    def test_single_subgroup_flagged(self):
        rows = [row(f"item_{i:05d}", "standard", 1 + (i % 2), 0, 0)
                for i in range(10)]
        self.assertTrue(any("fewer than 2 subgroups" in i
                            for i in validate_dataset(rows)))

    def test_missing_batch_flagged(self):
        rows = [row(f"item_{i:05d}", "standard" if i % 2 else "dialect",
                    1, 0, 0) for i in range(10)]
        self.assertTrue(any("both batches" in i
                            for i in validate_dataset(rows)))

    def test_empty_dataset(self):
        self.assertEqual(validate_dataset([]), ["dataset: no rows"])

    def test_non_dict_row(self):
        self.assertTrue(validate_row("nope"))


if __name__ == "__main__":
    unittest.main()
