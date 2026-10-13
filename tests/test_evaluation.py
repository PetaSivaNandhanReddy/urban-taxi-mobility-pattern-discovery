"""
test_evaluation.py
==================
Comprehensive automated test suite for evaluation methodology and synthetic benchmarks.

Covers all required edge cases:
PHASE 2 REQUIREMENTS:
1. Perfect recovery (all predicted match reference, none missed)
2. No predictions (empty predictions DataFrame)
3. No reference pairs (empty reference DataFrame)
4. Partial recovery (some match, some missed, some false positives)
5. Reversed ordered pairs ((A, B) vs (B, A) strictly separated)
6. Duplicate pairs across multiple patterns (deduplicated; no double credit)
7. Gapped pairs in PrefixSpan sequences ((A, X, B) implies (A, B))
8. Repeated-cell patterns (e.g., A -> B -> C -> A)
9. Different methods evaluated in identical metric units (pair-level TP, FP, FN)
10. Zero-denominator handling (documented zero_division policy)

PHASE 3 REQUIREMENTS:
11. Synthetic generator is deterministic and reproducible with fixed seed
12. Realized support matches actual occurrence count in generated sample
13. Planted frequent patterns can be recovered by PrefixSpan
14. Below-threshold planted pattern is NOT treated as required ground truth
15. Synthetic evaluation does not count duplicate predictions more than once
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
# pyrefly: ignore [missing-import]
import pattern_evaluation as pe
import synthetic_benchmark as sb



class TestRealDataEvaluationMetrics(unittest.TestCase):
    """Test suite verifying metric correctness, pair extraction, and edge cases."""

    def setUp(self):
        self.ref_fixture = pd.DataFrame({
            "origin_cell": ["CELL_A", "CELL_B", "CELL_A"],
            "destination_cell": ["CELL_B", "CELL_C", "CELL_C"],
            "raw_occurrence_count": [25, 20, 15],
            "n_taxis_involved": [8, 7, 5],
            "n_taxi_days_involved": [12, 10, 8],
            "include_in_reference": [True, True, True],
        })
        self.ref_pairs = {("CELL_A", "CELL_B"), ("CELL_B", "CELL_C"), ("CELL_A", "CELL_C")}

    def test_perfect_recovery(self):
        """Test case where predicted pairs perfectly match reference pairs."""
        patterns_df = pd.DataFrame({
            "pattern": [("CELL_A", "CELL_B", "CELL_C")],
            "pattern_length": [3],
        })
        pred_pairs = pe.extract_ordered_pairs_from_patterns(patterns_df, is_subsequence=True)
        counts = pe.confusion_counts_pair_level(pred_pairs, self.ref_pairs)
        metrics = pe.precision_recall_f1(counts)

        self.assertEqual(counts["TP"], 3)
        self.assertEqual(counts["FP"], 0)
        self.assertEqual(counts["FN"], 0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)

    def test_no_predictions(self):
        """Test behavior when no patterns are discovered (empty prediction set)."""
        empty_df = pd.DataFrame(columns=["pattern", "pattern_length"])
        pred_pairs = pe.extract_ordered_pairs_from_patterns(empty_df, is_subsequence=True)
        counts = pe.confusion_counts_pair_level(pred_pairs, self.ref_pairs)

        self.assertEqual(counts["TP"], 0)
        self.assertEqual(counts["FP"], 0)
        self.assertEqual(counts["FN"], 3)

        # Under default zero_division=None:
        metrics_none = pe.precision_recall_f1(counts, zero_division=None)
        self.assertIsNone(metrics_none["precision"])
        self.assertEqual(metrics_none["recall"], 0.0)
        self.assertIsNone(metrics_none["f1"])

        # Under zero_division=0.0:
        metrics_zero = pe.precision_recall_f1(counts, zero_division=0.0)
        self.assertEqual(metrics_zero["precision"], 0.0)
        self.assertEqual(metrics_zero["recall"], 0.0)
        self.assertEqual(metrics_zero["f1"], 0.0)

    def test_no_reference_pairs(self):
        """Test behavior when reference set is empty."""
        empty_ref = set()
        patterns_df = pd.DataFrame({
            "pattern": [("CELL_A", "CELL_B")],
            "pattern_length": [2],
        })
        pred_pairs = pe.extract_ordered_pairs_from_patterns(patterns_df, is_subsequence=True)
        counts = pe.confusion_counts_pair_level(pred_pairs, empty_ref)

        self.assertEqual(counts["TP"], 0)
        self.assertEqual(counts["FP"], 1)
        self.assertEqual(counts["FN"], 0)

        metrics = pe.precision_recall_f1(counts, zero_division=None)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertIsNone(metrics["recall"])
        self.assertIsNone(metrics["f1"])

    def test_partial_recovery(self):
        """Test partial recovery with both false positives and false negatives."""
        patterns_df = pd.DataFrame({
            "pattern": [("CELL_A", "CELL_B"), ("CELL_X", "CELL_Y")],
            "pattern_length": [2, 2],
        })
        pred_pairs = pe.extract_ordered_pairs_from_patterns(patterns_df, is_subsequence=True)
        counts = pe.confusion_counts_pair_level(pred_pairs, self.ref_pairs)
        metrics = pe.precision_recall_f1(counts)

        # Predicted: (CELL_A, CELL_B), (CELL_X, CELL_Y)
        # Reference: (CELL_A, CELL_B), (CELL_B, CELL_C), (CELL_A, CELL_C)
        self.assertEqual(counts["TP"], 1)  # (CELL_A, CELL_B)
        self.assertEqual(counts["FP"], 1)  # (CELL_X, CELL_Y)
        self.assertEqual(counts["FN"], 2)  # (CELL_B, CELL_C), (CELL_A, CELL_C)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 1 / 3, places=4)
        self.assertAlmostEqual(metrics["f1"], 2 * 0.5 * (1/3) / (0.5 + 1/3), places=4)

    def test_reversed_ordered_pairs(self):
        """Test that (A, B) and (B, A) are treated as distinct directional pairs."""
        ref_single = {("CELL_A", "CELL_B")}
        # Predicted is reverse direction (CELL_B, CELL_A)
        patterns_df = pd.DataFrame({
            "pattern": [("CELL_B", "CELL_A")],
            "pattern_length": [2],
        })
        pred_pairs = pe.extract_ordered_pairs_from_patterns(patterns_df, is_subsequence=True)
        counts = pe.confusion_counts_pair_level(pred_pairs, ref_single)

        self.assertEqual(counts["TP"], 0)
        self.assertEqual(counts["FP"], 1)  # (CELL_B, CELL_A) is a FP
        self.assertEqual(counts["FN"], 1)  # (CELL_A, CELL_B) was missed

    def test_duplicate_pairs_across_multiple_patterns(self):
        """Test that a pair appearing in multiple patterns is deduplicated and not double-credited."""
        patterns_df = pd.DataFrame({
            "pattern": [
                ("CELL_A", "CELL_B"),
                ("CELL_A", "CELL_B", "CELL_X"),
                ("CELL_A", "CELL_B", "CELL_Y"),
                ("CELL_Z", "CELL_A", "CELL_B"),
            ],
            "pattern_length": [2, 3, 3, 3],
        })
        ref_single = {("CELL_A", "CELL_B")}
        pred_pairs = pe.extract_ordered_pairs_from_patterns(patterns_df, is_subsequence=True)
        counts = pe.confusion_counts_pair_level(pred_pairs, ref_single)

        # (CELL_A, CELL_B) appears in all 4 patterns, but must count as exactly 1 TP
        self.assertEqual(counts["TP"], 1)
        self.assertEqual(counts["FN"], 0)
        # Other pairs: (A,X), (B,X), (A,Y), (B,Y), (Z,A), (Z,B) -> 6 FP
        self.assertEqual(counts["FP"], 6)

    def test_gapped_pairs_in_prefixspan(self):
        """Test that PrefixSpan extracts gapped pairs (A, C) from sequence (A, X, C)."""
        patterns_df = pd.DataFrame({
            "pattern": [("CELL_A", "CELL_X", "CELL_C")],
            "pattern_length": [3],
        })
        # With is_subsequence=True, (A, C) is extracted
        ps_pairs = pe.extract_ordered_pairs_from_patterns(patterns_df, is_subsequence=True)
        self.assertIn(("CELL_A", "CELL_C"), ps_pairs)
        self.assertIn(("CELL_A", "CELL_X"), ps_pairs)
        self.assertIn(("CELL_X", "CELL_C"), ps_pairs)

        # With is_subsequence=False (direct transitions), (A, C) is NOT extracted
        direct_pairs = pe.extract_ordered_pairs_from_patterns(patterns_df, is_subsequence=False)
        self.assertNotIn(("CELL_A", "CELL_C"), direct_pairs)
        self.assertIn(("CELL_A", "CELL_X"), direct_pairs)
        self.assertIn(("CELL_X", "CELL_C"), direct_pairs)

    def test_repeated_cell_patterns(self):
        """Test return routes (e.g. A -> B -> C -> A) to ensure order-sensitive matching works."""
        patterns_df = pd.DataFrame({
            "pattern": [("CELL_A", "CELL_B", "CELL_C", "CELL_A")],
            "pattern_length": [4],
        })
        pred_pairs = pe.extract_ordered_pairs_from_patterns(patterns_df, is_subsequence=True)
        # Should include (A, B), (B, C), (C, A), (B, A), (A, C)
        self.assertIn(("CELL_A", "CELL_B"), pred_pairs)
        self.assertIn(("CELL_B", "CELL_C"), pred_pairs)
        self.assertIn(("CELL_C", "CELL_A"), pred_pairs)
        self.assertIn(("CELL_B", "CELL_A"), pred_pairs)
        # Self-loops (A, A) must not be included
        self.assertNotIn(("CELL_A", "CELL_A"), pred_pairs)

    def test_different_methods_identical_units(self):
        """Test that PrefixSpan and Direct Transitions use identical metric units."""
        ps_patterns = pd.DataFrame({
            "pattern": [("CELL_A", "CELL_B", "CELL_C")],
            "pattern_length": [3],
        })
        direct_patterns = pd.DataFrame({
            "pattern": [("CELL_A", "CELL_B"), ("CELL_B", "CELL_C")],
            "pattern_length": [2, 2],
        })

        ps_summary = pe.summarize_method_consistent_pairs("PrefixSpan", ps_patterns, self.ref_fixture, 0.01, is_subsequence=True)
        direct_summary = pe.summarize_method_consistent_pairs("Direct", direct_patterns, self.ref_fixture, 0.01, is_subsequence=False)

        # Both report evaluation_unit as pair-level
        self.assertEqual(ps_summary["evaluation_unit"], "Distinct Ordered Pairs (Pair-Level)")
        self.assertEqual(direct_summary["evaluation_unit"], "Distinct Ordered Pairs (Pair-Level)")
        # In both summaries, TP + FP equals predicted_pairs
        self.assertEqual(ps_summary["TP"] + ps_summary["FP"], ps_summary["predicted_pairs"])
        self.assertEqual(direct_summary["TP"] + direct_summary["FP"], direct_summary["predicted_pairs"])

    def test_zero_denominator_handling(self):
        """Test zero-denominator behavior across all edge cases."""
        # 1. TP=0, FP=0, FN=0
        c0 = {"TP": 0, "FP": 0, "FN": 0}
        m0 = pe.precision_recall_f1(c0, zero_division=0.0)
        self.assertEqual(m0["precision"], 0.0)
        self.assertEqual(m0["recall"], 0.0)
        self.assertEqual(m0["f1"], 0.0)

        # 2. TP=0, FP=5, FN=0 (all predictions wrong, empty reference)
        c1 = {"TP": 0, "FP": 5, "FN": 0}
        m1 = pe.precision_recall_f1(c1, zero_division=0.0)
        self.assertEqual(m1["precision"], 0.0)
        self.assertEqual(m1["recall"], 0.0)
        self.assertEqual(m1["f1"], 0.0)


class TestSyntheticBenchmark(unittest.TestCase):
    """Test suite verifying synthetic benchmark reproducibility and realized support calculation."""

    def test_generator_reproducibility(self):
        """Test that same random seed produces identical synthetic datasets."""
        cfg1 = sb.SyntheticBenchmarkConfig(n_taxis=15, n_days=3, seed=999)
        cfg2 = sb.SyntheticBenchmarkConfig(n_taxis=15, n_days=3, seed=999)

        data1 = sb.generate_synthetic_dataset(cfg1)
        data2 = sb.generate_synthetic_dataset(cfg2)

        self.assertEqual(len(data1["sequences"]), len(data2["sequences"]))
        self.assertEqual(data1["sequences"], data2["sequences"])

    def test_realized_support_calculation(self):
        """Test that realized support exactly counts finite sequence occurrences."""
        cfg = sb.SyntheticBenchmarkConfig(n_taxis=20, n_days=3, seed=123)
        data = sb.generate_synthetic_dataset(cfg)
        sequences = data["sequences"]
        n_seqs = len(sequences)

        min_support = 0.10
        table = sb.compute_realized_support_table(sequences, cfg.planted_patterns, min_support)

        for _, row in table.iterrows():
            pat = row["pattern"]
            actual_count = sum(1 for s in sequences.values() if sb.is_subsequence(pat, s))
            self.assertEqual(row["realized_match_count"], actual_count)
            self.assertAlmostEqual(row["realized_support_pct"], round(100 * actual_count / n_seqs, 2))
            min_count_req = max(2, math.ceil(min_support * n_seqs))
            self.assertEqual(row["min_support_count_required"], min_count_req)
            self.assertEqual(row["is_realized_frequent"], actual_count >= min_count_req)

    def test_planted_frequent_pattern_recovery(self):
        """Test that high-support planted patterns are recovered by PrefixSpan."""
        cfg = sb.SyntheticBenchmarkConfig(n_taxis=25, n_days=4, seed=42)
        data = sb.generate_synthetic_dataset(cfg)
        res = sb.run_synthetic_evaluation(data, min_support=0.10, max_pattern_length=4)

        ps_pair_row = res["method_comparison_pair_level"][
            res["method_comparison_pair_level"]["method"] == "PrefixSpan (Gapped Sequential)"
        ].iloc[0]

        # PrefixSpan should achieve very high recall on frequent planted patterns
        self.assertGreater(ps_pair_row["recall"], 0.85)

    def test_below_threshold_pattern_not_ground_truth(self):
        """Test that rare planted patterns (below min_support) are excluded from ground truth."""
        cfg = sb.SyntheticBenchmarkConfig(n_taxis=30, n_days=4, seed=42)
        data = sb.generate_synthetic_dataset(cfg)
        res = sb.run_synthetic_evaluation(data, min_support=0.10, max_pattern_length=4)

        planted_table = res["realized_planted_table"]
        rare_row = planted_table[planted_table["pattern"] == ("RARE_D1", "RARE_D2")].iloc[0]

        # RARE_D1 -> RARE_D2 was injected with 0.03 prob, which is below 0.10 min_support
        self.assertFalse(rare_row["is_realized_frequent"])
        # Should not be in realized frequent patterns
        frequent_patterns = set(planted_table.loc[planted_table["is_realized_frequent"], "pattern"])
        self.assertNotIn(("RARE_D1", "RARE_D2"), frequent_patterns)

    def test_no_duplicate_counting_in_synthetic_evaluation(self):
        """Test that duplicate pairs in synthetic evaluation are not double counted."""
        cfg = sb.SyntheticBenchmarkConfig(n_taxis=20, n_days=3, seed=123)
        data = sb.generate_synthetic_dataset(cfg)
        res = sb.run_synthetic_evaluation(data, min_support=0.10, max_pattern_length=4)

        for _, row in res["method_comparison_pair_level"].iterrows():
            # In strictly unified pair evaluation: TP + FP == predicted_pairs
            self.assertEqual(row["TP"] + row["FP"], row["predicted_pairs"])
            # TP <= reference_pairs
            self.assertLessEqual(row["TP"], row["reference_pairs"])


def run_all_tests():
    """Run all automated unit tests and print summary."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestRealDataEvaluationMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestSyntheticBenchmark))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("One or more automated evaluation tests failed!")
    print("\nALL 15 EVALUATION & SYNTHETIC BENCHMARK TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_all_tests()
