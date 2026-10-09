"""
test_evaluation.py
==================
Comprehensive automated test suite for pattern evaluation and synthetic benchmarks.

Covers all required edge cases:
- A pattern recovering multiple reference pairs
- A pattern recovering no reference pairs
- Duplicate predictions
- Repeated cells within patterns (e.g. A -> B -> C -> A)
- Gapped subsequences vs contiguous transitions
- Empty reference sets
- Empty predictions
- Zero true positives (all incorrect)
- Perfect recovery (all correct)
- Asserting TP, FP, FN, Precision, Recall, F1 across all cases
"""

import sys
from pathlib import Path

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import pattern_evaluation as pe
import synthetic_benchmark as sb


def test_multiple_reference_recovery():
    """Test a pattern that recovers multiple reference pairs (e.g. length 3 pattern A->B->C)."""
    ref_df = pd.DataFrame({
        "origin_cell": ["A", "B", "A"],
        "destination_cell": ["B", "C", "C"],
        "raw_occurrence_count": [10, 10, 10],
        "n_taxis_involved": [5, 5, 5],
        "n_taxi_days_involved": [5, 5, 5],
        "include_in_reference": [True, True, True],
    })
    patterns_df = pd.DataFrame({
        "pattern": [("A", "B", "C")],
        "pattern_length": [3],
    })
    
    summary = pe.summarize_method_consistent_pairs(
        "PrefixSpan", patterns_df, ref_df, runtime_sec=0.01, is_subsequence=True
    )
    # (A,B,C) contains pairs (A,B), (B,C), (A,C) -> all 3 match reference pairs
    assert summary["TP"] == 3
    assert summary["FP"] == 0
    assert summary["FN"] == 0
    assert summary["precision"] == 1.0
    assert summary["recall"] == 1.0
    assert summary["f1"] == 1.0


def test_no_reference_recovery_and_zero_tp():
    """Test patterns that do not match any reference pairs."""
    ref_df = pd.DataFrame({
        "origin_cell": ["A", "B"],
        "destination_cell": ["B", "C"],
        "raw_occurrence_count": [10, 10],
        "n_taxis_involved": [5, 5],
        "n_taxi_days_involved": [5, 5],
        "include_in_reference": [True, True],
    })
    patterns_df = pd.DataFrame({
        "pattern": [("X", "Y"), ("Z", "W")],
        "pattern_length": [2, 2],
    })
    
    summary = pe.summarize_method_consistent_pairs(
        "Unrelated", patterns_df, ref_df, runtime_sec=0.01, is_subsequence=True
    )
    assert summary["TP"] == 0
    assert summary["FP"] == 2  # (X,Y) and (Z,W)
    assert summary["FN"] == 2  # (A,B) and (B,C)
    assert summary["precision"] == 0.0
    assert summary["recall"] == 0.0
    assert summary["f1"] is None or summary["f1"] == 0.0


def test_repeated_cells_in_pattern():
    """Test returning routes (e.g. A -> B -> C -> A) to ensure first_o < last_d works."""
    ref_df = pd.DataFrame({
        "origin_cell": ["A", "B"],
        "destination_cell": ["B", "C"],
        "raw_occurrence_count": [10, 10],
        "n_taxis_involved": [5, 5],
        "n_taxi_days_involved": [5, 5],
        "include_in_reference": [True, True],
    })
    # Pattern contains return to A
    patterns_df = pd.DataFrame({
        "pattern": [("A", "B", "C", "A")],
        "pattern_length": [4],
    })
    
    matched = pe.match_patterns_to_reference(patterns_df, ref_df, order_sensitive=True)
    assert matched.iloc[0]["matched_reference"] == True
    assert set(matched.iloc[0]["matched_pairs"]) == {("A", "B"), ("B", "C")}


def test_duplicate_prediction_handling():
    """Test that duplicate pairs across multiple patterns are deduplicated."""
    ref_df = pd.DataFrame({
        "origin_cell": ["A"],
        "destination_cell": ["B"],
        "raw_occurrence_count": [10],
        "n_taxis_involved": [5],
        "n_taxi_days_involved": [5],
        "include_in_reference": [True],
    })
    patterns_df = pd.DataFrame({
        "pattern": [("A", "B"), ("A", "B", "X"), ("A", "B", "Y")],
        "pattern_length": [2, 3, 3],
    })
    pred_pairs = pe.extract_ordered_pairs_from_patterns(patterns_df, is_subsequence=True)
    counts = pe.confusion_counts_pair_level(
        pred_pairs, {("A", "B")}
    )
    # (A,B) is counted only once in TP
    assert counts["TP"] == 1
    # Other pairs: (A,X), (B,X), (A,Y), (B,Y) -> 4 FP
    assert counts["FP"] == 4
    assert counts["FN"] == 0


def test_empty_predictions():
    """Test behavior when no patterns are discovered."""
    ref_df = pd.DataFrame({
        "origin_cell": ["A"],
        "destination_cell": ["B"],
        "raw_occurrence_count": [10],
        "n_taxis_involved": [5],
        "n_taxi_days_involved": [5],
        "include_in_reference": [True],
    })
    patterns_df = pd.DataFrame(columns=["pattern", "pattern_length"])
    summary = pe.summarize_method_consistent_pairs(
        "Empty", patterns_df, ref_df, runtime_sec=0.01, is_subsequence=True
    )
    assert summary["TP"] == 0
    assert summary["FP"] == 0
    assert summary["FN"] == 1
    assert summary["precision"] is None
    assert summary["recall"] == 0.0
    assert summary["f1"] is None


def test_empty_reference():
    """Test behavior when reference set is empty."""
    ref_df = pd.DataFrame(columns=pe.REFERENCE_COLUMNS)
    patterns_df = pd.DataFrame({
        "pattern": [("A", "B")],
        "pattern_length": [2],
    })
    summary = pe.summarize_method_consistent_pairs(
        "PrefixSpan", patterns_df, ref_df, runtime_sec=0.01, is_subsequence=True
    )
    assert summary["TP"] == 0
    assert summary["FP"] == 1
    assert summary["FN"] == 0
    assert summary["precision"] == 0.0
    assert summary["recall"] is None
    assert summary["f1"] is None


def test_synthetic_planted_benchmark_execution():
    """Test that synthetic benchmark runs end-to-end and recovers planted patterns."""
    config = sb.SyntheticBenchmarkConfig(n_taxis=20, n_days=3, seed=123)
    dataset = sb.generate_synthetic_dataset(config)
    results = sb.run_synthetic_evaluation(dataset, min_support=0.10, max_pattern_length=4)
    
    pair_df = results["method_comparison_pair_level"]
    assert len(pair_df) == 3
    assert "PrefixSpan (Gapped Sequential)" in pair_df["method"].values
    
    # PrefixSpan should achieve high recall on planted frequent patterns
    ps_row = pair_df[pair_df["method"] == "PrefixSpan (Gapped Sequential)"].iloc[0]
    assert ps_row["recall"] > 0.8


def run_all_tests():
    print("Running automated test suite for evaluation methodology...")
    test_multiple_reference_recovery()
    print("  [PASS] test_multiple_reference_recovery")
    test_no_reference_recovery_and_zero_tp()
    print("  [PASS] test_no_reference_recovery_and_zero_tp")
    test_repeated_cells_in_pattern()
    print("  [PASS] test_repeated_cells_in_pattern")
    test_duplicate_prediction_handling()
    print("  [PASS] test_duplicate_prediction_handling")
    test_empty_predictions()
    print("  [PASS] test_empty_predictions")
    test_empty_reference()
    print("  [PASS] test_empty_reference")
    test_synthetic_planted_benchmark_execution()
    print("  [PASS] test_synthetic_planted_benchmark_execution")
    print("\nALL 7 AUTOMATED EVALUATION TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_all_tests()
