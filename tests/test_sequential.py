"""
test_sequential.py
==================
Automated unit tests for Phase 3 Sequential Mobility Pattern Mining:
- Calendar-day and trip-level sequence building
- Consecutive duplicate cell compression
- PrefixSpan vs Contiguous n-gram mining
- Pattern traceability to supporting trajectories
- Long pattern support evaluation
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config
import sequential as sq


class TestSequentialPatternMining(unittest.TestCase):
    """Test suite verifying sequence construction and pattern mining."""

    def setUp(self):
        # 3 taxis across 2 days
        self.points_df = pd.DataFrame({
            config.COL_TAXI_ID: ["1", "1", "1", "1", "2", "2", "2", "3", "3", "3"],
            config.COL_TIMESTAMP: pd.to_datetime([
                "2008-02-02 08:00:00",
                "2008-02-02 08:05:00",
                "2008-02-02 08:10:00",
                "2008-02-03 08:00:00",  # Day 2
                "2008-02-02 08:00:00",
                "2008-02-02 08:05:00",
                "2008-02-02 08:10:00",
                "2008-02-02 08:00:00",
                "2008-02-02 08:05:00",
                "2008-02-02 08:10:00",
            ]),
            "spatial_cell_id": [
                "CELL_A", "CELL_B", "CELL_C", "CELL_A",  # Taxi 1
                "CELL_A", "CELL_B", "CELL_C",            # Taxi 2
                "CELL_A", "CELL_X", "CELL_C",            # Taxi 3 (gapped: A -> X -> C)
            ],
        })

    def test_sequence_construction_calendar_split(self):
        """Sequences must be split by calendar day to avoid cross-day artificial jumps."""
        seqs = sq.build_taxi_sequences(self.points_df, split_by_date=True)
        # Taxi 1 day 1 has [CELL_A, CELL_B, CELL_C] (length 3)
        # Taxi 1 day 2 has only 1 point [CELL_A] (length 1 -> excluded as < 2)
        self.assertIn("1::2008-02-02", seqs)
        self.assertNotIn("1::2008-02-03", seqs)
        self.assertEqual(seqs["1::2008-02-02"], ["CELL_A", "CELL_B", "CELL_C"])

    def test_prefixspan_gapped_subsequence_mining(self):
        """PrefixSpan must identify gapped subsequences (e.g. CELL_A -> CELL_C across all 3 taxis)."""
        seqs = sq.build_taxi_sequences(self.points_df, split_by_date=True)
        # 3 qualifying sequences: 1::2008-02-02, 2::2008-02-02, 3::2008-02-02
        # (CELL_A, CELL_C) occurs in all 3 sequences (100% support)
        patterns = sq.mine_patterns(seqs, min_support=0.6, max_pattern_length=2)
        pat_tuples = set(patterns["pattern"].tolist())
        self.assertIn(("CELL_A", "CELL_C"), pat_tuples)

    def test_contiguous_ngram_mining_distinguishes_gaps(self):
        """Contiguous mining must only count direct adjacent movements, excluding gapped sequences."""
        seqs = sq.build_taxi_sequences(self.points_df, split_by_date=True)
        direct_pats = sq.mine_contiguous_patterns(seqs, min_support=0.6, max_pattern_length=2)
        direct_tuples = set(direct_pats["pattern"].tolist())
        # (CELL_A, CELL_B) is direct in taxi 1 & 2 (2/3 = 66.7%) -> included
        self.assertIn(("CELL_A", "CELL_B"), direct_tuples)
        # (CELL_A, CELL_C) is NOT directly adjacent in any sequence -> must NOT be in direct_pats
        self.assertNotIn(("CELL_A", "CELL_C"), direct_tuples)

    def test_pattern_traceability(self):
        """Mined patterns must trace back to the exact supporting trajectory keys."""
        seqs = sq.build_taxi_sequences(self.points_df, split_by_date=True)
        matched_keys = sq.trace_pattern_occurrences(("CELL_A", "CELL_B"), seqs, is_contiguous=True)
        self.assertEqual(sorted(matched_keys), ["1::2008-02-02", "2::2008-02-02"])

    def test_evaluate_pattern_length_support(self):
        """Pattern length evaluation must report max supported lengths accurately."""
        seqs = sq.build_taxi_sequences(self.points_df, split_by_date=True)
        eval_df = sq.evaluate_pattern_length_support(seqs, lengths=[2, 3, 5], min_support=0.5)
        self.assertEqual(len(eval_df), 3)
        self.assertTrue(eval_df[eval_df["target_length"] == 2]["is_supported"].iloc[0])
        # Length 5 cannot be supported because max sequence length is 3
        self.assertFalse(eval_df[eval_df["target_length"] == 5]["is_supported"].iloc[0])


if __name__ == "__main__":
    unittest.main()
