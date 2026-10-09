"""
synthetic_benchmark.py
======================
Objective Synthetic Evaluation Benchmark with Known Planted Sequential Patterns.

This module provides a reproducible, controlled testing environment for evaluating
sequential pattern mining (PrefixSpan), direct contiguous-transition mining, and
unordered co-occurrence baseline against known, planted ground-truth patterns.

WHY A SYNTHETIC BENCHMARK IS ESSENTIAL
--------------------------------------
Real-world urban mobility data lacks objective ground-truth labels for frequent routes.
Evaluating real-data patterns against an empirical reference constructed from adjacent
transitions provides useful descriptive validation, but structurally biases results
toward adjacent methods and cannot verify gapped sequence recovery.

This synthetic benchmark generates controlled trajectories with known ground truth to
rigorously test:
1. Exact pattern recovery (Pattern-Level Precision, Recall, F1).
2. Ordered transition recovery (Pair-Level Precision, Recall, F1).
3. Resilience to non-systematic noise and gapped sequences.
4. Sensitivity across varying support thresholds and max pattern lengths.

IMPORTANT: This benchmark is strictly isolated from real T-Drive data and is never
presented as real-world model accuracy.
"""

from __future__ import annotations

import math
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from prefixspan import PrefixSpan

import pattern_evaluation as pe
import sequential


@dataclass
class SyntheticBenchmarkConfig:
    """Configuration for synthetic trajectory generation."""
    n_taxis: int = 50
    n_days: int = 7
    seq_min_length: int = 8
    seq_max_length: int = 16
    noise_cells: list[str] = field(
        default_factory=lambda: [f"NOISE_{i:02d}" for i in range(1, 21)]
    )
    # Planted patterns with their target injection probabilities across taxi-days
    # Pattern format: tuple of cell IDs -> injection probability (0.0 - 1.0)
    planted_patterns: dict[tuple[str, ...], float] = field(
        default_factory=lambda: {
            ("CORRIDOR_A1", "CORRIDOR_A2", "CORRIDOR_A3"): 0.35,  # Length 3, frequent
            ("CORRIDOR_B1", "CORRIDOR_B2"): 0.25,                 # Length 2, frequent
            ("CORRIDOR_C1", "CORRIDOR_C2", "CORRIDOR_C3", "CORRIDOR_C4"): 0.15, # Length 4, moderate
            ("RARE_D1", "RARE_D2"): 0.03,                         # Length 2, rare / below min_support
        }
    )
    gap_probability: float = 0.3  # Probability of inserting random noise between pattern items
    seed: int = 42


def generate_synthetic_dataset(
    config: SyntheticBenchmarkConfig = SyntheticBenchmarkConfig(),
) -> dict[str, Any]:
    """
    Generate synthetic taxi-day sequences with planted patterns and controlled noise.

    Returns
    -------
    dict containing:
        - sequences: dict[str, list[str]] (taxi_id::date -> list of cells)
        - planted_patterns: dict[tuple[str, ...], float]
        - true_frequent_patterns: set of planted patterns with injection rate >= threshold
        - ground_truth_pairs: set of all ordered pairs contained in planted patterns
    """
    random.seed(config.seed)
    np.random.seed(config.seed)

    sequences: dict[str, list[str]] = {}
    injection_records: list[dict] = []

    dates = [f"2008-02-{d:02d}" for d in range(2, 2 + config.n_days)]

    for taxi_num in range(1, config.n_taxis + 1):
        taxi_id = str(taxi_num)
        for date_str in dates:
            seq_key = f"{taxi_id}::{date_str}"
            current_seq: list[str] = []

            # 1. Start with baseline random noise
            base_len = random.randint(config.seq_min_length // 2, config.seq_max_length // 2)
            current_seq.extend(random.choices(config.noise_cells, k=base_len))

            # 2. Inquire whether each planted pattern should be injected
            injected_in_this_seq: list[tuple[str, ...]] = []
            for pattern, prob in config.planted_patterns.items():
                if random.random() < prob:
                    injected_in_this_seq.append(pattern)
                    # Insert pattern with optional gaps
                    insert_idx = random.randint(0, len(current_seq))
                    pattern_items = list(pattern)
                    
                    if config.gap_probability > 0:
                        # With probability gap_prob, insert a noise cell inside the pattern
                        expanded_pattern: list[str] = []
                        for i, item in enumerate(pattern_items):
                            expanded_pattern.append(item)
                            if i < len(pattern_items) - 1 and random.random() < config.gap_probability:
                                expanded_pattern.append(random.choice(config.noise_cells))
                        pattern_items = expanded_pattern

                    current_seq[insert_idx:insert_idx] = pattern_items

            # 3. Add trailing background noise
            tail_noise = random.choices(config.noise_cells, k=random.randint(1, 4))
            current_seq.extend(tail_noise)

            # 4. Compress consecutive duplicate cells (same as project pipeline)
            compressed_seq = sequential.compress_consecutive_duplicates(current_seq)

            if len(compressed_seq) >= 2:
                sequences[seq_key] = compressed_seq
                injection_records.append({
                    "sequence_key": seq_key,
                    "injected_patterns": injected_in_this_seq,
                    "length": len(compressed_seq),
                })

    # Derive true ground-truth ordered transition pairs from planted patterns
    ground_truth_pairs: set[tuple[str, str]] = set()
    for pattern in config.planted_patterns:
        for i in range(len(pattern)):
            for j in range(i + 1, len(pattern)):
                ground_truth_pairs.add((pattern[i], pattern[j]))

    return {
        "sequences": sequences,
        "config": config,
        "ground_truth_patterns": set(config.planted_patterns.keys()),
        "ground_truth_pairs": ground_truth_pairs,
        "injection_records": pd.DataFrame(injection_records),
    }


def run_synthetic_evaluation(
    synthetic_data: dict[str, Any],
    min_support: float = 0.10,
    max_pattern_length: int = 4,
) -> dict[str, pd.DataFrame]:
    """
    Run PrefixSpan, direct contiguous transitions, and co-occurrence on synthetic data
    and calculate both Pattern-Level and Pair-Level metrics against known ground truth.

    Returns
    -------
    dict with:
        - method_comparison_pair_level: pd.DataFrame
        - method_comparison_pattern_level: pd.DataFrame
        - mined_patterns: dict of DataFrames
    """
    sequences = synthetic_data["sequences"]
    gt_patterns = synthetic_data["ground_truth_patterns"]
    gt_pairs = synthetic_data["ground_truth_pairs"]

    # Filter ground truth patterns that meet min_support
    expected_frequent_patterns = {
        pat for pat, prob in synthetic_data["config"].planted_patterns.items()
        if prob >= min_support
    }
    expected_frequent_pairs: set[tuple[str, str]] = set()
    for pat in expected_frequent_patterns:
        for i in range(len(pat)):
            for j in range(i + 1, len(pat)):
                expected_frequent_pairs.add((pat[i], pat[j]))

    # 1. Mine PrefixSpan
    t0 = time.time()
    ps_df = sequential.mine_patterns(
        sequences,
        min_support=min_support,
        max_pattern_length=max_pattern_length,
    )
    ps_runtime = time.time() - t0

    # 2. Mine Direct Transitions (Contiguous N-grams)
    t0 = time.time()
    direct_df = sequential.mine_contiguous_patterns(
        sequences,
        min_support=min_support,
        max_pattern_length=max_pattern_length,
    )
    direct_runtime = time.time() - t0

    # 3. Mine Unordered Co-occurrence
    t0 = time.time()
    cooc_df = pe.mine_cooccurrence_pairs(
        sequences,
        min_support=min_support,
    )
    cooc_runtime = time.time() - t0

    # --- PAIR-LEVEL EVALUATION ---
    # Extract ordered pairs from each method
    def extract_pairs(df: pd.DataFrame, is_subsequence: bool = True) -> set[tuple[str, str]]:
        pairs: set[tuple[str, str]] = set()
        for pat in df["pattern"]:
            pat_t = tuple(pat)
            if is_subsequence:
                for i in range(len(pat_t)):
                    for j in range(i + 1, len(pat_t)):
                        pairs.add((pat_t[i], pat_t[j]))
            else:
                for i in range(len(pat_t) - 1):
                    pairs.add((pat_t[i], pat_t[i + 1]))
        return pairs

    ps_pairs = extract_pairs(ps_df, is_subsequence=True)
    direct_pairs = extract_pairs(direct_df, is_subsequence=False)
    
    # Co-occurrence pairs (unordered treated as both directions or symmetric)
    cooc_pairs: set[tuple[str, str]] = set()
    for pat in cooc_df["pattern"]:
        cooc_pairs.add((pat[0], pat[1]))
        cooc_pairs.add((pat[1], pat[0]))

    def compute_pair_row(name: str, pred_pairs: set[tuple[str, str]], runtime: float) -> dict:
        tp = len(pred_pairs & expected_frequent_pairs)
        fp = len(pred_pairs - expected_frequent_pairs)
        fn = len(expected_frequent_pairs - pred_pairs)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        return {
            "method": name,
            "evaluation_level": "Pair-Level (Ordered Transitions)",
            "predicted_pairs_count": len(pred_pairs),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "runtime_sec": round(runtime, 4),
        }

    pair_rows = [
        compute_pair_row("PrefixSpan (Gapped Sequential)", ps_pairs, ps_runtime),
        compute_pair_row("Direct Adjacent Transition", direct_pairs, direct_runtime),
        compute_pair_row("Unordered Co-occurrence", cooc_pairs, cooc_runtime),
    ]
    pair_eval_df = pd.DataFrame(pair_rows)

    # --- PATTERN-LEVEL EVALUATION ---
    def compute_pattern_row(name: str, df: pd.DataFrame, runtime: float) -> dict:
        pred_patterns = set(df["pattern"].apply(tuple))
        tp = len(pred_patterns & expected_frequent_patterns)
        fp = len(pred_patterns - expected_frequent_patterns)
        fn = len(expected_frequent_patterns - pred_patterns)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        return {
            "method": name,
            "evaluation_level": "Pattern-Level (Exact Pattern Match)",
            "patterns_mined": len(pred_patterns),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "runtime_sec": round(runtime, 4),
        }

    pattern_rows = [
        compute_pattern_row("PrefixSpan (Gapped Sequential)", ps_df, ps_runtime),
        compute_pattern_row("Direct Adjacent Transition", direct_df, direct_runtime),
    ]
    pattern_eval_df = pd.DataFrame(pattern_rows)

    return {
        "method_comparison_pair_level": pair_eval_df,
        "method_comparison_pattern_level": pattern_eval_df,
        "mined_patterns": {
            "prefixspan": ps_df,
            "direct": direct_df,
            "cooccurrence": cooc_df,
        },
    }


if __name__ == "__main__":
    print("Generating synthetic benchmark dataset...")
    synthetic = generate_synthetic_dataset()
    print(f"Generated {len(synthetic['sequences'])} synthetic taxi-day sequences.")
    print("Planted patterns:", list(synthetic["ground_truth_patterns"]))
    
    results = run_synthetic_evaluation(synthetic, min_support=0.10, max_pattern_length=4)
    print("\n=== SYNTHETIC BENCHMARK RESULTS (PAIR-LEVEL) ===")
    print(results["method_comparison_pair_level"].to_string(index=False))
    print("\n=== SYNTHETIC BENCHMARK RESULTS (PATTERN-LEVEL) ===")
    print(results["method_comparison_pattern_level"].to_string(index=False))
