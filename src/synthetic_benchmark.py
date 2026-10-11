"""
synthetic_benchmark.py
======================
Objective Synthetic Evaluation Benchmark with Known Planted Sequential Patterns.

METHODOLOGICAL PURPOSE:
-----------------------
Real-world urban mobility data lacks external, objective ground truth for recurring routes.
Evaluating real-data patterns against an empirical reference constructed from adjacent
transitions provides useful descriptive validation, but structurally biases results
toward adjacent methods and cannot verify gapped sequence recovery.

This synthetic benchmark establishes a controlled, fully reproducible ground-truth
environment to rigorously test:
1. Exact pattern recovery (Pattern-Level Precision, Recall, F1).
2. Ordered transition recovery (Pair-Level Precision, Recall, F1).
3. Resilience to non-systematic noise, shared/overlapping cells, and gapped sequences.
4. Calculation of REALIZED support from the actual finite generated sample, rather than
   the theoretical generation probability.

CRITICAL METHODOLOGICAL RULES:
------------------------------
- Realized Support: A planted pattern is only considered ground truth if its realized
  occurrence count in the generated finite dataset meets or exceeds the exact support threshold:
    min_support_count = max(2, math.ceil(min_support * n_sequences))
- Clear Distinction of Pattern Categories:
  * Intended Planted Patterns: Defined in generator configuration.
  * Realized Frequent Patterns: Planted patterns meeting the support threshold in the sample.
  * Recovered Patterns: Patterns correctly discovered by PrefixSpan.
  * Below-Threshold Patterns: Planted patterns that legitimately failed the support threshold.
  * Unexpected / Noise Patterns: Patterns mined from background noise or overlapping sequences.
- Completely Isolated: Kept strictly separate from real T-Drive empirical results.
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
    """Configuration for reproducible synthetic trajectory generation."""
    n_taxis: int = 50
    n_days: int = 7
    seq_min_length: int = 8
    seq_max_length: int = 16
    noise_cells: list[str] = field(
        default_factory=lambda: [f"NOISE_{i:02d}" for i in range(1, 21)]
    )
    # Planted patterns with their intended injection probabilities:
    # Includes frequent corridors, moderate corridors, overlapping/shared cells, and rare below-threshold routes.
    planted_patterns: dict[tuple[str, ...], float] = field(
        default_factory=lambda: {
            ("CORRIDOR_A1", "CORRIDOR_A2", "CORRIDOR_A3"): 0.35,          # Frequent corridor A (length 3)
            ("CORRIDOR_B1", "CORRIDOR_B2"): 0.25,                         # Frequent corridor B (length 2)
            ("CORRIDOR_C1", "CORRIDOR_C2", "CORRIDOR_C3", "CORRIDOR_C4"): 0.15, # Moderate corridor C (length 4)
            ("CORRIDOR_A2", "SHARED_S1", "CORRIDOR_B2"): 0.20,            # Shared-cell corridor (overlapping A & B)
            ("RARE_D1", "RARE_D2"): 0.03,                                 # Rare pattern (intentionally below 10% support)
        }
    )
    gap_probability: float = 0.25  # Probability of inserting background noise between pattern elements
    seed: int = 42


def is_subsequence(pattern: tuple[str, ...] | list[str], sequence: list[str]) -> bool:
    """Check if pattern is an ordered subsequence of sequence (gaps allowed)."""
    it = iter(sequence)
    return all(cell in it for cell in pattern)


def is_contiguous_subsequence(pattern: tuple[str, ...] | list[str], sequence: list[str]) -> bool:
    """Check if pattern appears contiguously in sequence."""
    k = len(pattern)
    for i in range(len(sequence) - k + 1):
        if tuple(sequence[i : i + k]) == tuple(pattern):
            return True
    return False


def generate_synthetic_dataset(
    config: SyntheticBenchmarkConfig = SyntheticBenchmarkConfig(),
) -> dict[str, Any]:
    """
    Generate synthetic taxi-day sequences with planted patterns and controlled noise.

    Returns
    -------
    dict containing:
        - sequences: dict[str, list[str]] (taxi_id::date -> sequence of cells)
        - config: SyntheticBenchmarkConfig
        - planted_patterns: dict[tuple[str, ...], float]
        - injection_records: pd.DataFrame
    """
    rng = random.Random(config.seed)
    np_rng = np.random.default_rng(config.seed)

    sequences: dict[str, list[str]] = {}
    injection_records: list[dict] = []

    dates = [f"2008-02-{d:02d}" for d in range(2, 2 + config.n_days)]

    for taxi_num in range(1, config.n_taxis + 1):
        taxi_id = str(taxi_num)
        for date_str in dates:
            seq_key = f"{taxi_id}::{date_str}"
            current_seq: list[str] = []

            # 1. Baseline random noise
            base_len = rng.randint(config.seq_min_length // 2, config.seq_max_length // 2)
            current_seq.extend(rng.choices(config.noise_cells, k=base_len))

            # 2. Planted pattern injection with optional noise gaps
            injected_in_this_seq: list[tuple[str, ...]] = []
            for pattern, prob in config.planted_patterns.items():
                if rng.random() < prob:
                    injected_in_this_seq.append(pattern)
                    insert_idx = rng.randint(0, len(current_seq))
                    pattern_items = list(pattern)

                    if config.gap_probability > 0:
                        expanded_pattern: list[str] = []
                        for i, item in enumerate(pattern_items):
                            expanded_pattern.append(item)
                            if i < len(pattern_items) - 1 and rng.random() < config.gap_probability:
                                expanded_pattern.append(rng.choice(config.noise_cells))
                        pattern_items = expanded_pattern

                    current_seq[insert_idx:insert_idx] = pattern_items

            # 3. Trailing background noise
            tail_noise = rng.choices(config.noise_cells, k=rng.randint(1, 4))
            current_seq.extend(tail_noise)

            # 4. Compress consecutive duplicate cells (matching real pipeline)
            compressed_seq = sequential.compress_consecutive_duplicates(current_seq)

            if len(compressed_seq) >= 2:
                sequences[seq_key] = compressed_seq
                injection_records.append({
                    "sequence_key": seq_key,
                    "injected_patterns": injected_in_this_seq,
                    "length": len(compressed_seq),
                })

    return {
        "sequences": sequences,
        "config": config,
        "planted_patterns": config.planted_patterns,
        "injection_records": pd.DataFrame(injection_records),
    }


def compute_realized_support_table(
    sequences: dict[str, list[str]],
    planted_patterns: dict[tuple[str, ...], float],
    min_support: float,
) -> pd.DataFrame:
    """
    Calculate realized support for all planted patterns directly from generated sequences.

    Applies exact project support conversion: ceil(min_support * n_sequences).
    """
    n_sequences = len(sequences)
    min_support_count = max(2, math.ceil(min_support * n_sequences))

    rows = []
    for pat, target_prob in planted_patterns.items():
        # Count taxi-days containing the pattern as a subsequence
        match_count = sum(1 for seq in sequences.values() if is_subsequence(pat, seq))
        realized_support_frac = match_count / n_sequences
        is_frequent = match_count >= min_support_count

        rows.append({
            "pattern": pat,
            "pattern_length": len(pat),
            "target_injection_prob": target_prob,
            "realized_match_count": match_count,
            "total_sequences": n_sequences,
            "realized_support_pct": round(100 * realized_support_frac, 2),
            "min_support_count_required": min_support_count,
            "is_realized_frequent": is_frequent,
        })

    return pd.DataFrame(rows).sort_values("realized_match_count", ascending=False).reset_index(drop=True)


def run_synthetic_evaluation(
    synthetic_data: dict[str, Any],
    min_support: float = 0.10,
    max_pattern_length: int = 4,
) -> dict[str, Any]:
    """
    Run PrefixSpan, Direct Transitions, and Co-occurrence baselines on synthetic data
    against REALIZED ground truth.
    """
    sequences = synthetic_data["sequences"]
    planted_patterns = synthetic_data["planted_patterns"]
    n_sequences = len(sequences)
    min_support_count = max(2, math.ceil(min_support * n_sequences))

    # 1. Compute realized ground truth from actual generated sample
    realized_table = compute_realized_support_table(sequences, planted_patterns, min_support)

    realized_frequent_patterns = set(
        realized_table.loc[realized_table["is_realized_frequent"], "pattern"]
    )

    # Derive realized reference ordered pairs from realized frequent patterns
    realized_frequent_pairs: set[tuple[str, str]] = set()
    for pat in realized_frequent_patterns:
        for i in range(len(pat)):
            for j in range(i + 1, len(pat)):
                if pat[i] != pat[j]:
                    realized_frequent_pairs.add((pat[i], pat[j]))

    # 2. Mine patterns with PrefixSpan
    t0 = time.time()
    ps_df = sequential.mine_patterns(
        sequences,
        min_support=min_support,
        max_pattern_length=max_pattern_length,
    )
    ps_runtime = time.time() - t0

    # 3. Mine Direct Transitions (Contiguous N-grams)
    t0 = time.time()
    direct_df = sequential.mine_contiguous_patterns(
        sequences,
        min_support=min_support,
        max_pattern_length=max_pattern_length,
    )
    direct_runtime = time.time() - t0

    # 4. Mine Unordered Co-occurrence
    t0 = time.time()
    cooc_df = pe.mine_cooccurrence_pairs(
        sequences,
        min_support=min_support,
    )
    cooc_runtime = time.time() - t0

    # Update realized table with discovery status
    mined_ps_patterns = set(ps_df["pattern"].apply(tuple)) if len(ps_df) > 0 else set()
    realized_table["recovered_by_prefixspan"] = realized_table["pattern"].isin(mined_ps_patterns)

    # --- PAIR-LEVEL EVALUATION ---
    ps_pairs = pe.extract_ordered_pairs_from_patterns(ps_df, is_subsequence=True)
    direct_pairs = pe.extract_ordered_pairs_from_patterns(direct_df, is_subsequence=False)
    cooc_pairs = pe.extract_ordered_pairs_from_cooccurrence(cooc_df)

    def compute_pair_metrics(name: str, pred_pairs: set[tuple[str, str]], runtime: float) -> dict:
        counts = pe.confusion_counts_pair_level(pred_pairs, realized_frequent_pairs)
        metrics = pe.precision_recall_f1(counts, zero_division=0.0)
        return {
            "method": name,
            "evaluation_unit": "Distinct Ordered Pairs (Pair-Level)",
            "predicted_pairs": len(pred_pairs),
            "reference_pairs": len(realized_frequent_pairs),
            "TP": counts["TP"],
            "FP": counts["FP"],
            "FN": counts["FN"],
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1": round(metrics["f1"], 4),
            "runtime_sec": round(runtime, 4),
        }

    pair_eval_df = pd.DataFrame([
        compute_pair_metrics("PrefixSpan (Gapped Sequential)", ps_pairs, ps_runtime),
        compute_pair_metrics("Direct Adjacent Transition", direct_pairs, direct_runtime),
        compute_pair_metrics("Unordered Co-occurrence (Symmetric Pairs)", cooc_pairs, cooc_runtime),
    ])

    # --- PATTERN-LEVEL EVALUATION ---
    def compute_pattern_metrics(name: str, mined_set: set[tuple[str, ...]], runtime: float) -> dict:
        tp = len(mined_set & realized_frequent_patterns)
        fp = len(mined_set - realized_frequent_patterns)
        fn = len(realized_frequent_patterns - mined_set)
        counts = {"TP": tp, "FP": fp, "FN": fn}
        metrics = pe.precision_recall_f1(counts, zero_division=0.0)
        return {
            "method": name,
            "evaluation_unit": "Full Sequential Pattern (Pattern-Level)",
            "patterns_mined": len(mined_set),
            "reference_patterns": len(realized_frequent_patterns),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1": round(metrics["f1"], 4),
            "runtime_sec": round(runtime, 4),
        }

    direct_patterns_set = set(direct_df["pattern"].apply(tuple)) if len(direct_df) > 0 else set()

    pattern_eval_df = pd.DataFrame([
        compute_pattern_metrics("PrefixSpan (Gapped Sequential)", mined_ps_patterns, ps_runtime),
        compute_pattern_metrics("Direct Adjacent Transition", direct_patterns_set, direct_runtime),
    ])

    return {
        "metadata": {
            "n_sequences": n_sequences,
            "min_support_fraction": min_support,
            "min_support_count_required": min_support_count,
            "n_planted_patterns": len(planted_patterns),
            "n_realized_frequent_patterns": len(realized_frequent_patterns),
            "n_realized_frequent_pairs": len(realized_frequent_pairs),
            "random_seed": synthetic_data["config"].seed,
        },
        "realized_planted_table": realized_table,
        "method_comparison_pair_level": pair_eval_df,
        "method_comparison_pattern_level": pattern_eval_df,
        "mined_patterns": {
            "prefixspan": ps_df,
            "direct": direct_df,
            "cooccurrence": cooc_df,
        },
    }


if __name__ == "__main__":
    print("Testing synthetic benchmark with realized support calculation...")
    cfg = SyntheticBenchmarkConfig(n_taxis=40, n_days=5, seed=42)
    syn_data = generate_synthetic_dataset(cfg)
    print(f"Generated {len(syn_data['sequences'])} sequences.")
    res = run_synthetic_evaluation(syn_data, min_support=0.10, max_pattern_length=4)
    print("\nRealized Planted Patterns:")
    print(res["realized_planted_table"][["pattern", "realized_match_count", "realized_support_pct", "is_realized_frequent", "recovered_by_prefixspan"]])
    print("\nPair-Level Comparison:")
    print(res["method_comparison_pair_level"][["method", "TP", "FP", "FN", "precision", "recall", "f1"]])
    print("\nPattern-Level Comparison:")
    print(res["method_comparison_pattern_level"][["method", "TP", "FP", "FN", "precision", "recall", "f1"]])
