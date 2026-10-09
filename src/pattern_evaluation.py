"""
pattern_evaluation.py
======================
Supporting module for pattern evaluation, baseline comparisons, and robustness analysis.

This module provides two distinct, scientifically defensible evaluation protocols:

1. ORDERED-TRANSITION-PAIR EVALUATION (Pair-Level):
   Evaluates methods based on their ability to recover physical, directed origin-to-destination
   transition pairs.
   - Prediction Unit: Unique ordered cell pair (o, d).
   - Reference Unit: Unique reference transition pair (o, d).
   - TP: Distinct reference pairs recovered by at least one predicted pattern.
   - FP: Distinct predicted pairs that are not in the reference set.
   - FN: Distinct reference pairs not recovered by any predicted pattern.
   - Precision = TP / (TP + FP), Recall = TP / (TP + FN), F1 = 2*P*R / (P+R).
   All counts are in identical units (distinct ordered cell pairs).

2. PATTERN-LEVEL EVALUATION (Exact Pattern Match):
   Evaluates whether full multi-cell discovered sequences (e.g. A -> B -> C) exactly match
   independently specified multi-cell reference patterns.
   - Prediction Unit: Full tuple pattern (c1, c2, ...).
   - Reference Unit: Full tuple pattern (c1, c2, ...).
   - TP: Mined patterns that exactly equal a reference pattern.
   - FP: Mined patterns not in the reference set.
   - FN: Reference patterns not discovered.

WHY AN EMPIRICAL REFERENCE IS USED (AND ITS LIMITATIONS)
---------------------------------------------------------
Urban taxi mobility pattern discovery is unsupervised: there is no external ground truth
specifying "the real recurring routes." To evaluate real T-Drive data without circularity
(i.e. without using PrefixSpan's own output as ground truth), an empirical reference set
is constructed from raw, physically observed adjacent cell-to-cell transitions that meet
strict frequency and multi-taxi thresholds.

DISCLOSED LIMITATIONS:
1. Because the empirical reference is constructed from observed adjacent transitions, it
   structurally favors contiguous adjacent-transition mining over gapped sequential mining.
2. Unordered co-occurrence baseline is order-blind and therefore produces high recall but
   low precision.
3. This empirical reference serves as a descriptive validation benchmark, not an objective
   absolute ground truth.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# REFERENCE / EMPIRICAL BENCHMARK CONSTRUCTION
# ---------------------------------------------------------------------------

REFERENCE_COLUMNS = [
    "origin_cell",
    "destination_cell",
    "raw_occurrence_count",
    "n_taxis_involved",
    "n_taxi_days_involved",
    "include_in_reference",
]


def build_reference_od_patterns(
    sequences: dict[str, list[str]],
    min_occurrence_count: int,
    min_taxis: int = 2,
) -> pd.DataFrame:
    """
    Build reference (origin, destination) transition pairs from raw physical transitions.
    
    Counts every adjacent pair (seq[i], seq[i+1]) across all taxi-day sequences.
    A transition qualifies as a reference candidate if:
      (a) raw occurrence count >= min_occurrence_count, and
      (b) observed across >= min_taxis distinct taxis.
    """
    if not sequences:
        raise ValueError("No sequences provided — check upstream filtering.")

    if min_occurrence_count < 1:
        raise ValueError("min_occurrence_count must be at least 1.")

    trans_counts: Counter[tuple[str, str]] = Counter()
    trans_taxis: dict[tuple[str, str], set[str]] = {}
    trans_taxi_days: dict[tuple[str, str], set[str]] = {}

    for seq_key, seq in sequences.items():
        taxi_id = seq_key.split("::", 1)[0]
        for i in range(len(seq) - 1):
            if seq[i] == seq[i + 1]:
                continue
            pair = (seq[i], seq[i + 1])
            trans_counts[pair] += 1
            trans_taxis.setdefault(pair, set()).add(taxi_id)
            trans_taxi_days.setdefault(pair, set()).add(seq_key)

    rows = [
        {
            "origin_cell": o,
            "destination_cell": d,
            "raw_occurrence_count": count,
            "n_taxis_involved": len(trans_taxis[(o, d)]),
            "n_taxi_days_involved": len(trans_taxi_days[(o, d)]),
            "include_in_reference": True,
        }
        for (o, d), count in trans_counts.items()
        if count >= min_occurrence_count and len(trans_taxis[(o, d)]) >= min_taxis
    ]

    columns = REFERENCE_COLUMNS
    if not rows:
        return pd.DataFrame(columns=columns)

    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values("raw_occurrence_count", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# BASELINE: UNORDERED CO-OCCURRENCE PAIRS
# ---------------------------------------------------------------------------

def mine_cooccurrence_pairs(
    sequences: dict[str, list[str]],
    min_support: float,
) -> pd.DataFrame:
    """
    Frequent unordered 2-cell co-occurrence baseline.
    
    Counts unordered pairs of cells visited on the same taxi-day (at most once per sequence).
    Order-blind baseline used to demonstrate what movement order adds over association.
    """
    if not sequences:
        raise ValueError("No sequences provided — check upstream filtering.")

    if not 0 < min_support <= 1:
        raise ValueError("min_support must be in the interval (0, 1].")

    n_sequences = len(sequences)
    min_support_count = max(2, math.ceil(min_support * n_sequences))

    pair_counts: Counter[tuple[str, str]] = Counter()

    for seq in sequences.values():
        distinct_cells = sorted(set(seq))
        if len(distinct_cells) < 2:
            continue
        for i in range(len(distinct_cells)):
            for j in range(i + 1, len(distinct_cells)):
                pair_counts[(distinct_cells[i], distinct_cells[j])] += 1

    rows = [
        {
            "pattern": pair,
            "support_count": count,
            "support_percentage": 100 * count / n_sequences,
            "pattern_length": 2,
            "mining_method": "cooccurrence_unordered",
        }
        for pair, count in pair_counts.items()
        if count >= min_support_count
    ]

    columns = ["pattern", "support_count", "support_percentage", "pattern_length", "mining_method"]
    if not rows:
        return pd.DataFrame(columns=columns)

    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values("support_count", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# ORDERED PAIR EXTRACTION & PAIR-LEVEL EVALUATION
# ---------------------------------------------------------------------------

def extract_ordered_pairs_from_patterns(
    patterns_df: pd.DataFrame,
    is_subsequence: bool = True,
) -> set[tuple[str, str]]:
    """
    Extract unique ordered pairs (o, d) from a mined patterns DataFrame.
    
    Parameters
    ----------
    patterns_df : pd.DataFrame
        DataFrame with a 'pattern' column (tuples of cell strings).
    is_subsequence : bool
        If True (PrefixSpan), extracts all ordered pairs (p[i], p[j]) with i < j (gaps allowed).
        If False (Contiguous), extracts only strictly adjacent pairs (p[i], p[i+1]).
    
    Returns
    -------
    set[tuple[str, str]]
        Deduplicated set of predicted ordered cell transitions.
    """
    pairs: set[tuple[str, str]] = set()
    for pat in patterns_df["pattern"]:
        pat_t = tuple(pat)
        if is_subsequence:
            for i in range(len(pat_t)):
                for j in range(i + 1, len(pat_t)):
                    if pat_t[i] != pat_t[j]:
                        pairs.add((pat_t[i], pat_t[j]))
        else:
            for i in range(len(pat_t) - 1):
                if pat_t[i] != pat_t[i + 1]:
                    pairs.add((pat_t[i], pat_t[i + 1]))
    return pairs


def match_patterns_to_reference(
    patterns_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    order_sensitive: bool,
) -> pd.DataFrame:
    """
    Annotate each pattern in patterns_df with reference pairs it recovers.
    """
    reference_pairs = list(
        zip(reference_df["origin_cell"], reference_df["destination_cell"])
    )

    matched_reference = []
    matched_pairs_col = []

    for pattern in patterns_df["pattern"]:
        pattern = tuple(pattern)
        found = []

        if order_sensitive:
            for (o, d) in reference_pairs:
                if o not in pattern or d not in pattern:
                    continue
                # first occurrence of o must be strictly before last occurrence of d
                first_o = min(i for i, c in enumerate(pattern) if c == o)
                last_d = max(i for i, c in enumerate(pattern) if c == d)
                if first_o < last_d:
                    found.append((o, d))
        else:
            pattern_set = set(pattern)
            for (o, d) in reference_pairs:
                if o in pattern_set and d in pattern_set:
                    found.append((o, d))

        matched_reference.append(len(found) > 0)
        matched_pairs_col.append(found)

    result = patterns_df.copy()
    result["matched_reference"] = matched_reference
    result["matched_pairs"] = matched_pairs_col
    return result


def confusion_counts_pair_level(
    predicted_pairs: set[tuple[str, str]],
    reference_pairs: set[tuple[str, str]],
) -> dict[str, int]:
    """
    Compute TP, FP, FN strictly at the PAIR level (identical units).
    
    TP: Reference pairs recovered by predicted pairs.
    FP: Predicted pairs that are not in the reference set.
    FN: Reference pairs not present in predicted pairs.
    """
    tp = len(predicted_pairs & reference_pairs)
    fp = len(predicted_pairs - reference_pairs)
    fn = len(reference_pairs - predicted_pairs)
    return {"TP": tp, "FP": fp, "FN": fn}


def confusion_counts(
    matched_patterns_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> dict:
    """
    Compute confusion counts. In this updated version, TP and FN represent
    reference pair recovery, and FP counts unmatched predicted items.
    """
    reference_pairs = set(
        zip(reference_df["origin_cell"], reference_df["destination_cell"])
    )
    recovered_pairs: set[tuple[str, str]] = set()
    for pairs in matched_patterns_df["matched_pairs"]:
        recovered_pairs.update(pairs)

    tp = len(recovered_pairs & reference_pairs)
    fn = len(reference_pairs - recovered_pairs)
    fp = int((~matched_patterns_df["matched_reference"]).sum())

    return {"TP": tp, "FP": fp, "FN": fn}


def precision_recall_f1(counts: dict) -> dict:
    """
    Compute Precision, Recall, and F1 from a confusion dictionary.
    Returns None if denominator is 0.
    """
    tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None

    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def bounded_universe_accuracy(
    matched_patterns_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    patterns_df: pd.DataFrame,
) -> dict | None:
    """
    Supplementary pair-level accuracy over the bounded universe of cells in play.
    
    Universe: All distinct ordered pairs (a, b) with a != b for cells appearing
    in either reference pairs or predicted patterns.
    
    NOTE: TN strongly dominates because the vast majority of cell pairs are unvisited.
    Precision, Recall, and F1 remain the primary evaluation metrics.
    """
    cells_in_play: set[str] = set()
    cells_in_play.update(reference_df["origin_cell"])
    cells_in_play.update(reference_df["destination_cell"])
    for pattern in patterns_df["pattern"]:
        cells_in_play.update(pattern)

    if len(cells_in_play) < 2:
        return None

    universe_size = len(cells_in_play) * (len(cells_in_play) - 1)

    reference_pairs = set(
        zip(reference_df["origin_cell"], reference_df["destination_cell"])
    )
    
    # Extract predicted pairs
    predicted_pairs: set[tuple[str, str]] = set()
    for pattern in patterns_df["pattern"]:
        pat = tuple(pattern)
        for i in range(len(pat)):
            for j in range(i + 1, len(pat)):
                if pat[i] != pat[j]:
                    predicted_pairs.add((pat[i], pat[j]))

    tp = len(predicted_pairs & reference_pairs)
    fp = len(predicted_pairs - reference_pairs)
    fn = len(reference_pairs - predicted_pairs)
    tn = universe_size - tp - fp - fn

    if tn < 0:
        return None

    accuracy = (tp + tn) / universe_size
    return {
        "universe_size": universe_size,
        "n_cells_in_play": len(cells_in_play),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "accuracy": accuracy,
    }


# ---------------------------------------------------------------------------
# STABILITY: JACCARD OVERLAP
# ---------------------------------------------------------------------------

def jaccard_similarity(set_a: set, set_b: set) -> float | None:
    """Jaccard(A, B) = |A ∩ B| / |A ∪ B|."""
    union = set_a | set_b
    if not union:
        return None
    return len(set_a & set_b) / len(union)


def pairwise_jaccard_table(pattern_sets: dict[str, set]) -> pd.DataFrame:
    """Pairwise Jaccard similarity across parameter settings with identical vocabulary."""
    labels = list(pattern_sets.keys())
    rows = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            rows.append({
                "set_a": a,
                "set_b": b,
                "n_a": len(pattern_sets[a]),
                "n_b": len(pattern_sets[b]),
                "n_intersection": len(pattern_sets[a] & pattern_sets[b]),
                "jaccard": jaccard_similarity(pattern_sets[a], pattern_sets[b]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# METHOD COMPARISON TABLE
# ---------------------------------------------------------------------------

def summarize_method(
    method_name: str,
    patterns_df: pd.DataFrame,
    matched_patterns_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    runtime_sec: float,
    pattern_col: str = "pattern",
) -> dict:
    """Summarize a single method's evaluation against reference."""
    counts = confusion_counts(matched_patterns_df, reference_df)
    metrics = precision_recall_f1(counts)

    if len(patterns_df) == 0:
        avg_len, max_len, n_unique_cells = None, None, 0
    else:
        avg_len = float(patterns_df["pattern_length"].mean())
        max_len = int(patterns_df["pattern_length"].max())
        n_unique_cells = len({cell for p in patterns_df[pattern_col] for cell in p})

    return {
        "method": method_name,
        "patterns_found": len(patterns_df),
        "TP": counts["TP"],
        "FP": counts["FP"],
        "FN": counts["FN"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "runtime_sec": round(runtime_sec, 4),
        "avg_pattern_length": round(avg_len, 2) if avg_len is not None else None,
        "max_pattern_length": max_len,
        "n_unique_cells_involved": n_unique_cells,
    }


def summarize_method_consistent_pairs(
    method_name: str,
    patterns_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    runtime_sec: float,
    is_subsequence: bool = True,
) -> dict:
    """
    Summarize a method using STRICT pair-level units for TP, FP, FN, Precision, Recall, F1.
    """
    ref_pairs = set(zip(reference_df["origin_cell"], reference_df["destination_cell"]))
    pred_pairs = extract_ordered_pairs_from_patterns(patterns_df, is_subsequence=is_subsequence)
    counts = confusion_counts_pair_level(pred_pairs, ref_pairs)
    metrics = precision_recall_f1(counts)

    return {
        "method": method_name,
        "evaluation_unit": "Distinct Ordered Pairs (Pair-Level)",
        "patterns_found": len(patterns_df),
        "predicted_pairs": len(pred_pairs),
        "reference_pairs": len(ref_pairs),
        "TP": counts["TP"],
        "FP": counts["FP"],
        "FN": counts["FN"],
        "precision": round(metrics["precision"], 4) if metrics["precision"] is not None else None,
        "recall": round(metrics["recall"], 4) if metrics["recall"] is not None else None,
        "f1": round(metrics["f1"], 4) if metrics["f1"] is not None else None,
        "runtime_sec": round(runtime_sec, 4),
    }


if __name__ == "__main__":
    # Synthetic verification tests
    fixture_sequences = {
        "1::2008-02-02": ["A", "B", "C", "D"],
        "1::2008-02-03": ["A", "B", "C", "D"],
        "2::2008-02-02": ["A", "B", "C", "D", "B", "C"],
        "3::2008-02-02": ["A", "X", "Y"],
    }

    ref = build_reference_od_patterns(
        fixture_sequences, min_occurrence_count=3, min_taxis=2
    )
    assert set(zip(ref["origin_cell"], ref["destination_cell"])) == {
        ("A", "B"), ("B", "C"), ("C", "D"),
    }

    fake_patterns = pd.DataFrame({
        "pattern": [("A", "B", "D"), ("X", "Y"), ("A", "B", "C", "A")],
        "pattern_length": [3, 2, 4],
    })
    matched = match_patterns_to_reference(fake_patterns, ref, order_sensitive=True)
    counts = confusion_counts(matched, ref)
    assert counts["TP"] == 2
    assert counts["FN"] == 1
    assert counts["FP"] == 1

    pair_summary = summarize_method_consistent_pairs("Test PrefixSpan", fake_patterns, ref, 0.05, is_subsequence=True)
    assert pair_summary["TP"] == 2
    assert pair_summary["FN"] == 1

    print("pattern_evaluation.py enhanced tests passed successfully.")
