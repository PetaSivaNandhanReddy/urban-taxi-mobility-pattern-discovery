"""
pattern_evaluation.py
======================
Comprehensive evaluation module for urban mobility pattern discovery.

METHODOLOGICAL FOUNDATIONS & CRITICAL EVALUATION RULES:
-------------------------------------------------------
1. SINGLE COMMON EVALUATION UNIT (Ordered Cell Pairs):
   To avoid mixing mismatched units, all models are evaluated on distinct, directed
   ordered cell-to-cell transition pairs:
     Unit: (origin_cell, destination_cell)
     - (A, B) and (B, A) represent distinct directional movements.
     - For PrefixSpan: Discovered sequences (c_1, c_2, ..., c_k) imply all ordered pairs
       (c_i, c_j) with i < j (gaps permitted). Each distinct pair is counted at most once
       (no double-counting across overlapping patterns).
     - For Direct Transitions (Contiguous N-Grams): Mined contiguous sequences imply
       strictly adjacent ordered pairs (c_i, c_{i+1}).
     - For Unordered Co-occurrence Baseline: Does not model direction. In a directed
       evaluation, each unordered pair {u, v} is represented symmetrically as {(u, v), (v, u)},
       which naturally incurs false positives due to direction-blindness. An order-agnostic
       evaluation helper is also provided for separate fair comparison.

2. EMPIRICAL REFERENCE BENCHMARK (Not Absolute Ground Truth):
   Urban taxi mobility pattern discovery on unannotated GPS data is unsupervised.
   The real-data reference set is constructed from observed adjacent cell-to-cell
   transitions meeting strict occurrence and multi-taxi thresholds.
   - It is an observed historical reference set, NOT independent external ground truth.
   - Because it is constructed from observed adjacent transitions, it structurally
     favors direct-transition mining over gapped sequential mining.
   - Agreement with this reference measures descriptive reference agreement, not
     absolute model accuracy.

3. CONSISTENT METRICS & ZERO-DENOMINATOR POLICY:
   - TP: Distinct predicted ordered pairs that match reference ordered pairs.
   - FP: Distinct predicted ordered pairs not in the reference set.
   - FN: Distinct reference ordered pairs not recovered by the method.
   - Precision = TP / (TP + FP)
   - Recall = TP / (TP + FN)
   - F1 = 2 * Precision * Recall / (Precision + Recall)
   - Zero-denominator policy: When denominator is 0 (e.g., no predictions made, or empty
     reference set), precision/recall/F1 return a documented default (0.0 or None).
"""

from __future__ import annotations

import math
import time
import warnings
from collections import Counter
from typing import Any

import pandas as pd

# Standard columns for empirical reference table
REFERENCE_COLUMNS = [
    "origin_cell",
    "destination_cell",
    "raw_occurrence_count",
    "n_taxis_involved",
    "n_taxi_days_involved",
    "include_in_reference",
]


# ---------------------------------------------------------------------------
# REFERENCE BENCHMARK CONSTRUCTION
# ---------------------------------------------------------------------------

def build_reference_od_patterns(
    sequences: dict[str, list[str]],
    min_occurrence_count: int,
    min_taxis: int = 2,
) -> pd.DataFrame:
    """
    Construct empirical reference transition pairs from physically observed movements.

    A transition (origin, destination) qualifies for the reference candidate set if:
      (a) raw occurrence count >= min_occurrence_count across all taxi-day sequences, and
      (b) observed across >= min_taxis distinct taxis.

    Limitations:
    - This is an empirical observed-transition reference, not external ground truth.
    - It structurally favors contiguous direct transitions over gapped sequential patterns.
    - All thresholds are recorded and configurable.
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

    if not rows:
        return pd.DataFrame(columns=REFERENCE_COLUMNS)

    return (
        pd.DataFrame(rows, columns=REFERENCE_COLUMNS)
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

    Counts distinct cell pairs visited on the same taxi-day (at most once per sequence).
    Order-blind baseline used to demonstrate what movement direction adds over association.
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
# ORDERED PAIR EXTRACTION & UNIFIED PAIR-LEVEL EVALUATION
# ---------------------------------------------------------------------------

def extract_ordered_pairs_from_patterns(
    patterns_df: pd.DataFrame,
    is_subsequence: bool = True,
    allow_self_loops: bool = False,
) -> set[tuple[str, str]]:
    """
    Extract a deduplicated set of predicted ordered cell pairs (origin, destination).

    Parameters
    ----------
    patterns_df : pd.DataFrame
        DataFrame with a 'pattern' column containing tuples/lists of cell IDs.
    is_subsequence : bool
        If True (PrefixSpan): extracts all ordered pairs (p[i], p[j]) with i < j (gapped).
        If False (Direct transitions): extracts only strictly adjacent pairs (p[i], p[i+1]).
    allow_self_loops : bool
        If False, excludes pairs where origin == destination.

    Returns
    -------
    set[tuple[str, str]]
        Deduplicated set of predicted ordered cell transitions.
    """
    pairs: set[tuple[str, str]] = set()
    if patterns_df is None or len(patterns_df) == 0 or "pattern" not in patterns_df.columns:
        return pairs

    for pat in patterns_df["pattern"]:
        pat_t = tuple(pat)
        if is_subsequence:
            for i in range(len(pat_t)):
                for j in range(i + 1, len(pat_t)):
                    if allow_self_loops or pat_t[i] != pat_t[j]:
                        pairs.add((pat_t[i], pat_t[j]))
        else:
            for i in range(len(pat_t) - 1):
                if allow_self_loops or pat_t[i] != pat_t[i + 1]:
                    pairs.add((pat_t[i], pat_t[i + 1]))
    return pairs


def extract_ordered_pairs_from_cooccurrence(
    cooc_df: pd.DataFrame,
) -> set[tuple[str, str]]:
    """
    Convert unordered co-occurrence pairs into bidirectional ordered pairs.

    Because co-occurrence does not establish travel direction, each unordered
    pair {u, v} generates both directed pairs (u, v) and (v, u).
    """
    pairs: set[tuple[str, str]] = set()
    if cooc_df is None or len(cooc_df) == 0 or "pattern" not in cooc_df.columns:
        return pairs

    for pat in cooc_df["pattern"]:
        u, v = pat[0], pat[1]
        if u != v:
            pairs.add((u, v))
            pairs.add((v, u))
    return pairs


def confusion_counts_pair_level(
    predicted_pairs: set[tuple[str, str]],
    reference_pairs: set[tuple[str, str]],
) -> dict[str, int]:
    """
    Calculate TP, FP, FN strictly at the ordered pair level (identical units).

    TP: Predicted ordered pairs that match reference ordered pairs.
    FP: Predicted ordered pairs not found in the reference set.
    FN: Reference ordered pairs missed by predicted pairs.
    """
    tp = len(predicted_pairs & reference_pairs)
    fp = len(predicted_pairs - reference_pairs)
    fn = len(reference_pairs - predicted_pairs)
    return {"TP": tp, "FP": fp, "FN": fn}


def precision_recall_f1(
    counts: dict[str, int],
    zero_division: float | None = None,
) -> dict[str, float | None]:
    """
    Compute Precision, Recall, and F1 from confusion counts.

    Zero-Denominator Policy:
    ------------------------
    - Precision: TP / (TP + FP). If TP + FP == 0, returns zero_division.
    - Recall:    TP / (TP + FN). If TP + FN == 0, returns zero_division.
    - F1:        2 * P * R / (P + R). If P is None, R is None, or P + R == 0, returns zero_division.
    """
    tp = counts["TP"]
    fp = counts["FP"]
    fn = counts["FN"]

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else zero_division
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else zero_division

    if precision is None or recall is None:
        f1 = zero_division
    elif (precision + recall) == 0:
        f1 = zero_division
    else:
        f1 = (2 * precision * recall) / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def summarize_method_consistent_pairs(
    method_name: str,
    patterns_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    runtime_sec: float,
    is_subsequence: bool = True,
    zero_division: float | None = None,
    pattern_col: str = "pattern",
) -> dict[str, Any]:
    """
    Authoritative evaluation summary using strict pair-level units for all metrics.
    """
    ref_pairs = set(zip(reference_df["origin_cell"], reference_df["destination_cell"]))

    if "cooccurrence" in method_name.lower():
        pred_pairs = extract_ordered_pairs_from_cooccurrence(patterns_df)
    else:
        pred_pairs = extract_ordered_pairs_from_patterns(
            patterns_df, is_subsequence=is_subsequence
        )

    counts = confusion_counts_pair_level(pred_pairs, ref_pairs)
    metrics = precision_recall_f1(counts, zero_division=zero_division)

    if patterns_df is None or len(patterns_df) == 0:
        avg_len, max_len, n_unique_cells = None, None, 0
    else:
        avg_len = (
            float(patterns_df["pattern_length"].mean())
            if "pattern_length" in patterns_df.columns
            else None
        )
        max_len = (
            int(patterns_df["pattern_length"].max())
            if "pattern_length" in patterns_df.columns
            else None
        )
        n_unique_cells = len(
            {cell for p in patterns_df[pattern_col] for cell in p}
        )

    return {
        "method": method_name,
        "evaluation_unit": "Distinct Ordered Pairs (Pair-Level)",
        "patterns_found": len(patterns_df) if patterns_df is not None else 0,
        "predicted_pairs": len(pred_pairs),
        "reference_pairs": len(ref_pairs),
        "TP": counts["TP"],
        "FP": counts["FP"],
        "FN": counts["FN"],
        "precision": round(metrics["precision"], 4) if metrics["precision"] is not None else None,
        "recall": round(metrics["recall"], 4) if metrics["recall"] is not None else None,
        "f1": round(metrics["f1"], 4) if metrics["f1"] is not None else None,
        "runtime_sec": round(runtime_sec, 4),
        "avg_pattern_length": round(avg_len, 2) if avg_len is not None else None,
        "max_pattern_length": max_len,
        "n_unique_cells_involved": n_unique_cells,
    }


def summarize_method(
    method_name: str,
    patterns_df: pd.DataFrame,
    matched_patterns_df: pd.DataFrame | None,
    reference_df: pd.DataFrame,
    runtime_sec: float,
    pattern_col: str = "pattern",
    zero_division: float | None = None,
) -> dict[str, Any]:
    """
    Standardized method summary.

    NOTE: Automatically delegates to summarize_method_consistent_pairs to enforce
    consistent pair-level units and prevent legacy unit-mixing bugs.
    """
    is_subsequence = not (
        "direct" in method_name.lower() or "contiguous" in method_name.lower()
    )
    return summarize_method_consistent_pairs(
        method_name=method_name,
        patterns_df=patterns_df,
        reference_df=reference_df,
        runtime_sec=runtime_sec,
        is_subsequence=is_subsequence,
        zero_division=zero_division,
        pattern_col=pattern_col,
    )


# ---------------------------------------------------------------------------
# PATTERN MATCHING & LEGACY DEPRECATION HELPERS
# ---------------------------------------------------------------------------

def match_patterns_to_reference(
    patterns_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    order_sensitive: bool = True,
) -> pd.DataFrame:
    """
    Annotate each pattern in patterns_df with reference pairs it recovers.
    """
    if patterns_df is None or len(patterns_df) == 0:
        res = pd.DataFrame(columns=["pattern", "pattern_length", "matched_reference", "matched_pairs"])
        return res

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


def confusion_counts(
    matched_patterns_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    legacy_mixed_units: bool = False,
) -> dict[str, int]:
    """
    Compute confusion counts.

    WARNING: The legacy behavior counted FP as unmatched whole patterns while counting
    TP and FN as cell pairs. By default (legacy_mixed_units=False), this helper now computes
    strictly consistent pair-level counts.
    """
    reference_pairs = set(
        zip(reference_df["origin_cell"], reference_df["destination_cell"])
    )
    recovered_pairs: set[tuple[str, str]] = set()
    for pairs in matched_patterns_df["matched_pairs"]:
        recovered_pairs.update(pairs)

    tp = len(recovered_pairs & reference_pairs)
    fn = len(reference_pairs - recovered_pairs)

    if legacy_mixed_units:
        warnings.warn(
            "legacy_mixed_units=True mixes pair-level TP/FN with pattern-level FP. "
            "This produces mathematically invalid precision and F1 scores.",
            DeprecationWarning,
            stacklevel=2,
        )
        fp = int((~matched_patterns_df["matched_reference"]).sum())
    else:
        # Strict pair-level FP
        predicted_pairs = extract_ordered_pairs_from_patterns(matched_patterns_df, is_subsequence=True)
        fp = len(predicted_pairs - reference_pairs)

    return {"TP": tp, "FP": fp, "FN": fn}


# ---------------------------------------------------------------------------
# ORDER-AGNOSTIC BASELINE COMPARISON
# ---------------------------------------------------------------------------

def evaluate_order_agnostic_cooccurrence(
    cooc_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    runtime_sec: float = 0.0,
    zero_division: float | None = None,
) -> dict[str, Any]:
    """
    Evaluate unordered co-occurrence against an order-agnostic reference set.

    Collapses both predictions and reference pairs into undirected sets {origin, destination},
    providing an honest evaluation of association without claiming directional discovery.
    """
    ref_undirected: set[frozenset[str]] = {
        frozenset([o, d])
        for o, d in zip(reference_df["origin_cell"], reference_df["destination_cell"])
        if o != d
    }

    pred_undirected: set[frozenset[str]] = set()
    if cooc_df is not None and len(cooc_df) > 0 and "pattern" in cooc_df.columns:
        for pat in cooc_df["pattern"]:
            if len(pat) >= 2 and pat[0] != pat[1]:
                pred_undirected.add(frozenset([pat[0], pat[1]]))

    tp = len(pred_undirected & ref_undirected)
    fp = len(pred_undirected - ref_undirected)
    fn = len(ref_undirected - pred_undirected)

    counts = {"TP": tp, "FP": fp, "FN": fn}
    metrics = precision_recall_f1(counts, zero_division=zero_division)

    return {
        "method": "Unordered Co-occurrence (Order-Agnostic)",
        "evaluation_unit": "Undirected Cell Pairs {u, v}",
        "patterns_found": len(cooc_df) if cooc_df is not None else 0,
        "predicted_pairs": len(pred_undirected),
        "reference_pairs": len(ref_undirected),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": round(metrics["precision"], 4) if metrics["precision"] is not None else None,
        "recall": round(metrics["recall"], 4) if metrics["recall"] is not None else None,
        "f1": round(metrics["f1"], 4) if metrics["f1"] is not None else None,
        "runtime_sec": round(runtime_sec, 4),
    }


# ---------------------------------------------------------------------------
# BOUNDED-UNIVERSE ACCURACY (AUDITED & DOCUMENTED)
# ---------------------------------------------------------------------------

def bounded_universe_accuracy(
    matched_patterns_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    patterns_df: pd.DataFrame,
) -> dict[str, Any] | None:
    """
    Supplementary pair-level accuracy over the explicitly bounded universe of active cells.

    Universe Definition:
    --------------------
    All distinct ordered cell pairs (c_i, c_j) with c_i != c_j among cells appearing in
    either reference transitions or predicted patterns:
        Universe Size = N_active * (N_active - 1)

    Methodological Notice:
    ----------------------
    In spatial mobility networks, unvisited pairs drastically dominate the universe (TN >> TP, FP, FN).
    Consequently, raw accuracy is near 100% simply because neither the model nor the reference
    proposes the vast majority of cell pairs. Precision, recall, and F1 over the active
    transition set are the only scientifically defensible primary evaluation metrics.
    """
    cells_in_play: set[str] = set()
    cells_in_play.update(reference_df["origin_cell"])
    cells_in_play.update(reference_df["destination_cell"])
    if patterns_df is not None and len(patterns_df) > 0 and "pattern" in patterns_df.columns:
        for pattern in patterns_df["pattern"]:
            cells_in_play.update(pattern)

    if len(cells_in_play) < 2:
        return None

    universe_size = len(cells_in_play) * (len(cells_in_play) - 1)
    reference_pairs = set(
        zip(reference_df["origin_cell"], reference_df["destination_cell"])
    )
    predicted_pairs = extract_ordered_pairs_from_patterns(patterns_df, is_subsequence=True)

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
        "tn_dominance_ratio": round(tn / universe_size, 4),
        "note": "High accuracy driven by true negative dominance; do not headline as primary model accuracy.",
    }


# ---------------------------------------------------------------------------
# STABILITY: JACCARD SIMILARITY
# ---------------------------------------------------------------------------

def jaccard_similarity(set_a: set, set_b: set) -> float | None:
    """Jaccard(A, B) = |A ∩ B| / |A ∪ B|."""
    union = set_a | set_b
    if not union:
        return None
    return len(set_a & set_b) / len(union)


def pairwise_jaccard_table(pattern_sets: dict[str, set]) -> pd.DataFrame:
    """
    Calculate pairwise Jaccard similarity across parameter settings with identical vocabulary.

    IMPORTANT: Direct Jaccard comparison across DIFFERENT grid resolutions (e.g. 300m vs 500m)
    is meaningless because cell identifiers do not represent the same geographic entities.
    Use this table only across runs sharing the identical grid resolution.
    """
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
                "n_union": len(pattern_sets[a] | pattern_sets[b]),
                "jaccard": jaccard_similarity(pattern_sets[a], pattern_sets[b]),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Running self-check for pattern_evaluation.py...")
    fixture_sequences = {
        "1::2008-02-02": ["A", "B", "C", "D"],
        "1::2008-02-03": ["A", "B", "C", "D"],
        "2::2008-02-02": ["A", "B", "C", "D", "B", "C"],
        "3::2008-02-02": ["A", "X", "Y"],
    }
    ref = build_reference_od_patterns(fixture_sequences, min_occurrence_count=3, min_taxis=2)
    assert set(zip(ref["origin_cell"], ref["destination_cell"])) == {
        ("A", "B"), ("B", "C"), ("C", "D"),
    }

    test_patterns = pd.DataFrame({
        "pattern": [("A", "B", "D"), ("X", "Y")],
        "pattern_length": [3, 2],
    })
    summary = summarize_method_consistent_pairs("PrefixSpan", test_patterns, ref, 0.05, is_subsequence=True)
    assert summary["TP"] == 1  # ("A", "B")
    assert summary["FN"] == 2  # ("B", "C"), ("C", "D")
    assert summary["FP"] == 3  # ("A", "D"), ("B", "D"), ("X", "Y")
    assert summary["predicted_pairs"] == 4
    print("Self-check completed successfully!")
