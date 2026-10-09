"""
pattern_evaluation.py
======================
Supporting module for notebooks/05_model_evaluation.ipynb.

This module exists ONLY to support the new evaluation notebook. It does not
change, replace, or duplicate anything in sequential.py, evaluation.py, or
pattern_ranking.py — it adds the pieces those modules do not already
provide:

1. A reference/"ground truth" pattern construction procedure that is
   independent of PrefixSpan's own output (see build_reference_od_patterns).
2. A cross-method matching procedure so PrefixSpan, the direct-transition
   baseline, and the co-occurrence baseline can all be scored against the
   SAME reference set on a common footing (see match_patterns_to_reference).
3. Precision / Recall / F1 (and an optional, explicitly bounded-universe
   accuracy) computed from real TP/FP/FN counts — nothing here fabricates a
   metric value (see confusion_counts, precision_recall_f1).
4. A simple, order-agnostic co-occurrence baseline (see
   mine_cooccurrence_pairs), added specifically for the method comparison —
   it is deliberately kept to 2-cell pairs rather than full Apriori-style
   itemsets of arbitrary size, since the notebook only needs a clean,
   order-blind contrast to PrefixSpan and mine_contiguous_patterns(), not a
   general frequent-itemset miner.
5. Jaccard pattern-set-overlap stability helpers (see jaccard_similarity /
   pairwise_jaccard_table), used to check how stable the mined pattern set
   is across nearby parameter settings (same grid, different min_support).

WHY A REFERENCE, NOT "GROUND TRUTH LABELS"
-------------------------------------------
Urban taxi mobility pattern discovery is unsupervised: there is no external
label saying "this is a real recurring route." Treating PrefixSpan's own
frequent patterns as ground truth would be circular (it would just measure
whether PrefixSpan agrees with itself). Instead, this module builds a
reference set from an INDEPENDENTLY MOTIVATED, reproducible rule that does
not call PrefixSpan, and does not use PrefixSpan's own support statistic
(fraction of DISTINCT SEQUENCES containing a pattern):

    Count every literal, physically-observed adjacent cell-to-cell
    transition (A immediately followed by B, after the project's existing
    consecutive-duplicate compression), TOTALLED ACROSS ALL OCCURRENCES —
    including a transition that repeats several times within the SAME
    taxi-day (PrefixSpan's and the contiguous-baseline's support statistic
    caps this at one count per sequence; this reference statistic
    deliberately does not). A transition becomes a reference candidate if
    it (a) occurs at least MIN_REFERENCE_OCCURRENCES times in total, and
    (b) is observed for at least MIN_REFERENCE_TAXIS distinct taxis, so a
    single taxi's idiosyncratic repeated loop cannot alone create a
    reference pattern.

An earlier candidate rule — matching each taxi-day sequence's own
start/end cell pair — was tried first and is kept out of the final
notebook because it produced ZERO qualifying pairs at every support
threshold from 5% down to 0.5% (taxi-day start/end locations are simply
too varied across 630 sequences to recur at a meaningful rate on this
sample). That negative result is itself reported in the notebook rather
than silently discarded, and the raw-occurrence-count transition rule
above was adopted instead because it is populated on the real data (see
the notebook's Ground Truth Design section for the actual counts).

IMPORTANT LIMITATION OF THIS CHOICE (stated again in the notebook):
because the reference is itself built from ADJACENT transitions, it
structurally resembles what the direct adjacent-transition baseline
measures more closely than what PrefixSpan measures (PrefixSpan's real
distinguishing ability — finding gapped, non-adjacent recurring order —
is not what this reference is designed to reward). This is disclosed as
an explicit limitation, not hidden: the reference validates "was this a
real, physically repeated movement," not "was this specifically a
PrefixSpan-style gapped pattern," and the notebook is explicit that a
strong showing by the adjacent-transition baseline against this
particular reference is partly a product of how the reference was built,
not proof that adjacency-only mining is the better method in general.

The reference table this module produces is meant to be inspected, and
edited if necessary, by a human before being treated as final (see the
notebook's marked manual-review section) — it is not asserted to be
objectively "correct" on its own.
"""

from __future__ import annotations

import math
import time
from collections import Counter

import pandas as pd


# ---------------------------------------------------------------------------
# REFERENCE / "GROUND TRUTH" CONSTRUCTION
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
    Build reference (origin, destination) transition patterns from raw,
    literal adjacent cell-to-cell movement — NOT from PrefixSpan output,
    and NOT using PrefixSpan's own support statistic.

    For every taxi-day sequence (already consecutive-duplicate compressed
    by sequential.build_taxi_sequences()), every adjacent pair
    (seq[i], seq[i+1]) is counted, and — unlike PrefixSpan's and the
    contiguous baseline's support, which count AT MOST ONCE PER SEQUENCE —
    every occurrence is counted, including repeats within the same
    taxi-day. A transition qualifies as a reference candidate only if it
    (a) occurs at least min_occurrence_count times in total and (b) is
    observed across at least min_taxis distinct taxis, so one taxi's
    repeated local loop cannot alone manufacture a reference pattern.

    See this module's docstring for why this specific rule was chosen
    (an earlier taxi-day start/end rule produced zero candidates on this
    sample) and for the limitation this introduces (the reference is
    itself adjacency-based, which structurally favors the direct
    adjacent-transition baseline in the comparison — disclosed, not
    hidden).

    Returns
    -------
    pd.DataFrame with columns REFERENCE_COLUMNS, sorted by
    raw_occurrence_count descending. `include_in_reference` defaults to
    True for every row and is the column a human reviewer should edit
    (set to False) to exclude a candidate before it is used as the
    evaluation reference — see the notebook's manual-review section.
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
# ALTERNATIVE BASELINE: ORDER-AGNOSTIC CO-OCCURRENCE PAIRS
# ---------------------------------------------------------------------------

def mine_cooccurrence_pairs(
    sequences: dict[str, list[str]],
    min_support: float,
) -> pd.DataFrame:
    """
    Frequent unordered 2-cell co-occurrence baseline.

    Counts, per taxi-day sequence, every distinct UNORDERED pair of cells
    that both appear somewhere in that sequence (at most once per
    sequence, regardless of how many times the pair co-occurs within it).
    This deliberately throws away all ordering and adjacency information —
    unlike PrefixSpan (ordered, gaps allowed) and the contiguous n-gram
    baseline (ordered, adjacent only), a co-occurrence pair "A, B" means
    only that a taxi visited both A and B on the same day, in either
    order, at any distance apart in the sequence. It is included only as
    a simple, clean baseline to show what "movement order" analysis adds
    over a plain association measure — it is NOT a substitute for the
    project's own PrefixSpan/contiguous distinction, and is intentionally
    kept to pairs rather than larger itemsets to avoid unnecessary
    Apriori-style complexity for a supporting contrast.
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
# MATCHING DISCOVERED PATTERNS AGAINST THE REFERENCE SET
# ---------------------------------------------------------------------------

def match_patterns_to_reference(
    patterns_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    order_sensitive: bool,
) -> pd.DataFrame:
    """
    Label each discovered pattern with every reference (origin,
    destination) transition it recovers.

    order_sensitive=True (PrefixSpan, contiguous n-gram):
        A pattern recovers reference pair (o, d) if o appears at some
        position in the pattern and d appears at a LATER position — i.e.
        (o, d) occurs as an ordered (possibly non-adjacent) subsequence of
        the discovered pattern. This is the same "ordered, gaps allowed"
        semantics PrefixSpan itself uses, applied one reference pair at a
        time, so a single length-4 discovered pattern can recover more
        than one reference transition if it happens to contain several.

    order_sensitive=False (unordered co-occurrence baseline):
        A pattern recovers reference pair (o, d) if BOTH o and d are
        members of the pattern's cell set, in EITHER order. This is
        deliberately more lenient, because the co-occurrence baseline has
        no direction to check — this is exactly why its precision/recall
        are not directly comparable to the ordered methods on equal
        terms, and the notebook states that explicitly next to the
        results.

    Returns
    -------
    patterns_df with two extra columns: matched_reference (bool — True if
    at least one reference pair was recovered) and matched_pairs (a list
    of the (o, d) tuples recovered by this pattern; empty list if none).
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


def confusion_counts(
    matched_patterns_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> dict:
    """
    Compute TP, FP, FN from a matched-patterns table produced by
    match_patterns_to_reference().

    TP: distinct reference pairs recovered by at least one discovered
        pattern (counted once per reference pair, not once per matching
        discovered pattern, so a reference pair rediscovered by several
        overlapping patterns is not double-counted).
    FP: discovered patterns that do not match any reference pair.
    FN: reference pairs recovered by none of the discovered patterns.
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
    Compute precision, recall, and F1 from a TP/FP/FN dict. Returns None
    for any metric whose denominator is zero, rather than dividing by
    zero or silently reporting 0.0 (which would misleadingly look like a
    real measured failure rather than an undefined case).
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
    Optional supplementary accuracy, computed ONLY over an explicitly
    bounded universe of ordered pairs — never over "all possible spatial
    sequences" (which is open-ended and cannot support a TN count; see
    the notebook's discussion of why plain accuracy is not used as the
    primary metric).

    The universe is defined as: all ordered pairs (a, b), a != b, drawn
    from the set of cells that appear EITHER in the reference transitions
    OR in a discovered pattern being scored against them. This is a
    finite, reproducible, and domain-meaningful universe — it is exactly
    the pool of cells actually in play for this particular reference/
    method comparison, not an arbitrary shrink-to-fit set chosen to make
    accuracy look good. It will still classify most of the universe as
    TN, which is expected and is exactly why accuracy is reported only as
    a supplementary figure, not the headline metric — see the notebook.

    Returns None if the universe would contain fewer than 2 candidate
    cells (accuracy is not meaningful over a near-empty universe).
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
    recovered_pairs: set[tuple[str, str]] = set()
    for pairs in matched_patterns_df["matched_pairs"]:
        recovered_pairs.update(pairs)

    tp = len(recovered_pairs & reference_pairs)
    fn = len(reference_pairs - recovered_pairs)
    fp = int((~matched_patterns_df["matched_reference"]).sum())
    tn = universe_size - tp - fn - fp

    if tn < 0:
        # A discovered pattern's endpoints fell outside the bounded
        # universe (e.g. an endpoint below the frequency cutoff for THIS
        # universe definition, even though the pattern itself met the
        # separate PrefixSpan support rule). Accuracy is not well-defined
        # against this universe in that case.
        return None

    accuracy = (tp + tn) / universe_size
    return {
        "universe_size": universe_size,
        "n_cells_in_play": len(cells_in_play),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "accuracy": accuracy,
    }


# ---------------------------------------------------------------------------
# STABILITY: JACCARD PATTERN-SET OVERLAP
# ---------------------------------------------------------------------------

def jaccard_similarity(set_a: set, set_b: set) -> float | None:
    """Jaccard(A, B) = |A ∩ B| / |A ∪ B|. Returns None if both sets are
    empty (undefined, not 0.0 — two empty pattern sets are not "totally
    dissimilar")."""
    union = set_a | set_b
    if not union:
        return None
    return len(set_a & set_b) / len(union)


def pairwise_jaccard_table(pattern_sets: dict[str, set]) -> pd.DataFrame:
    """
    Given {label: set_of_patterns, ...} (e.g. one entry per min_support
    threshold tested on the SAME grid), return a long-format DataFrame of
    pairwise Jaccard similarities between every pair of labels.

    Only meaningful when every pattern set uses the SAME underlying cell-
    id space — comparing pattern sets mined at different grid cell sizes
    is not included here, because different grid sizes produce different,
    non-comparable cell-id vocabularies (see the notebook's grid-size
    section for why that comparison is reported as separate summary
    statistics instead of a literal-pattern Jaccard score).
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
    """One row of the final method-comparison table for a single method."""
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


if __name__ == "__main__":
    # Small synthetic sanity check ONLY — not real T-Drive data. Confirms
    # the reference/matching/metric plumbing behaves as documented before
    # it is trusted on the real project sample.
    fixture_sequences = {
        "1::2008-02-02": ["A", "B", "C", "D"],
        "1::2008-02-03": ["A", "B", "C", "D"],
        "2::2008-02-02": ["A", "B", "C", "D", "B", "C"],  # repeats B->C twice
        "3::2008-02-02": ["A", "X", "Y"],
    }

    # A->B, B->C, and C->D each occur 3 times across 2 taxis.
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
    # ("A", "B", "C", "A") recovers (A, B) and (B, C) even with the returning 'A' at the end
    assert counts["TP"] == 2  # (A, B) and (B, C) recovered
    assert counts["FN"] == 1  # (C, D) not recovered
    assert counts["FP"] == 1  # ("X","Y") does not match any reference pair

    metrics = precision_recall_f1(counts)
    assert metrics["precision"] == 2 / 3
    assert abs(metrics["recall"] - 2 / 3) < 1e-9

    print("pattern_evaluation.py synthetic checks passed.")
