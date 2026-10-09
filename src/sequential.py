"""
sequential.py
=============
Sequence construction and sequential pattern mining — Phase 2.

This module supports two different analyses:

1. PrefixSpan subsequence mining:
   Finds ordered recurrence. A -> B means A occurred before B somewhere
   in a sequence; it does NOT mean a direct movement.

2. Contiguous n-gram mining:
   Finds adjacent cell-to-cell movements. A -> B means the two cells were
   consecutive after repeated adjacent cells were compressed. Use this for
   direct transition and mobility-corridor analysis.
"""

from __future__ import annotations

import math
from collections import Counter

import pandas as pd
from prefixspan import PrefixSpan

from config import COL_TAXI_ID, COL_TIMESTAMP, TIME_SEGMENTS
from spatial import COL_CELL_ID, compress_consecutive_duplicates


PATTERN_COLUMNS = [
    "pattern",
    "support_count",
    "support_percentage",
    "pattern_length",
]

DIRECT_PATTERN_COLUMNS = PATTERN_COLUMNS + [
    "mining_method",
    "is_direct_transition",
]


def assign_time_segment(hour: int, segments: dict = TIME_SEGMENTS) -> str:
    """
    Return the configured time-segment label for an hour from 0 through 23.

    Handles a segment that wraps midnight, for example night: (21, 6).
    """
    for name, (start, end) in segments.items():
        if start < end and start <= hour < end:
            return name

        if start >= end and (hour >= start or hour < end):
            return name

    raise ValueError(f"Hour {hour} is not covered by configured segments.")


def add_temporal_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy with time_segment and day_type columns.

    day_type is weekday for Monday-Friday and weekend for Saturday-Sunday.
    """
    work_df = df.copy()

    work_df["hour"] = work_df[COL_TIMESTAMP].dt.hour
    work_df["time_segment"] = work_df["hour"].apply(assign_time_segment)

    work_df["day_type"] = work_df[COL_TIMESTAMP].dt.dayofweek.map(
        lambda day: "weekend" if day >= 5 else "weekday"
    )

    return work_df


def build_taxi_sequences(
    df: pd.DataFrame,
    segment_by: str | None = None,
    split_by_date: bool = True,
) -> dict[str, list[str]]:
    """
    Build ordered, consecutive-duplicate-compressed cell sequences.

    By default, one sequence is built per taxi per calendar day. This avoids
    artificial transitions between the final observation of one day and the
    first observation of another day.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain taxi_id, timestamp, and spatial_cell_id.

    segment_by : str | None
        Optional grouping column, normally "time_segment" or "day_type".

    split_by_date : bool
        If True, build taxi-day sequences. This should remain True for
        movement analysis.

    Returns
    -------
    dict[str, list[str]]
        Keys have one of these forms:

        taxi_id::date
        taxi_id::date::time_segment
        taxi_id::date::day_type

    Sequences shorter than two cells are excluded because they contain no
    movement transition.
    """
    required_columns = {
        COL_TAXI_ID,
        COL_TIMESTAMP,
        COL_CELL_ID,
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required sequence columns: {sorted(missing_columns)}"
        )

    if segment_by is not None and segment_by not in df.columns:
        raise ValueError(
            f"segment_by='{segment_by}' is not present in df."
        )

    work_df = df.copy()
    work_df["_sequence_date"] = work_df[COL_TIMESTAMP].dt.date

    group_columns = [COL_TAXI_ID]

    if split_by_date:
        group_columns.append("_sequence_date")

    if segment_by is not None:
        group_columns.append(segment_by)

    sequences: dict[str, list[str]] = {}

    for group_key, group_df in (
        work_df.sort_values([COL_TAXI_ID, COL_TIMESTAMP])
        .groupby(group_columns, sort=False)
    ):
        raw_sequence = group_df[COL_CELL_ID].dropna().astype(str).tolist()

        compressed_sequence = compress_consecutive_duplicates(raw_sequence)

        if len(compressed_sequence) < 2:
            continue

        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        sequence_key = "::".join(map(str, group_key))
        sequences[sequence_key] = compressed_sequence

    return sequences


def _empty_patterns(
    columns: list[str] = PATTERN_COLUMNS,
) -> pd.DataFrame:
    """Return an empty DataFrame with a consistent pattern schema."""
    return pd.DataFrame(columns=columns)


def mine_patterns(
    sequences: dict[str, list[str]],
    min_support: float,
    max_pattern_length: int | None = None,
) -> pd.DataFrame:
    """
    Mine non-contiguous frequent subsequences with PrefixSpan.

    Important
    ---------
    A PrefixSpan pattern A -> B means A occurred before B somewhere in an
    ordered sequence. It does not prove that A and B were adjacent GPS-cell
    movements.

    Use mine_contiguous_patterns() for direct transition analysis.
    """
    if not sequences:
        raise ValueError(
            "No sequences to mine — check upstream filtering."
        )

    if not 0 < min_support <= 1:
        raise ValueError(
            "min_support must be in the interval (0, 1]."
        )

    if max_pattern_length is not None and max_pattern_length < 2:
        raise ValueError(
            "max_pattern_length must be at least 2 or None."
        )

    sequence_list = list(sequences.values())
    n_sequences = len(sequence_list)

    # Use ceil(), not round(), when converting a support FRACTION to an
    # absolute COUNT. round() can produce a count that fails the stated
    # requirement "support >= min_support" — e.g. 0.05 * 446 = 22.3,
    # round(22.3) = 22, but 22 / 446 = 4.93% < 5%. ceil() guarantees the
    # resulting count always satisfies the >= threshold.
    min_support_count = max(
        2,
        math.ceil(min_support * n_sequences),
    )

    # A cell occurring in fewer than min_support_count sequences cannot
    # participate in a qualifying frequent PrefixSpan pattern.
    cell_sequence_counts = Counter(
        cell
        for sequence in sequence_list
        for cell in set(sequence)
    )

    frequent_cells = {
        cell
        for cell, count in cell_sequence_counts.items()
        if count >= min_support_count
    }

    filtered_sequences = [
        [cell for cell in sequence if cell in frequent_cells]
        for sequence in sequence_list
    ]

    filtered_sequences = [
        sequence
        for sequence in filtered_sequences
        if sequence
    ]

    if not filtered_sequences:
        return _empty_patterns()

    prefixspan = PrefixSpan(filtered_sequences)
    prefixspan.minlen = 2

    if max_pattern_length is not None:
        prefixspan.maxlen = max_pattern_length

    rows = [
        {
            "pattern": tuple(pattern),
            "support_count": count,
            "support_percentage": 100 * count / n_sequences,
            "pattern_length": len(pattern),
        }
        for count, pattern in prefixspan.frequent(min_support_count)
    ]

    if not rows:
        return _empty_patterns()

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["support_count", "pattern_length"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )


def mine_contiguous_patterns(
    sequences: dict[str, list[str]],
    min_support: float,
    max_pattern_length: int = 2,
    exclude_repeated_cells: bool = True,
) -> pd.DataFrame:
    """
    Mine frequent contiguous cell patterns with sequence-level support.

    Unlike PrefixSpan, this function only counts adjacent observations.

    For a two-cell pattern:

        CELL_A -> CELL_B

    CELL_A and CELL_B were consecutive after duplicate adjacent cells were
    compressed. This is appropriate for direct movement-transition analysis.

    Each taxi-day sequence contributes at most one support count per pattern,
    even if it uses the same transition several times.

    Parameters
    ----------
    sequences : dict[str, list[str]]
        Taxi-day or taxi-day-segment sequences.

    min_support : float
        Fraction of sequences that must contain the pattern.

    max_pattern_length : int
        Maximum contiguous route length. Start with 2 for direct transitions.

    exclude_repeated_cells : bool
        If True, exclude patterns such as A -> B -> A. These can be examined
        later as return movements, but should not initially be called
        directional mobility corridors.
    """
    if not sequences:
        raise ValueError(
            "No sequences to mine — check upstream filtering."
        )

    if not 0 < min_support <= 1:
        raise ValueError(
            "min_support must be in the interval (0, 1]."
        )

    if max_pattern_length < 2:
        raise ValueError(
            "max_pattern_length must be at least 2."
        )

    sequence_list = list(sequences.values())
    n_sequences = len(sequence_list)

    # Same ceil()-not-round() correction as mine_patterns() above — see
    # that function's comment for why round() can understate the count
    # needed to actually satisfy "support >= min_support".
    min_support_count = max(
        2,
        math.ceil(min_support * n_sequences),
    )

    support_counts: Counter[tuple[str, ...]] = Counter()

    for sequence in sequence_list:
        patterns_in_this_sequence: set[tuple[str, ...]] = set()

        for pattern_length in range(2, max_pattern_length + 1):
            if len(sequence) < pattern_length:
                continue

            for start in range(
                len(sequence) - pattern_length + 1
            ):
                pattern = tuple(
                    sequence[start:start + pattern_length]
                )

                if (
                    exclude_repeated_cells
                    and len(set(pattern)) != len(pattern)
                ):
                    continue

                patterns_in_this_sequence.add(pattern)

        support_counts.update(patterns_in_this_sequence)

    rows = [
        {
            "pattern": pattern,
            "support_count": support_count,
            "support_percentage": 100 * support_count / n_sequences,
            "pattern_length": len(pattern),
            "mining_method": "contiguous_ngram",
            "is_direct_transition": len(pattern) == 2,
        }
        for pattern, support_count in support_counts.items()
        if support_count >= min_support_count
    ]

    if not rows:
        return _empty_patterns(DIRECT_PATTERN_COLUMNS)

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["support_count", "pattern_length"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )


def analyze_patterns(
    patterns_df: pd.DataFrame,
    top_n: int = 20,
) -> dict:
    """
    Return summary tables needed for pattern reporting and plots.

    Returns
    -------
    dict
        top_patterns:
            Highest-support patterns.

        length_distribution:
            Number of patterns at each sequence length.

        support_stats:
            Descriptive statistics for support_percentage.
    """
    if patterns_df.empty:
        return {
            "top_patterns": patterns_df.copy(),
            "length_distribution": pd.DataFrame(
                columns=["pattern_length", "n_patterns"]
            ),
            "support_stats": pd.Series(dtype=float),
        }

    top_patterns = patterns_df.head(top_n).copy()

    top_patterns["pattern_str"] = top_patterns["pattern"].apply(
        " -> ".join
    )

    length_distribution = (
        patterns_df.groupby("pattern_length")
        .size()
        .reset_index(name="n_patterns")
        .sort_values("pattern_length")
        .reset_index(drop=True)
    )

    return {
        "top_patterns": top_patterns,
        "length_distribution": length_distribution,
        "support_stats": patterns_df["support_percentage"].describe(),
    }


if __name__ == "__main__":
    # Synthetic fixture only. This does not use project taxi data.
    test_fixture = pd.DataFrame(
        {
            COL_TAXI_ID: ["1"] * 5 + ["2"] * 3 + ["3"] * 3,
            COL_TIMESTAMP: pd.to_datetime(
                [
                    "2008-02-02 08:00:00",
                    "2008-02-02 08:05:00",
                    "2008-02-02 08:10:00",
                    "2008-02-02 08:15:00",
                    "2008-02-03 08:00:00",
                    "2008-02-02 08:00:00",
                    "2008-02-02 08:05:00",
                    "2008-02-02 08:10:00",
                    "2008-02-02 08:00:00",
                    "2008-02-02 08:05:00",
                    "2008-02-02 08:10:00",
                ]
            ),
            COL_CELL_ID: [
                "CELL_A",
                "CELL_A",
                "CELL_B",
                "CELL_C",
                "CELL_D",
                "CELL_A",
                "CELL_B",
                "CELL_C",
                "CELL_A",
                "CELL_B",
                "CELL_D",
            ],
        }
    )

    sequences = build_taxi_sequences(test_fixture)

    assert "1::2008-02-02" in sequences
    assert "1::2008-02-03" not in sequences
    assert sequences["1::2008-02-02"] == [
        "CELL_A",
        "CELL_B",
        "CELL_C",
    ]

    direct_patterns = mine_contiguous_patterns(
        sequences,
        min_support=0.5,
    )

    assert ("CELL_A", "CELL_B") in set(
        direct_patterns["pattern"]
    )

    assert ("CELL_A", "CELL_A") not in set(
        direct_patterns["pattern"]
    )

    print("Sequential-mining synthetic checks passed.")