"""
pattern_ranking.py
===================
Phase 4 — Parts 1, 2, 3, 4, 5: enriched sequence construction, pattern
occurrence matching, the composite pattern-ranking score, and the
morning/evening and weekday/weekend comparison helpers.

WHY "ENRICHED" SEQUENCES
--------------------------
`sequential.build_taxi_sequences()` returns plain lists of cell-id strings
— enough for PrefixSpan, but not enough to answer "WHEN did this pattern
happen" or "on WHICH calendar date". This module builds a parallel
enriched sequence per TAXI-DAY (mirroring sequential.build_taxi_sequences()
exactly — same "taxi_id::date" grouping and key format): a list of records
(taxi_id, cell_id, timestamp, time_segment, day_type, date), compressed
the same way (one record per run of consecutive same-cell visits, keeping
that run's FIRST record). Any mined pattern can then be located inside a
taxi-day's enriched sequence to recover the timing of that specific
occurrence.

Grouping by taxi-day here (not just by taxi) matters: without it, a
taxi's last observation on one calendar day and its first observation on
the next day would be treated as directly adjacent, producing artificial
cross-day "transitions" and pattern occurrences — exactly the problem the
project's taxi-day methodology exists to avoid (see sequential.py's
module docstring and build_taxi_sequences()).

THE COMPOSITE SCORE (Part 2)
-------------------------------
For each frequent pattern, six components are computed and each is
normalized to [0, 1] before combining:

  1. support               — pattern's support_percentage / 100 (already
                              a 0-1 fraction of taxi-day sequences
                              containing it — same denominator
                              sequential.mine_patterns() used)
  2. n_taxis                — number of DISTINCT TAXIS (not taxi-days)
                              whose enriched sequence contains the
                              pattern at least once, divided by the total
                              number of distinct taxis mined. This is
                              deliberately NOT the same thing as support:
                              support counts taxi-DAYS, n_taxis counts
                              distinct taxis, and a taxi can contribute
                              multiple taxi-day sequences. The raw
                              taxi-day match count is also computed and
                              kept as its own column, `n_taxi_days_matched`
                              (see compute_pattern_metrics()), so the two
                              are never conflated.
  3. temporal_consistency   — how concentrated the pattern's occurrences
                              are in one named time segment. Computed as
                              the MODE SHARE: (occurrences in the most
                              common time_segment) / (total occurrences
                              located). 1.0 = every occurrence happened in
                              the same segment; 0.25 = spread evenly
                              across all four segments.
  4. spatial_density        — mean GPS-point count of the pattern's own
                              cells (from spatial.cell_grid_summary()),
                              min-max normalized against all occupied
                              cells' point counts. Higher = the pattern
                              passes through busier cells.
  5. pattern_length         — pattern_length min-max normalized against
                              the full mined pattern set's min/max length.
  6. recurrence             — number of DISTINCT CALENDAR DATES on which
                              the pattern was observed, divided by the
                              total number of distinct calendar dates
                              present in the data. Distinct from `support`
                              (which counts taxis, not days) — a pattern
                              could have high support (many taxis do it)
                              but low recurrence (only on one day), or vice
                              versa.

composite_score = sum(weight[c] * normalized_value[c] for c in components),
weights from config.PATTERN_SCORE_WEIGHTS (default: equal, 1/6 each).
"""

from __future__ import annotations

import pandas as pd

from clustering import COL_CLUSTER_ID
from config import COL_LAT, COL_LON, COL_TAXI_ID, COL_TIMESTAMP, PATTERN_SCORE_WEIGHTS
from spatial import COL_CELL_ID


# ---------------------------------------------------------------------------
# Enriched sequences + occurrence matching
# ---------------------------------------------------------------------------
def build_enriched_sequences(
    df: pd.DataFrame,
    split_by_date: bool = True,
) -> dict[str, list[dict]]:
    """Build one compressed, metadata-carrying sequence per TAXI-DAY.

    Each record in a taxi-day's sequence is a dict:
        {taxi_id, cell_id, timestamp, time_segment, day_type, date}
    Consecutive visits to the SAME cell are compressed to the FIRST such
    record in the run (mirrors spatial.compress_consecutive_duplicates(),
    applied here to full records instead of bare cell-id strings).

    Requires df to already have spatial_cell_id, time_segment, and
    day_type columns (i.e. output of spatial.assign_spatial_cells() +
    sequential.add_temporal_labels()).

    Parameters
    ----------
    split_by_date : bool
        If True (default), group by (taxi_id, calendar_date) — the
        project's primary sequence unit — exactly mirroring
        sequential.build_taxi_sequences(split_by_date=True). This MUST
        stay True for any analysis (ranking, anomaly detection) that is
        meant to be consistent with the PrefixSpan patterns mined via
        sequential.build_taxi_sequences(), since both then share the same
        taxi-day sequence set and the same support denominator.

    Returns
    -------
    dict[str, list[dict]]
        Keys are "taxi_id::date" (same format as
        sequential.build_taxi_sequences()) when split_by_date=True, else
        plain "taxi_id". Taxi-days with fewer than 2 distinct cells
        visited are dropped (mirrors sequential.build_taxi_sequences()'s
        length filter).
    """
    required = {COL_CELL_ID, "time_segment", "day_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"build_enriched_sequences requires columns {required}, missing: {missing}. "
            f"Run spatial.assign_spatial_cells() and sequential.add_temporal_labels() first."
        )

    work_df = df.copy()
    work_df["_date"] = work_df[COL_TIMESTAMP].dt.date

    group_columns = [COL_TAXI_ID]
    if split_by_date:
        group_columns.append("_date")

    sequences: dict[str, list[dict]] = {}

    for group_key, group in (
        work_df.sort_values([COL_TAXI_ID, COL_TIMESTAMP])
        .groupby(group_columns, sort=False)
    ):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        taxi_id = str(group_key[0])

        records = []
        for _, row in group.iterrows():
            record = {
                "taxi_id": taxi_id,
                "cell_id": row[COL_CELL_ID],
                "timestamp": row[COL_TIMESTAMP],
                "time_segment": row["time_segment"],
                "day_type": row["day_type"],
                "date": row[COL_TIMESTAMP].date(),
            }
            if not records or records[-1]["cell_id"] != record["cell_id"]:
                records.append(record)

        if len(records) >= 2:
            sequence_key = "::".join(map(str, group_key))
            sequences[sequence_key] = records

    return sequences


def find_pattern_occurrence(enriched_seq: list[dict], pattern: tuple[str, ...]) -> list[dict] | None:
    """Greedily locate ONE occurrence of `pattern` as a subsequence within
    a taxi's enriched sequence, returning the matched records (in order),
    or None if the pattern does not occur.

    Greedy left-to-right subsequence matching: for each pattern element in
    order, advance through the sequence until a matching cell_id is found.
    This finds *a* valid occurrence (not necessarily the only one, and not
    necessarily the "best" one) — sufficient for recovering a
    representative timing/date for the pattern in this taxi's day.
    """
    matched = []
    seq_idx = 0
    for target_cell in pattern:
        while seq_idx < len(enriched_seq) and enriched_seq[seq_idx]["cell_id"] != target_cell:
            seq_idx += 1
        if seq_idx >= len(enriched_seq):
            return None
        matched.append(enriched_seq[seq_idx])
        seq_idx += 1
    return matched


# ---------------------------------------------------------------------------
# Per-pattern metric computation
# ---------------------------------------------------------------------------
def _dominant_and_consistency(values: list) -> tuple:
    """Return (mode_value, mode_share) for a list of categorical values."""
    if not values:
        return None, 0.0
    counts = pd.Series(values).value_counts()
    mode_value = counts.index[0]
    mode_share = counts.iloc[0] / len(values)
    return mode_value, float(mode_share)


def compute_pattern_metrics(
    patterns_df: pd.DataFrame,
    enriched_sequences: dict[str, list[dict]],
    grid_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the six raw (not-yet-normalized) ranking components for
    every pattern in patterns_df, plus dominant_time_segment /
    dominant_day_type for downstream use (Part 5's hotspot table, Part 3's
    morning/evening comparison).

    Parameters
    ----------
    patterns_df : pd.DataFrame
        Output of sequential.mine_patterns() — must have `pattern`,
        `support_count`, `support_percentage`, `pattern_length`.
    enriched_sequences : dict[str, list[dict]]
        Output of build_enriched_sequences(), built from the SAME data
        patterns_df was mined from.
    grid_summary : pd.DataFrame
        Output of spatial.cell_grid_summary() — used for spatial_density.

    Returns
    -------
    pd.DataFrame
        patterns_df with added columns: n_taxi_days_matched (number of
        taxi-day sequences containing the pattern — the same denominator
        basis as support_count/support_percentage), n_taxis_matched
        (number of DISTINCT taxis, always <= n_taxi_days_matched),
        dominant_time_segment, temporal_consistency_raw, dominant_day_type,
        mean_cell_density, n_distinct_dates, total_distinct_dates.
    """
    total_taxi_days = len(enriched_sequences)
    all_taxi_ids = {seq[0]["taxi_id"] for seq in enriched_sequences.values() if seq}
    total_taxis = max(len(all_taxi_ids), 1)
    all_dates = {r["date"] for seq in enriched_sequences.values() for r in seq}
    total_distinct_dates = max(len(all_dates), 1)

    density_lookup = grid_summary.set_index(COL_CELL_ID)["n_points"].to_dict()

    rows = []
    for _, prow in patterns_df.iterrows():
        pattern = prow["pattern"]
        matched_segments, matched_day_types, matched_dates = [], [], set()
        matched_taxi_ids = set()
        n_taxi_days_matched = 0

        for seq in enriched_sequences.values():
            occurrence = find_pattern_occurrence(seq, pattern)
            if occurrence is None:
                continue
            n_taxi_days_matched += 1
            matched_taxi_ids.add(occurrence[-1]["taxi_id"])
            # Use the LAST matched record's segment/day_type as "when the
            # corridor was completed" — a single representative label per
            # occurrence, consistent across taxi-days.
            matched_segments.append(occurrence[-1]["time_segment"])
            matched_day_types.append(occurrence[-1]["day_type"])
            matched_dates.add(occurrence[-1]["date"])

        dominant_segment, consistency = _dominant_and_consistency(matched_segments)
        dominant_day_type, _ = _dominant_and_consistency(matched_day_types)

        cell_densities = [density_lookup.get(cell, 0) for cell in pattern]
        mean_density = sum(cell_densities) / len(cell_densities) if cell_densities else 0.0

        rows.append(
            {
                "n_taxi_days_matched": n_taxi_days_matched,
                "n_taxis_matched": len(matched_taxi_ids),
                "dominant_time_segment": dominant_segment,
                "temporal_consistency_raw": consistency,
                "dominant_day_type": dominant_day_type,
                "mean_cell_density": mean_density,
                "n_distinct_dates": len(matched_dates),
                "total_distinct_dates": total_distinct_dates,
                "total_taxi_days": total_taxi_days,
                "total_taxis": total_taxis,
            }
        )

    metrics_df = pd.DataFrame(rows)
    return pd.concat([patterns_df.reset_index(drop=True), metrics_df], axis=1)


def normalize_and_score(
    patterns_with_metrics: pd.DataFrame,
    weights: dict = PATTERN_SCORE_WEIGHTS,
) -> pd.DataFrame:
    """Normalize each of the six components to [0, 1] and compute the
    weighted composite_score. See module docstring for the exact
    definition of each component.
    """
    if abs(sum(weights.values()) - 1.0) > 1e-6:
        raise ValueError(f"PATTERN_SCORE_WEIGHTS must sum to 1.0, got {sum(weights.values())}")

    df = patterns_with_metrics.copy()

    df["norm_support"] = df["support_percentage"] / 100.0
    df["norm_n_taxis"] = df["n_taxis_matched"] / df["total_taxis"].replace(0, 1)
    df["norm_temporal_consistency"] = df["temporal_consistency_raw"]

    max_density = df["mean_cell_density"].max()
    df["norm_spatial_density"] = (
        df["mean_cell_density"] / max_density if max_density > 0 else 0.0
    )

    min_len, max_len = df["pattern_length"].min(), df["pattern_length"].max()
    df["norm_pattern_length"] = (
        (df["pattern_length"] - min_len) / (max_len - min_len) if max_len > min_len else 1.0
    )

    df["norm_recurrence"] = df["n_distinct_dates"] / df["total_distinct_dates"].replace(0, 1)

    df["composite_score"] = (
        weights["support"] * df["norm_support"]
        + weights["n_taxis"] * df["norm_n_taxis"]
        + weights["temporal_consistency"] * df["norm_temporal_consistency"]
        + weights["spatial_density"] * df["norm_spatial_density"]
        + weights["pattern_length"] * df["norm_pattern_length"]
        + weights["recurrence"] * df["norm_recurrence"]
    )

    return df.sort_values("composite_score", ascending=False).reset_index(drop=True)


def rank_patterns(
    patterns_df: pd.DataFrame,
    enriched_sequences: dict[str, list[dict]],
    grid_summary: pd.DataFrame,
    weights: dict = PATTERN_SCORE_WEIGHTS,
) -> pd.DataFrame:
    """Convenience wrapper: compute_pattern_metrics() + normalize_and_score()."""
    with_metrics = compute_pattern_metrics(patterns_df, enriched_sequences, grid_summary)
    return normalize_and_score(with_metrics, weights=weights)


def evaluate_weight_sensitivity(
    patterns_with_metrics: pd.DataFrame,
    weight_schemes: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Evaluate how pattern rankings shift across different weighting strategies.

    Computes pairwise Spearman rank correlation across different weighting schemes
    to verify whether rankings are robust or hyper-sensitive to specific weights.
    """
    if weight_schemes is None:
        weight_schemes = {
            "equal_weights": PATTERN_SCORE_WEIGHTS,
            "support_heavy": {
                "support": 0.40, "n_taxis": 0.20, "temporal_consistency": 0.10,
                "spatial_density": 0.10, "pattern_length": 0.10, "recurrence": 0.10,
            },
            "temporal_heavy": {
                "support": 0.10, "n_taxis": 0.10, "temporal_consistency": 0.40,
                "spatial_density": 0.10, "pattern_length": 0.10, "recurrence": 0.20,
            },
            "spatial_heavy": {
                "support": 0.10, "n_taxis": 0.10, "temporal_consistency": 0.10,
                "spatial_density": 0.40, "pattern_length": 0.20, "recurrence": 0.10,
            },
        }

    ranks_df = pd.DataFrame()
    for scheme_name, weights in weight_schemes.items():
        scored = normalize_and_score(patterns_with_metrics, weights=weights)
        # Store rank index
        ranks_df[scheme_name] = scored["composite_score"].rank(ascending=False)

    corr_matrix = ranks_df.corr(method="spearman").round(4)
    return corr_matrix



# ---------------------------------------------------------------------------
# Part 5 — Hotspot + sequence integration table
# ---------------------------------------------------------------------------
def build_hotspot_sequence_table(ranked_corridors_df: pd.DataFrame) -> pd.DataFrame:
    """Build the Part 5 table: Pattern | Start hotspot | Intermediate
    hotspot(s) | End hotspot | Support | Number of taxis | Dominant time period.

    Parameters
    ----------
    ranked_corridors_df : pd.DataFrame
        Rows that are BOTH corridors (integration.identify_corridors()'s
        is_corridor == True) AND already ranked (pattern_ranking.rank_patterns()'s
        output) — i.e. filter/merge both upstream before calling this.
        Must have: pattern, cluster_pattern, support_count,
        support_percentage, n_taxis_matched, n_taxi_days_matched,
        dominant_time_segment.
    """
    rows = []
    for _, row in ranked_corridors_df.iterrows():
        cluster_path = [c for c in row["cluster_pattern"] if c != -1]
        start = cluster_path[0] if cluster_path else None
        end = cluster_path[-1] if cluster_path else None
        intermediate = cluster_path[1:-1] if len(cluster_path) > 2 else []

        rows.append(
            {
                "pattern": " -> ".join(row["pattern"]),
                "start_hotspot": f"Cluster {start}" if start is not None else "N/A",
                "intermediate_hotspot": ", ".join(f"Cluster {c}" for c in intermediate) or "(none)",
                "end_hotspot": f"Cluster {end}" if end is not None else "N/A",
                "support_percentage": row["support_percentage"],
                "n_taxis": row["n_taxis_matched"],
                "n_taxi_days": row.get("n_taxi_days_matched", row["support_count"]),
                "dominant_time_period": row["dominant_time_segment"],
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "pattern",
                "start_hotspot",
                "intermediate_hotspot",
                "end_hotspot",
                "support_percentage",
                "n_taxis",
                "n_taxi_days",
                "dominant_time_period",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values("support_percentage", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Parts 3 & 4 — comparison helpers
# ---------------------------------------------------------------------------
def compare_pattern_sets(patterns_a: pd.DataFrame, patterns_b: pd.DataFrame, label_a: str, label_b: str) -> dict:
    """Compare two mined-pattern sets (e.g. morning vs evening, or weekday
    vs weekend): which patterns are unique to each, which are common, and
    how support changed for the common ones.

    Returns a dict with:
      - unique_to_a, unique_to_b: DataFrames of patterns only in one set
      - common: DataFrame with both sets' support side by side + support_diff
    """
    set_a = set(patterns_a["pattern"]) if len(patterns_a) else set()
    set_b = set(patterns_b["pattern"]) if len(patterns_b) else set()

    only_a = set_a - set_b
    only_b = set_b - set_a
    common = set_a & set_b

    unique_to_a = patterns_a[patterns_a["pattern"].isin(only_a)].copy()
    unique_to_b = patterns_b[patterns_b["pattern"].isin(only_b)].copy()

    common_rows = []
    a_lookup = patterns_a.set_index("pattern")["support_percentage"].to_dict()
    b_lookup = patterns_b.set_index("pattern")["support_percentage"].to_dict()
    for pattern in common:
        common_rows.append(
            {
                "pattern": " -> ".join(pattern),
                f"support_pct_{label_a}": a_lookup[pattern],
                f"support_pct_{label_b}": b_lookup[pattern],
                "support_diff": a_lookup[pattern] - b_lookup[pattern],
            }
        )
    common_df = pd.DataFrame(common_rows)
    if len(common_df):
        common_df = common_df.sort_values("support_diff", key=abs, ascending=False).reset_index(drop=True)

    return {"unique_to_a": unique_to_a, "unique_to_b": unique_to_b, "common": common_df}


if __name__ == "__main__":
    # Quick manual sanity check with a tiny synthetic fixture (clearly
    # labelled as a test fixture, not real T-Drive data).
    import numpy as np

    test_df = pd.DataFrame(
        {
            COL_TAXI_ID: ["1", "1", "1", "1", "2", "2", "2", "3", "3", "3"],
            COL_TIMESTAMP: pd.to_datetime(
                [
                    "2008-02-02 08:00:00", "2008-02-02 08:05:00",
                    "2008-02-02 08:10:00", "2008-02-02 08:15:00",
                    "2008-02-03 08:00:00", "2008-02-03 08:05:00", "2008-02-03 08:10:00",
                    "2008-02-02 20:00:00", "2008-02-02 20:05:00", "2008-02-02 20:10:00",
                ]
            ),
            COL_CELL_ID: [
                "CELL_A", "CELL_A", "CELL_B", "CELL_C",
                "CELL_A", "CELL_B", "CELL_C",
                "CELL_A", "CELL_B", "CELL_D",
            ],
        }
    )
    test_df["hour"] = test_df[COL_TIMESTAMP].dt.hour
    test_df["time_segment"] = test_df["hour"].apply(lambda h: "morning" if h < 12 else "evening")
    test_df["day_type"] = "weekday"

    grid_summary_fixture = pd.DataFrame(
        {COL_CELL_ID: ["CELL_A", "CELL_B", "CELL_C", "CELL_D"], "n_points": [10, 8, 4, 2]}
    )

    print("Building enriched sequences from a synthetic TEST FIXTURE (not real data):\n")
    enriched = build_enriched_sequences(test_df)
    for taxi, seq in enriched.items():
        print(f"  Taxi {taxi}: {[(r['cell_id'], r['time_segment']) for r in seq]}")

    from sequential import mine_patterns
    plain_sequences = {k: [r["cell_id"] for r in v] for k, v in enriched.items()}
    patterns = mine_patterns(plain_sequences, min_support=0.5)
    print(f"\nMined patterns:\n{patterns}")

    print("\nRanking patterns:")
    ranked = rank_patterns(patterns, enriched, grid_summary_fixture)
    print(ranked[[
        "pattern", "composite_score", "n_taxi_days_matched", "n_taxis_matched",
        "dominant_time_segment", "n_distinct_dates",
    ]])

    # n_taxi_days_matched must exactly equal support_count: both count how
    # many taxi-day sequences contain the pattern as a subsequence, computed
    # independently (PrefixSpan vs. greedy occurrence matching) — a
    # mismatch would indicate the two are no longer looking at the same
    # taxi-day sequence set.
    mismatches = ranked[ranked["n_taxi_days_matched"] != ranked["support_count"]]
    assert mismatches.empty, f"n_taxi_days_matched/support_count mismatch:\n{mismatches}"
    print("\nConsistency check passed: n_taxi_days_matched == support_count for every pattern.")

    print("\nComparing morning vs evening sub-mining (manual split for the test):")
    morning_seqs = {k: v for k, v in enriched.items() if v[0]["time_segment"] == "morning"}
    evening_seqs = {k: v for k, v in enriched.items() if v[0]["time_segment"] == "evening"}
    print(f"  morning taxis: {list(morning_seqs.keys())}, evening taxis: {list(evening_seqs.keys())}")
