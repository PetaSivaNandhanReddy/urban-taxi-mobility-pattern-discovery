"""
integration.py
===============
Preliminary sequential + spatial integration — Phase 2, Part 9.

THREE DISTINCT CONCEPTS (kept explicitly separate throughout this module)
-----------------------------------------------------------------------------
1. SPATIAL HOTSPOT — a single location with high point/taxi density
   (from clustering.py's DBSCAN clusters). Says nothing about movement or
   order — just "lots of activity happens here."
2. FREQUENT SEQUENCE — an ordered chain of spatial GRID CELLS that recurs
   across many taxis' trajectories (from sequential.py's PrefixSpan
   output). Says nothing about whether those cells are dense hotspots —
   a frequent sequence can pass through low-density cells.
3. RECURRING SPATIAL MOBILITY CORRIDOR — what this module constructs: a
   frequent sequence whose cells have been mapped onto DBSCAN cluster
   IDs, i.e. "Cluster 3 -> Cluster 7 -> Cluster 2" recurring across many
   taxis. A corridor is only interesting when it links two or more actual
   hotspot clusters — that is what distinguishes a corridor from an
   arbitrary frequent path through empty space.

This module does NOT claim that any corridor causes demand elsewhere, or
that taxis take these routes for any particular reason — it only reports
that the pattern recurs with a given support, which is a purely
descriptive/statistical claim.
"""

from __future__ import annotations

import pandas as pd

from clustering import COL_CLUSTER_ID
from spatial import COL_CELL_ID


def map_cells_to_clusters(cell_summary: pd.DataFrame, clustered_df: pd.DataFrame) -> pd.DataFrame:
    """Map each spatial_cell_id to its dominant DBSCAN cluster_id.

    Parameters
    ----------
    cell_summary : pd.DataFrame
        Output of spatial.cell_grid_summary() — one row per cell, with a
        centroid_lat/centroid_lon.
    clustered_df : pd.DataFrame
        The DataFrame returned by clustering.run_dbscan() (row-level, has
        cluster_id per GPS point) BEFORE grid discretization — used here
        only to look up which cluster a cell's centroid falls nearest to
        isn't done geometrically; instead we assign each raw point's
        cluster to its cell and take the most common cluster per cell,
        which is more faithful to the actual clustering than a nearest-
        centroid lookup.

    Returns
    -------
    pd.DataFrame
        cell_summary with an added `dominant_cluster_id` column. A cell
        with no clustered points, or whose points are mostly noise (-1),
        will show cluster_id -1 (not part of any hotspot).
    """
    if COL_CELL_ID not in clustered_df.columns:
        raise ValueError(
            "clustered_df must already have spatial_cell_id assigned — "
            "run spatial.assign_spatial_cells() and clustering.run_dbscan() "
            "on the SAME DataFrame before calling this."
        )

    dominant = (
        clustered_df.groupby(COL_CELL_ID)[COL_CLUSTER_ID]
        .agg(lambda s: s.value_counts().idxmax())
        .reset_index()
        .rename(columns={COL_CLUSTER_ID: "dominant_cluster_id"})
    )

    return cell_summary.merge(dominant, on=COL_CELL_ID, how="left")


def translate_pattern_to_clusters(
    pattern: tuple[str, ...],
    cell_to_cluster: dict[str, int],
) -> tuple[int, ...]:
    """Translate a (CELL_x, CELL_y, ...) pattern into (cluster_id, ...).

    Consecutive cells mapping to the SAME cluster are compressed (a
    corridor is about movement BETWEEN clusters, not within one), mirroring
    spatial.compress_consecutive_duplicates() but applied to clusters.
    """
    clusters = [cell_to_cluster.get(cell, -1) for cell in pattern]
    compressed = [clusters[0]]
    for c in clusters[1:]:
        if c != compressed[-1]:
            compressed.append(c)
    return tuple(compressed)


def identify_corridors(
    patterns_df: pd.DataFrame,
    cell_to_cluster: dict[str, int],
    min_clusters_in_corridor: int = 2,
) -> pd.DataFrame:
    """Build the sequential + spatial integration: frequent sequences whose
    cells map onto 2+ distinct, non-noise DBSCAN clusters become candidate
    "recurring spatial mobility corridors".

    Parameters
    ----------
    patterns_df : pd.DataFrame
        Output of sequential.mine_patterns() (has a `pattern` column of
        cell-id tuples, plus support_count / support_percentage).
    cell_to_cluster : dict[str, int]
        Mapping from spatial_cell_id to dominant_cluster_id, e.g. from
        map_cells_to_clusters() (converted to a dict for fast lookup).
    min_clusters_in_corridor : int
        Minimum number of DISTINCT, non-noise clusters a translated
        pattern must touch to count as a corridor (default 2 — the pattern
        must actually connect two different hotspots, not sit inside one).

    Returns
    -------
    pd.DataFrame
        patterns_df with added columns: cluster_pattern (tuple of cluster
        ids), n_distinct_clusters, is_corridor (bool). Only rows with
        is_corridor == True represent genuine hotspot-to-hotspot corridors;
        others are frequent sequences that don't (yet) qualify as such.
    """
    df = patterns_df.copy()

    df["cluster_pattern"] = df["pattern"].apply(
        lambda p: translate_pattern_to_clusters(p, cell_to_cluster)
    )
    df["n_distinct_clusters"] = df["cluster_pattern"].apply(
        lambda cp: len({c for c in cp if c != -1})
    )
    df["is_corridor"] = df["n_distinct_clusters"] >= min_clusters_in_corridor

    return df


CORRIDOR_SUMMARY_COLUMNS = [
    "cell_pattern",
    "cluster_path",
    "n_distinct_clusters",
    "support_count",
    "support_percentage",
    "n_taxi_days_matched",
    "n_taxis_matched",
    "dominant_time_segment",
]


def corridor_summary(corridors_df: pd.DataFrame, cluster_centers: pd.DataFrame) -> pd.DataFrame:
    """Produce a readable summary table of identified corridors: which
    clusters they connect (with geographic centers), and their support.

    Parameters
    ----------
    corridors_df : pd.DataFrame
        Output of identify_corridors(), filtered by the caller to
        is_corridor == True (kept as an explicit caller step so it's clear
        this table is a SUBSET, not the full mined-pattern set). May be
        EMPTY — zero identified corridors is a valid analytical result,
        not an error, and this function must return an empty DataFrame
        with the expected schema rather than raise.
    cluster_centers : pd.DataFrame
        Output of clustering.cluster_summary() — cluster_id, center_lat,
        center_lon. Not currently joined into the summary row-by-row
        (the cluster path already names each cluster id), kept as a
        parameter so a future geographic join has an explicit home.

    Returns
    -------
    pd.DataFrame
        Always has the CORRIDOR_SUMMARY_COLUMNS schema, even when empty.
        If `corridors_df` carries ranking metadata (n_taxi_days_matched,
        n_taxis_matched, dominant_time_segment — i.e. it was produced by
        calling identify_corridors() on an already-RANKED pattern table),
        those columns are carried through directly rather than merged
        back in later. Never merge tables on `support_count` alone: it is
        not a unique key, and doing so previously caused an accidental
        many-to-many merge (~110GB memory blowup) — direct column
        propagation avoids that class of bug entirely.
    """
    if len(corridors_df) == 0:
        return pd.DataFrame(columns=CORRIDOR_SUMMARY_COLUMNS)

    rows = []
    for _, row in corridors_df.iterrows():
        cluster_path = " -> ".join(f"Cluster {c}" for c in row["cluster_pattern"])
        rows.append(
            {
                "cell_pattern": " -> ".join(row["pattern"]),
                "cluster_path": cluster_path,
                "n_distinct_clusters": row["n_distinct_clusters"],
                "support_count": row["support_count"],
                "support_percentage": row["support_percentage"],
                # These three columns only exist if corridors_df came from
                # ranked patterns (pattern_ranking.rank_patterns() output).
                # Fall back gracefully rather than KeyError if it didn't.
                "n_taxi_days_matched": row.get("n_taxi_days_matched", row["support_count"]),
                "n_taxis_matched": row.get("n_taxis_matched", None),
                "dominant_time_segment": row.get("dominant_time_segment", None),
            }
        )
    return (
        pd.DataFrame(rows, columns=CORRIDOR_SUMMARY_COLUMNS)
        .sort_values("support_count", ascending=False)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    # Quick manual sanity check with a tiny synthetic fixture (clearly
    # labelled as a test fixture, not real T-Drive data).
    patterns_df = pd.DataFrame(
        {
            "pattern": [
                ("CELL_A", "CELL_B", "CELL_C"),  # spans clusters 0 -> 1 -> 1 => corridor (2 distinct)
                ("CELL_A", "CELL_A2"),            # both in cluster 0 => NOT a corridor
                ("CELL_D", "CELL_E"),             # cluster -1 (noise) -> 1 => only 1 real cluster, not a corridor
            ],
            "support_count": [10, 8, 4],
            "support_percentage": [50.0, 40.0, 20.0],
            "pattern_length": [3, 2, 2],
        }
    )
    cell_to_cluster = {
        "CELL_A": 0, "CELL_A2": 0, "CELL_B": 1, "CELL_C": 1,
        "CELL_D": -1, "CELL_E": 1,
    }
    cluster_centers = pd.DataFrame(
        {COL_CLUSTER_ID: [0, 1], "center_lat": [39.90, 39.95], "center_lon": [116.40, 116.45]}
    )

    print("Identifying corridors from a synthetic TEST FIXTURE (not real data):\n")
    result = identify_corridors(patterns_df, cell_to_cluster)
    print(result[["pattern", "cluster_pattern", "n_distinct_clusters", "is_corridor"]])

    print("\nCorridor summary (is_corridor == True only):")
    print(corridor_summary(result[result["is_corridor"]], cluster_centers))

    print("\nZero-corridor case must be safe, not raise:")
    empty_summary = corridor_summary(result[result["is_corridor"] == False].iloc[0:0], cluster_centers)
    print(empty_summary)
    assert list(empty_summary.columns) == CORRIDOR_SUMMARY_COLUMNS
    assert len(empty_summary) == 0
    print("Zero-corridor safety check passed.")
