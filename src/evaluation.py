"""
evaluation.py
=============
Phase 4, Part 9 — Research evaluation: how do results change across
different grid sizes, DBSCAN parameters, and PrefixSpan support
thresholds?

This module runs the actual pipeline (spatial discretization -> DBSCAN ->
sequence building -> PrefixSpan) across each candidate parameter value
and records outcome metrics, producing an experiment table a reader can
use to see the sensitivity of the results to each design choice — rather
than asserting one setting is "correct".
"""

from __future__ import annotations

import time

import pandas as pd

import clustering
import sequential
import spatial
from config import (
    COL_LAT,
    COL_LON,
    DBSCAN_EPS_METERS,
    DBSCAN_MIN_SAMPLES,
    DEFAULT_MIN_SUPPORT,
    GRID_CELL_SIZE_METERS,
    GRID_SIZES_TO_COMPARE_METERS,
    MAX_PATTERN_LENGTH,
)


def evaluate_grid_sizes(
    traj_df: pd.DataFrame,
    grid_sizes_m: list = GRID_SIZES_TO_COMPARE_METERS,
) -> pd.DataFrame:
    """For each candidate grid cell size: assign cells, build sequences at
    the default min_support, and record the resulting cell count, average
    cell density, and mined pattern count.
    """
    rows = []
    for cell_size in grid_sizes_m:
      t0 = time.time()

      # Notebook 05 may receive traj_df that already contains
      # spatial_cell_id from Notebook 02. Remove the old assignment
      # before recomputing the grid for each candidate cell size.
      grid_input = traj_df.drop(
        columns=[spatial.COL_CELL_ID],
        errors="ignore"
      )

      cell_df = spatial.assign_spatial_cells(
        grid_input,
        cell_size_meters=cell_size
      )

      grid_summary = spatial.cell_grid_summary(cell_df)

      sequences = sequential.build_taxi_sequences(cell_df)
      try:
          patterns = sequential.mine_patterns(
              sequences,
              min_support=DEFAULT_MIN_SUPPORT,
              max_pattern_length=MAX_PATTERN_LENGTH,
          )
          n_patterns = len(patterns)
          avg_pattern_len = float(patterns["pattern_length"].mean()) if len(patterns) else None
      except ValueError:
          n_patterns, avg_pattern_len = 0, None

      rows.append({
          "grid_cell_size_m": cell_size,
          "n_occupied_cells": len(grid_summary),
          "avg_points_per_cell": round(grid_summary["n_points"].mean(), 2) if len(grid_summary) else None,
          "n_sequences_built": len(sequences),
          "n_patterns_at_default_support": n_patterns,
          "avg_pattern_length": round(avg_pattern_len, 2) if avg_pattern_len else None,
          "runtime_sec": round(time.time() - t0, 2),
      })

    return pd.DataFrame(rows)


def evaluate_dbscan_parameters(
    cell_df: pd.DataFrame,
    eps_candidates_m: list,
    min_samples_candidates: list,
) -> pd.DataFrame:
    """Wraps clustering.compare_dbscan_parameters() — kept here as the
    single entry point for the Part 9 evaluation notebook section, so all
    three experiment tables are produced through this module.
    """
    return clustering.compare_dbscan_parameters(
        cell_df, eps_candidates_m=eps_candidates_m, min_samples_candidates=min_samples_candidates
    )


def evaluate_support_thresholds(
    sequences: dict,
    thresholds: list,
    max_pattern_length: int = MAX_PATTERN_LENGTH,
) -> pd.DataFrame:
    """For each candidate min_support: mine patterns and record pattern
    count and length statistics. (Duplicates the logic already used ad hoc
    in notebook 03, exposed here as a reusable, testable function so Part 9
    can call it directly.)

    max_pattern_length is bounded (default config.MAX_PATTERN_LENGTH) on
    every call — lower support thresholds mine more candidate patterns,
    and pairing a low threshold with an unbounded PrefixSpan run is
    exactly the combination that caused this project's earlier ~4.5 hour
    runaway run (see README performance history). Support sensitivity
    should test the support axis only, with pattern length held fixed at
    the project's bounded default.
    """
    rows = []
    for thresh in thresholds:
        t0 = time.time()
        try:
            patterns = sequential.mine_patterns(
                sequences, min_support=thresh, max_pattern_length=max_pattern_length
            )
            rows.append({
                "min_support": thresh,
                "n_patterns": len(patterns),
                "avg_pattern_length": round(patterns["pattern_length"].mean(), 2) if len(patterns) else None,
                "max_pattern_length": int(patterns["pattern_length"].max()) if len(patterns) else None,
                "runtime_sec": round(time.time() - t0, 2),
            })
        except ValueError:
            rows.append({
                "min_support": thresh, "n_patterns": 0, "avg_pattern_length": None,
                "max_pattern_length": None, "runtime_sec": round(time.time() - t0, 2),
            })
    return pd.DataFrame(rows)


def build_experiment_summary(
    grid_eval: pd.DataFrame,
    dbscan_eval: pd.DataFrame,
    support_eval: pd.DataFrame,
) -> pd.DataFrame:
    """Combine the three experiment tables' headline takeaways into one
    compact summary row per experiment type, for the README/report.
    """
    rows = []

    if len(grid_eval):
        smallest = grid_eval.loc[grid_eval["grid_cell_size_m"].idxmin()]
        largest = grid_eval.loc[grid_eval["grid_cell_size_m"].idxmax()]
        direction = (
            "more" if smallest["n_patterns_at_default_support"] > largest["n_patterns_at_default_support"]
            else "fewer"
        )
        rows.append({
            "experiment": "Grid cell size",
            "candidates_tested": ", ".join(f"{v}m" for v in grid_eval["grid_cell_size_m"]),
            "metric_range": (
                f"{grid_eval['n_patterns_at_default_support'].min()}-"
                f"{grid_eval['n_patterns_at_default_support'].max()} patterns found"
            ),
            "observation": (
                f"In this run, the smallest tested cell size "
                f"({smallest['grid_cell_size_m']}m, {int(smallest['n_occupied_cells'])} occupied cells) "
                f"produced {direction} mined patterns than the largest "
                f"({largest['grid_cell_size_m']}m, {int(largest['n_occupied_cells'])} occupied cells) at "
                f"the same min_support — finer grids create more distinct cell IDs, which can either "
                f"fragment movement into rarer sequences OR (if taxis wobble within a small area) "
                f"generate many short, trivially-repeated patterns; this direction is data-dependent "
                f"and should be checked on your own sample rather than assumed."
            ),
        })

    if len(dbscan_eval):
        max_clusters_row = dbscan_eval.loc[dbscan_eval["n_clusters"].idxmax()]
        min_clusters_row = dbscan_eval.loc[dbscan_eval["n_clusters"].idxmin()]
        rows.append({
            "experiment": "DBSCAN (eps, min_samples)",
            "candidates_tested": f"{len(dbscan_eval)} combinations",
            "metric_range": (
                f"{dbscan_eval['n_clusters'].min()}-{dbscan_eval['n_clusters'].max()} clusters, "
                f"{dbscan_eval['pct_noise'].min():.1f}-{dbscan_eval['pct_noise'].max():.1f}% noise"
            ),
            "observation": (
                f"Most clusters ({int(max_clusters_row['n_clusters'])}) came from "
                f"eps={max_clusters_row['eps_meters']}m, min_samples={max_clusters_row['min_samples']} "
                f"({max_clusters_row['pct_noise']:.1f}% noise); fewest "
                f"({int(min_clusters_row['n_clusters'])}) came from "
                f"eps={min_clusters_row['eps_meters']}m, min_samples={min_clusters_row['min_samples']} "
                f"({min_clusters_row['pct_noise']:.1f}% noise). See the full table for the complete "
                f"eps/min_samples grid before picking a value for downstream analysis."
            ),
        })

    if len(support_eval):
        lowest = support_eval.loc[support_eval["min_support"].idxmin()]
        highest = support_eval.loc[support_eval["min_support"].idxmax()]
        rows.append({
            "experiment": "PrefixSpan min_support",
            "candidates_tested": ", ".join(str(v) for v in support_eval["min_support"]),
            "metric_range": f"{support_eval['n_patterns'].min()}-{support_eval['n_patterns'].max()} patterns found",
            "observation": (
                f"min_support={lowest['min_support']} found {int(lowest['n_patterns']):,} patterns "
                f"(avg length {lowest['avg_pattern_length']}); min_support={highest['min_support']} found "
                f"{int(highest['n_patterns']):,} patterns (avg length {highest['avg_pattern_length']}). "
                f"Lower thresholds surface more candidate patterns at higher computational cost and lower "
                f"per-pattern statistical robustness; higher thresholds trade coverage for robustness."
            ),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Quick manual sanity check with a tiny synthetic fixture (clearly
    # labelled as a test fixture, not real T-Drive data).
    from config import COL_TAXI_ID, COL_TIMESTAMP
    import numpy as np

    rng = np.random.default_rng(42)
    n_taxis = 10
    rows = []
    for taxi_id in range(1, n_taxis + 1):
        base_lat, base_lon = 39.900, 116.400
        t = pd.Timestamp("2008-02-02 08:00:00")
        for step in range(20):
            lat = base_lat + 0.001 * step + rng.normal(0, 0.0003)
            lon = base_lon + 0.001 * step + rng.normal(0, 0.0003)
            t += pd.Timedelta(minutes=2)
            rows.append({COL_TAXI_ID: str(taxi_id), COL_TIMESTAMP: t, COL_LAT: lat, COL_LON: lon})
    fixture_df = pd.DataFrame(rows)

    print("Evaluating grid sizes on a synthetic TEST FIXTURE (not real data):\n")
    grid_results = evaluate_grid_sizes(fixture_df, grid_sizes_m=[200.0, 500.0])
    print(grid_results)

    print("\nEvaluating support thresholds:")
    cell_df_fixture = spatial.assign_spatial_cells(fixture_df, cell_size_meters=300.0)
    seqs_fixture = sequential.build_taxi_sequences(cell_df_fixture)
    support_results = evaluate_support_thresholds(seqs_fixture, thresholds=[0.3, 0.5])
    print(support_results)

    print("\nExperiment summary:")
    summary = build_experiment_summary(grid_results, pd.DataFrame(), support_results)
    print(summary.to_string())
