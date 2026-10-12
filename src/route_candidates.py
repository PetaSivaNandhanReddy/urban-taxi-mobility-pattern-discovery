"""
route_candidates.py
===================
Historical route candidate discovery between spatial grid cells — Phase 5.

Given an origin cell and destination cell, this module:
1. Locates all observed historical trajectory trip segments traversing from origin to destination.
2. Extracts the exact subpath taken (sequence of cells).
3. Verifies physical path continuity using spatial.verify_path_continuity().
4. Computes observed historical journey duration (seconds) from origin entry to destination arrival.
5. Aggregates candidate routes, reporting trip counts, taxi counts, median duration, and variability.
6. Returns an explicit "INSUFFICIENT_EVIDENCE" status when observations fall below configurable thresholds.

Important Rules:
- Distinguishes observed taxi movement from passenger demand or live traffic.
- Validates cell adjacency before calling a pattern a viable physical route.
- Reuses T-Drive data only (Feb 2-8, 2008).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from config import (
    COL_LAT,
    COL_LON,
    COL_TAXI_ID,
    COL_TIMESTAMP,
    MIN_ROUTE_EVIDENCE_TRIPS,
)
from spatial import COL_CELL_ID, build_cell_lookup, decode_cell, verify_path_continuity


def find_od_journeys(
    segmented_df: pd.DataFrame,
    origin_cell: str,
    destination_cell: str,
) -> list[dict[str, Any]]:
    """Extract all individual historical journeys between origin and destination cells.

    Parameters
    ----------
    segmented_df : pd.DataFrame
        DataFrame with at least [taxi_id, timestamp, spatial_cell_id, trip_id].
    origin_cell : str
        Starting spatial grid cell ID (e.g. 'CELL_03758').
    destination_cell : str
        Target spatial grid cell ID (e.g. 'CELL_03757').

    Returns
    -------
    list[dict]
        List of journeys, each containing:
        trip_id, taxi_id, origin_time, dest_time, duration_sec, cell_path.
    """
    if "trip_id" not in segmented_df.columns:
        raise ValueError("segmented_df must have 'trip_id' from trajectory.segment_trajectories().")
    if COL_CELL_ID not in segmented_df.columns:
        raise ValueError(f"segmented_df must have '{COL_CELL_ID}' from spatial.assign_spatial_cells().")

    # Fast pre-filtering: identify only trips that contain BOTH origin and destination cells
    trips_with_orig = set(segmented_df.loc[segmented_df[COL_CELL_ID] == origin_cell, "trip_id"])
    trips_with_dest = set(segmented_df.loc[segmented_df[COL_CELL_ID] == destination_cell, "trip_id"])
    candidate_trip_ids = trips_with_orig.intersection(trips_with_dest)

    if not candidate_trip_ids:
        return []

    subset_df = segmented_df[segmented_df["trip_id"].isin(candidate_trip_ids)]
    journeys = []

    # Group by qualifying trips only
    for trip_id, group in subset_df.groupby("trip_id", sort=False):
        cells = group[COL_CELL_ID].tolist()
        if origin_cell not in cells or destination_cell not in cells:
            continue


        # Find first occurrence of origin and last occurrence of destination after origin
        orig_indices = [i for i, c in enumerate(cells) if c == origin_cell]
        for orig_idx in orig_indices:
            dest_indices = [j for j in range(orig_idx + 1, len(cells)) if cells[j] == destination_cell]
            if not dest_indices:
                continue

            dest_idx = dest_indices[0]  # First arrival at destination after origin
            sub_group = group.iloc[orig_idx : dest_idx + 1]

            t_start = sub_group.iloc[0][COL_TIMESTAMP]
            t_end = sub_group.iloc[-1][COL_TIMESTAMP]
            duration_sec = (t_end - t_start).total_seconds()

            if duration_sec <= 0:
                continue

            sub_path = tuple(sub_group[COL_CELL_ID].tolist())

            journeys.append({
                "trip_id": trip_id,
                "taxi_id": str(sub_group.iloc[0][COL_TAXI_ID]),
                "date": t_start.date(),
                "origin_time": t_start,
                "destination_time": t_end,
                "duration_sec": float(duration_sec),
                "cell_path": sub_path,
                "path_length": len(sub_path),
            })
            break  # Record at most one journey per trip segment

    return journeys


def discover_route_candidates(
    segmented_df: pd.DataFrame,
    origin_cell: str,
    destination_cell: str,
    min_evidence_trips: int = MIN_ROUTE_EVIDENCE_TRIPS,
    cell_lookup: dict[str, dict[str, float]] | None = None,
    max_step_distance_m: float = 1200.0,
) -> dict[str, Any]:
    """Discover and aggregate candidate historical routes between origin and destination cells.

    Parameters
    ----------
    segmented_df : pd.DataFrame
        Cleaned, cell-assigned, and segmented trajectory DataFrame.
    origin_cell : str
        Starting spatial cell.
    destination_cell : str
        Destination spatial cell.
    min_evidence_trips : int
        Minimum supporting trips required to consider a candidate route viable.
    cell_lookup : dict or None
        Lookup from cell_id to centroid coordinates for continuity checking.
    max_step_distance_m : float
        Maximum allowable step distance between consecutive cells in a continuous path.

    Returns
    -------
    dict
        Dictionary containing:
        - status: "SUFFICIENT_EVIDENCE" or "INSUFFICIENT_EVIDENCE"
        - origin_cell: str
        - destination_cell: str
        - total_observed_journeys: int
        - candidate_routes: pd.DataFrame (aggregated routes with durations and evidence)
        - raw_journeys: pd.DataFrame (individual journey records)
    """
    if cell_lookup is None:
        cell_lookup = build_cell_lookup(segmented_df)

    journeys = find_od_journeys(segmented_df, origin_cell, destination_cell)
    journeys_df = pd.DataFrame(journeys)

    if journeys_df.empty:
        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "origin_cell": origin_cell,
            "destination_cell": destination_cell,
            "total_observed_journeys": 0,
            "candidate_routes": pd.DataFrame(),
            "raw_journeys": pd.DataFrame(),
            "message": f"Zero historical journeys found between {origin_cell} and {destination_cell}.",
        }

    # Group journeys by distinct path taken
    path_records = []
    for path_tuple, p_group in journeys_df.groupby("cell_path"):
        n_trips = len(p_group)
        n_taxis = p_group["taxi_id"].nunique()
        durations = p_group["duration_sec"].values

        is_cont, dists = verify_path_continuity(path_tuple, cell_lookup, max_gap_m=max_step_distance_m)

        path_records.append({
            "cell_path": path_tuple,
            "path_str": " -> ".join(path_tuple),
            "path_length": len(path_tuple),
            "is_continuous": is_cont,
            "max_step_gap_m": max(dists) if dists else 0.0,
            "n_trips": n_trips,
            "n_taxis": n_taxis,
            "median_duration_sec": float(np.median(durations)),
            "mean_duration_sec": float(np.mean(durations)),
            "std_duration_sec": float(np.std(durations)) if len(durations) > 1 else 0.0,
            "iqr_duration_sec": float(np.percentile(durations, 75) - np.percentile(durations, 25)) if len(durations) > 1 else 0.0,
            "min_duration_sec": float(np.min(durations)),
            "max_duration_sec": float(np.max(durations)),
        })

    candidates_df = (
        pd.DataFrame(path_records)
        .sort_values(["n_trips", "median_duration_sec"], ascending=[False, True])
        .reset_index(drop=True)
    )

    supported_candidates = candidates_df[candidates_df["n_trips"] >= min_evidence_trips].copy()
    status = "SUFFICIENT_EVIDENCE" if len(supported_candidates) > 0 else "INSUFFICIENT_EVIDENCE"

    return {
        "status": status,
        "origin_cell": origin_cell,
        "destination_cell": destination_cell,
        "total_observed_journeys": len(journeys_df),
        "candidate_routes": supported_candidates.reset_index(drop=True),
        "all_observed_routes": candidates_df,
        "raw_journeys": journeys_df,
        "message": (
            f"Found {len(supported_candidates)} supported route(s) with >= {min_evidence_trips} trips."
            if status == "SUFFICIENT_EVIDENCE"
            else f"Observed {len(journeys_df)} journeys across {len(candidates_df)} distinct path(s), "
                 f"but none meet the minimum evidence threshold of {min_evidence_trips} trips."
        ),
    }


def find_frequent_od_pairs(
    segmented_df: pd.DataFrame,
    min_trips: int = 5,
) -> pd.DataFrame:
    """Discover popular OD spatial cell pairs across all segmented historical trips.

    Helps identify origin-destination pairs with sufficient data for route and departure evaluation.
    """
    if "trip_id" not in segmented_df.columns or COL_CELL_ID not in segmented_df.columns:
        raise ValueError("segmented_df must contain trip_id and spatial_cell_id.")

    valid_pts = segmented_df.dropna(subset=[COL_CELL_ID])
    if valid_pts.empty:
        return pd.DataFrame(columns=["origin_cell", "destination_cell", "trip_count"])

    # Vectorized first and last cell per trip
    first_cells = valid_pts.groupby("trip_id")[COL_CELL_ID].first()
    last_cells = valid_pts.groupby("trip_id")[COL_CELL_ID].last()

    trip_od = pd.DataFrame({
        "origin_cell": first_cells,
        "destination_cell": last_cells,
    })
    trip_od = trip_od[trip_od["origin_cell"] != trip_od["destination_cell"]]

    if trip_od.empty:
        return pd.DataFrame(columns=["origin_cell", "destination_cell", "trip_count"])

    counts = (
        trip_od.groupby(["origin_cell", "destination_cell"])
        .size()
        .reset_index(name="trip_count")
    )
    return (
        counts[counts["trip_count"] >= min_trips]
        .sort_values("trip_count", ascending=False)
        .reset_index(drop=True)
    )

