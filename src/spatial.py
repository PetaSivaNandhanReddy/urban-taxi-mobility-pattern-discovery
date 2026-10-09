"""
spatial.py
==========
Grid-based spatial discretization — Phase 2, Part 1 & 2.

WHY DISCRETIZATION IS NECESSARY FOR SEQUENTIAL PATTERN MINING
---------------------------------------------------------------
Sequential pattern mining algorithms (PrefixSpan included) operate over
sequences of discrete, repeatable ITEMS drawn from a finite alphabet — a
pattern like "A -> B -> C" is only meaningful, and support counting only
works, if a taxi actually revisits the *same* item "A" multiple times
across many taxis' sequences. Raw GPS coordinates are continuous and
effectively unique to many decimal places (no two points are ever exactly
equal), so treating raw (lat, lon) pairs as sequence items would mean
every "item" has a support of 1 — no pattern could ever repeat, and mining
would be meaningless. Discretizing space into a finite grid of cells turns
"the taxi was somewhere in this 500m x 500m patch of the city" into a
reusable, countable event, which is what makes frequent-pattern support
counting possible in the first place.

WHAT THIS MODULE DOES
----------------------
1. Builds a configurable square grid (`GRID_CELL_SIZE_METERS` in config.py)
   over the bounding box of the input data.
2. Assigns every GPS observation to a spatial_cell_id, e.g. "CELL_00001".
   Cell IDs are assigned by sorted (row, col) grid position, over only the
   cells that actually contain data — so IDs are compact and deterministic
   (not sparse over the full possible grid).
3. Provides `compress_consecutive_duplicates()` (Part 2): collapses runs of
   the SAME cell repeated back-to-back (e.g. a taxi idling) into one visit,
   while leaving later, non-consecutive repeats of a cell untouched — e.g.
   A, A, A, B, B, C -> A, B, C, but A, B, A stays A, B, A.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from config import COL_LAT, COL_LON, COL_TAXI_ID, COL_TIMESTAMP, GRID_CELL_SIZE_METERS

COL_CELL_ID = "spatial_cell_id"
COL_CELL_ROW = "_cell_row"
COL_CELL_COL = "_cell_col"

EARTH_RADIUS_M = 6_371_000  # mean Earth radius in metres, consistent with trajectory.py


def meters_to_degrees(cell_size_m: float, reference_lat_deg: float) -> tuple[float, float]:
    """Convert a metre-based cell size into (delta_lat_deg, delta_lon_deg).

    A degree of latitude is ~constant everywhere (~111.32 km); a degree of
    longitude shrinks toward the poles by a factor of cos(latitude). Both
    deltas are computed at a single reference latitude (the data's median
    latitude) so that grid cells are approximately square in real-world
    metres across the (small, city-scale) area covered by the dataset.
    """
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(reference_lat_deg))
    if meters_per_deg_lon <= 0:
        raise ValueError(
            f"Invalid reference latitude {reference_lat_deg} — too close to a pole "
            f"for a metre-based longitude grid to be meaningful."
        )
    delta_lat = cell_size_m / meters_per_deg_lat
    delta_lon = cell_size_m / meters_per_deg_lon
    return delta_lat, delta_lon


def assign_spatial_cells(
    df: pd.DataFrame,
    cell_size_meters: float = GRID_CELL_SIZE_METERS,
) -> pd.DataFrame:
    """Assign every row a spatial_cell_id from a configurable square grid.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain COL_LAT and COL_LON columns (cleaned data).
    cell_size_meters : float
        Side length of each grid cell in metres.

    Returns
    -------
    pd.DataFrame
        Original columns preserved, plus `spatial_cell_id` (e.g. "CELL_00001").
        Cell IDs are assigned in sorted (row, col) order over only the grid
        cells that actually contain at least one point — deterministic given
        the same input data and cell size.
    """
    if len(df) == 0:
        raise ValueError("Cannot assign spatial cells to an empty DataFrame.")

    df = df.copy()

    reference_lat = float(df[COL_LAT].median())
    delta_lat, delta_lon = meters_to_degrees(cell_size_meters, reference_lat)

    lat_min = df[COL_LAT].min()
    lon_min = df[COL_LON].min()

    df[COL_CELL_ROW] = np.floor((df[COL_LAT] - lat_min) / delta_lat).astype(int)
    df[COL_CELL_COL] = np.floor((df[COL_LON] - lon_min) / delta_lon).astype(int)

    # Deterministic, compact cell IDs: sort unique (row, col) pairs and
    # number them in order, rather than leaving huge gaps for a sparse grid.
    unique_cells = (
        df[[COL_CELL_ROW, COL_CELL_COL]]
        .drop_duplicates()
        .sort_values([COL_CELL_ROW, COL_CELL_COL])
        .reset_index(drop=True)
    )
    n_digits = max(5, len(str(len(unique_cells))))
    unique_cells[COL_CELL_ID] = [
        f"CELL_{i + 1:0{n_digits}d}" for i in range(len(unique_cells))
    ]

    df = df.merge(unique_cells, on=[COL_CELL_ROW, COL_CELL_COL], how="left")
    df = df.drop(columns=[COL_CELL_ROW, COL_CELL_COL])

    return df


def compress_consecutive_duplicates(sequence: list) -> list:
    """Collapse consecutive repeated items; leave non-consecutive repeats.

    Example
    -------
    >>> compress_consecutive_duplicates(['A','A','A','B','B','C'])
    ['A', 'B', 'C']
    >>> compress_consecutive_duplicates(['A','B','A'])
    ['A', 'B', 'A']
    """
    if not sequence:
        return []
    compressed = [sequence[0]]
    for item in sequence[1:]:
        if item != compressed[-1]:
            compressed.append(item)
    return compressed


def cell_grid_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize point counts per spatial cell, with each cell's centroid.

    Useful for sanity-checking the grid (e.g. spotting a dominant cell that
    suggests too-coarse a resolution) before moving on to sequence building.
    """
    summary = (
        df.groupby(COL_CELL_ID)
        .agg(
            n_points=(COL_CELL_ID, "count"),
            n_taxis=(COL_TAXI_ID, "nunique"),
            centroid_lat=(COL_LAT, "mean"),
            centroid_lon=(COL_LON, "mean"),
        )
        .reset_index()
        .sort_values("n_points", ascending=False)
        .reset_index(drop=True)
    )
    return summary


if __name__ == "__main__":
    # Quick manual sanity check with a tiny synthetic fixture (clearly
    # labelled as a test fixture, not real T-Drive data).
    test_fixture = pd.DataFrame(
        {
            COL_TAXI_ID: ["1", "1", "1", "1", "2", "2"],
            COL_TIMESTAMP: pd.to_datetime(
                [
                    "2008-02-02 15:36:00",
                    "2008-02-02 15:36:30",
                    "2008-02-02 15:40:00",
                    "2008-02-02 15:41:00",
                    "2008-02-02 15:36:00",
                    "2008-02-02 15:37:00",
                ]
            ),
            # Points 1-2 in the same tiny patch, point 3 far away, point 4
            # back near points 1-2 -> expect a non-consecutive repeat.
            COL_LAT: [39.9000, 39.9001, 39.9500, 39.9002, 39.9000, 39.9500],
            COL_LON: [116.4000, 116.4001, 116.4500, 116.4002, 116.4000, 116.4500],
        }
    )
    print("Assigning spatial cells to a synthetic TEST FIXTURE (not real data):\n")
    cell_df = assign_spatial_cells(test_fixture, cell_size_meters=200.0)
    print(cell_df[[COL_TAXI_ID, COL_TIMESTAMP, COL_LAT, COL_LON, COL_CELL_ID]])

    print("\nGrid summary:")
    print(cell_grid_summary(cell_df))

    print("\nCompression check:")
    taxi1_seq = cell_df[cell_df[COL_TAXI_ID] == "1"].sort_values(COL_TIMESTAMP)[COL_CELL_ID].tolist()
    print(f"  Raw sequence:        {taxi1_seq}")
    print(f"  Compressed sequence: {compress_consecutive_duplicates(taxi1_seq)}")
