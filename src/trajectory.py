"""
trajectory.py
==============
Trajectory reconstruction and diagnostics for cleaned taxi GPS data.

For each taxi: groups by taxi_id, sorts by timestamp, and derives
per-point fields describing the transition from the previous point:

  * time_diff_sec  — seconds since the previous point (same taxi)
  * distance_m      — great-circle (haversine) distance from previous point, metres
  * speed_kmh        — approximate implied speed (distance / time)
  * is_suspicious_jump — flag for speed_kmh exceeding SUSPICIOUS_SPEED_KMH

Nothing is removed here — flagged points are preserved with a boolean flag,
per the "do not aggressively remove GPS jumps yet" requirement. Later
phases can decide how to use `is_suspicious_jump`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    COL_LAT,
    COL_LON,
    COL_TAXI_ID,
    COL_TIMESTAMP,
    MIN_TRIP_POINTS,
    TRIP_DWELL_TIME_THRESHOLD_SEC,
    TRIP_MAX_TIME_GAP_SEC,
)


def segment_trajectories(
    df: pd.DataFrame,
    dwell_threshold_sec: float = TRIP_DWELL_TIME_THRESHOLD_SEC,
    max_gap_sec: float = TRIP_MAX_TIME_GAP_SEC,
    min_points: int = MIN_TRIP_POINTS,
) -> pd.DataFrame:
    """Segment continuous taxi trajectories into discrete trips/sessions.

    Since T-Drive does not include passenger occupancy flags, trips are
    inferred objectively from physical movement boundaries:
      1. Calendar-day boundaries (preventing cross-day artificial jumps).
      2. Temporal signal loss or long off-duty gaps (time_diff >= max_gap_sec).
      3. Stationary dwell periods (time_diff >= dwell_threshold_sec with distance < 50m).

    Parameters
    ----------
    df : pd.DataFrame
        Trajectory DataFrame (output of build_trajectories or run_cleaning_pipeline).
    dwell_threshold_sec : float
        Stationary duration (seconds) indicating a potential stop/session boundary.
    max_gap_sec : float
        Maximum allowable time gap (seconds) between pings within one trip.
    min_points : int
        Minimum points required for a valid trip segment.

    Returns
    -------
    pd.DataFrame
        DataFrame with an added `trip_id` column (e.g. "1::20080202::T001").
    """
    if "time_diff_sec" not in df.columns or "distance_m" not in df.columns:
        df = build_trajectories(df)
    else:
        df = df.copy()

    # Identify boundary conditions
    taxi_changed = df[COL_TAXI_ID] != df[COL_TAXI_ID].shift(1)
    date_changed = df[COL_TIMESTAMP].dt.date != df[COL_TIMESTAMP].shift(1).dt.date
    large_gap = df["time_diff_sec"] >= max_gap_sec
    dwell_stop = (df["time_diff_sec"] >= dwell_threshold_sec) & (df["distance_m"] < 50.0)

    # First point of dataframe is always a boundary
    new_segment_flag = taxi_changed | date_changed | large_gap | dwell_stop
    new_segment_flag.iloc[0] = True

    # Assign segment index per taxi
    df["_seg_id"] = new_segment_flag.cumsum()
    df["trip_id"] = (
        df[COL_TAXI_ID].astype(str)
        + "::"
        + df[COL_TIMESTAMP].dt.strftime("%Y%m%d")
        + "::T"
        + df["_seg_id"].astype(str)
    )
    df = df.drop(columns=["_seg_id"])

    # Flag incomplete / too-short segments
    trip_counts = df.groupby("trip_id")["trip_id"].transform("count")
    df["is_valid_trip"] = trip_counts >= min_points

    return df


def summarize_trips(segmented_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize segmented trips into an OD journey table with durations.

    Filters to valid trips (is_valid_trip == True).

    Returns
    -------
    pd.DataFrame
        trip_id, taxi_id, date, start_time, end_time, duration_sec,
        start_lat, start_lon, end_lat, end_lon, total_distance_m, n_points.
    """
    if "trip_id" not in segmented_df.columns:
        raise ValueError("segmented_df must have 'trip_id' column from segment_trajectories().")

    valid_df = segmented_df[segmented_df["is_valid_trip"]].copy()
    if len(valid_df) == 0:
        return pd.DataFrame(
            columns=[
                "trip_id", "taxi_id", "date", "start_time", "end_time", "duration_sec",
                "start_lat", "start_lon", "end_lat", "end_lon", "total_distance_m", "n_points",
            ]
        )

    trips = []
    for trip_id, group in valid_df.groupby("trip_id", sort=False):
        group = group.sort_values(COL_TIMESTAMP)
        start_pt = group.iloc[0]
        end_pt = group.iloc[-1]
        duration = (end_pt[COL_TIMESTAMP] - start_pt[COL_TIMESTAMP]).total_seconds()
        dist = group["distance_m"].sum(skipna=True)

        trips.append({
            "trip_id": trip_id,
            "taxi_id": str(start_pt[COL_TAXI_ID]),
            "date": start_pt[COL_TIMESTAMP].date(),
            "start_time": start_pt[COL_TIMESTAMP],
            "end_time": end_pt[COL_TIMESTAMP],
            "duration_sec": max(0.0, float(duration)),
            "start_lat": float(start_pt[COL_LAT]),
            "start_lon": float(start_pt[COL_LON]),
            "end_lat": float(end_pt[COL_LAT]),
            "end_lon": float(end_pt[COL_LON]),
            "total_distance_m": float(dist),
            "n_points": len(group),
        })

    return pd.DataFrame(trips)


# A speed above this is flagged as a suspicious GPS jump. 150 km/h is well
# above realistic urban taxi speeds but avoids over-flagging highway travel.
SUSPICIOUS_SPEED_KMH = 150.0

EARTH_RADIUS_M = 6_371_000  # mean Earth radius in metres


def haversine_distance_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorized haversine distance in metres between two lat/lon arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return EARTH_RADIUS_M * c


def build_trajectories(
    df: pd.DataFrame,
    suspicious_speed_kmh: float = SUSPICIOUS_SPEED_KMH,
) -> pd.DataFrame:
    """Reconstruct per-taxi trajectories with derived transition fields.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned data with at least [taxi_id, timestamp, lat, lon] columns
        (as produced by preprocessing.run_cleaning_pipeline).
    suspicious_speed_kmh : float
        Implied speed above which a transition is flagged as suspicious.

    Returns
    -------
    pd.DataFrame
        Original columns preserved, plus: time_diff_sec, distance_m,
        speed_kmh, is_suspicious_jump. The first point of each taxi has
        NaN for the derived fields (no previous point to compare to).
    """
    df = df.sort_values([COL_TAXI_ID, COL_TIMESTAMP]).reset_index(drop=True).copy()

    grouped = df.groupby(COL_TAXI_ID, sort=False)

    prev_lat = grouped[COL_LAT].shift(1)
    prev_lon = grouped[COL_LON].shift(1)
    prev_ts = grouped[COL_TIMESTAMP].shift(1)

    df["time_diff_sec"] = (df[COL_TIMESTAMP] - prev_ts).dt.total_seconds()
    df["distance_m"] = haversine_distance_m(prev_lat, prev_lon, df[COL_LAT], df[COL_LON])

    # Avoid divide-by-zero: speed is NaN where time_diff is 0 or missing.
    with np.errstate(divide="ignore", invalid="ignore"):
        speed_mps = df["distance_m"] / df["time_diff_sec"]
    df["speed_kmh"] = speed_mps * 3.6
    df.loc[df["time_diff_sec"].isin([0, np.nan]), "speed_kmh"] = np.nan

    df["is_suspicious_jump"] = df["speed_kmh"] > suspicious_speed_kmh
    df["is_suspicious_jump"] = df["is_suspicious_jump"].fillna(False)

    return df


def trajectory_diagnostics(df: pd.DataFrame) -> dict:
    """Compute summary diagnostics across all reconstructed trajectories.

    Returns a dict of scalar/summary statistics suitable for printing or
    logging in the EDA notebook — does not print anything itself.
    """
    per_taxi_counts = df.groupby(COL_TAXI_ID).size()

    diagnostics = {
        "n_taxis": df[COL_TAXI_ID].nunique(),
        "n_points": len(df),
        "points_per_taxi_mean": float(per_taxi_counts.mean()),
        "points_per_taxi_median": float(per_taxi_counts.median()),
        "points_per_taxi_min": int(per_taxi_counts.min()),
        "points_per_taxi_max": int(per_taxi_counts.max()),
        "time_diff_sec_median": float(df["time_diff_sec"].median(skipna=True)),
        "time_diff_sec_mean": float(df["time_diff_sec"].mean(skipna=True)),
        "distance_m_median": float(df["distance_m"].median(skipna=True)),
        "speed_kmh_median": float(df["speed_kmh"].median(skipna=True)),
        "n_suspicious_jumps": int(df["is_suspicious_jump"].sum()),
        "pct_suspicious_jumps": float(100 * df["is_suspicious_jump"].mean()),
    }
    return diagnostics


if __name__ == "__main__":
    # Quick manual sanity check with a tiny synthetic fixture (clearly
    # labelled as a test fixture, not real T-Drive data).
    test_fixture = pd.DataFrame(
        {
            COL_TAXI_ID: ["1", "1", "1", "2", "2"],
            COL_TIMESTAMP: pd.to_datetime(
                [
                    "2008-02-02 15:36:00",
                    "2008-02-02 15:36:30",
                    "2008-02-02 15:37:00",
                    "2008-02-02 15:36:00",
                    "2008-02-02 15:36:05",  # very short gap -> may flag as suspicious
                ]
            ),
            COL_LON: [116.510, 116.512, 116.514, 116.400, 116.500],
            COL_LAT: [39.920, 39.921, 39.922, 39.900, 39.950],
        }
    )
    print("Building trajectories from a synthetic TEST FIXTURE (not real data):\n")
    traj = build_trajectories(test_fixture)
    print(traj)
    print("\nDiagnostics:")
    for k, v in trajectory_diagnostics(traj).items():
        print(f"  {k}: {v}")
