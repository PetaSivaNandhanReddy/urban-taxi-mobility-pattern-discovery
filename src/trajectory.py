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

from config import COL_LAT, COL_LON, COL_TAXI_ID, COL_TIMESTAMP

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
